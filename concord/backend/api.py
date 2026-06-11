"""
Step 9: FastAPI layer.

Single endpoint: POST /reconcile/{source_ref_id}
Runs the full agentic loop (Steps 6-8) and returns a structured JSON response.

Start the server:
    uv run uvicorn api:app --reload --port 8000

Then call it:
    curl -X POST http://localhost:8000/reconcile/CLN-001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from action_executor import ExecutionReport, execute_actions
from adjudicator import ReconciliationResult, reconcile
from escalation_reviewer import EscalationReport, run_escalation_review

app = FastAPI(
    title="Concord",
    description="Autonomous clinical record reconciliation for Sri Lankan patients",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Response schemas
# ------------------------------------------------------------------ #

class ConflictOut(BaseModel):
    conflict_type: str
    field: str
    source_a: str
    value_a: str
    source_b: str
    value_b: str
    description: str


class ResolutionOut(BaseModel):
    conflict_type: str
    field: str
    action: str
    chosen_value: str | None
    rationale: str
    confidence: float


class EscalationOut(BaseModel):
    field: str
    reason: str
    urgency: str


class ReconcileResponse(BaseModel):
    source_ref_id: str
    patient_name: str
    cluster_id: str
    conflicts_detected: int
    conflicts: list[ConflictOut]
    resolutions: list[ResolutionOut]
    changes_applied: int
    escalations: list[EscalationOut]
    overall_safe: bool
    adjudication_summary: str
    escalation_ids: list[str]


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    return {"status": "ok", "service": "concord"}


@app.post("/reconcile/{source_ref_id}", response_model=ReconcileResponse)
def reconcile_patient(source_ref_id: str):
    """
    Runs the full Concord agentic loop for one patient:
      1. Identity match across clinic / lab / pharmacy
      2. Deterministic conflict detection
      3. LLM Call 1 — batch adjudication
      4. Action execution
      5. LLM Call 2 — escalation review

    source_ref_id can be from any source: CLN-001, LAB-001, PHM-001, etc.
    """
    try:
        result: ReconciliationResult = reconcile(source_ref_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconciliation failed: {e}")

    try:
        execution: ExecutionReport = execute_actions(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action execution failed: {e}")

    try:
        escalation: EscalationReport = run_escalation_review(result, execution)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Escalation review failed: {e}")

    return ReconcileResponse(
        source_ref_id=source_ref_id,
        patient_name=result.patient_name,
        cluster_id=result.cluster_id,
        conflicts_detected=len(result.conflicts),
        conflicts=[ConflictOut(**c) for c in result.conflicts],
        resolutions=[
            ResolutionOut(
                conflict_type=r.conflict_type,
                field=r.field,
                action=r.action.value,
                chosen_value=r.chosen_value,
                rationale=r.rationale,
                confidence=r.confidence,
            )
            for r in result.adjudication.resolutions
        ],
        changes_applied=len(execution.applied),
        escalations=[
            EscalationOut(
                field=e.field,
                reason=e.reason,
                urgency=e.urgency,
            )
            for e in escalation.review.escalations
        ],
        overall_safe=escalation.overall_safe,
        adjudication_summary=result.adjudication.summary,
        escalation_ids=escalation.escalation_ids,
    )
