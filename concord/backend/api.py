"""
Step 9: FastAPI layer.

Single endpoint: POST /reconcile/{source_ref_id}
Runs the full agentic loop (Steps 6-8) and returns a structured JSON response.

Start the server:
    uv run uvicorn api:app --reload --port 8000

Then call it:
    curl -X POST http://localhost:8000/reconcile/CLN-001
"""

import concurrent.futures

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from action_executor import ExecutionReport, execute_actions
from adjudicator import ReconciliationResult, reconcile
from agent import AgentReport, run_agent
from escalation_reviewer import EscalationReport, run_escalation_review
from identity_agent import IdentityValidationResult, validate_identity

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
    mode: str = "pipeline"
    turns_taken: int | None = None
    guidelines_used: list[str] | None = None


class FieldStatusOut(BaseModel):
    field: str
    provided: str
    stored: str
    match: bool


class IdentityValidationOut(BaseModel):
    given_id: str
    is_correct: bool
    correct_id: str
    confidence: float
    mismatch_fields: list[str]
    field_details: list[FieldStatusOut]
    explanation: str
    patient_name_found: str


class VerifiedReconcileRequest(BaseModel):
    source_ref_id: str
    patient_name: str
    dob: str = ""
    nic: str = ""
    phone: str = ""
    address: str = ""


class VerifiedReconcileResponse(BaseModel):
    identity: IdentityValidationOut
    reconciliation: ReconcileResponse
    id_was_corrected: bool


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    return {"status": "ok", "service": "concord"}


@app.post("/reconcile-agent/{source_ref_id}", response_model=ReconcileResponse)
def reconcile_patient_agent(source_ref_id: str):
    """
    Agentic reconciliation using Gemini function calling + RAG.
    The LLM drives the tool-use loop, retrieves medical guidelines per conflict,
    and decides resolutions dynamically — not a fixed pipeline.
    """
    try:
        report: AgentReport = run_agent(source_ref_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e!s}")

    all_guidelines: list[str] = []
    for res in report.resolutions:
        all_guidelines.extend(res.guidelines_used)

    return ReconcileResponse(
        source_ref_id=source_ref_id,
        patient_name=report.patient_name,
        cluster_id=report.cluster_id,
        conflicts_detected=len(report.conflicts),
        conflicts=[ConflictOut(**c) for c in report.conflicts],
        resolutions=[
            ResolutionOut(
                conflict_type=r.conflict_type,
                field=r.field,
                action=r.action,
                chosen_value=r.chosen_value,
                rationale=r.rationale,
                confidence=r.confidence,
            )
            for r in report.resolutions
        ],
        changes_applied=report.changes_applied,
        escalations=[
            EscalationOut(field=e.field, reason=e.reason, urgency=e.urgency)
            for e in report.escalations
        ],
        overall_safe=report.overall_safe,
        adjudication_summary=report.summary,
        escalation_ids=report.escalation_ids,
        mode="agent",
        turns_taken=report.turns_taken,
        guidelines_used=list(dict.fromkeys(all_guidelines)),
    )


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


def _build_reconcile_response(report: AgentReport) -> ReconcileResponse:
    all_guidelines = [g for r in report.resolutions for g in r.guidelines_used]
    return ReconcileResponse(
        source_ref_id=report.source_ref_id,
        patient_name=report.patient_name,
        cluster_id=report.cluster_id,
        conflicts_detected=len(report.conflicts),
        conflicts=[ConflictOut(**c) for c in report.conflicts],
        resolutions=[
            ResolutionOut(
                conflict_type=r.conflict_type,
                field=r.field,
                action=r.action,
                chosen_value=r.chosen_value,
                rationale=r.rationale,
                confidence=r.confidence,
            )
            for r in report.resolutions
        ],
        changes_applied=report.changes_applied,
        escalations=[
            EscalationOut(field=e.field, reason=e.reason, urgency=e.urgency)
            for e in report.escalations
        ],
        overall_safe=report.overall_safe,
        adjudication_summary=report.summary,
        escalation_ids=report.escalation_ids,
        mode="agent",
        turns_taken=report.turns_taken,
        guidelines_used=list(dict.fromkeys(all_guidelines)),
    )


@app.post("/reconcile-verified", response_model=VerifiedReconcileResponse)
def reconcile_verified(req: VerifiedReconcileRequest):
    """
    Dual-agent endpoint.

    Runs two agents concurrently:
      Agent 1 — Identity Validator: checks if the given source_ref_id actually
                belongs to the patient based on their stated name/dob/nic.
                If wrong, finds the correct ID.
      Agent 2 — Reconciliation Agent: runs RAG + conflict resolution.

    If Agent 1 finds the ID was wrong, Agent 2 is re-run with the correct ID.
    Returns both the identity validation result and the reconciliation result.
    """
    print(f"[api] Starting dual-agent reconciliation for {req.source_ref_id}")

    # Run both agents concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        identity_future = pool.submit(
            validate_identity,
            req.source_ref_id,
            req.patient_name,
            req.dob,
            req.nic,
            req.phone,
            req.address,
        )
        recon_future = pool.submit(run_agent, req.source_ref_id)

        try:
            identity_result: IdentityValidationResult = identity_future.result()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Identity agent failed: {e!s}")

        try:
            recon_report: AgentReport = recon_future.result()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reconciliation agent failed: {e!s}")

    id_was_corrected = not identity_result.is_correct

    # If the ID was wrong, re-run reconciliation with the correct ID
    if id_was_corrected:
        correct_id = identity_result.correct_id
        print(f"[api] ID corrected: {req.source_ref_id} → {correct_id}. Re-running reconciliation.")
        try:
            recon_report = run_agent(correct_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Re-reconciliation failed: {e!s}")

    return VerifiedReconcileResponse(
        identity=IdentityValidationOut(
            given_id=identity_result.given_id,
            is_correct=identity_result.is_correct,
            correct_id=identity_result.correct_id,
            confidence=identity_result.confidence,
            mismatch_fields=identity_result.mismatch_fields,
            field_details=[
                FieldStatusOut(
                    field=f.field,
                    provided=f.provided,
                    stored=f.stored,
                    match=f.match,
                )
                for f in identity_result.field_details
            ],
            explanation=identity_result.explanation,
            patient_name_found=identity_result.patient_name_found,
        ),
        reconciliation=_build_reconcile_response(recon_report),
        id_was_corrected=id_was_corrected,
    )
