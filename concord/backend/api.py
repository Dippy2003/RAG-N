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
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

from action_executor import ExecutionReport, execute_actions
from adjudicator import ReconciliationResult, reconcile
from agent import AgentReport, run_agent
from escalation_reviewer import EscalationReport, run_escalation_review
from identity_agent import IdentityValidationResult, validate_identity
from rag_retriever import format_guidelines_context, retrieve_guidelines

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"

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


class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    source_ref_id: str = ""
    reconciliation_context: dict | None = None   # full ReconcileResponse JSON from frontend


class ChatResponse(BaseModel):
    reply: str
    guidelines_used: list[str] = []


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


_CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_patient_record",
            "description": (
                "Fetch the raw patient record stored under a source_ref_id directly from the database. "
                "Returns name, DOB, NIC, phone, address, medications, allergies, blood type, diagnoses. "
                "Use this for factual questions about a specific patient: "
                "'what is the DOB of CLN-001', 'what medications does PHM-001 have', 'show me the record for LAB-002'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {
                        "type": "string",
                        "description": "The source_ref_id to look up, e.g. CLN-001"
                    }
                },
                "required": ["source_ref_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reconcile_patient",
            "description": (
                "Run a full AI reconciliation for a patient — detects conflicts across clinic/lab/pharmacy, "
                "applies RAG guidelines, resolves or escalates each conflict. "
                "Use this when asked about conflicts, drug interactions, safety, or a full patient summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {
                        "type": "string",
                        "description": "The patient source_ref_id to reconcile, e.g. CLN-001"
                    }
                },
                "required": ["source_ref_id"]
            }
        }
    }
]


def _reconciliation_to_text(report: AgentReport) -> str:
    lines = [
        f"RECONCILIATION RESULT — {report.patient_name} ({report.source_ref_id})",
        f"Cluster: {report.cluster_id}",
        f"Overall safe: {report.overall_safe}",
        f"Summary: {report.summary}",
        f"Turns taken: {report.turns_taken}",
    ]
    if report.conflicts:
        lines.append(f"\nConflicts ({len(report.conflicts)}):")
        for c in report.conflicts:
            lines.append(f"  [{c['conflict_type']}] {c['field']}: {c['source_a']}={c['value_a']} vs {c['source_b']}={c['value_b']}")
            lines.append(f"    {c['description']}")
    if report.resolutions:
        lines.append(f"\nResolutions ({len(report.resolutions)}):")
        for r in report.resolutions:
            lines.append(f"  [{r.field}] {r.action} → {r.chosen_value} (confidence {r.confidence:.0%})")
            lines.append(f"    Rationale: {r.rationale}")
    if report.escalations:
        lines.append(f"\nEscalations ({len(report.escalations)}):")
        for e in report.escalations:
            lines.append(f"  [{e.urgency.upper()}] {e.field}: {e.reason}")
    all_guidelines = list(dict.fromkeys(g for r in report.resolutions for g in r.guidelines_used))
    if all_guidelines:
        lines.append(f"\nGuidelines used: {', '.join(all_guidelines)}")
    return "\n".join(lines)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Conversational endpoint. Answers clinical questions using RAG + optional
    reconcile_patient tool so users can ask about any patient directly in chat.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    guidelines = retrieve_guidelines(req.message, top_k=4, threshold=0.25)
    context_block = format_guidelines_context(guidelines)
    guideline_ids = [g["guideline_id"] for g in guidelines]

    # Build patient context from already-reconciled data if provided
    patient_context = ""
    if req.reconciliation_context:
        rc = req.reconciliation_context
        lines = [f"\nALREADY RECONCILED — {rc.get('patient_name', '')} ({rc.get('source_ref_id', '')})"]
        lines.append(f"Overall safe: {rc.get('overall_safe')}")
        lines.append(f"AI Summary: {rc.get('adjudication_summary', '')}")
        for c in rc.get("conflicts", []):
            lines.append(f"  Conflict [{c.get('conflict_type')}] {c.get('field')}: {c.get('source_a')}={c.get('value_a')} vs {c.get('source_b')}={c.get('value_b')} — {c.get('description')}")
        for r in rc.get("resolutions", []):
            lines.append(f"  Resolution [{r.get('field')}]: {r.get('action')} → {r.get('chosen_value')} — {r.get('rationale')}")
        for e in rc.get("escalations", []):
            lines.append(f"  Escalation [{e.get('urgency','').upper()}] {e.get('field')}: {e.get('reason')}")
        patient_context = "\n".join(lines)

    system_prompt = (
        "You are Concord Assistant, an AI clinical advisor for Sri Lankan healthcare. "
        "You help clinicians understand patient records, conflicts, drug interactions, and clinical guidelines. "
        "Answer clearly and concisely. Flag safety-critical issues urgently."
        f"\n\n{context_block}"
        f"{patient_context}"
    )

    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": system_prompt}]
    for m in req.history[-12:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    # If the message contains a patient ID, pre-fetch the record and inject it
    # so the LLM always has real data without needing to call a tool itself.
    import re as _re
    _id_match = _re.search(r'\b(CLN|LAB|PHM)-\d+\b', req.message, _re.IGNORECASE)
    if _id_match and not patient_context:
        sid = _id_match.group(0).upper()
        try:
            from supabase import create_client as _sc
            _sb = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
            resp = (
                _sb.table("source_records")
                .select("source_ref_id, source, name, dob, nic, phone, address, medications, allergies, blood_type, diagnoses")
                .eq("source_ref_id", sid)
                .execute()
            )
            if resp.data:
                injected = json.dumps(resp.data[0], indent=2, default=str)
                messages.append({
                    "role": "system",
                    "content": f"PATIENT RECORD FOR {sid} (fetched directly from database — use this data to answer):\n{injected}"
                })
        except Exception:
            pass

    # Agentic loop — LLM can still call tools for reconciliation/deeper queries
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=_CHAT_TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as e:
            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                try:
                    response = client.chat.completions.create(
                        model=GROQ_FALLBACK_MODEL,
                        messages=messages,
                        tools=_CHAT_TOOLS,
                        tool_choice="auto",
                        temperature=0.3,
                        max_tokens=1024,
                    )
                except Exception as e2:
                    raise HTTPException(status_code=429, detail=f"Rate limit on both models. Wait ~15 min and try again. ({e2!s})")
            else:
                raise HTTPException(status_code=500, detail=f"LLM error: {e!s}")

        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            return ChatResponse(reply=msg.content or "", guidelines_used=guideline_ids)

        messages.append(msg)
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if tc.function.name == "lookup_patient_record":
                sid = args.get("source_ref_id", "").strip()
                try:
                    from supabase import create_client as _sc
                    _sb = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
                    resp = (
                        _sb.table("source_records")
                        .select("source_ref_id, source, name, dob, nic, phone, address, medications, allergies, blood_type, diagnoses")
                        .eq("source_ref_id", sid)
                        .execute()
                    )
                    if resp.data:
                        tool_result = json.dumps(resp.data[0], indent=2, default=str)
                    else:
                        tool_result = f"No record found for source_ref_id: {sid}"
                except Exception as e:
                    tool_result = f"Lookup failed: {e}"

            elif tc.function.name == "reconcile_patient":
                sid = args.get("source_ref_id", "").strip()
                try:
                    report: AgentReport = run_agent(sid)
                    tool_result = _reconciliation_to_text(report)
                    # Collect any new guideline IDs from the reconciliation
                    for r in report.resolutions:
                        for g in r.guidelines_used:
                            if g not in guideline_ids:
                                guideline_ids.append(g)
                except Exception as e:
                    tool_result = f"Reconciliation failed for {sid}: {e}"
            else:
                tool_result = "Unknown tool."

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    # Fallback — get final answer after tool results
    try:
        final = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return ChatResponse(reply=final.choices[0].message.content or "", guidelines_used=guideline_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e!s}")


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
