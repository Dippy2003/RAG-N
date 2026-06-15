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
import re as _re
import time

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
from rag_retriever import (
    format_guidelines_context, retrieve_guidelines, retrieve_for_chat,
    retrieve_alternatives, retrieve_registration_risks,
)
from router_agent import route
from registration_agent import register_patient_from_details, RegistrationResult
from prescription_agent import process_prescription, PrescriptionResult
from database_agent import process_db_update, DbUpdateResult
from query_agent import run_query, QueryResult
from curator_agent import add_guideline, CuratorResult
from notification_agent import (
    create_notification, get_notifications, mark_read,
    notify_prescription_blocked, notify_escalation,
)

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
    forced_intent: str = ""                       # bypass router: "register"|"prescribe"|"query"|"reconcile"|"db_update"|"chat"


class Citation(BaseModel):
    id: str
    title: str
    severity: str = ""
    category: str = ""
    relevance: float = 0.0   # cosine similarity 0..1


class ChatResponse(BaseModel):
    reply: str
    guidelines_used: list[str] = []
    citations: list[Citation] = []     # rich provenance for retrieved guidelines
    action: str | None = None          # "registered" | "duplicate" | None
    action_data: dict | None = None    # registration result details


def _to_citations(guidelines: list[dict]) -> list[Citation]:
    """Build rich citation objects from retrieved guideline rows."""
    out: list[Citation] = []
    for g in guidelines:
        out.append(Citation(
            id=g.get("guideline_id", ""),
            title=g.get("title", ""),
            severity=g.get("severity", ""),
            category=g.get("category", ""),
            relevance=round(float(g.get("similarity", 0)), 3),
        ))
    return out


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


# ── Reconciliation cache (5-min TTL) ──────────────────────────────────────
_recon_cache: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 300


def _fast_reconcile_text(source_ref_id: str) -> str:
    """
    Pipeline-mode reconciliation (2 LLM calls, not 5-10).
    Uses cache so the same patient isn't re-reconciled within 5 minutes.
    """
    cached = _recon_cache.get(source_ref_id)
    if cached and (time.time() - cached[0]) < _CACHE_TTL:
        return cached[1]

    result: ReconciliationResult = reconcile(source_ref_id)
    execution: ExecutionReport = execute_actions(result)
    escalation: EscalationReport = run_escalation_review(result, execution)

    lines = [
        f"RECONCILIATION — {result.patient_name} ({source_ref_id})",
        f"Overall safe: {escalation.overall_safe}",
        f"Summary: {result.adjudication.summary}",
    ]
    if result.conflicts:
        lines.append(f"\nConflicts ({len(result.conflicts)}):")
        for c in result.conflicts:
            lines.append(f"  [{c['conflict_type']}] {c['field']}: {c['source_a']}={c['value_a']} vs {c['source_b']}={c['value_b']}")
            lines.append(f"    {c['description']}")
    else:
        lines.append("\nNo conflicts found — records are consistent across all sources.")
    for r in result.adjudication.resolutions:
        lines.append(f"  Resolution [{r.field}]: {r.action.value} — {r.rationale}")
    for e in escalation.review.escalations:
        lines.append(f"  Escalation [{e.urgency.upper()}] {e.field}: {e.reason}")

    text = "\n".join(lines)
    _recon_cache[source_ref_id] = (time.time(), text)

    # Auto-notify all sources involved in conflicts
    if result.conflicts:
        # Collect the ACTUAL source_ref_ids from conflict records (e.g. "CLN-001", "LAB-002")
        # source_a / source_b are the real source names like "clinic", "lab", "pharmacy"
        # but we need the actual IDs — start with the queried patient and find linked records
        involved_refs: set[str] = {source_ref_id}

        # Pull all records in the same cluster so we can notify each location
        try:
            from supabase import create_client as _sc
            _sb2 = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
            cluster_resp = _sb2.table("source_records").select("source_ref_id").eq("cluster_id", result.cluster_id).execute()
            for row in (cluster_resp.data or []):
                ref = row.get("source_ref_id", "")
                prefix = ref.split("-")[0].upper()
                if prefix in ("CLN", "LAB", "PHM"):
                    involved_refs.add(ref)
        except Exception:
            pass

        conflict_summary = "; ".join(
            f"{c['field']} ({c['conflict_type']})" for c in result.conflicts[:5]
        )
        urgency = "critical" if not escalation.overall_safe else "medium"

        for notif_ref in involved_refs:
            create_notification(
                source_ref_id=notif_ref,
                patient_name=result.patient_name,
                title=f"Conflict Detected — {len(result.conflicts)} issue(s)",
                message=(
                    f"Patient: {result.patient_name} ({source_ref_id})\n"
                    f"Conflicts: {conflict_summary}\n"
                    f"Overall safe: {escalation.overall_safe}"
                ),
                urgency=urgency,
                notification_type="escalation",
            )

    # Also notify for each escalation
    for e in escalation.review.escalations:
        notify_escalation(
            source_ref_id=source_ref_id,
            patient_name=result.patient_name,
            field=e.field,
            reason=e.reason,
            urgency=e.urgency,
        )

    return text


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Conversational endpoint. Answers clinical questions using RAG + optional
    reconcile_patient tool so users can ask about any patient directly in chat.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    # ── Router: classify intent (or use forced_intent to bypass) ──────────
    _valid_intents = {"register", "update", "db_update", "prescribe", "query", "reconcile", "add_guideline", "chat"}
    _history_dicts = [{"role": m.role, "content": m.content} for m in req.history[-6:]]
    if req.forced_intent and req.forced_intent in _valid_intents:
        intent = req.forced_intent
        # Still run router to extract params (with history for reference resolution)
        route_result = route(req.message, history=_history_dicts)
        params = route_result.get("params", {})
    else:
        route_result = route(req.message, history=_history_dicts)
        intent = route_result.get("intent", "chat")
        params = route_result.get("params", {})

    if intent == "add_guideline":
        rule_text = (params.get("text") or req.message).strip()
        try:
            cur: CuratorResult = add_guideline(rule_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Curator failed: {e!s}")

        if cur.action == "added":
            reply = (
                f"**Guideline added to the knowledge base.**\n\n"
                f"**ID:** {cur.guideline_id}\n"
                f"**Title:** {cur.title}\n"
                f"**Category:** {cur.category.replace('_', ' ')}\n"
                f"**Severity:** {cur.severity.upper()}\n\n"
                f"{cur.content}\n\n"
                f"_This guideline is now live — the prescription, reconciliation, and "
                f"chat agents will retrieve it automatically in future safety checks._"
            )
            return ChatResponse(
                reply=reply,
                action="guideline_added",
                action_data={"guideline_id": cur.guideline_id, "patient_name": cur.title},
                citations=[Citation(
                    id=cur.guideline_id, title=cur.title,
                    severity=cur.severity, category=cur.category, relevance=1.0,
                )],
            )

        if cur.action == "duplicate":
            reply = (
                f"**Already covered.**\n\n{cur.message}\n\n"
                f"If you want to change the existing rule, edit guideline "
                f"**{cur.duplicate_of}** instead of adding a new one."
            )
            return ChatResponse(
                reply=reply,
                action="guideline_duplicate",
                action_data={"guideline_id": cur.duplicate_of, "patient_name": cur.title},
                citations=[Citation(
                    id=cur.duplicate_of, title=cur.title,
                    severity=cur.severity, category=cur.category,
                    relevance=cur.duplicate_similarity,
                )],
            )

        raise HTTPException(status_code=500, detail=cur.message)

    if intent in ("register", "update"):
        # For update: find the patient by name if no source_ref_id given
        if intent == "update" and not params.get("source_ref_id") and params.get("name"):
            from supabase import create_client as _sc
            _sb = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
            found = _sb.table("source_records").select("source_ref_id, name").ilike("name", f"%{params['name']}%").limit(1).execute()
            if found.data:
                params["source_ref_id"] = found.data[0]["source_ref_id"]

        try:
            reg: RegistrationResult = register_patient_from_details(params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registration failed: {e!s}")

        if reg.action == "registered":
            # RAG: check the new patient's medications + allergies for known risks
            risk_guidelines: list[dict] = []
            try:
                _meds     = params.get("medications") or []
                _allergies_reg = params.get("allergies") or []
                _blood    = params.get("blood_type") or ""
                if _meds or _allergies_reg:
                    risk_guidelines = retrieve_registration_risks(_meds, _allergies_reg, _blood, top_k=4)
            except Exception:
                pass

            # Build risk block
            risk_block = ""
            risk_citations: list[dict] = []
            if risk_guidelines:
                risk_lines = []
                for g in risk_guidelines:
                    risk_lines.append(f"- **[{g['guideline_id']}]** [{g['severity'].upper()}] {g['title']}")
                risk_block = "\n\n⚠️ **Risk flags detected at registration:**\n" + "\n".join(risk_lines)
                risk_citations = risk_guidelines

                # Auto-create a notification for each critical/high risk
                for g in risk_guidelines:
                    if g.get("severity") in ("critical", "high"):
                        create_notification(
                            source_ref_id=reg.source_ref_id,
                            patient_name=reg.patient_name,
                            title=f"Registration Risk — {g['title']}",
                            message=g.get("content", "")[:300],
                            urgency=g["severity"],
                            notification_type="registration",
                        )

            reply = (
                f"Patient registered successfully!\n\n"
                f"**Name:** {reg.patient_name}\n"
                f"**New ID:** {reg.source_ref_id}\n\n"
                f"You can now ask about this patient using their ID."
                f"{risk_block}"
            )
            return ChatResponse(
                reply=reply,
                action="registered",
                action_data={"source_ref_id": reg.source_ref_id, "patient_name": reg.patient_name},
                citations=_to_citations(risk_citations),
            )

        if reg.action == "updated":
            _recon_cache.pop(reg.source_ref_id, None)  # invalidate stale cache
            reply = f"Record updated successfully for **{reg.patient_name}** ({reg.source_ref_id})."
            return ChatResponse(reply=reply, action="updated", action_data={"source_ref_id": reg.source_ref_id, "patient_name": reg.patient_name})

        if reg.action == "duplicate":
            reply = (
                f"A patient with these details already exists.\n"
                f"**Existing ID:** {reg.existing_id}\n"
                f"No duplicate was created. If you want to update that record, say \"update {reg.existing_id}\"."
            )
            return ChatResponse(reply=reply, action="duplicate", action_data={"existing_id": reg.existing_id, "patient_name": reg.patient_name})

        raise HTTPException(status_code=500, detail=reg.message)

    if intent == "prescribe":
        sid = params.get("source_ref_id", "").strip()
        drug = params.get("drug", "").strip()
        if not sid or not drug:
            return ChatResponse(reply="Please specify the patient ID and the drug name. Example: 'Prescribe aspirin 100mg daily for CLN-001'")
        try:
            rx: PrescriptionResult = process_prescription(
                source_ref_id=sid,
                drug=drug,
                dosage=params.get("dosage", ""),
                notes=params.get("notes", ""),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prescription failed: {e!s}")

        if rx.action == "issued":
            _recon_cache.pop(rx.source_ref_id, None)  # medications changed — invalidate cache
            # Auto-notify (low urgency — successful prescription)
            create_notification(
                source_ref_id=rx.source_ref_id, patient_name=rx.patient_name,
                title=f"Prescription Issued — {rx.drug}",
                message=rx.reason, urgency="low", notification_type="prescription_issued",
            )
            reply = f"Prescription issued successfully.\n\n**Drug:** {rx.drug}\n**Patient:** {rx.patient_name} ({rx.source_ref_id})\n\n{rx.reason}"
            return ChatResponse(reply=reply, action="prescription_issued",
                                action_data={"source_ref_id": rx.source_ref_id, "patient_name": rx.patient_name, "drug": rx.drug})

        if rx.action == "blocked":
            notify_prescription_blocked(
                source_ref_id=rx.source_ref_id, patient_name=rx.patient_name,
                drug=rx.drug, reason=rx.reason, interactions=rx.interactions_found,
            )

            # RAG: find safe alternative drugs for this patient
            try:
                from supabase import create_client as _sc
                _sb_rx = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
                _pt = _sb_rx.table("source_records").select("medications, allergies").eq("source_ref_id", rx.source_ref_id).execute()
                _existing_meds = (_pt.data[0].get("medications") or []) if _pt.data else []
                _allergies     = (_pt.data[0].get("allergies")   or []) if _pt.data else []
                alt_guidelines = retrieve_alternatives(rx.drug, _existing_meds, _allergies, top_k=3)
            except Exception:
                alt_guidelines = []

            # Build alternatives section
            alt_lines = []
            for g in alt_guidelines:
                alt_lines.append(f"- **[{g['guideline_id']}]** {g['title']} _{g.get('severity','').upper()}_")
            alt_block = (
                "\n\n**Suggested alternatives (from knowledge base):**\n" + "\n".join(alt_lines)
                if alt_lines else ""
            )

            # Single LLM call to turn RAG results into a concrete recommendation
            alt_reply_extra = ""
            if alt_guidelines:
                try:
                    _alt_context = format_guidelines_context(alt_guidelines)
                    _client = Groq(api_key=GROQ_API_KEY)
                    _alt_resp = _client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": (
                                "You are a clinical pharmacist. A prescription was BLOCKED. "
                                "Based ONLY on the guidelines provided, suggest 1-3 concrete safe alternatives "
                                "for the blocked drug. Be specific: name the drug, typical dose, and why it's safer. "
                                "Keep it under 80 words. Do not invent drugs not mentioned in the guidelines.\n\n"
                                + _alt_context
                            )},
                            {"role": "user", "content": f"Blocked drug: {rx.drug}. Reason: {rx.reason}. Suggest alternatives."},
                        ],
                        temperature=0.2, max_tokens=200,
                    )
                    alt_reply_extra = "\n\n**AI Recommendation:**\n" + (_alt_resp.choices[0].message.content or "")
                except Exception:
                    pass

            reply = (
                f"Prescription **BLOCKED** — {rx.drug} cannot be safely prescribed.\n\n"
                f"**Reason:** {rx.reason}\n"
                f"**Interactions:** {', '.join(rx.interactions_found) if rx.interactions_found else 'allergy conflict'}\n\n"
                f"A notification has been sent to the clinical team."
                f"{alt_block}"
                f"{alt_reply_extra}"
            )
            return ChatResponse(
                reply=reply,
                action="prescription_blocked",
                action_data={"source_ref_id": rx.source_ref_id, "patient_name": rx.patient_name, "drug": rx.drug},
                citations=_to_citations(alt_guidelines),
            )

        raise HTTPException(status_code=500, detail=rx.reason)

    if intent == "db_update" and params.get("operation") == "rename_id":
        old_id = (params.get("source_ref_id") or "").strip().upper()
        new_id = (params.get("new_id") or "").strip().upper()
        if not old_id or not new_id:
            return ChatResponse(reply="Please provide both the current ID and the new ID. Example: 'rename CLN-004 to CLN-006'")
        if old_id == new_id:
            return ChatResponse(reply=f"The ID is already {old_id}. No change needed.")
        try:
            from supabase import create_client as _sc
            _sb = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
            # Check old ID exists
            existing = _sb.table("source_records").select("source_ref_id, name").eq("source_ref_id", old_id).execute()
            if not existing.data:
                return ChatResponse(reply=f"No patient found with ID **{old_id}**. Please check the ID and try again.")
            patient_name = existing.data[0].get("name", "")
            # Check new ID not already taken
            conflict = _sb.table("source_records").select("source_ref_id").eq("source_ref_id", new_id).execute()
            if conflict.data:
                return ChatResponse(reply=f"ID **{new_id}** is already in use by another patient. Choose a different ID.")
            # Cascade rename across all tables
            _sb.table("source_records").update({"source_ref_id": new_id}).eq("source_ref_id", old_id).execute()
            for tbl in ("prescriptions", "escalations", "notifications"):
                try:
                    _sb.table(tbl).update({"source_ref_id": new_id}).eq("source_ref_id", old_id).execute()
                except Exception:
                    pass  # table may not have the column or may not exist
            _recon_cache.pop(old_id, None)
            _recon_cache.pop(new_id, None)
            return ChatResponse(
                reply=f"Patient ID renamed successfully.\n\n**{old_id}** → **{new_id}**\n**Patient:** {patient_name}\n\nAll prescriptions, escalations, and notifications have been updated to the new ID.",
                action="db_updated",
                action_data={"source_ref_id": new_id, "patient_name": patient_name},
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Rename failed: {e!s}")

    if intent == "db_update":
        try:
            db: DbUpdateResult = process_db_update(params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database update failed: {e!s}")

        if db.action in ("db_updated", "db_deleted"):
            _recon_cache.pop(db.record_id, None)
            fields_str = ", ".join(db.updated_fields) if db.updated_fields else "record"
            verb = "deleted" if db.action == "db_deleted" else "updated"
            reply = (
                f"Database {verb} successfully.\n\n"
                f"**Table:** {db.table}\n"
                f"**Record:** {db.record_id}\n\n"
                f"{db.message}"
            )
            return ChatResponse(
                reply=reply,
                action=db.action,
                action_data={"source_ref_id": db.record_id, "table": db.table, "fields": fields_str},
            )

        raise HTTPException(status_code=500, detail=db.message)

    if intent == "query":
        try:
            qr: QueryResult = run_query(params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query failed: {e!s}")

        if not qr.success:
            return ChatResponse(reply=qr.message)

        # Format rows as a readable table for the LLM to present
        rows_text = json.dumps(qr.rows[:30], indent=2, default=str)
        query_context = (
            f"QUERY RESULT — {qr.query_type.replace('_', ' ').upper()}\n"
            f"{qr.message}\n\n"
            f"Data:\n{rows_text}"
        )

        # Single LLM call to present the results nicely
        client = Groq(api_key=GROQ_API_KEY)
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are Concord Assistant. Present the following database query results clearly and concisely. "
                        "Use a markdown table if there are multiple rows. Highlight important values like blocked prescriptions or unresolved escalations."
                    )},
                    {"role": "system", "content": query_context},
                    {"role": "user", "content": req.message},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
        except Exception as e:
            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                resp = client.chat.completions.create(
                    model=GROQ_FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": "Present these query results clearly."},
                        {"role": "system", "content": query_context},
                        {"role": "user", "content": req.message},
                    ],
                    temperature=0.2, max_tokens=1024,
                )
            else:
                raise HTTPException(status_code=500, detail=f"LLM error: {e!s}")

        return ChatResponse(
            reply=resp.choices[0].message.content or qr.message,
            action="query_result",
            action_data={"query_type": qr.query_type, "total": str(qr.total)},
        )

    guidelines = retrieve_for_chat(req.message, top_k=4)
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
        "Answer clearly and concisely. Flag safety-critical issues urgently.\n"
        "When a retrieved guideline below informs your answer, cite its ID inline in square "
        "brackets, e.g. [LK-003] or [DI-001], immediately after the relevant claim. "
        "Only cite IDs that appear in the guidelines provided. Do not invent guideline IDs."
        f"\n\n{context_block}"
        f"{patient_context}"
    )

    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": system_prompt}]
    for m in req.history[-12:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    # Extract ALL patient IDs mentioned in the message (supports multi-patient queries)
    _id_matches = list(dict.fromkeys(
        m.upper() for m in _re.findall(r'\b(CLN|LAB|PHM)-\d+\b', req.message, _re.IGNORECASE)
    ))
    _conflict_keywords = {"conflict", "conflicts", "interaction", "drug", "safe", "safety",
                          "escalat", "reconcil", "medic", "allerg", "risk", "danger"}
    _msg_lower = req.message.lower()
    _is_conflict_query = any(k in _msg_lower for k in _conflict_keywords)
    _is_compare = "compare" in _msg_lower or "vs" in _msg_lower or "versus" in _msg_lower or "difference" in _msg_lower

    # ── Compare path: 2+ IDs + compare keyword → fetch all records side-by-side ──
    if _id_matches and len(_id_matches) >= 2 and _is_compare and not patient_context:
        from supabase import create_client as _sc
        _sb_cmp = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        _cols = "source_ref_id, source, name, dob, nic, phone, blood_type, medications, allergies"
        for sid in _id_matches:
            try:
                _r = _sb_cmp.table("source_records").select(_cols).eq("source_ref_id", sid).execute()
                if _r.data:
                    messages.append({
                        "role": "system",
                        "content": f"PATIENT RECORD — {sid}:\n{json.dumps(_r.data[0], indent=2, default=str)}",
                    })
                else:
                    messages.append({
                        "role": "system",
                        "content": f"PATIENT RECORD — {sid}: NOT FOUND in database.",
                    })
            except Exception as _e:
                messages.append({
                    "role": "system",
                    "content": f"PATIENT RECORD — {sid}: fetch failed ({_e}).",
                })
        # Also pull RAG guidelines for any shared medications/allergies
        try:
            _all_meds: list[str] = []
            for _msg_sys in messages:
                if "PATIENT RECORD" in _msg_sys.get("content", ""):
                    import re as _re2
                    _all_meds += _re2.findall(r'"medications": \[([^\]]*)\]', _msg_sys["content"])
            if _all_meds:
                _compare_guidelines = retrieve_for_chat(f"compare patients medications allergies {' '.join(_all_meds)}", top_k=3)
                if _compare_guidelines:
                    messages.append({"role": "system", "content": format_guidelines_context(_compare_guidelines)})
        except Exception:
            pass

    elif _id_matches and not patient_context:
        from supabase import create_client as _sc
        _sb = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

        for sid in _id_matches:
            if _is_conflict_query:
                # Fast path: inject reconciliation result
                try:
                    recon_text = _fast_reconcile_text(sid)
                    messages.append({
                        "role": "system",
                        "content": f"RECONCILIATION RESULT FOR {sid}:\n{recon_text}",
                    })
                    continue
                except Exception:
                    pass  # fall through to raw record below

            # Raw record inject (info query or reconcile failed)
            try:
                resp = (
                    _sb.table("source_records")
                    .select("source_ref_id, source, name, dob, nic, phone, address, medications, allergies, blood_type")
                    .eq("source_ref_id", sid)
                    .execute()
                )
                if resp.data:
                    note = ""
                    if _is_conflict_query:
                        note = "\nNote: This patient was recently added — no cross-source conflict analysis available yet."
                    messages.append({
                        "role": "system",
                        "content": f"PATIENT RECORD FOR {sid}:\n{json.dumps(resp.data[0], indent=2, default=str)}{note}",
                    })
            except Exception:
                pass

    # Single LLM call — no tool loop needed since data is already injected
    def _llm_call(model: str):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )

    try:
        response = _llm_call(GROQ_MODEL)
    except Exception as e:
        if "rate_limit_exceeded" in str(e) or "429" in str(e):
            try:
                response = _llm_call(GROQ_FALLBACK_MODEL)
            except Exception as e2:
                raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Wait ~15 min. ({e2!s})")
        else:
            raise HTTPException(status_code=500, detail=f"LLM error: {e!s}")

    return ChatResponse(
        reply=response.choices[0].message.content or "",
        guidelines_used=guideline_ids,
        citations=_to_citations(guidelines),
    )


@app.get("/guideline/{guideline_id}")
def get_guideline(guideline_id: str):
    """Fetch one clinical guideline by its ID (for citation pop-overs)."""
    try:
        from supabase import create_client as _sc
        sb = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        resp = (
            sb.table("medical_guidelines")
            .select("guideline_id, category, title, content, severity, tags")
            .eq("guideline_id", guideline_id.upper())
            .execute()
        )
        if not resp.data:
            raise HTTPException(status_code=404, detail=f"Guideline {guideline_id} not found.")
        return resp.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/notifications")
def list_notifications(unread_only: bool = False, limit: int = 20, source: str = ""):
    """
    Fetch notifications. Optionally filter by source prefix.
    source = "CLN" | "LAB" | "PHM" | "" (all)
    """
    notifications = get_notifications(unread_only=unread_only, limit=limit)
    if source:
        prefix = source.upper()
        notifications = [n for n in notifications if n.get("source_ref_id", "").upper().startswith(prefix)]
    return notifications


@app.post("/notifications/read")
def read_notifications(ids: list[str]):
    mark_read(ids)
    return {"ok": True}


@app.delete("/notifications/{notification_id}")
def delete_notification(notification_id: str):
    """Delete a single notification by ID."""
    try:
        from supabase import create_client as _sc
        sb = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        sb.table("notifications").delete().eq("id", notification_id).execute()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/notifications")
def clear_notifications(source: str = ""):
    """Delete all notifications, optionally filtered by source prefix (CLN/LAB/PHM)."""
    try:
        from supabase import create_client as _sc
        sb = _sc(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
        query = sb.table("notifications").delete()
        if source:
            query = query.like("source_ref_id", f"{source.upper()}-%")
        else:
            query = query.neq("id", "00000000-0000-0000-0000-000000000000")  # delete all
        query.execute()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

    # Auto-create notifications for escalations
    for e in recon_report.escalations:
        notify_escalation(
            source_ref_id=recon_report.source_ref_id,
            patient_name=recon_report.patient_name,
            field=e.field,
            reason=e.reason,
            urgency=e.urgency,
        )

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
