"""
Prescription Agent — issues a new prescription for a patient.

Before saving, it automatically checks the new drug against:
  - the patient's existing medications (drug-drug interactions)
  - the patient's known allergies

Tools:
  get_patient_record     — fetch current meds, allergies, blood type
  check_interaction      — RAG search for interactions with existing meds
  issue_prescription     — add drug to patient record, log in prescriptions table
  block_prescription     — block with reason if dangerous
"""

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TURNS = 8


@dataclass
class PrescriptionResult:
    success: bool
    action: str           # "issued" | "blocked" | "error"
    source_ref_id: str
    patient_name: str
    drug: str
    reason: str
    interactions_found: list[str] = field(default_factory=list)
    guidelines_used: list[str] = field(default_factory=list)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_patient_record",
            "description": "Fetch a patient's current medications, allergies, and basic info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string"}
                },
                "required": ["source_ref_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_interaction",
            "description": (
                "Check whether a new drug is safe to prescribe given the patient's "
                "existing medications and allergies. Uses clinical guidelines (RAG). "
                "Returns any known interactions and their severity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "new_drug": {"type": "string", "description": "Drug being prescribed"},
                    "existing_medications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Patient's current medications",
                    },
                    "allergies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Patient's known allergies",
                    },
                },
                "required": ["new_drug", "existing_medications", "allergies"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_prescription",
            "description": "Add the drug to the patient's medications and log the prescription. Only call if interactions are safe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string"},
                    "drug": {"type": "string"},
                    "dosage": {"type": "string", "description": "e.g. '500mg twice daily'"},
                    "notes": {"type": "string"},
                },
                "required": ["source_ref_id", "drug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "block_prescription",
            "description": "Block the prescription due to a dangerous interaction or allergy. Record the reason.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string"},
                    "drug": {"type": "string"},
                    "reason": {"type": "string"},
                    "interactions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific interactions found",
                    },
                },
                "required": ["source_ref_id", "drug", "reason", "interactions"],
            },
        },
    },
]


class PrescriptionToolExecutor:
    def __init__(self):
        self._result: PrescriptionResult | None = None
        self._done = False
        self._patient_name = ""

    def execute(self, tool_name: str, args: dict) -> str:
        handlers = {
            "get_patient_record": self._get_patient_record,
            "check_interaction":  self._check_interaction,
            "issue_prescription": self._issue_prescription,
            "block_prescription": self._block_prescription,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        try:
            return handler(**args)
        except Exception as e:
            return f"Tool error: {e}"

    def _get_patient_record(self, source_ref_id: str) -> str:
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        resp = (
            sb.table("source_records")
            .select("source_ref_id, name, medications, allergies, blood_type, dob")
            .eq("source_ref_id", source_ref_id)
            .execute()
        )
        if not resp.data:
            self._result = PrescriptionResult(
                success=False, action="error",
                source_ref_id=source_ref_id, patient_name="", drug="",
                reason=f"Patient {source_ref_id} not found — prescription aborted.",
            )
            self._done = True
            return json.dumps({"error": f"Patient {source_ref_id} not found. Prescription aborted."})
        r = resp.data[0]
        self._patient_name = r.get("name", "")
        return json.dumps({
            "source_ref_id": r["source_ref_id"],
            "name": r.get("name"),
            "medications": r.get("medications") or [],
            "allergies": r.get("allergies") or [],
            "blood_type": r.get("blood_type"),
        }, indent=2)

    def _check_interaction(self, new_drug: str, existing_medications: list, allergies: list) -> str:
        from rag_retriever import retrieve_for_prescription, format_guidelines_context
        guidelines = retrieve_for_prescription(new_drug, existing_medications, allergies, top_k=6)

        # Allergy name match (direct name check)
        drug_lower = new_drug.lower()
        allergy_hit = [a for a in allergies if a.lower() in drug_lower or drug_lower in a.lower()]

        # Find critical/high-severity guidelines that mention the new drug AND an existing med or allergy
        interaction_guidelines = []
        for g in guidelines:
            text = (g.get("title", "") + " " + g.get("content", "") + " " + (g.get("tags") or "")).lower()
            # Check if this guideline is relevant to new drug
            drug_words = [w for w in drug_lower.split() if len(w) > 3]
            drug_mentioned = any(w in text for w in drug_words) or drug_lower in text
            if not drug_mentioned:
                continue
            # Check if it also mentions an existing med or allergy
            med_mentioned = any(
                any(w in text for w in med.lower().split() if len(w) > 3)
                for med in existing_medications
            )
            allergy_mentioned = any(a.lower() in text for a in allergies)
            if med_mentioned or allergy_mentioned or g.get("severity") == "critical":
                interaction_guidelines.append({
                    "guideline_id": g["guideline_id"],
                    "title": g["title"],
                    "severity": g.get("severity", "unknown"),
                    "summary": g.get("content", "")[:300],
                })

        critical_found = any(g["severity"] == "critical" for g in interaction_guidelines)
        safe = len(interaction_guidelines) == 0 and len(allergy_hit) == 0

        return json.dumps({
            "new_drug": new_drug,
            "existing_medications": existing_medications,
            "allergy_conflicts": allergy_hit,
            "interaction_guidelines": interaction_guidelines,
            "safe_to_prescribe": safe,
            "has_critical_interaction": critical_found,
            "rag_context": format_guidelines_context(guidelines),
        }, indent=2)

    def _issue_prescription(self, source_ref_id: str, drug: str, dosage: str = "", notes: str = "") -> str:
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

        # Fetch current medications
        resp = sb.table("source_records").select("medications, name").eq("source_ref_id", source_ref_id).execute()
        if not resp.data:
            return json.dumps({"error": "Patient not found"})

        current_meds = resp.data[0].get("medications") or []
        patient_name = resp.data[0].get("name", "")
        drug_entry = f"{drug} {dosage}".strip()

        # Dedup check: skip if drug name already present (case-insensitive, partial match)
        already_present = any(drug.lower() in str(m).lower() for m in current_meds)
        if not already_present:
            current_meds.append(drug_entry)
            sb.table("source_records").update({"medications": current_meds}).eq("source_ref_id", source_ref_id).execute()

        # Log in prescriptions table (create if needed)
        try:
            sb.table("prescriptions").insert({
                "source_ref_id": source_ref_id,
                "drug": drug,
                "dosage": dosage,
                "notes": notes,
                "status": "active",
            }).execute()
        except Exception:
            pass  # Table may not exist yet — medications array was already updated

        self._result = PrescriptionResult(
            success=True,
            action="issued",
            source_ref_id=source_ref_id,
            patient_name=patient_name or self._patient_name,
            drug=drug,
            reason=f"{drug} prescribed successfully. Added to patient's medication list.",
        )
        self._done = True
        return json.dumps({"success": True, "drug": drug, "added_to_record": True})

    def _block_prescription(self, source_ref_id: str, drug: str, reason: str, interactions: list) -> str:
        # Log the blocked attempt
        try:
            sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            sb.table("prescriptions").insert({
                "source_ref_id": source_ref_id,
                "drug": drug,
                "dosage": "",
                "notes": reason,
                "status": "blocked",
            }).execute()
        except Exception:
            pass

        self._result = PrescriptionResult(
            success=False,
            action="blocked",
            source_ref_id=source_ref_id,
            patient_name=self._patient_name,
            drug=drug,
            reason=reason,
            interactions_found=interactions,
        )
        self._done = True
        return json.dumps({"blocked": True, "reason": reason})


_SYSTEM_PROMPT = """You are the Prescription Agent for Concord, a Sri Lankan clinical record system.

Your job: safely issue or block a prescription.

Workflow:
1. Call get_patient_record to fetch current medications and allergies.
2. Call check_interaction with the new drug + existing meds + allergies.
3. If safe_to_prescribe is true → call issue_prescription.
4. If safe_to_prescribe is false OR interaction guidelines found → call block_prescription with a clear reason.

Never skip the interaction check. Patient safety is the priority."""


def process_prescription(source_ref_id: str, drug: str, dosage: str = "", notes: str = "") -> PrescriptionResult:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=GROQ_API_KEY)
    executor = PrescriptionToolExecutor()

    user_msg = (
        f"Issue a prescription for patient {source_ref_id}.\n"
        f"Drug: {drug}\n"
        f"Dosage: {dosage or 'standard'}\n"
        f"Notes: {notes or 'none'}\n\n"
        f"Check for interactions with existing medications first."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    turns = 0
    while turns < MAX_TURNS and not executor._done:
        turns += 1
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.1,
            )
        except Exception as e:
            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                response = client.chat.completions.create(
                    model=GROQ_FALLBACK_MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.1,
                )
            else:
                raise

        msg = response.choices[0].message
        messages.append(msg)
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            break

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            print(f"[prescription-agent] Turn {turns}: {tc.function.name}({list(args.keys())})")
            result = executor.execute(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    if executor._result is None:
        action = "timeout" if turns >= MAX_TURNS else "error"
        reason = (
            "Prescription check timed out (too many reasoning steps). Please retry."
            if action == "timeout"
            else "Could not complete prescription check."
        )
        print(f"[prescription-agent] {action.upper()}: {reason}")
        return PrescriptionResult(
            success=False, action=action, source_ref_id=source_ref_id,
            patient_name="", drug=drug, reason=reason,
        )

    return executor._result
