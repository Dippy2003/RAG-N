"""
Database Agent — lets clinicians update any table in Supabase via chat.

Supported operations:
  source_records   — update patient demographics, medications, allergies, blood type
  patient_clusters — update canonical name/dob/nic
  escalations      — resolve or reopen an escalation
  prescriptions    — update status (active/blocked/discontinued), add notes
  adjudications    — not directly editable (read-only audit trail)

Invoked when router_agent returns intent = "db_update".
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
MAX_TURNS = 10


@dataclass
class DbUpdateResult:
    success: bool
    table: str
    record_id: str
    updated_fields: list[str] = field(default_factory=list)
    message: str = ""
    action: str = "db_updated"   # "db_updated" | "db_error" | "db_not_found"


TOOLS = [
    # ── source_records ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "find_patient",
            "description": (
                "Find a patient record by source_ref_id (e.g. CLN-001) or by name. "
                "Always call this first to confirm the record exists before updating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string", "description": "e.g. CLN-001"},
                    "name":          {"type": "string", "description": "partial name search"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_source_record",
            "description": (
                "Update any field on a patient's source_record. "
                "Supports: name, dob, nic, phone, address, blood_type, "
                "medications (replaces full array), allergies (replaces full array). "
                "Use add_medication / remove_medication for single-item changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string", "description": "e.g. CLN-001"},
                    "name":          {"type": "string"},
                    "dob":           {"type": "string", "description": "YYYY-MM-DD"},
                    "nic":           {"type": "string"},
                    "phone":         {"type": "string"},
                    "address":       {"type": "string"},
                    "blood_type":    {"type": "string"},
                    "medications":   {"type": "array", "items": {"type": "string"}, "description": "full replacement list"},
                    "allergies":     {"type": "array", "items": {"type": "string"}, "description": "full replacement list"},
                },
                "required": ["source_ref_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_medication",
            "description": "Append a single medication to a patient's existing medications list without replacing the whole list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string"},
                    "medication":    {"type": "string", "description": "medication name / dosage string"},
                },
                "required": ["source_ref_id", "medication"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_medication",
            "description": "Remove a single medication from a patient's medications list by name (partial match).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string"},
                    "medication":    {"type": "string"},
                },
                "required": ["source_ref_id", "medication"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_allergy",
            "description": "Append a single allergy to a patient's existing allergies list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string"},
                    "allergy":       {"type": "string"},
                },
                "required": ["source_ref_id", "allergy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_allergy",
            "description": "Remove a single allergy from a patient's allergies list by name (partial match).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string"},
                    "allergy":       {"type": "string"},
                },
                "required": ["source_ref_id", "allergy"],
            },
        },
    },
    # ── patient_clusters ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "update_cluster",
            "description": (
                "Update the canonical (master) record for a patient cluster. "
                "Use when the canonical name, dob, or NIC needs correcting. "
                "Provide source_ref_id (e.g. CLN-001) and it auto-resolves the cluster."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id":  {"type": "string", "description": "e.g. CLN-001 — used to find cluster automatically"},
                    "cluster_id":     {"type": "string", "description": "UUID from patient_clusters (optional if source_ref_id given)"},
                    "canonical_name": {"type": "string"},
                    "canonical_dob":  {"type": "string", "description": "YYYY-MM-DD"},
                    "canonical_nic":  {"type": "string"},
                },
            },
        },
    },
    # ── escalations ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "resolve_escalation",
            "description": "Mark an escalation as resolved. Use when a clinician has reviewed and handled the conflict.",
            "parameters": {
                "type": "object",
                "properties": {
                    "escalation_id": {"type": "string", "description": "UUID of the escalation"},
                    "source_ref_id": {"type": "string", "description": "patient ID to find escalations if no UUID given"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reopen_escalation",
            "description": "Reopen a previously resolved escalation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "escalation_id": {"type": "string"},
                },
                "required": ["escalation_id"],
            },
        },
    },
    # ── prescriptions ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "update_prescription",
            "description": (
                "Update a prescription record. Can change status (active/blocked/discontinued) "
                "or add notes. Use when discontinuing a drug or adding clinical notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string", "description": "patient ID to find prescriptions"},
                    "drug":          {"type": "string", "description": "drug name to identify which prescription"},
                    "status":        {"type": "string", "description": "active / blocked / discontinued"},
                    "notes":         {"type": "string"},
                    "dosage":        {"type": "string"},
                },
                "required": ["source_ref_id", "drug"],
            },
        },
    },
    # ── delete operations ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "delete_patient",
            "description": (
                "Permanently delete a patient record from source_records. "
                "Only use when user explicitly asks to delete/remove a patient. "
                "This is irreversible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string", "description": "e.g. CLN-001"},
                },
                "required": ["source_ref_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_prescription",
            "description": "Delete a prescription record entirely. Use only when user asks to delete (not just discontinue).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string"},
                    "drug":          {"type": "string", "description": "drug name to identify which prescription"},
                },
                "required": ["source_ref_id", "drug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_notification",
            "description": "Delete a notification by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notification_id": {"type": "string", "description": "UUID of the notification"},
                },
                "required": ["notification_id"],
            },
        },
    },
    # ── terminal ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call this when all updates are complete to return the final result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "one sentence describing what was updated"},
                },
                "required": ["summary"],
            },
        },
    },
]


class DatabaseToolExecutor:
    def __init__(self):
        self._result: DbUpdateResult | None = None
        self._done = False
        self._sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    def execute(self, tool_name: str, args: dict) -> str:
        handlers = {
            "find_patient":         self._find_patient,
            "update_source_record": self._update_source_record,
            "add_medication":       self._add_medication,
            "remove_medication":    self._remove_medication,
            "add_allergy":          self._add_allergy,
            "remove_allergy":       self._remove_allergy,
            "update_cluster":       self._update_cluster,
            "resolve_escalation":   self._resolve_escalation,
            "reopen_escalation":    self._reopen_escalation,
            "update_prescription":  self._update_prescription,
            "delete_patient":       self._delete_patient,
            "delete_prescription":  self._delete_prescription,
            "delete_notification":  self._delete_notification,
            "done":                 self._done_tool,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return handler(**args)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── find_patient ──────────────────────────────────────────────────────
    def _find_patient(self, source_ref_id: str = "", name: str = "") -> str:
        if source_ref_id:
            resp = self._sb.table("source_records").select(
                "id, source_ref_id, source, name, dob, nic, phone, address, blood_type, medications, allergies, cluster_id"
            ).eq("source_ref_id", source_ref_id.upper()).execute()
        elif name:
            resp = self._sb.table("source_records").select(
                "id, source_ref_id, source, name, dob, nic, phone, address, blood_type, medications, allergies, cluster_id"
            ).ilike("name", f"%{name}%").limit(5).execute()
        else:
            return json.dumps({"error": "Provide source_ref_id or name"})

        if not resp.data:
            return json.dumps({"found": False, "records": []})
        return json.dumps({"found": True, "records": resp.data})

    # ── update_source_record ──────────────────────────────────────────────
    def _update_source_record(self, source_ref_id: str, **kwargs) -> str:
        allowed = {"name", "dob", "nic", "phone", "address", "blood_type", "medications", "allergies"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None and v != ""}
        if not updates:
            return json.dumps({"error": "No valid fields to update."})

        resp = self._sb.table("source_records").update(updates).eq("source_ref_id", source_ref_id.upper()).execute()
        if not resp.data:
            return json.dumps({"error": f"No record found for {source_ref_id}"})

        # Re-embed if any identity field changed (name, dob, nic, phone, address)
        if updates.keys() & {"name", "dob", "nic", "phone", "address"}:
            try:
                from embed_records import re_embed_record
                re_embed_record(source_ref_id.upper())
            except Exception as e:
                print(f"[database-agent] WARNING: re-embedding failed for {source_ref_id}: {e} — identity matching may be degraded")

        self._result = DbUpdateResult(
            success=True, table="source_records", record_id=source_ref_id,
            updated_fields=list(updates.keys()),
            message=f"Updated {source_ref_id}: {', '.join(updates.keys())}",
            action="db_updated",
        )
        return json.dumps({"success": True, "updated": list(updates.keys())})

    # ── add_medication ────────────────────────────────────────────────────
    def _add_medication(self, source_ref_id: str, medication: str) -> str:
        current = self._sb.table("source_records").select("medications, name").eq("source_ref_id", source_ref_id.upper()).execute()
        if not current.data:
            return json.dumps({"error": f"No record found for {source_ref_id}"})

        meds = list(current.data[0].get("medications") or [])
        if medication not in meds:
            meds.append(medication)

        self._sb.table("source_records").update({"medications": meds}).eq("source_ref_id", source_ref_id.upper()).execute()
        self._result = DbUpdateResult(
            success=True, table="source_records", record_id=source_ref_id,
            updated_fields=["medications"], message=f"Added {medication} to {source_ref_id} medications.",
            action="db_updated",
        )
        return json.dumps({"success": True, "medications": meds})

    # ── remove_medication ─────────────────────────────────────────────────
    def _remove_medication(self, source_ref_id: str, medication: str) -> str:
        current = self._sb.table("source_records").select("medications, name").eq("source_ref_id", source_ref_id.upper()).execute()
        if not current.data:
            return json.dumps({"error": f"No record found for {source_ref_id}"})

        meds = list(current.data[0].get("medications") or [])
        med_lower = medication.lower()
        meds = [m for m in meds if med_lower not in str(m).lower()]

        self._sb.table("source_records").update({"medications": meds}).eq("source_ref_id", source_ref_id.upper()).execute()
        self._result = DbUpdateResult(
            success=True, table="source_records", record_id=source_ref_id,
            updated_fields=["medications"], message=f"Removed {medication} from {source_ref_id} medications.",
            action="db_updated",
        )
        return json.dumps({"success": True, "medications": meds})

    # ── add_allergy ───────────────────────────────────────────────────────
    def _add_allergy(self, source_ref_id: str, allergy: str) -> str:
        current = self._sb.table("source_records").select("allergies, name").eq("source_ref_id", source_ref_id.upper()).execute()
        if not current.data:
            return json.dumps({"error": f"No record found for {source_ref_id}"})

        allergies = list(current.data[0].get("allergies") or [])
        if allergy not in allergies:
            allergies.append(allergy)

        self._sb.table("source_records").update({"allergies": allergies}).eq("source_ref_id", source_ref_id.upper()).execute()
        self._result = DbUpdateResult(
            success=True, table="source_records", record_id=source_ref_id,
            updated_fields=["allergies"], message=f"Added allergy '{allergy}' to {source_ref_id}.",
            action="db_updated",
        )
        return json.dumps({"success": True, "allergies": allergies})

    # ── remove_allergy ────────────────────────────────────────────────────
    def _remove_allergy(self, source_ref_id: str, allergy: str) -> str:
        current = self._sb.table("source_records").select("allergies, name").eq("source_ref_id", source_ref_id.upper()).execute()
        if not current.data:
            return json.dumps({"error": f"No record found for {source_ref_id}"})

        allergies = list(current.data[0].get("allergies") or [])
        al_lower = allergy.lower()
        allergies = [a for a in allergies if al_lower not in str(a).lower()]

        self._sb.table("source_records").update({"allergies": allergies}).eq("source_ref_id", source_ref_id.upper()).execute()
        self._result = DbUpdateResult(
            success=True, table="source_records", record_id=source_ref_id,
            updated_fields=["allergies"], message=f"Removed allergy '{allergy}' from {source_ref_id}.",
            action="db_updated",
        )
        return json.dumps({"success": True, "allergies": allergies})

    # ── update_cluster ────────────────────────────────────────────────────
    def _update_cluster(self, cluster_id: str = "", source_ref_id: str = "",
                        canonical_name: str = "", canonical_dob: str = "", canonical_nic: str = "") -> str:
        # Auto-resolve cluster_id from source_ref_id if not provided
        if not cluster_id and source_ref_id:
            src = self._sb.table("source_records").select("cluster_id").eq("source_ref_id", source_ref_id.upper()).execute()
            if src.data and src.data[0].get("cluster_id"):
                cluster_id = src.data[0]["cluster_id"]

        if not cluster_id:
            return json.dumps({"error": "Could not determine cluster_id. Provide source_ref_id or cluster_id."})

        updates = {}
        if canonical_name: updates["canonical_name"] = canonical_name
        if canonical_dob:  updates["canonical_dob"] = canonical_dob
        if canonical_nic:  updates["canonical_nic"] = canonical_nic
        if not updates:
            return json.dumps({"error": "No fields to update."})

        resp = self._sb.table("patient_clusters").update(updates).eq("id", cluster_id).execute()
        if not resp.data:
            return json.dumps({"error": f"No cluster found for id {cluster_id}"})

        self._result = DbUpdateResult(
            success=True, table="patient_clusters", record_id=cluster_id,
            updated_fields=list(updates.keys()),
            message=f"Cluster updated for {source_ref_id or cluster_id}: {', '.join(updates.keys())}",
            action="db_updated",
        )
        return json.dumps({"success": True, "updated": list(updates.keys())})

    # ── resolve_escalation ────────────────────────────────────────────────
    def _resolve_escalation(self, escalation_id: str = "", source_ref_id: str = "") -> str:
        if escalation_id:
            resp = self._sb.table("escalations").update({"resolved": True}).eq("id", escalation_id).execute()
            resolved_ids = [escalation_id]
        elif source_ref_id:
            # Find open escalations for this patient via adjudications → detected_conflicts → cluster
            src_resp = self._sb.table("source_records").select("cluster_id").eq("source_ref_id", source_ref_id.upper()).execute()
            if not src_resp.data or not src_resp.data[0].get("cluster_id"):
                return json.dumps({"error": f"No cluster found for {source_ref_id}"})
            cluster_id = src_resp.data[0]["cluster_id"]

            conflicts_resp = self._sb.table("detected_conflicts").select("id").eq("cluster_id", cluster_id).execute()
            conflict_ids = [r["id"] for r in (conflicts_resp.data or [])]
            if not conflict_ids:
                return json.dumps({"error": "No conflicts found for this patient."})

            adj_resp = self._sb.table("adjudications").select("id").in_("conflict_id", conflict_ids).execute()
            adj_ids = [r["id"] for r in (adj_resp.data or [])]
            if not adj_ids:
                return json.dumps({"error": "No adjudications found."})

            esc_resp = self._sb.table("escalations").select("id").in_("adjudication_id", adj_ids).eq("resolved", False).execute()
            resolved_ids = [r["id"] for r in (esc_resp.data or [])]
            for eid in resolved_ids:
                self._sb.table("escalations").update({"resolved": True}).eq("id", eid).execute()
        else:
            return json.dumps({"error": "Provide escalation_id or source_ref_id"})

        self._result = DbUpdateResult(
            success=True, table="escalations", record_id=escalation_id or source_ref_id,
            updated_fields=["resolved"],
            message=f"Resolved {len(resolved_ids)} escalation(s).",
            action="db_updated",
        )
        return json.dumps({"success": True, "resolved_count": len(resolved_ids)})

    # ── reopen_escalation ─────────────────────────────────────────────────
    def _reopen_escalation(self, escalation_id: str) -> str:
        resp = self._sb.table("escalations").update({"resolved": False}).eq("id", escalation_id).execute()
        if not resp.data:
            return json.dumps({"error": f"No escalation found for {escalation_id}"})

        self._result = DbUpdateResult(
            success=True, table="escalations", record_id=escalation_id,
            updated_fields=["resolved"], message=f"Escalation {escalation_id} reopened.",
            action="db_updated",
        )
        return json.dumps({"success": True})

    # ── update_prescription ───────────────────────────────────────────────
    def _update_prescription(self, source_ref_id: str, drug: str, status: str = "", notes: str = "", dosage: str = "") -> str:
        # Find the prescription by patient + drug name (partial match)
        resp = self._sb.table("prescriptions").select("id, drug, status, notes").eq("source_ref_id", source_ref_id.upper()).execute()
        if not resp.data:
            return json.dumps({"error": f"No prescriptions found for {source_ref_id}"})

        drug_lower = drug.lower()
        matches = [r for r in resp.data if drug_lower in r.get("drug", "").lower()]
        if not matches:
            return json.dumps({"error": f"No prescription matching '{drug}' found for {source_ref_id}"})

        updates = {}
        if status: updates["status"] = status.lower()
        if notes:  updates["notes"] = notes
        if dosage: updates["dosage"] = dosage
        if not updates:
            return json.dumps({"error": "No fields to update."})

        for row in matches:
            self._sb.table("prescriptions").update(updates).eq("id", row["id"]).execute()

        self._result = DbUpdateResult(
            success=True, table="prescriptions", record_id=source_ref_id,
            updated_fields=list(updates.keys()),
            message=f"Updated prescription '{drug}' for {source_ref_id}: {', '.join(updates.keys())}.",
            action="db_updated",
        )
        return json.dumps({"success": True, "updated": list(updates.keys()), "matched": len(matches)})

    # ── delete_patient ────────────────────────────────────────────────────
    def _delete_patient(self, source_ref_id: str) -> str:
        current = self._sb.table("source_records").select("name").eq("source_ref_id", source_ref_id.upper()).execute()
        if not current.data:
            return json.dumps({"error": f"No record found for {source_ref_id}"})

        patient_name = current.data[0].get("name", "Unknown")
        self._sb.table("source_records").delete().eq("source_ref_id", source_ref_id.upper()).execute()

        self._result = DbUpdateResult(
            success=True, table="source_records", record_id=source_ref_id,
            updated_fields=["deleted"],
            message=f"Patient {patient_name} ({source_ref_id}) has been permanently deleted.",
            action="db_deleted",
        )
        return json.dumps({"success": True, "deleted": source_ref_id})

    # ── delete_prescription ───────────────────────────────────────────────
    def _delete_prescription(self, source_ref_id: str, drug: str) -> str:
        resp = self._sb.table("prescriptions").select("id, drug").eq("source_ref_id", source_ref_id.upper()).execute()
        if not resp.data:
            return json.dumps({"error": f"No prescriptions found for {source_ref_id}"})

        drug_lower = drug.lower()
        matches = [r for r in resp.data if drug_lower in r.get("drug", "").lower()]
        if not matches:
            return json.dumps({"error": f"No prescription matching '{drug}' found"})

        for row in matches:
            self._sb.table("prescriptions").delete().eq("id", row["id"]).execute()

        self._result = DbUpdateResult(
            success=True, table="prescriptions", record_id=source_ref_id,
            updated_fields=["deleted"],
            message=f"Deleted {len(matches)} prescription(s) matching '{drug}' for {source_ref_id}.",
            action="db_deleted",
        )
        return json.dumps({"success": True, "deleted_count": len(matches)})

    # ── delete_notification ───────────────────────────────────────────────
    def _delete_notification(self, notification_id: str) -> str:
        resp = self._sb.table("notifications").select("id").eq("id", notification_id).execute()
        if not resp.data:
            return json.dumps({"error": f"No notification found with id {notification_id}"})

        self._sb.table("notifications").delete().eq("id", notification_id).execute()

        self._result = DbUpdateResult(
            success=True, table="notifications", record_id=notification_id,
            updated_fields=["deleted"],
            message=f"Notification {notification_id} deleted.",
            action="db_deleted",
        )
        return json.dumps({"success": True})

    # ── done ──────────────────────────────────────────────────────────────
    def _done_tool(self, summary: str = "") -> str:
        if self._result is None:
            self._result = DbUpdateResult(
                success=True, table="", record_id="",
                message=summary, action="db_updated",
            )
        else:
            self._result.message = summary or self._result.message
        self._done = True
        return json.dumps({"done": True, "summary": summary})


_SYSTEM_PROMPT = """You are the Database Agent for Concord, a Sri Lankan clinical record system.

Your job is to update or delete records in Supabase based on the user's request.

Tables you can update:
- source_records     → patient demographics, medications, allergies, blood type
- patient_clusters   → canonical name/dob/NIC (master identity)
- escalations        → resolve or reopen conflicts needing human review
- prescriptions      → update status (active/discontinued) or add notes; or delete entirely
- notifications      → delete a notification

Rules:
1. Always call find_patient first to confirm the record exists before updating or deleting.
2. For medications/allergies, prefer add_medication/remove_medication over replacing the whole list.
3. Only call delete_patient, delete_prescription when the user explicitly uses the word "delete" or "remove record".
4. After completing all operations, call done() with a one-sentence summary.
5. Never invent data — only change what the user explicitly asked.
"""


def process_db_update(params: dict) -> DbUpdateResult:
    """Entry point called from api.py for intent=db_update."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=GROQ_API_KEY)
    executor = DatabaseToolExecutor()

    user_msg = f"Process this database update request:\n{json.dumps(params, indent=2)}"
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
            print(f"[database-agent] Turn {turns}: {tc.function.name}({list(args.keys())})")
            result = executor.execute(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    if executor._result is None:
        return DbUpdateResult(
            success=False, table="", record_id="",
            message="Could not complete the update. Please provide more details.",
            action="db_error",
        )

    return executor._result
