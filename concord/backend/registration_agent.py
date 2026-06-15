"""
Registration Agent — registers a new patient or updates an existing one.

Tools:
  check_duplicate     — search by NIC or name+dob
  register_patient    — insert new record (sets result directly, no LLM ID passing)
  update_patient      — update fields on an existing record
"""

import json
import os
from dataclasses import dataclass

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
class RegistrationResult:
    success: bool
    source_ref_id: str
    patient_name: str
    message: str
    action: str = "registered"   # "registered" | "updated" | "duplicate" | "error"
    existing_id: str = ""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_duplicate",
            "description": (
                "Check if a patient already exists. Search by NIC (exact) or name+dob. "
                "Always call this before registering a new patient."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nic":  {"type": "string"},
                    "name": {"type": "string"},
                    "dob":  {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_patient",
            "description": "Insert a new patient record. Call only after check_duplicate confirms no duplicate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":        {"type": "string"},
                    "dob":         {"type": "string"},
                    "nic":         {"type": "string"},
                    "phone":       {"type": "string"},
                    "address":     {"type": "string"},
                    "medications": {"type": "array", "items": {"type": "string"}},
                    "allergies":   {"type": "array", "items": {"type": "string"}},
                    "blood_type":  {"type": "string"},
                    "source":      {"type": "string", "description": "clinic / lab / pharmacy"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_patient",
            "description": "Update specific fields on an existing patient record. Use when adding NIC, phone, address, etc. to an existing patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {"type": "string", "description": "The existing patient ID to update"},
                    "name":          {"type": "string"},
                    "dob":           {"type": "string"},
                    "nic":           {"type": "string"},
                    "phone":         {"type": "string"},
                    "address":       {"type": "string"},
                    "medications":   {"type": "array", "items": {"type": "string"}},
                    "allergies":     {"type": "array", "items": {"type": "string"}},
                    "blood_type":    {"type": "string"},
                },
                "required": ["source_ref_id"],
            },
        },
    },
]


class RegistrationToolExecutor:
    def __init__(self):
        self._result: RegistrationResult | None = None
        self._done = False

    def execute(self, tool_name: str, args: dict) -> str:
        handlers = {
            "check_duplicate": self._check_duplicate,
            "register_patient": self._register_patient,
            "update_patient": self._update_patient,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        try:
            return handler(**args)
        except Exception as e:
            return f"Tool error: {e}"

    def _check_duplicate(self, nic: str = "", name: str = "", dob: str = "") -> str:
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        if nic:
            resp = sb.table("source_records").select("source_ref_id, name, dob, nic, source").eq("nic", nic).execute()
            if resp.data:
                return json.dumps({"duplicate": True, "matches": resp.data})
        if name and dob:
            resp = sb.table("source_records").select("source_ref_id, name, dob, nic, source").ilike("name", f"%{name}%").eq("dob", dob).execute()
            if resp.data:
                return json.dumps({"duplicate": True, "matches": resp.data})
        if name and not dob:
            resp = sb.table("source_records").select("source_ref_id, name, dob, nic, source").ilike("name", f"%{name}%").execute()
            if resp.data:
                return json.dumps({"duplicate": True, "matches": resp.data})
        return json.dumps({"duplicate": False, "matches": []})

    def _register_patient(
        self,
        name: str,
        dob: str = "",
        nic: str = "",
        phone: str = "",
        address: str = "",
        medications: list = None,
        allergies: list = None,
        blood_type: str = "",
        source: str = "clinic",
    ) -> str:
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        prefix_map = {"clinic": "CLN", "lab": "LAB", "pharmacy": "PHM"}
        prefix = prefix_map.get(source.lower(), "CLN")

        existing = sb.table("source_records").select("source_ref_id").like("source_ref_id", f"{prefix}-%").execute()
        max_num = 0
        for r in (existing.data or []):
            try:
                num = int(r["source_ref_id"].split("-")[1])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                pass
        new_id = f"{prefix}-{str(max_num + 1).zfill(3)}"

        record = {"source_ref_id": new_id, "source": source.lower(), "name": name}
        if dob:         record["dob"] = dob
        if nic:         record["nic"] = nic
        if phone:       record["phone"] = phone
        if address:     record["address"] = address
        if medications: record["medications"] = medications
        if allergies:   record["allergies"] = allergies
        if blood_type:  record["blood_type"] = blood_type

        sb.table("source_records").insert(record).execute()

        # Generate embedding for the new record
        try:
            from embed_records import re_embed_record
            re_embed_record(new_id)
        except Exception as e:
            print(f"[registration-agent] WARNING: embedding failed for {new_id}: {e} — identity matching may be degraded")

        # Set result directly — don't rely on LLM to pass the ID back
        self._result = RegistrationResult(
            success=True,
            source_ref_id=new_id,
            patient_name=name,
            message=f"Patient {name} registered with ID {new_id}.",
            action="registered",
        )
        self._done = True
        return json.dumps({"success": True, "source_ref_id": new_id, "message": f"Registered as {new_id}"})

    def _update_patient(
        self,
        source_ref_id: str,
        name: str = "",
        dob: str = "",
        nic: str = "",
        phone: str = "",
        address: str = "",
        medications: list = None,
        allergies: list = None,
        blood_type: str = "",
    ) -> str:
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

        # Fetch current record to get name
        current = sb.table("source_records").select("name").eq("source_ref_id", source_ref_id).execute()
        if not current.data:
            return json.dumps({"error": f"No record found for {source_ref_id}"})

        patient_name = name or current.data[0].get("name", "Unknown")
        updates = {}
        if name:        updates["name"] = name
        if dob:         updates["dob"] = dob
        if nic:         updates["nic"] = nic
        if phone:       updates["phone"] = phone
        if address:     updates["address"] = address
        if medications: updates["medications"] = medications
        if allergies:   updates["allergies"] = allergies
        if blood_type:  updates["blood_type"] = blood_type

        if not updates:
            return json.dumps({"error": "No fields to update."})

        sb.table("source_records").update(updates).eq("source_ref_id", source_ref_id).execute()

        # Re-embed if identity fields changed
        if updates.keys() & {"name", "dob", "nic", "phone", "address"}:
            try:
                from embed_records import re_embed_record
                re_embed_record(source_ref_id)
            except Exception:
                pass

        self._result = RegistrationResult(
            success=True,
            source_ref_id=source_ref_id,
            patient_name=patient_name,
            message=f"Updated {patient_name} ({source_ref_id}): {', '.join(updates.keys())}.",
            action="updated",
        )
        self._done = True
        return json.dumps({"success": True, "updated_fields": list(updates.keys())})


_SYSTEM_PROMPT = """You are the Registration Agent for Concord, a Sri Lankan clinical record system.

Your job:
- To register a new patient: first call check_duplicate, then if no duplicate call register_patient.
- To update an existing patient: call update_patient with the source_ref_id and the fields to update.
- If a duplicate is found and the user just wants to add info (like NIC) to the existing patient, call update_patient on that existing record.

Do NOT call any confirmation tool. register_patient and update_patient complete the task automatically."""


def register_patient_from_details(params: dict) -> RegistrationResult:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=GROQ_API_KEY)
    executor = RegistrationToolExecutor()

    user_msg = f"Process this patient request:\n{json.dumps(params, indent=2)}"
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
            print(f"[registration-agent] Turn {turns}: {tc.function.name}({list(args.keys())})")
            result = executor.execute(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    if executor._result is None:
        return RegistrationResult(
            success=False,
            source_ref_id="",
            patient_name=params.get("name", "Unknown"),
            message="Could not complete. Please provide more details.",
            action="error",
        )

    return executor._result
