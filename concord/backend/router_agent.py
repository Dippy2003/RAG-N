"""
Router Agent — reads the user's chat message and decides which agent to invoke.

Returns:
  intent:  "register" | "reconcile" | "chat"
  params:  extracted parameters relevant to the intent
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"

_SYSTEM = """You are a clinical system router. Read the user message and return a JSON object with:
- "intent": one of "register", "update", "db_update", "prescribe", "query", "reconcile", or "chat"
- "params": extracted parameters

Intent rules (READ CAREFULLY — ambiguous cases are explained):

- "register": user wants to ADD a completely NEW patient who does not exist yet.
  Keywords: register, new patient, create patient, enroll, add new patient.

- "update": user wants to change DEMOGRAPHIC fields (name, dob, NIC, phone, address) on an existing patient.
  Keywords: update phone, change address, add NIC, set DOB.

- "db_update": user wants to modify CLINICAL DATA or RECORDS on an existing patient or system record.
  Covers: add medication, remove medication, add allergy, remove allergy, update blood type,
          resolve/reopen escalation, discontinue/update prescription, update cluster.
  IMPORTANT — "add medication" or "add [drug] to [patient]" WITHOUT the word "prescribe" = db_update (just appends to array, NO interaction check).
  Keywords: add medication, remove medication, add allergy, remove allergy, discontinue, resolve escalation, update blood type.

- "prescribe": user explicitly wants a CLINICAL PRESCRIPTION with safety/interaction check.
  MUST contain: prescribe, issue prescription, give prescription, administer drug.
  "prescribe aspirin to CLN-001" = prescribe (runs interaction check).
  "add aspirin to CLN-001 medications" = db_update (no interaction check, just append).

- "query": user wants to LIST or SEARCH records — not update, not prescribe.
  Keywords: show, list, find, search, tell me about, what is, which patients, all patients, history, what prescriptions, unresolved escalations.
  When the user asks about a specific patient ID (e.g. "tell me about CLN-004", "show CLN-004", "what is CLN-004"), use query_type="search" and set source_ref_id to that ID.
  Examples: "show all patients", "list prescriptions for CLN-001", "which patients have warfarin", "show unresolved escalations".

- "reconcile": user wants conflict/safety analysis for a specific patient ID.
  Keywords: conflicts, reconcile, safe, drug interaction analysis, check safety.

- "chat": everything else — general questions, explain, what is, how does, guidelines.

For "register", extract:
  name, dob (YYYY-MM-DD), nic, phone, address, medications (list), allergies (list), blood_type,
  source ("clinic"/"lab"/"pharmacy" — default "clinic", extract from message if mentioned)

For "update", extract:
  source_ref_id (if given), name (to find patient if no ID), fields being updated

For "db_update", extract:
  operation: "add_medication" | "remove_medication" | "add_allergy" | "remove_allergy" |
             "resolve_escalation" | "reopen_escalation" | "update_prescription" | "update_cluster" | "update_blood_type"
  source_ref_id: patient ID (e.g. CLN-001)
  name: patient name if no ID given
  value: the item being added/removed/changed (drug name, allergy name, new value)
  status: for update_prescription — "active" | "discontinued" | "blocked"

For "prescribe", extract:
  source_ref_id, drug, dosage, notes

For "query", extract:
  query_type: "all_patients" | "prescriptions" | "escalations" | "medications" | "allergies" | "notifications" | "search"
  source_ref_id: patient ID if filtering by patient
  filter: any filter value (e.g. drug name, status, "unresolved")
  source: "CLN" | "LAB" | "PHM" if filtering by location

For "reconcile", extract:
  source_ref_id

For "chat", params = {}

Return ONLY valid JSON. No explanation. No markdown. Examples:
{"intent": "register", "params": {"name": "Kasun Silva", "dob": "1990-05-14", "source": "clinic"}}
{"intent": "prescribe", "params": {"source_ref_id": "CLN-001", "drug": "aspirin", "dosage": "100mg daily"}}
{"intent": "update", "params": {"source_ref_id": "CLN-004", "phone": "0771234567"}}
{"intent": "db_update", "params": {"operation": "add_medication", "source_ref_id": "CLN-001", "value": "metformin 500mg"}}
{"intent": "db_update", "params": {"operation": "remove_medication", "source_ref_id": "CLN-001", "value": "warfarin"}}
{"intent": "db_update", "params": {"operation": "add_allergy", "source_ref_id": "CLN-002", "value": "penicillin"}}
{"intent": "db_update", "params": {"operation": "resolve_escalation", "source_ref_id": "CLN-003"}}
{"intent": "db_update", "params": {"operation": "update_prescription", "source_ref_id": "CLN-001", "value": "aspirin", "status": "discontinued"}}
{"intent": "query", "params": {"query_type": "all_patients"}}
{"intent": "query", "params": {"query_type": "search", "source_ref_id": "CLN-004"}}
{"intent": "query", "params": {"query_type": "prescriptions", "source_ref_id": "CLN-001"}}
{"intent": "query", "params": {"query_type": "escalations", "filter": "unresolved"}}
{"intent": "query", "params": {"query_type": "medications", "source_ref_id": "CLN-001"}}
{"intent": "query", "params": {"query_type": "allergies", "source_ref_id": "CLN-002"}}
{"intent": "query", "params": {"query_type": "search", "filter": "warfarin"}}
"""


def route(message: str) -> dict:
    """Returns {"intent": str, "params": dict}"""
    if not GROQ_API_KEY:
        return {"intent": "chat", "params": {}}

    client = Groq(api_key=GROQ_API_KEY)

    def _call(model: str):
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            max_tokens=300,
        )

    try:
        resp = _call(GROQ_MODEL)
    except Exception as e:
        if "rate_limit_exceeded" in str(e) or "429" in str(e):
            resp = _call(GROQ_FALLBACK_MODEL)
        else:
            return {"intent": "chat", "params": {}}

    raw = resp.choices[0].message.content or ""
    # Strip markdown code fences if present
    raw = raw.strip().strip("```json").strip("```").strip()
    try:
        result = json.loads(raw)
        if "intent" not in result:
            return {"intent": "chat", "params": {}}
        return result
    except json.JSONDecodeError:
        return {"intent": "chat", "params": {}}
