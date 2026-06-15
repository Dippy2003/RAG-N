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
- "intent": one of "register", "update", "db_update", "prescribe", "query", "reconcile", "add_guideline", or "chat"
- "params": extracted parameters

Intent rules (READ CAREFULLY — ambiguous cases are explained):

- "register": user wants to ADD a completely NEW patient who does not exist yet.
  Keywords: register, new patient, create patient, enroll, add new patient.

- "update": user wants to change DEMOGRAPHIC fields (name, dob, NIC, phone, address) on an existing patient.
  Keywords: update phone, change address, add NIC, set DOB.

- "db_update": user wants to modify CLINICAL DATA or RECORDS on an existing patient or system record.
  Covers: add medication, remove medication, add allergy, remove allergy, update blood type,
          resolve/reopen escalation, discontinue/update prescription, update cluster,
          rename/change/update a patient's ID to a new ID.
  IMPORTANT — "add medication" or "add [drug] to [patient]" WITHOUT the word "prescribe" = db_update (just appends to array, NO interaction check).
  Keywords: add medication, remove medication, add allergy, remove allergy, discontinue, resolve escalation, update blood type, change ID, rename ID, update ID.

- "prescribe": user explicitly wants a CLINICAL PRESCRIPTION with safety/interaction check.
  MUST contain: prescribe, issue prescription, give prescription, administer drug.
  "prescribe aspirin to CLN-001" = prescribe (runs interaction check).
  "add aspirin to CLN-001 medications" = db_update (no interaction check, just append).

- "query": user wants to LIST or SEARCH records — not update, not prescribe.
  Keywords: show, list, find, search, tell me about, what is, which patients, all patients, history, what prescriptions, unresolved escalations.
  When the user asks about a specific patient ID (e.g. "tell me about CLN-004", "show CLN-004"), use query_type="search" and set source_ref_id to that ID.
  When the user searches by NIC, phone, DOB, name, blood type, medication, or allergy — use query_type="search" and put the search term in "filter".
  Examples: "show all patients", "find NIC 200334511790", "search phone 0771234567", "who has warfarin", "find patient born 1990", "patient with blood type B+".

- "reconcile": user wants conflict/safety analysis for a specific patient ID.
  Keywords: conflicts, reconcile, safe, drug interaction analysis, check safety.

- "add_guideline": user wants to TEACH the system a new clinical rule / guideline to remember
  for future safety checks — NOT about a specific patient.
  Keywords: add a guideline, add a rule, remember that, new guideline, teach the system,
            from now on flag, the system should know, add to knowledge base.
  Examples: "Add a guideline: don't give ACE inhibitors in pregnancy",
            "Remember that metoclopramide should be avoided under 1 year",
            "Teach the system that grapefruit interacts with statins".
  Extract: text = the full clinical rule the user described (verbatim or lightly cleaned).

- "chat": everything else — general questions, explain, what is, how does, guidelines.

For "register", extract:
  name, dob (YYYY-MM-DD), nic, phone, address, medications (list), allergies (list), blood_type,
  source ("clinic"/"lab"/"pharmacy" — default "clinic", extract from message if mentioned)

For "update", extract:
  source_ref_id (if given), name (to find patient if no ID), fields being updated

For "db_update", extract:
  operation: "add_medication" | "remove_medication" | "add_allergy" | "remove_allergy" |
             "resolve_escalation" | "reopen_escalation" | "update_prescription" | "update_cluster" | "update_blood_type" | "rename_id"
  source_ref_id: patient ID (e.g. CLN-001) — the CURRENT/OLD id
  new_id: the NEW id the user wants (only for rename_id operation)
  name: patient name if no ID given
  value: the item being added/removed/changed (drug name, allergy name, new value)
  status: for update_prescription — "active" | "discontinued" | "blocked"

For "prescribe", extract:
  source_ref_id, drug, dosage, notes

For "query", extract:
  query_type: "all_patients" | "prescriptions" | "escalations" | "medications" | "allergies" | "notifications" | "search"
  source_ref_id: patient ID if filtering by a known CLN/LAB/PHM ID
  filter: search term — use this for NIC numbers, phone numbers, patient names, DOB, blood type, medication names, allergy names, or any freetext identifier
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
{"intent": "db_update", "params": {"operation": "rename_id", "source_ref_id": "CLN-004", "new_id": "CLN-006"}}
{"intent": "query", "params": {"query_type": "all_patients"}}
{"intent": "query", "params": {"query_type": "search", "source_ref_id": "CLN-004"}}
{"intent": "query", "params": {"query_type": "prescriptions", "source_ref_id": "CLN-001"}}
{"intent": "query", "params": {"query_type": "escalations", "filter": "unresolved"}}
{"intent": "query", "params": {"query_type": "medications", "source_ref_id": "CLN-001"}}
{"intent": "query", "params": {"query_type": "allergies", "source_ref_id": "CLN-002"}}
{"intent": "query", "params": {"query_type": "search", "filter": "warfarin"}}
{"intent": "query", "params": {"query_type": "search", "filter": "200334511790"}}
{"intent": "query", "params": {"query_type": "search", "filter": "0771234567"}}
{"intent": "query", "params": {"query_type": "search", "filter": "1990-05-14"}}
{"intent": "query", "params": {"query_type": "search", "filter": "B+"}}
{"intent": "query", "params": {"query_type": "search", "filter": "Nimal Perera"}}
{"intent": "add_guideline", "params": {"text": "Don't give ACE inhibitors to pregnant patients, risk of fetal harm"}}
"""


def route(message: str, history: list[dict] | None = None) -> dict:
    """Returns {"intent": str, "params": dict}

    history: optional list of {"role": "user"|"assistant", "content": str}
    Providing history lets the router resolve pronouns like "this patient",
    "their ID", "same drug" from prior conversation context.
    """
    if not GROQ_API_KEY:
        return {"intent": "chat", "params": {}}

    client = Groq(api_key=GROQ_API_KEY)

    # Build message list — inject up to 6 recent turns so the router can
    # resolve references like "this patient" or "their medications".
    def _build_messages(sys_prompt: str) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": sys_prompt}]
        if history:
            for h in history[-6:]:
                # Truncate long assistant replies to keep token cost low
                content = h["content"][:400] if h.get("role") == "assistant" else h["content"]
                msgs.append({"role": h["role"], "content": content})
        msgs.append({"role": "user", "content": message})
        return msgs

    def _call(model: str):
        return client.chat.completions.create(
            model=model,
            messages=_build_messages(_SYSTEM),
            temperature=0.0,
            max_tokens=300,
        )

    try:
        resp = _call(GROQ_MODEL)
    except Exception as e:
        if "rate_limit_exceeded" in str(e) or "429" in str(e):
            resp = _call(GROQ_FALLBACK_MODEL)
        else:
            print(f"[router-agent] ERROR: routing failed ({type(e).__name__}: {e}), defaulting to chat")
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
