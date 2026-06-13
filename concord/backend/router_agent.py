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
- "intent": one of "register", "update", "reconcile", or "chat"
- "params": extracted parameters

Intent rules:
- "register": user wants to ADD a completely new patient. Keywords: add, register, new patient, create patient, enroll
- "update": user wants to ADD or CHANGE a field on an EXISTING patient (e.g. "add NIC to Dipna", "update phone for CLN-001")
- "prescribe": user wants to prescribe or give a drug/medication to a patient. Keywords: prescribe, give, administer, add medication, issue
- "reconcile": user wants conflict/safety analysis for a patient ID (CLN-xxx, LAB-xxx, PHM-xxx)
- "chat": everything else

For "register", extract:
  name, dob (YYYY-MM-DD), nic, phone, address, medications (list), allergies (list), blood_type, source ("clinic"/"lab"/"pharmacy", default "clinic")

For "update", extract:
  source_ref_id (if given), name (to find patient if no ID), and all fields being updated

For "prescribe", extract:
  source_ref_id (e.g. CLN-001), drug (medication name), dosage (e.g. "500mg daily"), notes

For "reconcile", extract:
  source_ref_id

For "chat", params = {}

Return ONLY valid JSON. No explanation. No markdown. Examples:
{"intent": "register", "params": {"name": "Kasun Silva", "dob": "1990-05-14"}}
{"intent": "prescribe", "params": {"source_ref_id": "CLN-001", "drug": "aspirin", "dosage": "100mg daily"}}
{"intent": "update", "params": {"source_ref_id": "CLN-004", "phone": "0771234567"}}
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
