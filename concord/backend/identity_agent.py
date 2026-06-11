"""
Identity Validation Agent — Agent 1 in the multi-agent pipeline.

Given a source_ref_id AND the details the patient verbally provides
(name, dob, nic), this agent checks whether the ID actually belongs
to that person. If not, it searches for the correct record and returns
the correct source_ref_id for Agent 2 (reconciliation) to use.

Tools:
  lookup_record       — fetch the record stored under the given ID
  search_by_details   — vector + exact search by patient-provided details
  confirm_identity    — final answer: ID correct / incorrect + correct ID
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

MAX_TURNS = 10


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class FieldStatus:
    field: str
    provided: str       # what the patient said
    stored: str         # what's in the record
    match: bool         # do they match?


@dataclass
class IdentityValidationResult:
    given_id: str
    is_correct: bool            # ID belongs to this patient
    correct_id: str             # confirmed correct source_ref_id
    confidence: float
    mismatch_fields: list[str]  # names of fields that don't match
    field_details: list[FieldStatus]   # per-field comparison
    explanation: str
    patient_name_found: str


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI-compatible for Groq)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_record",
            "description": (
                "Fetches the patient record stored under a specific source_ref_id. "
                "Returns name, date of birth, NIC, phone, address, and source. "
                "Use this first to see what the ID currently points to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {
                        "type": "string",
                        "description": "The source_ref_id to look up (e.g. CLN-001)"
                    }
                },
                "required": ["source_ref_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_details",
            "description": (
                "Searches all records using the patient-provided details as a query. "
                "Embeds the provided details and does a vector similarity search, "
                "then also tries exact NIC match. Returns the top matching records "
                "with their source_ref_ids. Use this when the given ID doesn't match "
                "the patient's stated details, to find the correct record."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Patient name as stated by the patient"
                    },
                    "dob": {
                        "type": "string",
                        "description": "Date of birth as stated (YYYY-MM-DD)"
                    },
                    "nic": {
                        "type": "string",
                        "description": "NIC number as stated by the patient (optional)"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_identity",
            "description": (
                "Call this once you have compared all fields. Report the match/mismatch "
                "status for EVERY field the patient provided, not just the ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "is_correct": {
                        "type": "boolean",
                        "description": "True if the given source_ref_id belongs to this patient (ID itself is right)"
                    },
                    "correct_source_ref_id": {
                        "type": "string",
                        "description": "The correct source_ref_id to use (same as given if is_correct=true)"
                    },
                    "patient_name": {
                        "type": "string",
                        "description": "The patient name on the confirmed correct record"
                    },
                    "field_comparisons": {
                        "type": "array",
                        "description": "One entry per field the patient provided. Include ALL checked fields.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field":    {"type": "string", "description": "Field name: name, dob, nic, phone, address"},
                                "provided": {"type": "string", "description": "What the patient stated"},
                                "stored":   {"type": "string", "description": "What is in the record"},
                                "match":    {"type": "boolean", "description": "True if they match (allow minor name typos)"}
                            },
                            "required": ["field", "provided", "stored", "match"]
                        }
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0.0-1.0"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "1-2 sentence summary of the overall finding"
                    }
                },
                "required": ["is_correct", "correct_source_ref_id", "patient_name", "field_comparisons", "confidence", "explanation"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class IdentityToolExecutor:
    def __init__(self):
        self._result: IdentityValidationResult | None = None
        self._done = False

    def execute(self, tool_name: str, args: dict) -> str:
        handlers = {
            "lookup_record":    self._lookup_record,
            "search_by_details": self._search_by_details,
            "confirm_identity": self._confirm_identity,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        try:
            return handler(**args)
        except Exception as e:
            return f"Tool error: {e}"

    def _lookup_record(self, source_ref_id: str) -> str:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        resp = (
            supabase.table("source_records")
            .select("source_ref_id, source, name, dob, nic, phone, address")
            .eq("source_ref_id", source_ref_id)
            .execute()
        )
        if not resp.data:
            return json.dumps({"error": f"No record found for source_ref_id: {source_ref_id}"})
        r = resp.data[0]
        return json.dumps({
            "source_ref_id": r["source_ref_id"],
            "source": r["source"],
            "name": r["name"],
            "dob": str(r.get("dob")),
            "nic": r.get("nic"),
            "phone": r.get("phone"),
            "address": r.get("address"),
        }, indent=2)

    def _search_by_details(self, name: str, dob: str = "", nic: str = "") -> str:
        from sentence_transformers import SentenceTransformer
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

        # Try exact NIC match first
        if nic:
            resp = (
                supabase.table("source_records")
                .select("source_ref_id, source, name, dob, nic")
                .eq("nic", nic)
                .execute()
            )
            if resp.data:
                return json.dumps({
                    "search_method": "exact_nic_match",
                    "matches": [
                        {
                            "source_ref_id": r["source_ref_id"],
                            "source": r["source"],
                            "name": r["name"],
                            "dob": str(r.get("dob")),
                            "nic": r.get("nic"),
                            "similarity": 1.0,
                        }
                        for r in resp.data
                    ]
                }, indent=2)

        # Fall back to vector similarity search
        model = SentenceTransformer("all-MiniLM-L6-v2")
        parts = [p for p in [name, dob, nic] if p]
        query_text = " | ".join(parts)
        query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()

        resp = supabase.rpc(
            "match_records",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.6,
                "match_count": 5,
            }
        ).execute()

        matches = resp.data or []
        return json.dumps({
            "search_method": "vector_similarity",
            "query": query_text,
            "matches": [
                {
                    "source_ref_id": r["source_ref_id"],
                    "source": r["source"],
                    "name": r["name"],
                    "dob": str(r.get("dob")),
                    "nic": r.get("nic"),
                    "similarity": round(r.get("similarity", 0), 4),
                }
                for r in matches
            ]
        }, indent=2)

    def _confirm_identity(
        self,
        is_correct: bool,
        correct_source_ref_id: str,
        patient_name: str,
        field_comparisons: list,
        confidence: float,
        explanation: str,
    ) -> str:
        field_details = [
            FieldStatus(
                field=fc.get("field", ""),
                provided=fc.get("provided", ""),
                stored=fc.get("stored", ""),
                match=bool(fc.get("match", True)),
            )
            for fc in field_comparisons
        ]
        mismatch_fields = [f.field for f in field_details if not f.match]
        self._result = IdentityValidationResult(
            given_id="",
            is_correct=is_correct,
            correct_id=correct_source_ref_id,
            confidence=max(0.0, min(1.0, float(confidence))),
            mismatch_fields=mismatch_fields,
            field_details=field_details,
            explanation=explanation,
            patient_name_found=patient_name,
        )
        self._done = True
        return json.dumps({"status": "confirmed", "is_correct": is_correct, "correct_id": correct_source_ref_id})


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Identity Validation Agent for Concord, a clinical record reconciliation system in Sri Lanka.

Your job: check whether a given source_ref_id belongs to the patient AND whether every detail the patient provided is correct.

Workflow:
1. Call lookup_record with the given source_ref_id to retrieve what is stored.
2. Compare EVERY field the patient provided against the stored record:
   - name, dob, nic, phone, address — check each one individually.
3. If the ID belongs to the right person but some details are wrong → call confirm_identity with is_correct=true but list the mismatching fields in field_comparisons.
4. If the ID belongs to the WRONG person entirely → call search_by_details to find the correct record, then call confirm_identity with is_correct=false and the correct source_ref_id.

Matching rules per field:
- name: allow minor spelling differences or transliteration variants of Sinhala/Tamil names. Mark mismatch only if clearly a different person.
- dob: exact match required (YYYY-MM-DD). Any difference is a mismatch.
- nic: old format (9 digits + V/X) and new format (12 digits) may be the same person — check numerically. Otherwise exact match.
- phone: ignore spaces/dashes/+94 prefix vs 0 prefix differences. Core digits must match.
- address: partial match is acceptable (suburb or city match counts). Full mismatch only if completely different location.

IMPORTANT: In field_comparisons, include an entry for EVERY field the patient provided — even matching ones. Set match=true for matches, match=false for mismatches. If a field was not provided by the patient, omit it.

Always call confirm_identity — never end without it."""


def validate_identity(
    source_ref_id: str,
    patient_name: str,
    dob: str = "",
    nic: str = "",
    phone: str = "",
    address: str = "",
) -> IdentityValidationResult:
    """
    Runs the identity validation agent.
    Checks all provided fields against the stored record, not just the ID.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=GROQ_API_KEY)
    executor = IdentityToolExecutor()

    details = [f"  - name: {patient_name}"]
    if dob:     details.append(f"  - dob: {dob}")
    if nic:     details.append(f"  - nic: {nic}")
    if phone:   details.append(f"  - phone: {phone}")
    if address: details.append(f"  - address: {address}")

    user_msg = (
        f"Validate this patient identity.\n\n"
        f"Given source_ref_id: {source_ref_id}\n\n"
        f"Patient's stated details:\n" + "\n".join(details) +
        f"\n\nLook up the record for this ID and compare every detail above. "
        f"Report which fields match and which do not. "
        f"If the ID belongs to a completely different person, search for the correct record."
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
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception as e:
            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                response = client.chat.completions.create(
                    model=GROQ_FALLBACK_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                )
            else:
                raise
        msg = response.choices[0].message
        messages.append(msg)

        tool_calls = msg.tool_calls or []
        if not tool_calls:
            break

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            print(f"[identity-agent] Turn {turns}: {tool_name}({list(args.keys())})")
            result = executor.execute(tool_name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    if executor._result is None:
        return IdentityValidationResult(
            given_id=source_ref_id,
            is_correct=True,
            correct_id=source_ref_id,
            confidence=0.5,
            mismatch_fields=[],
            field_details=[],
            explanation="Identity validation inconclusive. Proceeding with given ID.",
            patient_name_found="",
        )

    executor._result.given_id = source_ref_id
    return executor._result


if __name__ == "__main__":
    # Correct ID test
    print("=== Test 1: Correct ID ===")
    r = validate_identity("CLN-001", patient_name="Nimal Perera", dob="1975-03-12")
    print(f"Correct: {r.is_correct} | ID to use: {r.correct_id} | Confidence: {r.confidence}")
    print(f"Explanation: {r.explanation}\n")

    # Wrong ID test
    print("=== Test 2: Wrong ID (CLN-002 but giving Nimal Perera's details) ===")
    r2 = validate_identity("CLN-002", patient_name="Nimal Perera", dob="1975-03-12")
    print(f"Correct: {r2.is_correct} | ID to use: {r2.correct_id} | Confidence: {r2.confidence}")
    print(f"Mismatches: {r2.mismatch_fields}")
    print(f"Explanation: {r2.explanation}")
