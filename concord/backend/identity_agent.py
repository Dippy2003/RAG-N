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

MAX_TURNS = 10


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class IdentityValidationResult:
    given_id: str               # what the user passed in
    is_correct: bool            # does the ID match the patient's details?
    correct_id: str             # confirmed correct source_ref_id to use
    confidence: float
    mismatch_fields: list[str]  # which fields didn't match
    explanation: str
    patient_name_found: str     # name on the record that was confirmed


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
                "Call this when you have enough information to make a final determination. "
                "Provide whether the given ID is correct, what the correct ID is, "
                "and which fields mismatched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "is_correct": {
                        "type": "boolean",
                        "description": "True if the given source_ref_id belongs to the patient described"
                    },
                    "correct_source_ref_id": {
                        "type": "string",
                        "description": "The correct source_ref_id to use (same as given if is_correct=true)"
                    },
                    "patient_name": {
                        "type": "string",
                        "description": "The patient name on the confirmed correct record"
                    },
                    "mismatch_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields that didn't match (e.g. ['name', 'dob']). Empty if correct."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0.0-1.0 in this determination"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "1-2 sentence explanation of the finding"
                    }
                },
                "required": ["is_correct", "correct_source_ref_id", "patient_name", "mismatch_fields", "confidence", "explanation"]
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
        mismatch_fields: list,
        confidence: float,
        explanation: str,
    ) -> str:
        self._result = IdentityValidationResult(
            given_id="",  # filled in by caller
            is_correct=is_correct,
            correct_id=correct_source_ref_id,
            confidence=max(0.0, min(1.0, float(confidence))),
            mismatch_fields=mismatch_fields,
            explanation=explanation,
            patient_name_found=patient_name,
        )
        self._done = True
        return json.dumps({"status": "confirmed", "is_correct": is_correct, "correct_id": correct_source_ref_id})


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Identity Validation Agent for Concord, a clinical record reconciliation system.

Your job: verify whether a given source_ref_id actually belongs to the patient who provided their details.

Workflow:
1. Call lookup_record with the given source_ref_id to see what's stored under that ID.
2. Compare the stored record's name, date of birth, and NIC with the patient-provided details.
3. If they match well → call confirm_identity with is_correct=true.
4. If they DON'T match (different name, DOB, or NIC) → call search_by_details with the patient's stated details to find the correct record.
5. From the search results, identify the best matching record and call confirm_identity with is_correct=false and the correct source_ref_id.

Matching rules:
- Names: allow minor spelling differences (typos, transliterations of Sinhala/Tamil names). Flag as mismatch only if clearly different person.
- DOB: exact match required. A 1-year difference is a mismatch.
- NIC: if provided, exact match is strongest signal. Old format (9 digits + V) and new format (12 digits) may represent the same person.
- If uncertain, use confidence < 0.7 and explain why.

Always call confirm_identity — never end without it."""


def validate_identity(
    source_ref_id: str,
    patient_name: str,
    dob: str = "",
    nic: str = "",
) -> IdentityValidationResult:
    """
    Runs the identity validation agent.
    Returns an IdentityValidationResult with the confirmed correct source_ref_id.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")

    client = Groq(api_key=GROQ_API_KEY)
    executor = IdentityToolExecutor()

    user_msg = (
        f"Validate this patient identity:\n"
        f"  Given source_ref_id: {source_ref_id}\n"
        f"  Patient states their name is: {patient_name}\n"
        f"  Patient states their DOB is: {dob or 'not provided'}\n"
        f"  Patient states their NIC is: {nic or 'not provided'}\n\n"
        f"Determine if this source_ref_id belongs to this patient."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    turns = 0
    while turns < MAX_TURNS and not executor._done:
        turns += 1
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )
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
        # Agent didn't call confirm_identity — treat as correct (fallback)
        return IdentityValidationResult(
            given_id=source_ref_id,
            is_correct=True,
            correct_id=source_ref_id,
            confidence=0.5,
            mismatch_fields=[],
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
