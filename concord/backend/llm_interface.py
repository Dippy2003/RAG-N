"""
Step 5: LLM interface.

Wraps Gemini 2.0 Flash (primary) + Groq llama-3.3-70b (fallback) behind a
single adjudicate() call. Provides strict Pydantic response models so the
rest of the pipeline never touches raw LLM strings.

Exactly 2 LLM calls are made per reconciliation:
  Call 1 — adjudicate(conflicts)   → AdjudicationResult   [Step 6 drives this]
  Call 2 — review_actions(actions) → EscalationReview     [Step 8 drives this]

This module owns only the transport + schema layer.
"""

import json
import os
from enum import Enum
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, field_validator

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ConflictAction(str, Enum):
    ACCEPT_A = "accept_a"        # take source_a's value as ground truth
    ACCEPT_B = "accept_b"        # take source_b's value as ground truth
    ESCALATE = "escalate"        # cannot resolve — flag for human review
    FLAG_CRITICAL = "flag_critical"  # resolved but patient safety risk, notify clinician


class ConflictResolution(BaseModel):
    conflict_type: str
    field: str
    action: ConflictAction
    chosen_value: str | None       # null when action is ESCALATE
    rationale: str                 # ≤ 2 sentences
    confidence: float              # 0.0 – 1.0

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class AdjudicationResult(BaseModel):
    """LLM Call 1 output: one resolution per conflict, plus an overall summary."""
    resolutions: list[ConflictResolution]
    summary: str                   # ≤ 3 sentences describing the overall reconciliation


class EscalationItem(BaseModel):
    field: str
    reason: str
    urgency: str                   # "routine" | "urgent" | "critical"


class EscalationReview(BaseModel):
    """LLM Call 2 output: which resolved actions still need human eyes."""
    approved_actions: list[str]    # fields whose resolutions look safe
    escalations: list[EscalationItem]
    overall_safe: bool             # True if no critical escalations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ADJUDICATION_SCHEMA = AdjudicationResult.model_json_schema()
_ESCALATION_SCHEMA = EscalationReview.model_json_schema()


def _build_adjudication_prompt(conflicts: list[dict]) -> str:
    conflicts_json = json.dumps(conflicts, indent=2)
    schema_json = json.dumps(_ADJUDICATION_SCHEMA, indent=2)
    return f"""You are a clinical informatics expert reconciling contradictory patient records from multiple Sri Lankan healthcare providers.

For each conflict below, decide the safest resolution. Always prioritise patient safety over data consistency — when in doubt, escalate.

CONFLICTS:
{conflicts_json}

Respond with ONLY valid JSON matching this schema (no markdown, no explanation outside the JSON):
{schema_json}"""


def _build_review_prompt(resolutions: list[dict], original_conflicts: list[dict]) -> str:
    res_json = json.dumps(resolutions, indent=2)
    orig_json = json.dumps(original_conflicts, indent=2)
    schema_json = json.dumps(_ESCALATION_SCHEMA, indent=2)
    return f"""You are a senior clinical safety reviewer. The reconciliation engine has resolved the following patient record conflicts. Review whether any resolutions are unsafe or need a human clinician's attention.

ORIGINAL CONFLICTS:
{orig_json}

PROPOSED RESOLUTIONS:
{res_json}

Respond with ONLY valid JSON matching this schema (no markdown, no explanation outside the JSON):
{schema_json}"""


def _call_gemini(prompt: str) -> str:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    response = model.generate_content(prompt)
    return response.text


def _call_groq(prompt: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return response.choices[0].message.content


def _llm_call(prompt: str) -> str:
    """Try Gemini first; fall back to Groq on any error."""
    if GEMINI_API_KEY:
        try:
            return _call_gemini(prompt)
        except Exception as e:
            print(f"[llm_interface] Gemini failed ({e}), falling back to Groq.")

    if GROQ_API_KEY:
        return _call_groq(prompt)

    raise RuntimeError(
        "No LLM provider available. Set GEMINI_API_KEY or GROQ_API_KEY in .env"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adjudicate(conflicts: list[dict]) -> AdjudicationResult:
    """
    LLM Call 1.
    Sends all conflicts to the LLM in a single request and returns
    structured resolutions. Raises ValueError on unparseable output.
    """
    if not conflicts:
        return AdjudicationResult(resolutions=[], summary="No conflicts to adjudicate.")

    prompt = _build_adjudication_prompt(conflicts)
    raw = _llm_call(prompt)

    try:
        data = json.loads(raw)
        return AdjudicationResult.model_validate(data)
    except Exception as e:
        raise ValueError(f"LLM returned unparseable adjudication output: {e}\nRaw: {raw[:500]}")


def review_actions(
    adjudication: AdjudicationResult,
    original_conflicts: list[dict],
) -> EscalationReview:
    """
    LLM Call 2.
    Reviews the proposed resolutions for safety and flags anything that
    needs a human clinician. Raises ValueError on unparseable output.
    """
    resolutions_as_dicts: list[dict[str, Any]] = [
        r.model_dump() for r in adjudication.resolutions
    ]

    prompt = _build_review_prompt(resolutions_as_dicts, original_conflicts)
    raw = _llm_call(prompt)

    try:
        data = json.loads(raw)
        return EscalationReview.model_validate(data)
    except Exception as e:
        raise ValueError(f"LLM returned unparseable escalation review: {e}\nRaw: {raw[:500]}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_conflicts = [
        {
            "conflict_type": "drug_interaction",
            "field": "medications",
            "source_a": "clinic",
            "value_a": "warfarin",
            "source_b": "pharmacy",
            "value_b": "aspirin",
            "description": "Dangerous interaction: warfarin (clinic) and aspirin (pharmacy) are both recorded for this patient.",
        },
        {
            "conflict_type": "allergy_mismatch",
            "field": "allergies",
            "source_a": "clinic",
            "value_a": "allergy: penicillin",
            "source_b": "pharmacy",
            "value_b": "prescribed: amoxicillin",
            "description": "Allergy mismatch: penicillin allergy at clinic but amoxicillin prescribed at pharmacy.",
        },
        {
            "conflict_type": "data_integrity",
            "field": "blood_type",
            "source_a": "clinic",
            "value_a": "A+",
            "source_b": "lab",
            "value_b": "B+",
            "description": "Blood type conflict: clinic records A+ but lab records B+. Critical for transfusion safety.",
        },
    ]

    print("=== LLM Call 1: Adjudication ===")
    result = adjudicate(sample_conflicts)
    print(result.model_dump_json(indent=2))

    print("\n=== LLM Call 2: Escalation Review ===")
    review = review_actions(result, sample_conflicts)
    print(review.model_dump_json(indent=2))
