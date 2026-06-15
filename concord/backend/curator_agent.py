"""
Curator Agent — turns a plain-English clinical rule into a structured,
embedded, deduplicated guideline that goes live in the RAG knowledge base.

Flow:
  1. CLASSIFY  — LLM extracts category, severity, a clean title, an expanded
                 content body, and search tags from the user's free text.
  2. EMBED     — encode the guideline with the same model the retriever uses.
  3. DEDUP     — vector-search existing guidelines; if a near-duplicate exists
                 (similarity > DUP_THRESHOLD) return it instead of inserting.
  4. INSERT    — write the new row to medical_guidelines. It is immediately
                 retrievable by the prescription / reconciliation / chat agents.

The moment a guideline is inserted it influences the very next safety check —
no redeploy, no re-seed, no LLM retraining.
"""

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from groq import Groq
from supabase import create_client

from rag_retriever import _get_model, retrieve_guidelines

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"

# Curated (agent-added) guidelines get the CU- prefix so they are easy to spot.
CURATED_PREFIX = "CU"

# Above this cosine similarity, treat the new guideline as a duplicate.
DUP_THRESHOLD = 0.86

_VALID_CATEGORIES = {"drug_interaction", "allergy_mismatch", "data_integrity"}
_VALID_SEVERITIES = {"critical", "high", "medium", "low"}


@dataclass
class CuratorResult:
    success: bool
    action: str                       # "added" | "duplicate" | "error"
    guideline_id: str = ""
    title: str = ""
    category: str = ""
    severity: str = ""
    content: str = ""
    message: str = ""
    duplicate_of: str = ""            # set when action == "duplicate"
    duplicate_similarity: float = 0.0
    tags: str = ""
    fields: dict = field(default_factory=dict)


_CLASSIFY_PROMPT = """You are a clinical knowledge curator. A clinician has given you a new
medical rule in free text. Turn it into a structured guideline.

Return ONLY valid JSON (no markdown, no prose) with these fields:
- "title": a concise clinical title (max ~12 words)
- "category": exactly one of "drug_interaction", "allergy_mismatch", "data_integrity"
- "severity": exactly one of "critical", "high", "medium", "low"
      critical = immediate risk to life (fatal interaction, contraindicated in pregnancy/dengue, anaphylaxis)
      high     = serious harm likely without action
      medium   = important but manageable
      low      = informational / myth-busting
- "content": a clear 2-4 sentence clinical explanation. Expand the user's text into a
      proper guideline body including the recommendation. Keep it factual.
- "tags": a space-separated list of lowercase keywords (drug names, conditions, mechanisms)
      that a search engine would use to find this guideline.

Category guidance:
- drug_interaction  → drug-drug, drug-food, drug-condition interactions and contraindications
- allergy_mismatch  → allergy/cross-reactivity protocols
- data_integrity    → identity, dosing rules, record/data conflict, blood type, renal/hepatic dosing

Example input: "don't give ace inhibitors to pregnant women, risk of fetal harm"
Example output:
{"title":"ACE Inhibitors Contraindicated in Pregnancy","category":"drug_interaction","severity":"critical","content":"ACE inhibitors (e.g. enalapril, lisinopril) are contraindicated in the 2nd and 3rd trimesters of pregnancy due to risk of fetal renal agenesis, oligohydramnios, and fetal death. Recommendation: switch to a pregnancy-safe antihypertensive such as methyldopa, labetalol, or nifedipine before conception or as soon as pregnancy is confirmed.","tags":"ace inhibitor pregnancy enalapril lisinopril teratogen fetal contraindicated antihypertensive"}"""


def _classify(text: str) -> dict:
    """LLM call → structured guideline fields."""
    client = Groq(api_key=GROQ_API_KEY)

    def _call(model: str):
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLASSIFY_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=500,
        )

    try:
        resp = _call(GROQ_MODEL)
    except Exception as e:
        if "rate_limit_exceeded" in str(e) or "429" in str(e):
            resp = _call(GROQ_FALLBACK_MODEL)
        else:
            raise

    raw = (resp.choices[0].message.content or "").strip()
    raw = raw.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # LLM returned malformed JSON — extract what we can, use safe defaults
        print(f"[curator-agent] WARNING: LLM returned non-JSON, using safe defaults. Raw: {raw[:200]}")
        data = {}

    # Normalise / validate
    category = str(data.get("category", "")).strip().lower()
    if category not in _VALID_CATEGORIES:
        category = "data_integrity"
    severity = str(data.get("severity", "")).strip().lower()
    if severity not in _VALID_SEVERITIES:
        severity = "medium"

    return {
        "title": str(data.get("title", "")).strip() or "Untitled Guideline",
        "category": category,
        "severity": severity,
        "content": str(data.get("content", "")).strip(),
        "tags": str(data.get("tags", "")).strip().lower(),
    }


def _next_guideline_id(sb) -> str:
    """Find the next free CU-NNN id."""
    resp = (
        sb.table("medical_guidelines")
        .select("guideline_id")
        .like("guideline_id", f"{CURATED_PREFIX}-%")
        .execute()
    )
    max_n = 0
    for row in (resp.data or []):
        gid = row.get("guideline_id", "")
        try:
            n = int(gid.split("-")[1])
            max_n = max(max_n, n)
        except (IndexError, ValueError):
            continue
    return f"{CURATED_PREFIX}-{max_n + 1:03d}"


def add_guideline(text: str) -> CuratorResult:
    """
    Entry point from api.py. `text` is the clinician's free-text rule.
    """
    if not GROQ_API_KEY:
        return CuratorResult(success=False, action="error", message="GROQ_API_KEY not set.")
    if not text or len(text.strip()) < 8:
        return CuratorResult(
            success=False, action="error",
            message="Please describe the guideline in a full sentence.",
        )

    # 1. CLASSIFY
    try:
        fields = _classify(text)
    except Exception as e:
        return CuratorResult(success=False, action="error", message=f"Could not classify guideline: {e}")

    if not fields["content"]:
        fields["content"] = text.strip()

    embed_text = f"{fields['title']}. {fields['content']} {fields['tags']}"

    # 2. EMBED
    model = _get_model()
    vector = model.encode(embed_text, normalize_embeddings=True).tolist()

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # 3. DEDUP — search the existing KB for a near-identical guideline
    try:
        similar = retrieve_guidelines(
            query=embed_text, top_k=1, threshold=DUP_THRESHOLD,
            category=fields["category"], rerank=False,
        )
    except Exception:
        similar = []

    if similar:
        match = similar[0]
        return CuratorResult(
            success=True, action="duplicate",
            guideline_id=match.get("guideline_id", ""),
            title=match.get("title", ""),
            category=fields["category"],
            severity=match.get("severity", ""),
            content=match.get("content", ""),
            duplicate_of=match.get("guideline_id", ""),
            duplicate_similarity=round(float(match.get("similarity", 0)), 3),
            tags=fields["tags"],
            fields=fields,
            message=(
                f"This looks like an existing guideline "
                f"[{match.get('guideline_id')}] {match.get('title')} "
                f"({round(float(match.get('similarity', 0)) * 100)}% similar). "
                f"No duplicate was added."
            ),
        )

    # 4. INSERT
    guideline_id = _next_guideline_id(sb)
    try:
        sb.table("medical_guidelines").upsert({
            "guideline_id": guideline_id,
            "category": fields["category"],
            "title": fields["title"],
            "content": fields["content"],
            "severity": fields["severity"],
            "tags": fields["tags"],
            "embedding": vector,
        }, on_conflict="guideline_id").execute()
    except Exception as e:
        return CuratorResult(success=False, action="error", message=f"Insert failed: {e}")

    return CuratorResult(
        success=True, action="added",
        guideline_id=guideline_id,
        title=fields["title"],
        category=fields["category"],
        severity=fields["severity"],
        content=fields["content"],
        tags=fields["tags"],
        fields=fields,
        message=f"Added {guideline_id} ({fields['severity'].upper()}). It is now live in safety checks.",
    )


if __name__ == "__main__":
    # Smoke test (does not insert unless DB is reachable)
    r = add_guideline("Don't give ACE inhibitors to pregnant patients — risk of fetal harm")
    print(json.dumps(r.__dict__, indent=2, default=str))
