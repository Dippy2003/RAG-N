"""
RAG retriever — vector search over the medical_guidelines knowledge base.

Features:
  - Category filtering: restrict to drug_interaction / allergy_mismatch / data_integrity
  - Per-use-case thresholds: prescription (0.30), conflict (0.25), chat (0.35)
  - Hybrid search: vector similarity + optional keyword pre-filter on tags
  - Re-ranking: sort by (similarity * severity_weight) to surface critical results first
"""

import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
MODEL_NAME = "all-MiniLM-L6-v2"

# Severity → numeric weight for re-ranking
_SEVERITY_WEIGHT = {"critical": 2.0, "high": 1.5, "medium": 1.0, "low": 0.7}

# Per-use-case default thresholds
THRESHOLD_PRESCRIPTION = 0.30   # prescription agent: needs precision
THRESHOLD_CONFLICT     = 0.25   # reconciliation: cast wider net
THRESHOLD_CHAT         = 0.35   # general chat: only strong matches

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def retrieve_guidelines(
    query: str,
    top_k: int = 4,
    threshold: float = THRESHOLD_CHAT,
    category: str = "",          # "" = all, "drug_interaction", "allergy_mismatch", "data_integrity"
    keyword: str = "",           # optional tag keyword pre-filter (e.g. "warfarin")
    rerank: bool = True,         # weight by severity
) -> list[dict]:
    """
    Embed `query` and return the top_k most relevant guidelines.

    Returns list of dicts:
        {guideline_id, category, title, content, severity, similarity, score}

    score = similarity * severity_weight (used for re-ranking)
    """
    model = _get_model()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # Fetch more candidates than needed so we can filter/re-rank
    fetch_count = max(top_k * 3, 15)

    result = supabase.rpc(
        "match_guidelines",
        {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": fetch_count,
        }
    ).execute()

    rows = result.data or []

    # Category filter
    if category:
        rows = [r for r in rows if r.get("category", "") == category]

    # Hybrid: tag keyword filter (keep rows where keyword appears in tags or title)
    if keyword:
        kw = keyword.lower()
        rows = [
            r for r in rows
            if kw in (r.get("tags") or "").lower() or kw in r.get("title", "").lower()
        ]

    # Re-rank by severity-weighted score
    if rerank:
        for r in rows:
            weight = _SEVERITY_WEIGHT.get(r.get("severity", "medium"), 1.0)
            r["score"] = r.get("similarity", 0) * weight
        rows.sort(key=lambda r: r["score"], reverse=True)
    else:
        for r in rows:
            r["score"] = r.get("similarity", 0)

    return rows[:top_k]


def retrieve_for_prescription(drug: str, existing_meds: list[str], allergies: list[str], top_k: int = 5) -> list[dict]:
    """
    Targeted RAG retrieval for the prescription agent.
    Searches drug_interaction + allergy_mismatch categories only.
    Uses a richer query combining the new drug with existing medications.
    """
    med_str = ", ".join(existing_meds) if existing_meds else "no current medications"
    allergy_str = ", ".join(allergies) if allergies else "no known allergies"
    query = (
        f"Prescribing {drug} — interactions with {med_str}. "
        f"Patient allergies: {allergy_str}. Safety and contraindications."
    )

    # Fetch drug interactions
    drug_results = retrieve_guidelines(
        query=query,
        top_k=top_k,
        threshold=THRESHOLD_PRESCRIPTION,
        category="drug_interaction",
        keyword=drug.split()[0].lower(),  # use first word of drug name as keyword hint
        rerank=True,
    )

    # Fetch allergy guidelines
    allergy_results = retrieve_guidelines(
        query=query,
        top_k=3,
        threshold=THRESHOLD_PRESCRIPTION,
        category="allergy_mismatch",
        rerank=True,
    )

    # Merge, deduplicate by guideline_id, keep top results
    seen = set()
    merged = []
    for r in drug_results + allergy_results:
        gid = r.get("guideline_id")
        if gid not in seen:
            seen.add(gid)
            merged.append(r)

    merged.sort(key=lambda r: r.get("score", 0), reverse=True)
    return merged[:top_k]


def retrieve_for_conflict(conflict_description: str, conflict_type: str = "", top_k: int = 4) -> list[dict]:
    """
    Targeted RAG retrieval for reconciliation/conflict resolution.
    Maps conflict_type to the right category.
    """
    category_map = {
        "drug_interaction": "drug_interaction",
        "allergy_mismatch": "allergy_mismatch",
        "data_integrity":   "data_integrity",
    }
    category = category_map.get(conflict_type, "")

    return retrieve_guidelines(
        query=conflict_description,
        top_k=top_k,
        threshold=THRESHOLD_CONFLICT,
        category=category,
        rerank=True,
    )


def retrieve_for_chat(query: str, top_k: int = 4) -> list[dict]:
    """
    RAG retrieval for general chat queries.
    Higher threshold — only confident matches injected.
    """
    return retrieve_guidelines(
        query=query,
        top_k=top_k,
        threshold=THRESHOLD_CHAT,
        rerank=True,
    )


def format_guidelines_context(guidelines: list[dict]) -> str:
    """Format retrieved guidelines into a readable context block for LLM prompts."""
    if not guidelines:
        return "No specific guidelines retrieved for this query."

    lines = ["RELEVANT CLINICAL GUIDELINES (retrieved via RAG):"]
    for i, g in enumerate(guidelines, 1):
        sim_pct = round(g.get("similarity", 0) * 100, 1)
        severity = g.get("severity", "").upper()
        lines.append(
            f"\n[{i}] [{severity}] {g['title']} (relevance: {sim_pct}%)\n"
            f"    {g['content']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test
    tests = [
        ("warfarin and aspirin prescribed together", "drug_interaction"),
        ("penicillin allergy patient needs antibiotic", "allergy_mismatch"),
        ("blood type conflict clinic versus lab", "data_integrity"),
        ("dengue fever patient needs pain relief", ""),
        ("metformin contrast CT scan kidney", "drug_interaction"),
        ("child 8 years old aspirin dosing", ""),
    ]
    for q, cat in tests:
        print(f"\nQuery: {q} (category={cat or 'all'})")
        results = retrieve_guidelines(q, top_k=2, threshold=0.2, category=cat)
        print(format_guidelines_context(results))
        print("-" * 60)
