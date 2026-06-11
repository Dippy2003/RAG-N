"""
RAG retriever — vector search over the medical_guidelines knowledge base.

Given a conflict description (or any clinical query), retrieves the top-k
most relevant guidelines from Supabase using pgvector cosine similarity.
The retrieved text is injected into LLM prompts as grounding context.
"""

import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def retrieve_guidelines(query: str, top_k: int = 3, threshold: float = 0.3) -> list[dict]:
    """
    Embed `query` and return the top_k most similar guidelines from the
    medical_guidelines table, each with a similarity score.

    Returns a list of dicts:
        {guideline_id, category, title, content, severity, similarity}
    """
    model = _get_model()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    result = supabase.rpc(
        "match_guidelines",
        {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": top_k,
        }
    ).execute()

    return result.data or []


def format_guidelines_context(guidelines: list[dict]) -> str:
    """
    Formats retrieved guidelines into a readable context block for LLM prompts.
    """
    if not guidelines:
        return "No specific guidelines retrieved for this conflict."

    lines = ["RELEVANT CLINICAL GUIDELINES (retrieved via RAG):"]
    for i, g in enumerate(guidelines, 1):
        sim_pct = round(g.get("similarity", 0) * 100, 1)
        lines.append(
            f"\n[{i}] {g['title']} (severity: {g['severity'].upper()}, relevance: {sim_pct}%)\n"
            f"    {g['content']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test
    queries = [
        "warfarin and aspirin prescribed together dangerous bleeding",
        "penicillin allergy amoxicillin prescribed",
        "blood type conflict clinic versus lab different ABO group",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = retrieve_guidelines(q, top_k=2)
        print(format_guidelines_context(results))
        print("-" * 60)
