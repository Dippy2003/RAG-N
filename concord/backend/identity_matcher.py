"""
Step 3: Identity matcher.
Given a source_ref_id (e.g. "CLN-001"), finds the record, then uses pgvector
cosine similarity to find all other records belonging to the same real patient
across all 3 sources. Returns a cluster of matched records ready for conflict
detection in Step 4.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Cosine similarity threshold: records above this score are considered the same person.
# 0.85 chosen because demographic text (name+dob+nic) is highly deterministic —
# a lower threshold risks false matches between different patients.
SIMILARITY_THRESHOLD = 0.78


def match_patient(source_ref_id: str) -> dict:
    """
    Given a source_ref_id, returns a cluster dict:
    {
        "anchor": <the record we started from>,
        "matches": [<all records for this patient across sources>],
        "sources_found": ["clinic", "lab", "pharmacy"]
    }
    Raises ValueError if the source_ref_id is not found.
    """
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # 1. Fetch the anchor record (the one the clinician looked up)
    response = (
        supabase.table("source_records")
        .select("*")
        .eq("source_ref_id", source_ref_id)
        .execute()
    )
    if not response.data:
        raise ValueError(f"No record found for source_ref_id: {source_ref_id}")

    anchor = response.data[0]
    anchor_embedding = anchor["embedding"]

    if anchor_embedding is None:
        raise ValueError(f"Record {source_ref_id} has no embedding. Run embed_records.py first.")

    # 2. Use pgvector to find all records within the similarity threshold.
    # We call the match_records RPC function (defined below in SQL).
    # This is a single DB call — no Python loop over all records.
    matches_response = supabase.rpc(
        "match_records",
        {
            "query_embedding": anchor_embedding,
            "match_threshold": SIMILARITY_THRESHOLD,
            "match_count": 10
        }
    ).execute()

    matches = matches_response.data

    sources_found = list({r["source"] for r in matches})

    return {
        "anchor": anchor,
        "matches": matches,
        "sources_found": sources_found
    }


if __name__ == "__main__":
    # Quick smoke test: look up Nimal Perera by his clinic ID
    result = match_patient("CLN-001")
    print(f"Anchor: {result['anchor']['name']} ({result['anchor']['source']})")
    print(f"Sources found: {result['sources_found']}")
    print(f"Matched {len(result['matches'])} records:")
    for r in result["matches"]:
        print(f"  [{r['source']}] {r['name']} | DOB: {r['dob']} | similarity: {r['similarity']:.4f}")
