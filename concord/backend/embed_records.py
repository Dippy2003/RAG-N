"""
Step 2: Embedding pipeline.
Loads all source_records that have no embedding yet, generates a 384-dim vector
from each record's demographic text using a local sentence-transformers model,
then writes the vector back to Supabase. No paid API — runs fully offline.
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# all-MiniLM-L6-v2: 384 dimensions, fast, free, downloads once (~90MB)
MODEL_NAME = "all-MiniLM-L6-v2"


def record_to_text(record: dict) -> str:
    """
    Converts a source_record row into a single string for embedding.
    We use only demographic fields — NOT clinical fields — because the purpose
    of this vector is identity matching, not clinical similarity.
    """
    parts = [
        record.get("name") or "",
        str(record.get("dob") or ""),
        record.get("nic") or "",
        record.get("phone") or "",
        record.get("address") or "",
    ]
    return " | ".join(p.strip() for p in parts if p.strip())


def re_embed_record(source_ref_id: str) -> bool:
    """
    Re-generate the embedding for a single patient record after a demographic update.
    Returns True if successful, False if record not found or model unavailable.
    """
    try:
        model = SentenceTransformer(MODEL_NAME)
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

        resp = supabase.table("source_records").select("id, name, dob, nic, phone, address").eq("source_ref_id", source_ref_id).execute()
        if not resp.data:
            return False

        record = resp.data[0]
        text = record_to_text(record)
        vector = model.encode([text], normalize_embeddings=True)[0]
        supabase.table("source_records").update({"embedding": vector.tolist()}).eq("id", record["id"]).execute()
        print(f"[embed] Re-embedded {source_ref_id} ({record.get('name', '')})")
        return True
    except Exception as e:
        print(f"[embed] Failed to re-embed {source_ref_id}: {e}")
        return False


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # Fetch all records — filter nulls in Python because pgvector columns
    # don't respond to .is_("embedding", "null") via the REST API
    response = (
        supabase.table("source_records")
        .select("id, name, dob, nic, phone, address, embedding")
        .execute()
    )
    all_records = response.data
    records = [r for r in all_records if r.get("embedding") is None]
    records = response.data

    if not records:
        print("All records already have embeddings.")
        return

    print(f"Embedding {len(records)} records...")

    texts = [record_to_text(r) for r in records]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    for record, vector in zip(records, embeddings):
        supabase.table("source_records").update(
            {"embedding": vector.tolist()}
        ).eq("id", record["id"]).execute()
        print(f"  ✓ {record['name']} ({record['id'][:8]}...)")

    print(f"\nDone. {len(records)} embeddings written to Supabase.")


if __name__ == "__main__":
    main()
