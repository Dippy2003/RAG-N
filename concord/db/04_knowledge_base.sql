-- Medical guidelines knowledge base table and RAG search function.
-- Run this in the Supabase SQL editor before using agent mode.

-- Table to store clinical guidelines with embeddings
create table if not exists medical_guidelines (
    id              uuid primary key default gen_random_uuid(),
    guideline_id    text unique not null,
    category        text not null,   -- "drug_interaction" | "allergy_mismatch" | "data_integrity"
    title           text not null,
    content         text not null,
    severity        text not null,   -- "critical" | "high" | "medium" | "low"
    tags            text,
    embedding       vector(384),
    created_at      timestamptz default now()
);

-- pgvector similarity search over medical guidelines (RAG retrieval)
create or replace function match_guidelines(
    query_embedding vector(384),
    match_threshold float,
    match_count int
)
returns table (
    guideline_id text,
    category     text,
    title        text,
    content      text,
    severity     text,
    similarity   float
)
language sql stable
as $$
    select
        guideline_id,
        category,
        title,
        content,
        severity,
        1 - (embedding <=> query_embedding) as similarity
    from medical_guidelines
    where embedding is not null
      and 1 - (embedding <=> query_embedding) > match_threshold
    order by similarity desc
    limit match_count;
$$;
