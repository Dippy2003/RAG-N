-- pgvector similarity search function.
-- Called by the Python identity matcher via supabase.rpc("match_records", ...).
-- Returns all records whose demographic embedding is within match_threshold
-- cosine similarity of the query vector.
create or replace function match_records(
    query_embedding vector(384),
    match_threshold float,
    match_count int
)
returns table (
    id uuid,
    source text,
    source_ref_id text,
    name text,
    dob date,
    nic text,
    phone text,
    address text,
    blood_type text,
    allergies text[],
    medications jsonb,
    similarity float
)
language sql stable
as $$
    select
        id,
        source::text,
        source_ref_id,
        name,
        dob,
        nic,
        phone,
        address,
        blood_type,
        allergies,
        medications,
        1 - (embedding <=> query_embedding) as similarity
    from source_records
    where 1 - (embedding <=> query_embedding) > match_threshold
    order by similarity desc
    limit match_count;
$$;
