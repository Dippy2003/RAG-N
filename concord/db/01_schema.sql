-- Enable pgvector extension for similarity search
create extension if not exists vector;

-- Enum for the 3 data sources
create type data_source as enum ('clinic', 'lab', 'pharmacy');

-- Enum for conflict types we detect deterministically
create type conflict_type as enum ('drug_interaction', 'allergy_mismatch', 'data_integrity');

-- Enum for adjudication severity (set by LLM Call 1)
create type severity_level as enum ('low', 'medium', 'high', 'critical');

-- Raw patient records as received from each source (no cleaning)
create table source_records (
    id            uuid primary key default gen_random_uuid(),
    source        data_source not null,
    source_ref_id text not null,           -- the ID this source uses internally
    name          text not null,
    dob           date not null,
    nic           text,                    -- Sri Lankan NIC, may be missing or malformed
    phone         text,
    address       text,
    blood_type    text,
    allergies     text[],                  -- free-text allergy list from this source
    medications   jsonb,                  -- [{ name, dose, frequency }]
    created_at    timestamptz default now(),
    embedding     vector(384)             -- sentence-transformers all-MiniLM-L6-v2 output
);

-- Reconciled patient clusters (one row = one real person across sources)
create table patient_clusters (
    id              uuid primary key default gen_random_uuid(),
    canonical_name  text not null,
    canonical_dob   date not null,
    canonical_nic   text,
    source_record_ids uuid[],             -- all source_records that belong to this person
    reconciled_at   timestamptz default now()
);

-- Conflicts detected deterministically (Step 2 of the agentic loop)
create table detected_conflicts (
    id               uuid primary key default gen_random_uuid(),
    cluster_id       uuid references patient_clusters(id),
    conflict_type    conflict_type not null,
    field            text not null,       -- which field conflicts (e.g. "medications", "blood_type")
    source_a         data_source not null,
    value_a          text not null,
    source_b         data_source not null,
    value_b          text not null,
    detected_at      timestamptz default now()
);

-- Adjudication results written by LLM Call 1
create table adjudications (
    id               uuid primary key default gen_random_uuid(),
    conflict_id      uuid references detected_conflicts(id),
    trusted_value    text not null,
    trusted_source   data_source not null,
    reasoning        text not null,
    severity         severity_level not null,
    action           text not null,       -- e.g. "alert_prescriber", "update_record", "flag_referral"
    confidence       float not null check (confidence between 0 and 1),
    adjudicated_at   timestamptz default now()
);

-- Escalations flagged by LLM Call 2 (low-confidence items needing human review)
create table escalations (
    id               uuid primary key default gen_random_uuid(),
    adjudication_id  uuid references adjudications(id),
    reason           text not null,
    escalated_at     timestamptz default now(),
    resolved         boolean default false
);

-- Index for fast vector similarity search (identity matching)
create index on source_records using ivfflat (embedding vector_cosine_ops) with (lists = 10);
