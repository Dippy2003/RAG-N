-- Run this in Supabase SQL Editor

create table if not exists notifications (
  id            uuid primary key default gen_random_uuid(),
  source_ref_id text not null,
  patient_name  text not null default '',
  title         text not null,
  message       text not null,
  urgency       text not null default 'medium',   -- critical / high / medium / low
  notification_type text not null default 'escalation', -- escalation / prescription_blocked / registration
  is_read       boolean not null default false,
  created_at    timestamptz not null default now()
);

-- Index for fast unread queries
create index if not exists notifications_is_read_idx on notifications (is_read, created_at desc);

-- Disable RLS so the service key can read/write freely
alter table notifications disable row level security;

-- Optional: prescriptions log table
create table if not exists prescriptions (
  id            uuid primary key default gen_random_uuid(),
  source_ref_id text not null,
  drug          text not null,
  dosage        text default '',
  notes         text default '',
  status        text not null default 'active',   -- active / blocked
  created_at    timestamptz not null default now()
);

alter table prescriptions disable row level security;
