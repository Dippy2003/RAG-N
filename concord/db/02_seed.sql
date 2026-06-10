-- Seed data: 3 real patients, each with records across sources.
-- Contradictions are deliberately planted for demo purposes.
-- Embeddings are left null here — the Python embedding pipeline fills them in Step 2.

-- ─── PATIENT 1: Nimal Perera ────────────────────────────────────────────────
-- Contradiction planted: warfarin (clinic) + aspirin (pharmacy) → drug interaction

insert into source_records (source, source_ref_id, name, dob, nic, phone, address, blood_type, allergies, medications)
values (
    'clinic', 'CLN-001',
    'Nimal Perera', '1978-04-12', '782031234V', '0771234567',
    '42 Galle Road, Colombo 03',
    'O+',
    array['sulfa drugs'],
    '[{"name":"warfarin","dose":"5mg","frequency":"daily"},{"name":"metformin","dose":"500mg","frequency":"twice daily"}]'::jsonb
);

insert into source_records (source, source_ref_id, name, dob, nic, phone, address, blood_type, allergies, medications)
values (
    'lab', 'LAB-001',
    'N. Perera', '1978-04-12', '782031234V', '0771234567',
    'Colombo 3',
    'O+',
    array['sulfa drugs'],
    '[]'::jsonb  -- lab doesn't record medications
);

insert into source_records (source, source_ref_id, name, dob, nic, phone, address, blood_type, allergies, medications)
values (
    'pharmacy', 'PHM-001',
    'Nimal K Perera', '1978-04-12', null, '0771234567',
    '42 Galle Rd, Colombo',
    null,
    array['sulfa drugs'],
    '[{"name":"aspirin","dose":"100mg","frequency":"daily"},{"name":"metformin","dose":"500mg","frequency":"twice daily"}]'::jsonb
    -- ⚠ aspirin + warfarin = dangerous bleeding risk
);

-- ─── PATIENT 2: Kumari Fernando ─────────────────────────────────────────────
-- Contradiction planted: penicillin allergy (lab) + amoxicillin prescription (clinic)
-- Amoxicillin is a penicillin-class antibiotic → allergy mismatch

insert into source_records (source, source_ref_id, name, dob, nic, phone, address, blood_type, allergies, medications)
values (
    'clinic', 'CLN-002',
    'Kumari Fernando', '1990-11-03', '902984521V', '0712345678',
    '15 Temple Road, Kandy',
    'A+',
    array[]::text[],   -- clinic has NO allergy on record
    '[{"name":"amoxicillin","dose":"500mg","frequency":"three times daily"}]'::jsonb
    -- ⚠ prescribing penicillin-class without knowing allergy
);

insert into source_records (source, source_ref_id, name, dob, nic, phone, address, blood_type, allergies, medications)
values (
    'lab', 'LAB-002',
    'K. Fernando', '1990-11-03', '902984521V', '0712345678',
    'Kandy',
    'A+',
    array['penicillin'],   -- ⚠ allergy is HERE, not at clinic
    '[]'::jsonb
);

insert into source_records (source, source_ref_id, name, dob, nic, phone, address, blood_type, allergies, medications)
values (
    'pharmacy', 'PHM-002',
    'Kumari R Fernando', '1990-11-03', null, '0712345678',
    'Temple Rd, Kandy',
    null,
    array[]::text[],
    '[{"name":"amoxicillin","dose":"500mg","frequency":"three times daily"}]'::jsonb
);

-- ─── PATIENT 3: Suresh Bandara ───────────────────────────────────────────────
-- Contradiction planted: blood type A+ (clinic) vs B+ (lab) → data integrity conflict

insert into source_records (source, source_ref_id, name, dob, nic, phone, address, blood_type, allergies, medications)
values (
    'clinic', 'CLN-003',
    'Suresh Bandara', '1985-07-22', '852033987V', '0769876543',
    '8 Kandy Road, Kurunegala',
    'A+',   -- ⚠ clinic says A+
    array[]::text[],
    '[{"name":"atorvastatin","dose":"10mg","frequency":"nightly"}]'::jsonb
);

insert into source_records (source, source_ref_id, name, dob, nic, phone, address, blood_type, allergies, medications)
values (
    'lab', 'LAB-003',
    'S. M. Bandara', '1985-07-22', '852033987V', '0769876543',
    'Kurunegala',
    'B+',   -- ⚠ lab says B+ — one of them is wrong, matters for transfusions
    array[]::text[],
    '[]'::jsonb
);

insert into source_records (source, source_ref_id, name, dob, nic, phone, address, blood_type, allergies, medications)
values (
    'pharmacy', 'PHM-003',
    'Suresh Bandara', '1985-07-22', null, '0769876543',
    'Kurunegala',
    null,
    array[]::text[],
    '[{"name":"atorvastatin","dose":"10mg","frequency":"nightly"}]'::jsonb
);
