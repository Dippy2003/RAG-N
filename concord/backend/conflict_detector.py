"""
Step 4: Deterministic conflict detector.
Takes a matched patient cluster (from identity_matcher.py) and applies rule-based
checks to find contradictions. No LLM involved — these are hard clinical rules.
Returns a list of structured conflicts ready for LLM Call 1 (adjudication).
"""

# Known dangerous drug interaction pairs (bidirectional).
# In a real system this would be a full drug interaction database.
DANGEROUS_PAIRS = [
    {"warfarin", "aspirin"},
    {"warfarin", "ibuprofen"},
    {"methotrexate", "aspirin"},
]

# Known penicillin-class antibiotics (for allergy cross-reactivity check).
PENICILLIN_CLASS = {"amoxicillin", "ampicillin", "flucloxacillin", "co-amoxiclav", "penicillin"}


def detect_conflicts(cluster: dict) -> list[dict]:
    """
    Runs all conflict rules against a patient cluster.
    Returns a list of conflict dicts, each with:
      - conflict_type: "drug_interaction" | "allergy_mismatch" | "data_integrity"
      - field: which field the conflict is on
      - source_a, value_a: first side of the conflict
      - source_b, value_b: second side of the conflict
      - description: human-readable summary for the LLM prompt
    """
    records = cluster["matches"]
    conflicts = []

    conflicts += _check_drug_interactions(records)
    conflicts += _check_allergy_mismatches(records)
    conflicts += _check_data_integrity(records)

    return conflicts


def _get_medication_names(record: dict) -> set[str]:
    meds = record.get("medications") or []
    return {m["name"].lower() for m in meds if isinstance(m, dict) and "name" in m}


def _check_drug_interactions(records: list[dict]) -> list[dict]:
    """
    Collects all medications across all sources for this patient,
    then checks every pair against the dangerous interactions list.
    """
    conflicts = []

    # Build a map of drug_name -> source for attribution
    drug_source_map: dict[str, str] = {}
    for record in records:
        for drug in _get_medication_names(record):
            drug_source_map[drug] = record["source"]

    all_drugs = set(drug_source_map.keys())

    for pair in DANGEROUS_PAIRS:
        if pair.issubset(all_drugs):
            drug_a, drug_b = tuple(pair)
            conflicts.append({
                "conflict_type": "drug_interaction",
                "field": "medications",
                "source_a": drug_source_map[drug_a],
                "value_a": drug_a,
                "source_b": drug_source_map[drug_b],
                "value_b": drug_b,
                "description": (
                    f"Dangerous interaction: {drug_a} (from {drug_source_map[drug_a]}) "
                    f"and {drug_b} (from {drug_source_map[drug_b]}) "
                    f"are both recorded for this patient."
                )
            })

    return conflicts


def _check_allergy_mismatches(records: list[dict]) -> list[dict]:
    """
    If any source records a known allergy, checks whether other sources
    are prescribing drugs in that allergy class.
    """
    conflicts = []

    # Collect all allergies across sources
    allergy_source_map: dict[str, str] = {}
    for record in records:
        for allergy in (record.get("allergies") or []):
            allergy_source_map[allergy.lower()] = record["source"]

    # Collect all prescribed drugs across sources
    drug_source_map: dict[str, str] = {}
    for record in records:
        for drug in _get_medication_names(record):
            drug_source_map[drug] = record["source"]

    # Check penicillin class specifically
    if "penicillin" in allergy_source_map:
        for drug, source in drug_source_map.items():
            if drug in PENICILLIN_CLASS:
                conflicts.append({
                    "conflict_type": "allergy_mismatch",
                    "field": "allergies",
                    "source_a": allergy_source_map["penicillin"],
                    "value_a": "allergy: penicillin",
                    "source_b": source,
                    "value_b": f"prescribed: {drug}",
                    "description": (
                        f"Allergy mismatch: penicillin allergy recorded at "
                        f"{allergy_source_map['penicillin']} but {drug} "
                        f"(penicillin-class) prescribed at {source}."
                    )
                })

    return conflicts


def _check_data_integrity(records: list[dict]) -> list[dict]:
    """
    Checks for factual contradictions in fields that should be consistent
    across sources: blood type, date of birth.
    """
    conflicts = []

    blood_types = {
        r["source"]: r["blood_type"]
        for r in records
        if r.get("blood_type")
    }

    unique_blood_types = set(blood_types.values())
    if len(unique_blood_types) > 1:
        sources = list(blood_types.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                s_a, s_b = sources[i], sources[j]
                if blood_types[s_a] != blood_types[s_b]:
                    conflicts.append({
                        "conflict_type": "data_integrity",
                        "field": "blood_type",
                        "source_a": s_a,
                        "value_a": blood_types[s_a],
                        "source_b": s_b,
                        "value_b": blood_types[s_b],
                        "description": (
                            f"Blood type conflict: {s_a} records {blood_types[s_a]} "
                            f"but {s_b} records {blood_types[s_b]}. "
                            f"This is critical for transfusion safety."
                        )
                    })

    return conflicts


if __name__ == "__main__":
    from identity_matcher import match_patient

    for ref_id in ["CLN-001", "CLN-002", "CLN-003"]:
        cluster = match_patient(ref_id)
        conflicts = detect_conflicts(cluster)
        print(f"\n--- {cluster['anchor']['name']} ({ref_id}) ---")
        if not conflicts:
            print("  No conflicts detected.")
        for c in conflicts:
            print(f"  [{c['conflict_type'].upper()}] {c['description']}")
