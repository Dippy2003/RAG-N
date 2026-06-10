"""
Step 6: Adjudicator.

Orchestrates LLM Call 1 of the agentic loop:
  1. Match patient records across sources (identity_matcher)
  2. Detect conflicts deterministically (conflict_detector)
  3. Send ALL conflicts to the LLM in ONE call (llm_interface.adjudicate)
  4. Write the cluster, detected conflicts, and adjudications to Supabase

Returns a ReconciliationResult with everything needed for Step 7 (action executor)
and Step 8 (LLM Call 2 escalation review).
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

from conflict_detector import detect_conflicts
from identity_matcher import match_patient
from llm_interface import AdjudicationResult, ConflictAction, adjudicate

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Map LLM ConflictAction → severity level stored in DB
_ACTION_TO_SEVERITY = {
    ConflictAction.ACCEPT_A: "medium",
    ConflictAction.ACCEPT_B: "medium",
    ConflictAction.ESCALATE: "high",
    ConflictAction.FLAG_CRITICAL: "critical",
}

# Map LLM ConflictAction → human-readable DB action string
_ACTION_TO_DB_ACTION = {
    ConflictAction.ACCEPT_A: "accept_source_a",
    ConflictAction.ACCEPT_B: "accept_source_b",
    ConflictAction.ESCALATE: "escalate_to_clinician",
    ConflictAction.FLAG_CRITICAL: "flag_critical_alert",
}


@dataclass
class ReconciliationResult:
    source_ref_id: str
    patient_name: str
    cluster_id: str                        # UUID of the patient_clusters row
    conflicts: list[dict]                  # raw conflict dicts from conflict_detector
    adjudication: AdjudicationResult       # structured LLM Call 1 output
    conflict_ids: list[str]                # UUIDs of detected_conflicts rows
    adjudication_ids: list[str]            # UUIDs of adjudications rows
    reconciled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def reconcile(source_ref_id: str) -> ReconciliationResult:
    """
    Full LLM Call 1 pipeline for a single patient lookup.
    Pass a source_ref_id like "CLN-001" (clinic), "LAB-001" (lab),
    or "PHM-001" (pharmacy) — any source works as the anchor.
    """
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # ------------------------------------------------------------------ #
    # Step 1: Identity matching — find all records for this patient
    # ------------------------------------------------------------------ #
    cluster = match_patient(source_ref_id)
    anchor = cluster["anchor"]
    matches = cluster["matches"]

    print(f"[adjudicator] Matched {len(matches)} records for {anchor['name']}")

    # ------------------------------------------------------------------ #
    # Step 2: Deterministic conflict detection
    # ------------------------------------------------------------------ #
    conflicts = detect_conflicts(cluster)
    print(f"[adjudicator] Detected {len(conflicts)} conflicts")
    for c in conflicts:
        print(f"  [{c['conflict_type'].upper()}] {c['description']}")

    # ------------------------------------------------------------------ #
    # Step 3: LLM Call 1 — adjudicate all conflicts in ONE request
    # ------------------------------------------------------------------ #
    print("[adjudicator] Sending conflicts to LLM (Call 1)...")
    adjudication = adjudicate(conflicts)
    print(f"[adjudicator] LLM adjudicated {len(adjudication.resolutions)} resolutions")

    # ------------------------------------------------------------------ #
    # Step 4: Persist to Supabase
    # ------------------------------------------------------------------ #

    # 4a. Upsert a patient_cluster row
    cluster_row = supabase.table("patient_clusters").insert({
        "canonical_name": anchor["name"],
        "canonical_dob": str(anchor["dob"]),
        "canonical_nic": anchor.get("nic"),
        "source_record_ids": [r["id"] for r in matches],
    }).execute()
    cluster_id = cluster_row.data[0]["id"]
    print(f"[adjudicator] Created cluster {cluster_id}")

    # 4b. Insert detected_conflicts rows (one per conflict)
    conflict_ids: list[str] = []
    for c in conflicts:
        row = supabase.table("detected_conflicts").insert({
            "cluster_id": cluster_id,
            "conflict_type": c["conflict_type"],
            "field": c["field"],
            "source_a": c["source_a"],
            "value_a": c["value_a"],
            "source_b": c["source_b"],
            "value_b": c["value_b"],
        }).execute()
        conflict_ids.append(row.data[0]["id"])

    print(f"[adjudicator] Inserted {len(conflict_ids)} conflict rows")

    # 4c. Insert adjudication rows (one per resolution)
    adjudication_ids: list[str] = []
    for i, resolution in enumerate(adjudication.resolutions):
        # Match resolution to conflict by position (LLM preserves order)
        conflict_id = conflict_ids[i] if i < len(conflict_ids) else conflict_ids[-1]
        original_conflict = conflicts[i] if i < len(conflicts) else conflicts[-1]

        # Determine which source is "trusted" based on the LLM's action
        if resolution.action == ConflictAction.ACCEPT_A:
            trusted_source = original_conflict["source_a"]
            trusted_value = resolution.chosen_value or original_conflict["value_a"]
        elif resolution.action == ConflictAction.ACCEPT_B:
            trusted_source = original_conflict["source_b"]
            trusted_value = resolution.chosen_value or original_conflict["value_b"]
        else:
            # ESCALATE or FLAG_CRITICAL — no single trusted source yet
            trusted_source = original_conflict["source_a"]
            trusted_value = resolution.chosen_value or "pending_human_review"

        row = supabase.table("adjudications").insert({
            "conflict_id": conflict_id,
            "trusted_value": trusted_value,
            "trusted_source": trusted_source,
            "reasoning": resolution.rationale,
            "severity": _ACTION_TO_SEVERITY[resolution.action],
            "action": _ACTION_TO_DB_ACTION[resolution.action],
            "confidence": resolution.confidence,
        }).execute()
        adjudication_ids.append(row.data[0]["id"])

    print(f"[adjudicator] Inserted {len(adjudication_ids)} adjudication rows")
    print(f"[adjudicator] Summary: {adjudication.summary}")

    return ReconciliationResult(
        source_ref_id=source_ref_id,
        patient_name=anchor["name"],
        cluster_id=cluster_id,
        conflicts=conflicts,
        adjudication=adjudication,
        conflict_ids=conflict_ids,
        adjudication_ids=adjudication_ids,
    )


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Reconciling CLN-001 (Nimal Perera)...")
    print("=" * 60)
    result = reconcile("CLN-001")

    print("\n--- Final ReconciliationResult ---")
    print(f"Patient : {result.patient_name}")
    print(f"Cluster : {result.cluster_id}")
    print(f"Conflicts detected : {len(result.conflicts)}")
    print(f"Resolutions from LLM:")
    for r in result.adjudication.resolutions:
        print(f"  [{r.conflict_type}] {r.action.value} | confidence={r.confidence:.2f}")
        print(f"    > {r.rationale}")
    print(f"\nCluster ID   : {result.cluster_id}")
    print(f"Conflict IDs : {result.conflict_ids}")
    print(f"Adjudication IDs: {result.adjudication_ids}")
