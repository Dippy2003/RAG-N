"""
Step 7: Action executor.

Reads the adjudication resolutions from Step 6 and applies them:
  - accept_source_a / accept_source_b  -> overwrite the losing source's field
  - flag_critical_alert                -> mark the conflict record, no field change
  - escalate_to_clinician              -> mark the conflict record, no field change

Only concrete accept actions touch source_records. Escalations and critical
flags are left for LLM Call 2 (Step 8) and human review.

Returns an ExecutionReport summarising what was changed and what was skipped.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from supabase import create_client

from adjudicator import ReconciliationResult
from llm_interface import ConflictAction

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Fields that can be directly overwritten on source_records
_UPDATABLE_FIELDS = {"blood_type", "allergies", "medications"}


@dataclass
class AppliedChange:
    conflict_type: str
    field: str
    source_updated: str        # which source record was corrected
    old_value: str
    new_value: str


@dataclass
class SkippedAction:
    conflict_type: str
    field: str
    reason: str                # "escalate_to_clinician" | "flag_critical_alert"


@dataclass
class ExecutionReport:
    patient_name: str
    cluster_id: str
    applied: list[AppliedChange] = field(default_factory=list)
    skipped: list[SkippedAction] = field(default_factory=list)

    @property
    def has_pending_escalations(self) -> bool:
        return any(
            s.reason in ("escalate_to_clinician", "flag_critical_alert")
            for s in self.skipped
        )


def execute_actions(reconciliation: ReconciliationResult) -> ExecutionReport:
    """
    Applies adjudication decisions to source_records in Supabase.
    Takes the ReconciliationResult produced by adjudicator.reconcile().
    """
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    report = ExecutionReport(
        patient_name=reconciliation.patient_name,
        cluster_id=reconciliation.cluster_id,
    )

    resolutions = reconciliation.adjudication.resolutions
    conflicts = reconciliation.conflicts

    for i, resolution in enumerate(resolutions):
        conflict = conflicts[i] if i < len(conflicts) else conflicts[-1]
        action = resolution.action

        # ---------------------------------------------------------------- #
        # Skip — needs human review, do not touch records
        # ---------------------------------------------------------------- #
        if action in (ConflictAction.ESCALATE, ConflictAction.FLAG_CRITICAL):
            report.skipped.append(SkippedAction(
                conflict_type=conflict["conflict_type"],
                field=conflict["field"],
                reason=_action_label(action),
            ))
            print(
                f"[executor] SKIP [{conflict['conflict_type']}] {conflict['field']} "
                f"— {_action_label(action)}"
            )
            continue

        # ---------------------------------------------------------------- #
        # Accept A or B — overwrite the losing source's field
        # ---------------------------------------------------------------- #
        if action == ConflictAction.ACCEPT_A:
            winning_source = conflict["source_a"]
            winning_value = resolution.chosen_value or conflict["value_a"]
            losing_source = conflict["source_b"]
        else:  # ACCEPT_B
            winning_source = conflict["source_b"]
            winning_value = resolution.chosen_value or conflict["value_b"]
            losing_source = conflict["source_a"]

        db_field = conflict["field"]
        if db_field not in _UPDATABLE_FIELDS:
            report.skipped.append(SkippedAction(
                conflict_type=conflict["conflict_type"],
                field=db_field,
                reason=f"field '{db_field}' is not directly updatable",
            ))
            continue

        # Fetch the losing source's record id from the cluster
        losing_record = _find_record(reconciliation, losing_source)
        if losing_record is None:
            print(f"[executor] WARNING: no record found for source '{losing_source}', skipping")
            continue

        old_value = str(losing_record.get(db_field, ""))

        # Apply the update
        supabase.table("source_records").update(
            {db_field: winning_value}
        ).eq("id", losing_record["id"]).execute()

        report.applied.append(AppliedChange(
            conflict_type=conflict["conflict_type"],
            field=db_field,
            source_updated=losing_source,
            old_value=old_value,
            new_value=str(winning_value),
        ))
        print(
            f"[executor] APPLIED [{conflict['conflict_type']}] "
            f"{losing_source}.{db_field}: '{old_value}' -> '{winning_value}'"
        )

    return report


def _find_record(reconciliation: ReconciliationResult, source: str) -> dict | None:
    """Looks up a matched record by source name from the reconciliation cluster."""
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    # Fetch the cluster's source_record_ids, then find the one matching source
    cluster_row = supabase.table("patient_clusters").select(
        "source_record_ids"
    ).eq("id", reconciliation.cluster_id).execute()

    if not cluster_row.data:
        return None

    record_ids = cluster_row.data[0]["source_record_ids"]
    records = supabase.table("source_records").select("*").in_(
        "id", record_ids
    ).eq("source", source).execute()

    return records.data[0] if records.data else None


def _action_label(action: ConflictAction) -> str:
    return {
        ConflictAction.ESCALATE: "escalate_to_clinician",
        ConflictAction.FLAG_CRITICAL: "flag_critical_alert",
    }.get(action, action.value)


if __name__ == "__main__":
    from adjudicator import reconcile

    print("=" * 60)
    print("Step 7 smoke test: reconcile CLN-001 then execute actions")
    print("=" * 60)

    result = reconcile("CLN-001")

    print("\n[executor] Executing actions...")
    report = execute_actions(result)

    print("\n--- Execution Report ---")
    print(f"Patient : {report.patient_name}")
    print(f"Applied : {len(report.applied)} change(s)")
    for c in report.applied:
        print(f"  [{c.conflict_type}] {c.source_updated}.{c.field}")
        print(f"    was: {c.old_value}")
        print(f"    now: {c.new_value}")
    print(f"Skipped : {len(report.skipped)} action(s) (need human review)")
    for s in report.skipped:
        print(f"  [{s.conflict_type}] {s.field} — {s.reason}")
    print(f"Escalations pending: {report.has_pending_escalations}")
