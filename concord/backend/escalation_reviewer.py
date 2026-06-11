"""
Step 8: Escalation reviewer — LLM Call 2.

Takes the ExecutionReport from Step 7 and the original ReconciliationResult,
sends the skipped/flagged items to the LLM for a safety review, then writes
escalation rows to Supabase for the clinician dashboard.

This is the final step of the agentic loop before the API layer (Step 9).
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from supabase import create_client

from action_executor import ExecutionReport
from adjudicator import ReconciliationResult
from llm_interface import EscalationReview, review_actions

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


@dataclass
class EscalationReport:
    patient_name: str
    cluster_id: str
    overall_safe: bool
    escalation_ids: list[str] = field(default_factory=list)
    review: EscalationReview = None


def run_escalation_review(
    reconciliation: ReconciliationResult,
    execution_report: ExecutionReport,
) -> EscalationReport:
    """
    LLM Call 2 — reviews all proposed adjudications for safety.
    Writes escalation rows to Supabase for any items needing clinician attention.
    """
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # ------------------------------------------------------------------ #
    # LLM Call 2: safety review of all resolutions
    # ------------------------------------------------------------------ #
    print("[escalation] Sending resolutions to LLM for safety review (Call 2)...")
    review = review_actions(reconciliation.adjudication, reconciliation.conflicts)

    print(f"[escalation] Overall safe: {review.overall_safe}")
    print(f"[escalation] Escalations flagged: {len(review.escalations)}")
    for e in review.escalations:
        print(f"  [{e.urgency.upper()}] {e.field}: {e.reason}")

    # ------------------------------------------------------------------ #
    # Write escalation rows to Supabase
    # ------------------------------------------------------------------ #
    escalation_ids: list[str] = []

    for esc in review.escalations:
        # Find the adjudication_id for this field
        adjudication_id = _find_adjudication_id(
            reconciliation, esc.field
        )
        if adjudication_id is None:
            print(f"[escalation] WARNING: no adjudication found for field '{esc.field}', skipping DB write")
            continue

        row = supabase.table("escalations").insert({
            "adjudication_id": adjudication_id,
            "reason": f"[{esc.urgency.upper()}] {esc.reason}",
            "resolved": False,
        }).execute()

        escalation_ids.append(row.data[0]["id"])
        print(f"[escalation] Wrote escalation {row.data[0]['id']} for field '{esc.field}'")

    return EscalationReport(
        patient_name=reconciliation.patient_name,
        cluster_id=reconciliation.cluster_id,
        overall_safe=review.overall_safe,
        escalation_ids=escalation_ids,
        review=review,
    )


def _find_adjudication_id(reconciliation: ReconciliationResult, field: str) -> str | None:
    """Match a field name back to its adjudication_id using conflict position."""
    for i, conflict in enumerate(reconciliation.conflicts):
        if conflict["field"] == field and i < len(reconciliation.adjudication_ids):
            return reconciliation.adjudication_ids[i]
    # fallback: return first adjudication if only one
    if reconciliation.adjudication_ids:
        return reconciliation.adjudication_ids[0]
    return None


if __name__ == "__main__":
    from action_executor import execute_actions
    from adjudicator import reconcile

    print("=" * 60)
    print("Full agentic loop: Steps 6 + 7 + 8")
    print("=" * 60)

    # Step 6: adjudicate
    reconciliation = reconcile("CLN-001")

    # Step 7: execute actions
    print("\n[Step 7] Executing actions...")
    execution_report = execute_actions(reconciliation)

    # Step 8: escalation review (LLM Call 2)
    print("\n[Step 8] Running escalation review...")
    escalation_report = run_escalation_review(reconciliation, execution_report)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Patient          : {escalation_report.patient_name}")
    print(f"Cluster ID       : {escalation_report.cluster_id}")
    print(f"Overall safe     : {escalation_report.overall_safe}")
    print(f"Changes applied  : {len(execution_report.applied)}")
    print(f"Escalations      : {len(escalation_report.escalation_ids)}")
    for e in escalation_report.review.escalations:
        print(f"  [{e.urgency.upper()}] {e.field}: {e.reason}")
    print(f"Escalation IDs   : {escalation_report.escalation_ids}")
    print("\nAgentic loop complete. Awaiting clinician review.")
