"""
Query Agent — handles all list/search operations against Supabase.

Supports:
  all_patients     — list all patients (optionally filtered by source CLN/LAB/PHM)
  prescriptions    — list prescriptions for a patient or all
  escalations      — list escalations (optionally filter: unresolved/resolved)
  medications      — list medications for a patient
  allergies        — list allergies for a patient
  notifications    — list notifications (optionally unread only)
  search           — find patients by medication, allergy, blood type, or name keyword
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


@dataclass
class QueryResult:
    success: bool
    query_type: str
    rows: list[dict] = field(default_factory=list)
    total: int = 0
    message: str = ""


def run_query(params: dict) -> QueryResult:
    """
    Entry point from api.py.
    params contains: query_type, source_ref_id, filter, source
    """
    query_type = params.get("query_type", "all_patients")
    source_ref_id = (params.get("source_ref_id") or "").strip().upper()
    filter_val = (params.get("filter") or "").strip().lower()
    source = (params.get("source") or "").strip().upper()

    try:
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

        if query_type == "all_patients":
            return _all_patients(sb, source)

        if query_type == "prescriptions":
            return _prescriptions(sb, source_ref_id, filter_val)

        if query_type == "escalations":
            return _escalations(sb, source_ref_id, filter_val)

        if query_type == "medications":
            return _medications(sb, source_ref_id)

        if query_type == "allergies":
            return _allergies(sb, source_ref_id)

        if query_type == "notifications":
            return _notifications(sb, source_ref_id, filter_val, source)

        if query_type == "search":
            return _search(sb, filter_val, source)

        return QueryResult(success=False, query_type=query_type, message=f"Unknown query type: {query_type}")

    except Exception as e:
        return QueryResult(success=False, query_type=query_type, message=str(e))


def _all_patients(sb, source: str) -> QueryResult:
    query = sb.table("source_records").select(
        "source_ref_id, source, name, dob, nic, blood_type, phone"
    ).order("source_ref_id")

    if source in ("CLN", "LAB", "PHM"):
        source_map = {"CLN": "clinic", "LAB": "lab", "PHM": "pharmacy"}
        query = query.eq("source", source_map[source])

    resp = query.limit(100).execute()
    rows = resp.data or []
    return QueryResult(
        success=True, query_type="all_patients",
        rows=rows, total=len(rows),
        message=f"Found {len(rows)} patient(s).",
    )


def _prescriptions(sb, source_ref_id: str, filter_val: str) -> QueryResult:
    query = sb.table("prescriptions").select("*").order("created_at", desc=True)
    if source_ref_id:
        query = query.eq("source_ref_id", source_ref_id)
    if filter_val in ("active", "blocked", "discontinued"):
        query = query.eq("status", filter_val)

    resp = query.limit(50).execute()
    rows = resp.data or []
    label = f"for {source_ref_id}" if source_ref_id else "across all patients"
    return QueryResult(
        success=True, query_type="prescriptions",
        rows=rows, total=len(rows),
        message=f"Found {len(rows)} prescription(s) {label}.",
    )


def _escalations(sb, source_ref_id: str, filter_val: str) -> QueryResult:
    # Escalations link: escalations → adjudications → detected_conflicts → patient_clusters
    # Simpler: fetch all escalations and annotate if source_ref_id given
    esc_query = sb.table("escalations").select("id, adjudication_id, reason, escalated_at, resolved").order("escalated_at", desc=True)

    if filter_val == "unresolved":
        esc_query = esc_query.eq("resolved", False)
    elif filter_val == "resolved":
        esc_query = esc_query.eq("resolved", True)

    esc_resp = esc_query.limit(50).execute()
    rows = esc_resp.data or []

    # If patient-specific, filter via adjudication → conflict → cluster
    if source_ref_id and rows:
        src_resp = sb.table("source_records").select("cluster_id").eq("source_ref_id", source_ref_id).execute()
        if src_resp.data and src_resp.data[0].get("cluster_id"):
            cluster_id = src_resp.data[0]["cluster_id"]
            conflict_resp = sb.table("detected_conflicts").select("id").eq("cluster_id", cluster_id).execute()
            conflict_ids = {r["id"] for r in (conflict_resp.data or [])}
            adj_resp = sb.table("adjudications").select("id").in_("conflict_id", list(conflict_ids)).execute()
            adj_ids = {r["id"] for r in (adj_resp.data or [])}
            rows = [r for r in rows if r.get("adjudication_id") in adj_ids]

    status_label = f" ({filter_val})" if filter_val else ""
    patient_label = f" for {source_ref_id}" if source_ref_id else ""
    return QueryResult(
        success=True, query_type="escalations",
        rows=rows, total=len(rows),
        message=f"Found {len(rows)} escalation(s){status_label}{patient_label}.",
    )


def _medications(sb, source_ref_id: str) -> QueryResult:
    if not source_ref_id:
        return QueryResult(success=False, query_type="medications", message="Please specify a patient ID.")

    resp = sb.table("source_records").select("source_ref_id, name, medications").eq("source_ref_id", source_ref_id).execute()
    if not resp.data:
        return QueryResult(success=False, query_type="medications", message=f"No record found for {source_ref_id}.")

    row = resp.data[0]
    meds = row.get("medications") or []
    return QueryResult(
        success=True, query_type="medications",
        rows=[{"medication": m} for m in meds], total=len(meds),
        message=f"{row.get('name', source_ref_id)} has {len(meds)} medication(s).",
    )


def _allergies(sb, source_ref_id: str) -> QueryResult:
    if not source_ref_id:
        return QueryResult(success=False, query_type="allergies", message="Please specify a patient ID.")

    resp = sb.table("source_records").select("source_ref_id, name, allergies").eq("source_ref_id", source_ref_id).execute()
    if not resp.data:
        return QueryResult(success=False, query_type="allergies", message=f"No record found for {source_ref_id}.")

    row = resp.data[0]
    allergies = row.get("allergies") or []
    return QueryResult(
        success=True, query_type="allergies",
        rows=[{"allergy": a} for a in allergies], total=len(allergies),
        message=f"{row.get('name', source_ref_id)} has {len(allergies)} known allergy/allergies.",
    )


def _notifications(sb, source_ref_id: str, filter_val: str, source: str) -> QueryResult:
    query = sb.table("notifications").select("*").order("created_at", desc=True)

    if source_ref_id:
        query = query.eq("source_ref_id", source_ref_id)
    elif source in ("CLN", "LAB", "PHM"):
        query = query.like("source_ref_id", f"{source}-%")

    if filter_val == "unread":
        query = query.eq("is_read", False)

    resp = query.limit(30).execute()
    rows = resp.data or []
    return QueryResult(
        success=True, query_type="notifications",
        rows=rows, total=len(rows),
        message=f"Found {len(rows)} notification(s).",
    )


def _search(sb, filter_val: str, source: str) -> QueryResult:
    """Search patients by medication name, allergy, blood type, or name fragment."""
    if not filter_val:
        return QueryResult(success=False, query_type="search", message="Provide a search term.")

    results = []

    # Search by name
    name_resp = sb.table("source_records").select(
        "source_ref_id, source, name, dob, blood_type, medications, allergies"
    ).ilike("name", f"%{filter_val}%").limit(20).execute()
    results.extend(name_resp.data or [])

    # Search by blood type
    bt_resp = sb.table("source_records").select(
        "source_ref_id, source, name, dob, blood_type, medications, allergies"
    ).ilike("blood_type", f"%{filter_val}%").limit(20).execute()
    for r in (bt_resp.data or []):
        if r["source_ref_id"] not in {x["source_ref_id"] for x in results}:
            results.append(r)

    # Search patients whose medications contain the term
    all_records = sb.table("source_records").select(
        "source_ref_id, source, name, dob, medications, allergies"
    ).limit(200).execute()

    for r in (all_records.data or []):
        ref = r["source_ref_id"]
        if ref in {x["source_ref_id"] for x in results}:
            continue
        meds = r.get("medications") or []
        allergies = r.get("allergies") or []
        med_strs = [str(m).lower() for m in meds]
        allergy_strs = [str(a).lower() for a in allergies]
        if any(filter_val in m for m in med_strs) or any(filter_val in a for a in allergy_strs):
            results.append(r)

    # Filter by source if given
    if source in ("CLN", "LAB", "PHM"):
        results = [r for r in results if r.get("source_ref_id", "").startswith(source)]

    return QueryResult(
        success=True, query_type="search",
        rows=results[:50], total=len(results),
        message=f"Found {len(results)} patient(s) matching '{filter_val}'.",
    )
