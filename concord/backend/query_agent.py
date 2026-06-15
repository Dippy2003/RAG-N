"""
Query Agent — handles all list/search operations against Supabase.

Supports:
  all_patients     — list all patients (optionally filtered by source CLN/LAB/PHM)
  prescriptions    — list prescriptions for a patient or all
  escalations      — list escalations (optionally filter: unresolved/resolved)
  medications      — list medications for a patient
  allergies        — list allergies for a patient
  notifications    — list notifications (optionally unread only)
  search           — find patients by any field:
                       • source_ref_id (CLN-001)
                       • name (partial, case-insensitive)
                       • NIC number (exact or partial)
                       • phone number (exact or partial)
                       • date of birth (YYYY-MM-DD or partial)
                       • blood type (A+, B-, O+…)
                       • medication name (any med in their list)
                       • allergy name (any allergy in their list)
                       • vector similarity (fuzzy name / phonetic match via embeddings)
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
            # Router sometimes puts the ID in source_ref_id instead of filter
            term = filter_val or source_ref_id
            return _search(sb, term, source)

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
    """Search patients by any identifier or clinical value, plus vector similarity fallback."""
    if not filter_val:
        return QueryResult(success=False, query_type="search", message="Provide a search term.")

    results: list[dict] = []
    seen: set[str] = set()

    def add(rows: list[dict]) -> None:
        for r in rows:
            ref = r.get("source_ref_id", "")
            if ref and ref not in seen:
                seen.add(ref)
                results.append(r)

    cols = "source_ref_id, source, name, dob, nic, phone, blood_type, medications, allergies"
    ref_upper = filter_val.upper()
    fv = filter_val.lower().strip()

    # 1. Exact source_ref_id  (CLN-004)
    add((sb.table("source_records").select(cols).eq("source_ref_id", ref_upper).execute()).data or [])

    # 2. Partial source_ref_id prefix  (CLN, LAB-0)
    if not results:
        add((sb.table("source_records").select(cols).ilike("source_ref_id", f"{ref_upper}%").limit(20).execute()).data or [])

    # 3. Exact NIC  (200334511790)
    add((sb.table("source_records").select(cols).eq("nic", filter_val).execute()).data or [])

    # 4. Partial NIC  (last 4 digits, prefix, etc.)
    add((sb.table("source_records").select(cols).ilike("nic", f"%{filter_val}%").limit(20).execute()).data or [])

    # 5. Phone number  (exact or partial)
    add((sb.table("source_records").select(cols).ilike("phone", f"%{filter_val}%").limit(20).execute()).data or [])

    # 7. Name  (partial, case-insensitive)
    add((sb.table("source_records").select(cols).ilike("name", f"%{filter_val}%").limit(20).execute()).data or [])

    # 8. Blood type  (A+, B-, O+, AB-)
    add((sb.table("source_records").select(cols).ilike("blood_type", f"%{filter_val}%").limit(20).execute()).data or [])

    # 9. Medication / allergy / DOB (scan up to 500 records, filter in Python)
    #    DOB is a date column — ilike doesn't work on it, so we cast to str here.
    scan = (sb.table("source_records").select(cols).limit(500).execute()).data or []
    for r in scan:
        if r.get("source_ref_id") in seen:
            continue
        meds      = [str(m).lower() for m in (r.get("medications") or [])]
        allergies = [str(a).lower() for a in (r.get("allergies")   or [])]
        dob_str   = str(r.get("dob") or "").lower()
        if (any(fv in m for m in meds)
                or any(fv in a for a in allergies)
                or (fv and fv in dob_str)):
            seen.add(r["source_ref_id"])
            results.append(r)

    # 10. Vector similarity fallback — catches phonetic / misspelling variants
    #     Only runs when the exact searches found nothing (avoids unnecessary compute)
    if not results:
        try:
            from rag_retriever import _get_model
            from supabase import create_client as _sc
            model = _get_model()
            query_vec = model.encode(filter_val, normalize_embeddings=True).tolist()
            # Call the Supabase RPC that does cosine similarity against patient embeddings
            sim_resp = sb.rpc("match_patient_records", {
                "query_embedding": query_vec,
                "match_threshold": 0.60,
                "match_count": 10,
            }).execute()
            for r in (sim_resp.data or []):
                if r.get("source_ref_id") not in seen:
                    seen.add(r["source_ref_id"])
                    results.append({**r, "_match_type": "vector_similarity"})
        except Exception:
            pass  # vector search is a best-effort fallback

    # 11. Filter by source prefix (CLN / LAB / PHM) if caller specified
    if source in ("CLN", "LAB", "PHM"):
        results = [r for r in results if r.get("source_ref_id", "").startswith(source)]

    # Annotate match types for the LLM to explain to the user
    msg = (
        f"Found {len(results)} patient(s) matching '{filter_val}'."
        if results
        else f"No patient found matching '{filter_val}'. Searched by ID, NIC, phone, DOB, name, blood type, medications, and allergies."
    )

    return QueryResult(
        success=True, query_type="search",
        rows=results[:50], total=len(results),
        message=msg,
    )
