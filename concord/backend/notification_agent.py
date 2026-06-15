"""
Notification Agent — creates and stores alerts for clinicians.

Triggered automatically when:
  - A prescription is blocked (drug interaction)
  - A reconciliation escalation is created
  - A patient is registered with high-risk medications

Tools:
  get_escalations     — fetch recent unnotified escalations
  create_notification — store a notification in the DB
  mark_sent           — mark notifications as sent
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


@dataclass
class NotificationResult:
    created: int
    notifications: list[dict] = field(default_factory=list)


def _sb():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def create_notification(
    source_ref_id: str,
    patient_name: str,
    title: str,
    message: str,
    urgency: str = "medium",   # critical / high / medium / low
    notification_type: str = "escalation",  # escalation / prescription_blocked / registration
) -> dict:
    """Insert a notification into the notifications table."""
    record = {
        "source_ref_id": source_ref_id,
        "patient_name": patient_name,
        "title": title,
        "message": message,
        "urgency": urgency.lower(),
        "notification_type": notification_type,
        "is_read": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        sb = _sb()
        resp = sb.table("notifications").insert(record).execute()
        if resp.data:
            return resp.data[0]
        print(f"[notification-agent] WARNING: insert returned no data for '{title}' — notification may not have persisted")
    except Exception as e:
        print(f"[notification-agent] ERROR: failed to store notification '{title}': {e}")
    # Return local record with a flag so callers know it wasn't persisted
    return {**record, "_persisted": False}


def get_notifications(unread_only: bool = False, limit: int = 20) -> list[dict]:
    """Fetch recent notifications."""
    try:
        sb = _sb()
        query = sb.table("notifications").select("*").order("created_at", desc=True).limit(limit)
        if unread_only:
            query = query.eq("is_read", False)
        resp = query.execute()
        return resp.data or []
    except Exception as e:
        print(f"[notification-agent] Failed to fetch notifications: {e}")
        return []


def mark_read(notification_ids: list[str]) -> bool:
    """Mark notifications as read."""
    try:
        sb = _sb()
        for nid in notification_ids:
            sb.table("notifications").update({"is_read": True}).eq("id", nid).execute()
        return True
    except Exception:
        return False


def notify_prescription_blocked(
    source_ref_id: str,
    patient_name: str,
    drug: str,
    reason: str,
    interactions: list[str],
) -> dict:
    """Called automatically when a prescription is blocked."""
    title = f"Prescription Blocked — {drug}"
    message = (
        f"A prescription for {drug} was blocked for patient {patient_name} ({source_ref_id}).\n"
        f"Reason: {reason}\n"
        f"Interactions detected: {', '.join(interactions) if interactions else 'allergy conflict'}"
    )
    return create_notification(
        source_ref_id=source_ref_id,
        patient_name=patient_name,
        title=title,
        message=message,
        urgency="high",
        notification_type="prescription_blocked",
    )


def notify_escalation(
    source_ref_id: str,
    patient_name: str,
    field: str,
    reason: str,
    urgency: str,
) -> dict:
    """Called automatically when a reconciliation conflict is escalated."""
    title = f"Escalation — {field.replace('_', ' ').title()}"
    message = (
        f"Patient: {patient_name} ({source_ref_id})\n"
        f"Field: {field}\n"
        f"Reason: {reason}"
    )
    return create_notification(
        source_ref_id=source_ref_id,
        patient_name=patient_name,
        title=title,
        message=message,
        urgency=urgency,
        notification_type="escalation",
    )
