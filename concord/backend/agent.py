"""
True agentic reconciliation loop using Gemini function calling.

The LLM is given a set of tools and decides:
  - Which patients records to fetch
  - Which conflicts to investigate
  - Which guidelines to retrieve (RAG) for each conflict
  - Whether to resolve autonomously or escalate each conflict
  - When the reconciliation is complete

This replaces the fixed adjudicator → action_executor → escalation_reviewer
pipeline with a dynamic tool-use loop. The LLM drives the sequence.
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from groq import Groq
from dotenv import load_dotenv
from supabase import create_client

from conflict_detector import detect_conflicts
from identity_matcher import match_patient
from rag_retriever import format_guidelines_context, retrieve_guidelines, retrieve_for_conflict

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"

MAX_TURNS = 20  # safety cap on agentic loop iterations


# ---------------------------------------------------------------------------
# Agent output schema
# ---------------------------------------------------------------------------

@dataclass
class AgentResolution:
    conflict_index: int
    conflict_type: str
    field: str
    action: str          # "accept_a" | "accept_b" | "escalate" | "flag_critical"
    chosen_value: str | None
    rationale: str
    confidence: float
    guidelines_used: list[str] = field(default_factory=list)


@dataclass
class AgentEscalation:
    field: str
    reason: str
    urgency: str         # "routine" | "urgent" | "critical"


@dataclass
class AgentReport:
    source_ref_id: str
    patient_name: str
    cluster_id: str
    conflicts: list[dict]
    resolutions: list[AgentResolution]
    escalations: list[AgentEscalation]
    overall_safe: bool
    summary: str
    changes_applied: int
    escalation_ids: list[str]
    turns_taken: int
    reconciled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Tool definitions (Gemini function calling schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "match_patient_records",
            "description": (
                "Fetches all records for a patient across clinic, lab, and pharmacy "
                "using vector similarity matching. Returns the patient cluster with "
                "all matched records and their sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {
                        "type": "string",
                        "description": "The source reference ID to look up (e.g. CLN-001, LAB-002, PHM-003)"
                    }
                },
                "required": ["source_ref_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_record_conflicts",
            "description": (
                "Runs deterministic conflict detection on the matched patient cluster. "
                "Returns a list of conflicts (drug interactions, allergy mismatches, "
                "data integrity issues) ready for investigation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_ref_id": {
                        "type": "string",
                        "description": "The source_ref_id used in match_patient_records"
                    }
                },
                "required": ["source_ref_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_medical_guidelines",
            "description": (
                "Searches the medical knowledge base for clinical guidelines relevant "
                "to a specific conflict. Uses semantic vector search (RAG) to find the "
                "most applicable drug interaction rules, allergy protocols, or data "
                "integrity standards. Always call this before resolving a conflict."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A clinical query describing the conflict, e.g. "
                            "'warfarin and aspirin co-prescribed bleeding risk'"
                        )
                    },
                    "conflict_index": {
                        "type": "integer",
                        "description": "Index of the conflict this retrieval is for (0-based)"
                    }
                },
                "required": ["query", "conflict_index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_conflict",
            "description": (
                "Records your resolution decision for a specific conflict. "
                "Use after retrieving relevant guidelines. "
                "Choose accept_a or accept_b when one source is clearly correct, "
                "flag_critical for serious patient safety issues resolved but needing "
                "clinician notification, or escalate when you cannot decide autonomously."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conflict_index": {
                        "type": "integer",
                        "description": "Index of the conflict being resolved (0-based)"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["accept_a", "accept_b", "escalate", "flag_critical"],
                        "description": "Resolution action"
                    },
                    "chosen_value": {
                        "type": "string",
                        "description": "The accepted value (omit if escalating)"
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Clinical rationale for this decision (1-2 sentences)"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0.0-1.0"
                    }
                },
                "required": ["conflict_index", "action", "rationale", "confidence"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_conflict",
            "description": (
                "Flags a conflict for urgent clinician review. Use when the conflict "
                "poses a safety risk that cannot be resolved autonomously."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conflict_index": {
                        "type": "integer",
                        "description": "Index of the conflict to escalate (0-based)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this needs clinician review"
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["routine", "urgent", "critical"],
                        "description": "Urgency level"
                    }
                },
                "required": ["conflict_index", "reason", "urgency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_reconciliation",
            "description": (
                "Call this when you have resolved or escalated every conflict. "
                "Provide a concise summary of the reconciliation outcome."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "2-3 sentence summary of the reconciliation outcome"
                    },
                    "overall_safe": {
                        "type": "boolean",
                        "description": "True if all critical issues are resolved or escalated"
                    }
                },
                "required": ["summary", "overall_safe"]
            }
        }
    },
]


# ---------------------------------------------------------------------------
# Tool executor (maps LLM tool calls → actual Python functions)
# ---------------------------------------------------------------------------

class ToolExecutor:
    def __init__(self, source_ref_id: str):
        self.source_ref_id = source_ref_id
        self._cluster: dict | None = None
        self._conflicts: list[dict] = []
        self._guidelines_cache: dict[int, list[dict]] = {}  # conflict_index -> guidelines
        self.resolutions: list[AgentResolution] = []
        self.escalations: list[AgentEscalation] = []
        self._done = False
        self._summary = ""
        self._overall_safe = True

    def execute(self, tool_name: str, args: dict) -> str:
        """Dispatch tool call and return string result for LLM."""
        handlers = {
            "match_patient_records":    self._match_patient,
            "detect_record_conflicts":  self._detect_conflicts,
            "retrieve_medical_guidelines": self._retrieve_guidelines,
            "resolve_conflict":         self._resolve_conflict,
            "escalate_conflict":        self._escalate_conflict,
            "complete_reconciliation":  self._complete,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        try:
            return handler(**args)
        except Exception as e:
            return f"Tool error: {e}"

    def _match_patient(self, source_ref_id: str) -> str:
        self._cluster = match_patient(source_ref_id)
        anchor = self._cluster["anchor"]
        sources = self._cluster["sources_found"]
        matches = self._cluster["matches"]
        return json.dumps({
            "patient_name": anchor["name"],
            "dob": str(anchor.get("dob")),
            "nic": anchor.get("nic"),
            "sources_found": sources,
            "record_count": len(matches),
            "records_summary": [
                {
                    "source": r["source"],
                    "name": r["name"],
                    "blood_type": r.get("blood_type"),
                    "allergies": r.get("allergies"),
                    "medications": r.get("medications"),
                    "similarity": round(r.get("similarity", 1.0), 4),
                }
                for r in matches
            ]
        }, indent=2)

    def _detect_conflicts(self, source_ref_id: str) -> str:
        if self._cluster is None:
            return "Error: call match_patient_records first."
        self._conflicts = detect_conflicts(self._cluster)
        if not self._conflicts:
            return json.dumps({"conflicts": [], "message": "No conflicts detected. Records are consistent."})
        return json.dumps({
            "conflict_count": len(self._conflicts),
            "conflicts": [
                {
                    "index": i,
                    "conflict_type": c["conflict_type"],
                    "field": c["field"],
                    "source_a": c["source_a"],
                    "value_a": c["value_a"],
                    "source_b": c["source_b"],
                    "value_b": c["value_b"],
                    "description": c["description"],
                }
                for i, c in enumerate(self._conflicts)
            ]
        }, indent=2)

    def _retrieve_guidelines(self, query: str, conflict_index: int) -> str:
        # Detect conflict type from context to apply category filter
        conflict_type = ""
        if conflict_index < len(self._conflicts):
            conflict_type = self._conflicts[conflict_index].get("conflict_type", "")
        guidelines = retrieve_for_conflict(query, conflict_type=conflict_type, top_k=4)
        self._guidelines_cache[conflict_index] = guidelines
        formatted = format_guidelines_context(guidelines)
        return formatted

    def _resolve_conflict(
        self,
        conflict_index: int,
        action: str,
        rationale: str,
        confidence: float,
        chosen_value: str | None = None,
    ) -> str:
        if conflict_index >= len(self._conflicts):
            return f"Error: conflict_index {conflict_index} out of range."
        c = self._conflicts[conflict_index]
        guidelines_used = [
            g["guideline_id"]
            for g in self._guidelines_cache.get(conflict_index, [])
        ]
        self.resolutions.append(AgentResolution(
            conflict_index=conflict_index,
            conflict_type=c["conflict_type"],
            field=c["field"],
            action=action,
            chosen_value=chosen_value,
            rationale=rationale,
            confidence=max(0.0, min(1.0, float(confidence))),
            guidelines_used=guidelines_used,
        ))
        return json.dumps({
            "status": "recorded",
            "conflict_index": conflict_index,
            "action": action,
            "guidelines_referenced": guidelines_used,
        })

    def _escalate_conflict(self, conflict_index: int, reason: str, urgency: str) -> str:
        if conflict_index >= len(self._conflicts):
            return f"Error: conflict_index {conflict_index} out of range."
        c = self._conflicts[conflict_index]
        self.escalations.append(AgentEscalation(
            field=c["field"],
            reason=reason,
            urgency=urgency,
        ))
        if urgency == "critical":
            self._overall_safe = False
        return json.dumps({
            "status": "escalated",
            "conflict_index": conflict_index,
            "field": c["field"],
            "urgency": urgency,
        })

    def _complete(self, summary: str, overall_safe: bool) -> str:
        self._summary = summary
        self._overall_safe = overall_safe
        self._done = True
        return json.dumps({"status": "complete", "summary": summary})


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are Concord, an autonomous clinical record reconciliation agent for Sri Lankan healthcare.

Your mission: reconcile fragmented patient records across clinic, lab, and pharmacy sources to ensure patient safety.

Workflow for EVERY reconciliation:
1. Call match_patient_records to fetch all records for the patient.
2. Call detect_record_conflicts to find all conflicts.
3. For EACH conflict:
   a. Call retrieve_medical_guidelines with a targeted clinical query about that specific conflict.
   b. Based on the retrieved guidelines, call either resolve_conflict or escalate_conflict.
4. Once ALL conflicts are handled, call complete_reconciliation with a summary.

Key principles:
- Always retrieve guidelines BEFORE resolving — never make clinical decisions without evidence.
- Prioritise patient safety. When in doubt, escalate rather than resolve autonomously.
- Drug interactions and allergy mismatches are critical — flag_critical or escalate unless guidelines clearly support one source.
- Be precise: use the exact conflict_index from detect_record_conflicts in all subsequent calls.
- Do not stop early. Resolve or escalate every single conflict before calling complete_reconciliation."""


def run_agent(source_ref_id: str) -> AgentReport:
    """
    Runs the full agentic reconciliation loop for one patient using Groq tool calling.
    Returns an AgentReport with all resolutions, escalations, and metadata.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Agentic mode requires Groq.")

    client = Groq(api_key=GROQ_API_KEY)
    executor = ToolExecutor(source_ref_id)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Reconcile patient records for source_ref_id: {source_ref_id}"},
    ]

    turns = 0
    while turns < MAX_TURNS and not executor._done:
        turns += 1

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception as e:
            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                response = client.chat.completions.create(
                    model=GROQ_FALLBACK_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                )
            else:
                raise

        msg = response.choices[0].message
        messages.append(msg)

        tool_calls = msg.tool_calls or []
        if not tool_calls:
            print(f"[agent] No tool calls in turn {turns}. Loop ending.")
            break

        # Execute each tool call and append results to message history
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            print(f"[agent] Turn {turns}: calling {tool_name}({list(args.keys())})")
            result_text = executor.execute(tool_name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

    print(f"[agent] Loop complete in {turns} turns.")

    # ---------------------------------------------------------------------------
    # Persist to Supabase
    # ---------------------------------------------------------------------------
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    anchor = executor._cluster["anchor"] if executor._cluster else {}

    cluster_row = supabase.table("patient_clusters").insert({
        "canonical_name": anchor.get("name", "unknown"),
        "canonical_dob": str(anchor.get("dob", "")),
        "canonical_nic": anchor.get("nic"),
        "source_record_ids": [r["id"] for r in executor._cluster.get("matches", [])],
    }).execute()
    cluster_id = cluster_row.data[0]["id"]

    conflict_ids: list[str] = []
    for c in executor._conflicts:
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

    _ACTION_TO_SEVERITY = {
        "accept_a": "medium", "accept_b": "medium",
        "escalate": "high", "flag_critical": "critical",
    }
    _ACTION_TO_DB = {
        "accept_a": "accept_source_a", "accept_b": "accept_source_b",
        "escalate": "escalate_to_clinician", "flag_critical": "flag_critical_alert",
    }

    adjudication_ids: list[str] = []
    changes_applied = 0

    for res in executor.resolutions:
        conflict_id = (
            conflict_ids[res.conflict_index]
            if res.conflict_index < len(conflict_ids)
            else conflict_ids[-1] if conflict_ids else None
        )
        if conflict_id is None:
            continue

        c = executor._conflicts[res.conflict_index]
        if res.action == "accept_a":
            trusted_source, trusted_value = c["source_a"], res.chosen_value or c["value_a"]
        elif res.action == "accept_b":
            trusted_source, trusted_value = c["source_b"], res.chosen_value or c["value_b"]
        else:
            trusted_source, trusted_value = c["source_a"], "pending_human_review"

        row = supabase.table("adjudications").insert({
            "conflict_id": conflict_id,
            "trusted_value": trusted_value,
            "trusted_source": trusted_source,
            "reasoning": res.rationale,
            "severity": _ACTION_TO_SEVERITY.get(res.action, "medium"),
            "action": _ACTION_TO_DB.get(res.action, res.action),
            "confidence": res.confidence,
        }).execute()
        adj_id = row.data[0]["id"]
        adjudication_ids.append(adj_id)

        if res.action in ("accept_a", "accept_b"):
            changes_applied += 1

    escalation_ids: list[str] = []
    for i, esc in enumerate(executor.escalations):
        adj_id = adjudication_ids[i] if i < len(adjudication_ids) else (adjudication_ids[-1] if adjudication_ids else None)
        if adj_id is None:
            continue
        row = supabase.table("escalations").insert({
            "adjudication_id": adj_id,
            "reason": f"[{esc.urgency.upper()}] {esc.reason}",
            "resolved": False,
        }).execute()
        escalation_ids.append(row.data[0]["id"])

    return AgentReport(
        source_ref_id=source_ref_id,
        patient_name=anchor.get("name", "Unknown"),
        cluster_id=cluster_id,
        conflicts=executor._conflicts,
        resolutions=executor.resolutions,
        escalations=executor.escalations,
        overall_safe=executor._overall_safe,
        summary=executor._summary or "Agent completed reconciliation.",
        changes_applied=changes_applied,
        escalation_ids=escalation_ids,
        turns_taken=turns,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Agentic reconciliation: CLN-001")
    print("=" * 60)
    report = run_agent("CLN-001")

    print(f"\nPatient        : {report.patient_name}")
    print(f"Cluster        : {report.cluster_id}")
    print(f"Conflicts      : {len(report.conflicts)}")
    print(f"Resolutions    : {len(report.resolutions)}")
    print(f"Escalations    : {len(report.escalations)}")
    print(f"Changes applied: {report.changes_applied}")
    print(f"Overall safe   : {report.overall_safe}")
    print(f"Turns taken    : {report.turns_taken}")
    print(f"\nSummary: {report.summary}")
