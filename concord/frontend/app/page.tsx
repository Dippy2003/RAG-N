"use client";

import { useState } from "react";

const API = "http://localhost:8080";

type Conflict = {
  conflict_type: string;
  field: string;
  source_a: string;
  value_a: string;
  source_b: string;
  value_b: string;
  description: string;
};

type Resolution = {
  conflict_type: string;
  field: string;
  action: string;
  chosen_value: string | null;
  rationale: string;
  confidence: number;
};

type Escalation = {
  field: string;
  reason: string;
  urgency: string;
};

type FieldStatus = {
  field: string;
  provided: string;
  stored: string;
  match: boolean;
};

type IdentityValidation = {
  given_id: string;
  is_correct: boolean;
  correct_id: string;
  confidence: number;
  mismatch_fields: string[];
  field_details: FieldStatus[];
  explanation: string;
  patient_name_found: string;
};

type VerifiedResponse = {
  identity: IdentityValidation;
  reconciliation: ReconcileResponse;
  id_was_corrected: boolean;
};

type ReconcileResponse = {
  source_ref_id: string;
  patient_name: string;
  cluster_id: string;
  conflicts_detected: number;
  conflicts: Conflict[];
  resolutions: Resolution[];
  changes_applied: number;
  escalations: Escalation[];
  overall_safe: boolean;
  adjudication_summary: string;
  escalation_ids: string[];
  mode?: string;
  turns_taken?: number | null;
  guidelines_used?: string[] | null;
};

const CONFLICT_TYPE_COLORS: Record<string, { bg: string; text: string; dot: string }> = {
  drug_interaction:  { bg: "bg-red-500/10",    text: "text-red-400",    dot: "bg-red-400" },
  allergy_mismatch:  { bg: "bg-amber-500/10",  text: "text-amber-400",  dot: "bg-amber-400" },
  data_integrity:    { bg: "bg-purple-500/10", text: "text-purple-400", dot: "bg-purple-400" },
};

const ACTION_STYLE: Record<string, { label: string; cls: string }> = {
  escalate:         { label: "Escalated",      cls: "bg-orange-500/20 text-orange-300 border border-orange-500/30" },
  flag_critical:    { label: "Critical Flag",  cls: "bg-red-500/20 text-red-300 border border-red-500/30" },
  accept_a:         { label: "Accepted A",     cls: "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30" },
  accept_b:         { label: "Accepted B",     cls: "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30" },
};

const URGENCY_STYLE: Record<string, string> = {
  critical: "border-red-500/40 bg-red-500/10 text-red-300",
  high:     "border-orange-500/40 bg-orange-500/10 text-orange-300",
  urgent:   "border-orange-500/40 bg-orange-500/10 text-orange-300",
  medium:   "border-yellow-500/40 bg-yellow-500/10 text-yellow-300",
  low:      "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  routine:  "border-blue-500/40 bg-blue-500/10 text-blue-300",
};

const SAMPLE_IDS = ["CLN-001", "CLN-002", "CLN-003", "LAB-001", "PHM-001"];

type Mode = "pipeline" | "agent" | "verified";

export default function Home() {
  const [refId, setRefId] = useState("");
  const [patientName, setPatientName] = useState("");
  const [dob, setDob] = useState("");
  const [nic, setNic] = useState("");
  const [phone, setPhone] = useState("");
  const [address, setAddress] = useState("");
  const [mode, setMode] = useState<Mode>("pipeline");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ReconcileResponse | null>(null);
  const [verifiedResult, setVerifiedResult] = useState<VerifiedResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleReconcile(id?: string) {
    const target = (id ?? refId).trim();
    if (!target) return;
    if (mode === "verified" && !patientName.trim()) {
      setError("Patient name is required for Verified mode.");
      return;
    }
    setLoading(true);
    setResult(null);
    setVerifiedResult(null);
    setError(null);
    try {
      if (mode === "verified") {
        const res = await fetch(`${API}/reconcile-verified`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source_ref_id: target, patient_name: patientName, dob, nic, phone, address }),
        });
        if (!res.ok) { const b = await res.json(); throw new Error(b.detail ?? `HTTP ${res.status}`); }
        const data: VerifiedResponse = await res.json();
        setVerifiedResult(data);
        setResult(data.reconciliation);
      } else {
        const endpoint = mode === "agent" ? `${API}/reconcile-agent/${target}` : `${API}/reconcile/${target}`;
        const res = await fetch(endpoint, { method: "POST" });
        if (!res.ok) { const b = await res.json(); throw new Error(b.detail ?? `HTTP ${res.status}`); }
        setResult(await res.json());
      }
      if (id) setRefId(id);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#0d1117] text-white font-sans">
      {/* Ambient glow */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-violet-600/10 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative border-b border-white/[0.06] bg-[#0d1117]/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-blue-500/25">
              C
            </div>
            <div>
              <span className="font-semibold text-white tracking-tight">Concord</span>
              <span className="ml-2 text-xs text-white/30">Clinical Record Reconciliation</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-white/40">AI Engine Online</span>
          </div>
        </div>
      </header>

      <main className="relative max-w-5xl mx-auto px-6 py-12">
        {/* Hero */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-medium mb-4">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
            Autonomous · 2 LLM Calls · Zero Manual Triage
          </div>
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-br from-white via-white to-white/50 bg-clip-text text-transparent mb-3">
            Patient Record Reconciliation
          </h1>
          <p className="text-white/40 text-sm max-w-md mx-auto">
            Instantly reconcile fragmented patient records across clinics, labs, and pharmacies using AI-powered conflict resolution.
          </p>
        </div>

        {/* Search card */}
        <div className="relative rounded-2xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-sm p-6 mb-8 shadow-2xl">
          {/* Mode toggle */}
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs text-white/40 font-medium uppercase tracking-widest">Mode</span>
            <div className="flex items-center gap-1 p-1 rounded-xl bg-white/[0.05] border border-white/[0.08]">
              {(["pipeline", "agent", "verified"] as Mode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all capitalize ${
                    mode === m
                      ? m === "verified"
                        ? "bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow"
                        : m === "agent"
                        ? "bg-gradient-to-r from-violet-600 to-blue-600 text-white shadow"
                        : "bg-white/10 text-white shadow"
                      : "text-white/35 hover:text-white/60"
                  }`}
                >
                  {m === "verified" ? "✦ Verified" : m === "agent" ? "⚡ RAG + Agent" : "Pipeline"}
                </button>
              ))}
            </div>
          </div>

          {mode === "agent" && (
            <div className="mb-4 px-4 py-3 rounded-xl bg-violet-500/10 border border-violet-500/20 text-xs text-violet-300 leading-relaxed">
              <span className="font-semibold">Agent mode:</span> LLM drives a tool-use loop — retrieves clinical guidelines (RAG) per conflict, then resolves dynamically.
            </div>
          )}
          {mode === "verified" && (
            <div className="mb-4 px-4 py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-300 leading-relaxed">
              <span className="font-semibold">Verified mode:</span> Two agents run in parallel — Agent 1 validates whether the ID matches the patient&apos;s stated details. If wrong, it finds the correct ID and Agent 2 re-reconciles with it.
            </div>
          )}

          {/* ID input */}
          <div className="flex gap-3 mb-3">
            <div className="flex-1 relative">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-white/25 text-sm">#</span>
              <input
                type="text"
                placeholder="Source Ref ID — CLN-001, LAB-002, PHM-003…"
                value={refId}
                onChange={(e) => setRefId(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleReconcile()}
                className="w-full bg-white/[0.05] border border-white/[0.1] rounded-xl pl-8 pr-4 py-3 text-sm text-white placeholder:text-white/25 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all"
              />
            </div>
            <button
              onClick={() => handleReconcile()}
              disabled={loading || !refId.trim()}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 text-white text-sm font-semibold disabled:opacity-40 hover:opacity-90 transition-all shadow-lg shadow-blue-500/20 active:scale-95"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                  </svg>
                  Running
                </span>
              ) : "Reconcile"}
            </button>
          </div>

          {/* Verified mode extra fields */}
          {mode === "verified" && (
            <div className="space-y-2 mb-3">
              <div className="grid grid-cols-2 gap-2">
                <input type="text" placeholder="Patient name (as stated) *" value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  className="bg-white/[0.05] border border-white/[0.1] rounded-xl px-4 py-2.5 text-sm text-white placeholder:text-white/25 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 transition-all" />
                <input type="text" placeholder="Date of birth (YYYY-MM-DD)" value={dob}
                  onChange={(e) => setDob(e.target.value)}
                  className="bg-white/[0.05] border border-white/[0.1] rounded-xl px-4 py-2.5 text-sm text-white placeholder:text-white/25 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 transition-all" />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <input type="text" placeholder="NIC number" value={nic}
                  onChange={(e) => setNic(e.target.value)}
                  className="bg-white/[0.05] border border-white/[0.1] rounded-xl px-4 py-2.5 text-sm text-white placeholder:text-white/25 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 transition-all" />
                <input type="text" placeholder="Phone number" value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  className="bg-white/[0.05] border border-white/[0.1] rounded-xl px-4 py-2.5 text-sm text-white placeholder:text-white/25 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 transition-all" />
                <input type="text" placeholder="Address" value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  className="bg-white/[0.05] border border-white/[0.1] rounded-xl px-4 py-2.5 text-sm text-white placeholder:text-white/25 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 transition-all" />
              </div>
            </div>
          )}

          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-white/25">Try:</span>
            {SAMPLE_IDS.map((id) => (
              <button
                key={id}
                onClick={() => handleReconcile(id)}
                disabled={loading}
                className="text-xs px-3 py-1 rounded-lg bg-white/[0.05] hover:bg-white/[0.09] border border-white/[0.08] text-white/50 hover:text-white/80 disabled:opacity-30 transition-all"
              >
                {id}
              </button>
            ))}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-5 py-4 mb-6 text-sm text-red-300 flex items-start gap-3">
            <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm-.707-4.293a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L10 11.586l-1.293-1.293a1 1 0 00-1.414 1.414l2 2z" clipRule="evenodd"/>
            </svg>
            <span>{error}</span>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-8">
            <div className="flex flex-col items-center gap-4 text-center">
              <div className="w-12 h-12 rounded-full border-2 border-blue-500/30 border-t-blue-500 animate-spin" />
              <div>
                <p className="text-sm font-medium text-white/70">
                  {mode === "verified" ? "Running dual-agent verification…" : "Running agentic loop…"}
                </p>
                <p className="text-xs text-white/30 mt-1">
                  {mode === "verified"
                    ? "Agent 1: Validating identity · Agent 2: Detecting conflicts · Reconciling"
                    : "Matching identities · Detecting conflicts · Adjudicating · Reviewing"}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="space-y-5">

            {/* Identity validation panel — verified mode only */}
            {verifiedResult && (
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.02] overflow-hidden">
                {/* Header */}
                <div className={`px-6 py-4 flex items-center justify-between border-b border-white/[0.06] ${
                  verifiedResult.identity.mismatch_fields.length > 0
                    ? "bg-amber-500/10"
                    : "bg-emerald-500/10"
                }`}>
                  <div className="flex items-center gap-3">
                    <span className="text-xl">{verifiedResult.identity.mismatch_fields.length > 0 ? "⚠️" : "✅"}</span>
                    <div>
                      <p className={`text-xs font-bold uppercase tracking-widest ${
                        verifiedResult.identity.mismatch_fields.length > 0 ? "text-amber-400" : "text-emerald-400"
                      }`}>
                        {verifiedResult.id_was_corrected
                          ? "Wrong Patient ID — Corrected"
                          : verifiedResult.identity.mismatch_fields.length > 0
                          ? "ID Correct · Detail Mismatches Found"
                          : "Identity Fully Verified"}
                      </p>
                      <p className="text-xs text-white/40 mt-0.5">{verifiedResult.identity.explanation}</p>
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <p className="text-xs text-white/30">Confidence</p>
                    <p className={`text-xl font-bold ${
                      verifiedResult.identity.mismatch_fields.length > 0 ? "text-amber-300" : "text-emerald-300"
                    }`}>{Math.round(verifiedResult.identity.confidence * 100)}%</p>
                  </div>
                </div>

                {/* ID correction row */}
                {verifiedResult.id_was_corrected && (
                  <div className="grid grid-cols-2 gap-px bg-white/[0.04] border-b border-white/[0.06]">
                    <div className="bg-[#0d1117] px-6 py-3">
                      <p className="text-xs text-red-400/60 uppercase tracking-widest mb-1">Given ID</p>
                      <p className="text-sm font-mono font-semibold text-red-300 line-through">{verifiedResult.identity.given_id}</p>
                    </div>
                    <div className="bg-[#0d1117] px-6 py-3">
                      <p className="text-xs text-emerald-400/60 uppercase tracking-widest mb-1">Correct ID Used</p>
                      <p className="text-sm font-mono font-semibold text-emerald-300">{verifiedResult.identity.correct_id}</p>
                    </div>
                  </div>
                )}

                {/* Per-field breakdown table */}
                {verifiedResult.identity.field_details.length > 0 && (
                  <div className="divide-y divide-white/[0.04]">
                    <div className="grid grid-cols-4 px-6 py-2 bg-white/[0.02]">
                      <p className="text-xs text-white/30 uppercase tracking-widest">Field</p>
                      <p className="text-xs text-white/30 uppercase tracking-widest">Patient Stated</p>
                      <p className="text-xs text-white/30 uppercase tracking-widest">On Record</p>
                      <p className="text-xs text-white/30 uppercase tracking-widest text-right">Status</p>
                    </div>
                    {verifiedResult.identity.field_details.map((f) => (
                      <div key={f.field} className={`grid grid-cols-4 px-6 py-3 items-center ${!f.match ? "bg-red-500/[0.04]" : ""}`}>
                        <p className="text-xs font-semibold text-white/50 uppercase tracking-wide">{f.field}</p>
                        <p className={`text-sm font-medium ${!f.match ? "text-red-300" : "text-white/70"}`}>{f.provided || "—"}</p>
                        <p className="text-sm text-white/50">{f.stored || "—"}</p>
                        <div className="flex justify-end">
                          {f.match
                            ? <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 font-semibold">✓ Match</span>
                            : <span className="text-xs px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 font-semibold">✗ Mismatch</span>
                          }
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Patient card */}
            <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-6 shadow-2xl">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <p className="text-xs text-white/35 uppercase tracking-widest mb-1">Reconciled Patient</p>
                  <h2 className="text-3xl font-bold tracking-tight">{result.patient_name}</h2>
                  <p className="text-sm text-white/35 mt-1">
                    {result.source_ref_id} &nbsp;·&nbsp; cluster <code className="text-white/50 font-mono text-xs">{result.cluster_id.slice(0, 8)}</code>
                  </p>
                </div>
                <StatusBadge safe={result.overall_safe} />
              </div>

              <div className="grid grid-cols-3 gap-4">
                <StatCard label="Conflicts" value={result.conflicts_detected} color="from-orange-500 to-red-500" />
                <StatCard label="Auto-Fixed" value={result.changes_applied} color="from-emerald-500 to-teal-500" />
                <StatCard label="Escalated" value={result.escalations.length} color="from-violet-500 to-purple-500" />
              </div>
            </div>

            {/* Agent metadata */}
            {result.mode === "agent" && (
              <div className="rounded-2xl border border-violet-500/20 bg-violet-500/[0.06] px-6 py-4 flex flex-wrap gap-4 items-center">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
                  <span className="text-xs font-semibold text-violet-300 uppercase tracking-widest">RAG + Agent</span>
                </div>
                {result.turns_taken != null && (
                  <div className="flex items-center gap-1.5 text-xs text-violet-300/70">
                    <span className="font-mono bg-violet-500/20 px-2 py-0.5 rounded">{result.turns_taken}</span>
                    <span>tool-use turns</span>
                  </div>
                )}
                {result.guidelines_used && result.guidelines_used.length > 0 && (
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="text-xs text-violet-300/50">Guidelines retrieved:</span>
                    {result.guidelines_used.map((g) => (
                      <span key={g} className="text-xs font-mono px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-300/80">
                        {g}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* AI Summary */}
            <div className="rounded-2xl border border-blue-500/20 bg-blue-500/[0.06] px-6 py-5">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-5 h-5 rounded-md bg-blue-500/20 flex items-center justify-center">
                  <svg className="w-3 h-3 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 2a8 8 0 100 16A8 8 0 0010 2zm1 11H9v-2h2v2zm0-4H9V7h2v2z"/>
                  </svg>
                </div>
                <span className="text-xs font-semibold text-blue-400 uppercase tracking-wide">AI Adjudication Summary</span>
              </div>
              <p className="text-sm text-blue-100/80 leading-relaxed">{result.adjudication_summary}</p>
            </div>

            {/* Conflicts */}
            {result.conflicts.length > 0 && (
              <section>
                <SectionHeader title="Conflicts & Resolutions" count={result.conflicts.length} />
                <div className="space-y-3 mt-3">
                  {result.conflicts.map((c, i) => {
                    const res = result.resolutions[i];
                    const style = CONFLICT_TYPE_COLORS[c.conflict_type] ?? { bg: "bg-white/5", text: "text-white/50", dot: "bg-white/30" };
                    const actionStyle = res ? (ACTION_STYLE[res.action] ?? { label: res.action, cls: "bg-white/10 text-white/50 border border-white/10" }) : null;
                    return (
                      <div key={i} className="rounded-2xl border border-white/[0.08] bg-white/[0.02] overflow-hidden">
                        {/* Top bar */}
                        <div className="flex items-center justify-between px-5 py-3 border-b border-white/[0.06]">
                          <div className="flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${style.dot}`} />
                            <span className={`text-xs font-semibold uppercase tracking-wider ${style.text}`}>
                              {c.conflict_type.replace(/_/g, " ")}
                            </span>
                            <span className="text-xs text-white/25 ml-1">· {c.field}</span>
                          </div>
                          {actionStyle && (
                            <span className={`text-xs font-semibold px-2.5 py-1 rounded-lg ${actionStyle.cls}`}>
                              {actionStyle.label}
                            </span>
                          )}
                        </div>

                        <div className="px-5 py-4">
                          <p className="text-sm text-white/60 mb-4 leading-relaxed">{c.description}</p>

                          {/* Source comparison */}
                          <div className="grid grid-cols-2 gap-3 mb-4">
                            <SourceBox source={c.source_a} value={c.value_a} />
                            <SourceBox source={c.source_b} value={c.value_b} />
                          </div>

                          {/* Resolution */}
                          {res && (
                            <div className="rounded-xl border border-white/[0.06] bg-white/[0.03] px-4 py-3">
                              <p className="text-xs text-white/35 mb-1.5">AI Rationale</p>
                              <p className="text-sm text-white/70 leading-relaxed">{res.rationale}</p>
                              <div className="flex items-center gap-3 mt-3">
                                <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-blue-500 to-violet-500 rounded-full transition-all duration-700"
                                    style={{ width: `${res.confidence * 100}%` }}
                                  />
                                </div>
                                <span className="text-xs text-white/40 whitespace-nowrap">
                                  {Math.round(res.confidence * 100)}% confidence
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            )}

            {/* Escalations */}
            {result.escalations.length > 0 && (
              <section>
                <SectionHeader title="Clinician Escalations" count={result.escalations.length} warn />
                <div className="space-y-2 mt-3">
                  {result.escalations.map((e, i) => (
                    <div
                      key={i}
                      className={`rounded-xl border px-5 py-4 ${URGENCY_STYLE[e.urgency.toLowerCase()] ?? "border-white/10 bg-white/5 text-white/60"}`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-bold uppercase tracking-widest">
                          {e.urgency} &nbsp;·&nbsp; {e.field}
                        </span>
                        <code className="text-xs opacity-40 font-mono">
                          {result.escalation_ids[i]?.slice(0, 8)}&hellip;
                        </code>
                      </div>
                      <p className="text-sm opacity-80">{e.reason}</p>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* All clear */}
            {result.conflicts.length === 0 && (
              <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/[0.06] px-6 py-8 text-center">
                <div className="w-12 h-12 rounded-full bg-emerald-500/20 flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="font-semibold text-emerald-300">No conflicts detected</p>
                <p className="text-sm text-emerald-400/60 mt-1">All records are consistent across sources.</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

function StatusBadge({ safe }: { safe: boolean }) {
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-semibold ${
      safe
        ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
        : "bg-red-500/10 border-red-500/30 text-red-400"
    }`}>
      <span className={`w-1.5 h-1.5 rounded-full ${safe ? "bg-emerald-400" : "bg-red-400 animate-pulse"}`} />
      {safe ? "Safe" : "Needs Review"}
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="rounded-xl border border-white/[0.06] bg-white/[0.03] px-4 py-4 text-center">
      <p className={`text-3xl font-bold bg-gradient-to-br ${color} bg-clip-text text-transparent`}>
        {value}
      </p>
      <p className="text-xs text-white/35 mt-1 uppercase tracking-wide">{label}</p>
    </div>
  );
}

function SourceBox({ source, value }: { source: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2.5">
      <p className="text-xs text-white/30 uppercase tracking-widest mb-1">{source}</p>
      <p className="text-sm font-medium text-white/80">{value}</p>
    </div>
  );
}

function SectionHeader({ title, count, warn }: { title: string; count: number; warn?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest">{title}</h3>
      <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
        warn ? "bg-red-500/20 text-red-400" : "bg-white/10 text-white/50"
      }`}>
        {count}
      </span>
      <div className="flex-1 h-px bg-white/[0.06]" />
    </div>
  );
}
