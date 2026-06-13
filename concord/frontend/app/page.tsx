"use client";

import { useEffect, useRef, useState } from "react";

const API = "http://localhost:8080";

type Notification = {
  id: string;
  source_ref_id: string;
  patient_name: string;
  title: string;
  message: string;
  urgency: string;
  notification_type: string;
  is_read: boolean;
  created_at: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  guidelines?: string[];
  action?: string;
  action_data?: { source_ref_id?: string; patient_name?: string; existing_id?: string; table?: string; fields?: string };
};

const STARTERS = [
  "Tell me about patient CLN-001",
  "What conflicts does CLN-001 have?",
  "Show all patients",
  "List unresolved escalations",
  "Add clinic patient: Kasun Silva, DOB 1990-05-14, medications: metformin",
  "Add pharmacy patient: Nimal Perera, DOB 1985-03-22",
  "Compare CLN-001 and CLN-002",
  "Search patients with warfarin",
];

function OrbitSpinner() {
  return (
    <div className="relative w-10 h-10">
      {[...Array(8)].map((_, i) => (
        <span
          key={i}
          className="absolute w-2.5 h-2.5 rounded-full"
          style={{
            background: `hsl(${260 + i * 15}, 80%, ${55 + i * 3}%)`,
            top: "50%",
            left: "50%",
            transform: `rotate(${i * 45}deg) translateY(-160%) translate(-50%, -50%)`,
            opacity: 0.25 + i * 0.1,
            animation: `orbit-fade 1.2s ease-in-out infinite`,
            animationDelay: `${i * 0.15}s`,
          }}
        />
      ))}
      <style>{`
        @keyframes orbit-fade {
          0%, 100% { opacity: 0.2; transform: rotate(var(--r)) translateY(-160%) translate(-50%, -50%) scale(0.8); }
          50% { opacity: 1; transform: rotate(var(--r)) translateY(-160%) translate(-50%, -50%) scale(1.1); }
        }
      `}</style>
    </div>
  );
}

function SpinnerRing() {
  return (
    <div className="relative w-8 h-8 animate-[spin_1.4s_linear_infinite]">
      {[...Array(8)].map((_, i) => (
        <span
          key={i}
          className="absolute rounded-full"
          style={{
            width: 9,
            height: 9,
            top: "50%",
            left: "50%",
            background: `hsl(${270 + i * 12}, 90%, 65%)`,
            opacity: (i + 1) / 8,
            transform: `rotate(${i * 45}deg) translateY(-15px) translate(-50%, -50%)`,
          }}
        />
      ))}
    </div>
  );
}

const URGENCY_COLOR: Record<string, string> = {
  critical: "bg-red-500/20 border-red-500/40 text-red-300",
  high:     "bg-orange-500/20 border-orange-500/40 text-orange-300",
  medium:   "bg-yellow-500/20 border-yellow-500/40 text-yellow-300",
  low:      "bg-blue-500/20 border-blue-500/40 text-blue-300",
  prescription_issued: "bg-emerald-500/20 border-emerald-500/40 text-emerald-300",
};

type SourceRole = "ALL" | "CLN" | "LAB" | "PHM";

const SOURCE_ROLES: { id: SourceRole; label: string; color: string }[] = [
  { id: "CLN", label: "Clinic",   color: "text-violet-400 border-violet-500/40 bg-violet-500/10" },
  { id: "LAB", label: "Lab",      color: "text-sky-400 border-sky-500/40 bg-sky-500/10" },
  { id: "PHM", label: "Pharmacy", color: "text-emerald-400 border-emerald-500/40 bg-emerald-500/10" },
];

const DESTRUCTIVE_KEYWORDS = ["delete", "remove patient", "remove record", "drop", "wipe", "erase"];

function isDestructive(msg: string): boolean {
  const lower = msg.toLowerCase();
  return DESTRUCTIVE_KEYWORDS.some((k) => lower.includes(k));
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const [confirmPending, setConfirmPending] = useState<string | null>(null);

  const [activeRole, setActiveRole] = useState<SourceRole>("CLN");
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [notifOpen, setNotifOpen] = useState(false);
  const unreadCount = notifications.filter((n) => !n.is_read).length;

  // Poll notifications every 10 seconds, filtered by active role
  useEffect(() => {
    async function fetchNotifs() {
      try {
        const params = new URLSearchParams({ limit: "30", source: activeRole });
        const res = await fetch(`${API}/notifications?${params}`);
        if (res.ok) setNotifications(await res.json());
      } catch { /* backend may not be ready */ }
    }
    fetchNotifs();
    const interval = setInterval(fetchNotifs, 10000);
    return () => clearInterval(interval);
  }, [activeRole]);

  async function markAllRead() {
    const unreadIds = notifications.filter((n) => !n.is_read).map((n) => n.id);
    if (!unreadIds.length) return;
    await fetch(`${API}/notifications/read`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(unreadIds),
    });
    setNotifications((prev) => prev.map((n) => ({ ...n, is_read: true })));
  }

  async function clearAllNotifications() {
    await fetch(`${API}/notifications?source=${activeRole}`, { method: "DELETE" });
    setNotifications([]);
  }

  async function deleteNotification(id: string) {
    await fetch(`${API}/notifications/${id}`, { method: "DELETE" });
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(text?: string, confirmed = false) {
    const msg = (text ?? input).trim();
    if (!msg || loading) return;

    // Intercept destructive operations — ask for confirmation first
    if (!confirmed && isDestructive(msg)) {
      setConfirmPending(msg);
      return;
    }

    const userMsg: ChatMessage = { role: "user", content: msg };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: msg,
          history: updated.slice(-12).map((m) => ({ role: m.role, content: m.content })),
          source_ref_id: "",
          reconciliation_context: null,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.reply,
          guidelines: data.guidelines_used,
          action: data.action,
          action_data: data.action_data,
        },
      ]);
    } catch (e: unknown) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Something went wrong: ${e instanceof Error ? e.message : "Unknown error"}`,
        },
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }

  return (
    <div className="flex flex-col h-screen bg-[#18191a] text-white font-sans">

      {/* Top bar */}
      <header className="shrink-0 px-6 py-4 flex items-center justify-between border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center text-white font-bold text-sm">
            C
          </div>
          <span className="font-semibold tracking-tight">Concord</span>
          <span className="text-xs text-white/30 hidden sm:block">· Clinical AI</span>
        </div>

        {/* Role selector */}
        <div className="flex items-center gap-1 bg-white/4 border border-white/8 rounded-full px-1 py-1">
          {SOURCE_ROLES.map((r) => (
            <button
              key={r.id}
              onClick={() => { setActiveRole(r.id); setNotifications([]); }}
              className={`text-xs px-3 py-1 rounded-full font-semibold transition-all ${
                activeRole === r.id
                  ? `border ${r.color}`
                  : "text-white/30 hover:text-white/60"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-white/35">Online</span>
          </div>
          {/* Notification bell */}
          <div className="relative">
            <button
              onClick={() => { setNotifOpen((o) => !o); if (!notifOpen) markAllRead(); }}
              className="relative w-8 h-8 rounded-full bg-white/5 border border-white/10 flex items-center justify-center hover:bg-white/10 transition-all"
            >
              <svg className="w-4 h-4 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6 6 0 00-9.33-5.004M9 17H4l1.405-1.405A2.032 2.032 0 006 14.158V11a6 6 0 016-6 6 6 0 016 6v3.159c0 .538.214 1.055.595 1.436L19 17h-4m-6 0v1a3 3 0 006 0v-1m-6 0h6" />
              </svg>
              {unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-red-500 text-white text-[9px] font-bold flex items-center justify-center">
                  {unreadCount > 9 ? "9+" : unreadCount}
                </span>
              )}
            </button>

            {/* Dropdown */}
            {notifOpen && (
              <div className="absolute right-0 top-10 w-80 max-h-96 overflow-y-auto rounded-2xl border border-white/10 bg-[#1e1f21] shadow-2xl z-50">
                <div className="flex items-center justify-between px-4 py-3 border-b border-white/8">
                  <div>
                    <span className="text-sm font-semibold text-white">Notifications</span>
                    <span className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full border font-semibold ${SOURCE_ROLES.find(r => r.id === activeRole)?.color ?? ""}`}>
                      {SOURCE_ROLES.find(r => r.id === activeRole)?.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={markAllRead} className="text-xs text-white/30 hover:text-white/60 transition-colors">Read</button>
                    <span className="text-white/15">·</span>
                    <button onClick={clearAllNotifications} className="text-xs text-red-400/50 hover:text-red-400 transition-colors">Clear all</button>
                  </div>
                </div>
                {notifications.length === 0 ? (
                  <div className="px-4 py-8 text-center text-sm text-white/30">No notifications for {SOURCE_ROLES.find(r => r.id === activeRole)?.label}</div>
                ) : (
                  <div className="divide-y divide-white/5">
                    {notifications.map((n) => (
                      <div key={n.id} className={`px-4 py-3 ${!n.is_read ? "bg-white/[0.03]" : ""}`}>
                        <div className="flex items-start gap-2">
                          <span className={`mt-0.5 text-[10px] px-1.5 py-0.5 rounded border font-semibold uppercase shrink-0 ${URGENCY_COLOR[n.urgency] ?? URGENCY_COLOR.low}`}>
                            {n.urgency}
                          </span>
                          <div className="min-w-0 flex-1">
                            <p className="text-xs font-semibold text-white/80 truncate">{n.title}</p>
                            <p className="text-xs text-white/40 mt-0.5 line-clamp-2">{n.message}</p>
                            <p className="text-[10px] text-white/20 mt-1">{n.patient_name} · {new Date(n.created_at).toLocaleTimeString()}</p>
                          </div>
                          <div className="flex flex-col items-end gap-1 shrink-0">
                            {!n.is_read && <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />}
                            <button onClick={() => deleteNotification(n.id)} className="text-white/20 hover:text-red-400 transition-colors text-xs leading-none">✕</button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Message area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && !loading ? (
          /* ── Empty / welcome screen ── */
          <div className="flex flex-col items-center justify-center h-full px-6 text-center">
            <div className="mb-6">
              <OrbitSpinner />
            </div>
            <h1 className="text-2xl font-bold mb-1">Ask Concord</h1>
            <p className="text-white/40 text-sm max-w-xs mb-10">
              Ask about any patient, drug interaction, conflict, or clinical guideline.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
              {STARTERS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="text-sm text-left px-4 py-3 rounded-2xl bg-white/[0.05] border border-white/[0.08] text-white/55 hover:text-white/90 hover:bg-white/[0.09] transition-all"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* ── Conversation ── */
          <div className="max-w-2xl mx-auto px-4 py-8 space-y-8">
            {messages.map((m, i) => (
              <div key={i} className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                {m.role === "assistant" && (
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center text-white text-xs font-bold shrink-0 mt-0.5">
                    C
                  </div>
                )}
                <div className={`max-w-[78%] ${m.role === "user" ? "items-end" : "items-start"} flex flex-col gap-1`}>
                  <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                    m.role === "user"
                      ? "bg-[#2d2f31] text-white/90 rounded-tr-sm"
                      : "bg-transparent text-white/85"
                  }`}>
                    <p className="whitespace-pre-wrap">{m.content}</p>
                  </div>

                  {/* Registration success card */}
                  {m.action === "registered" && m.action_data && (
                    <div className="mt-2 rounded-2xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 flex items-center gap-3">
                      <div className="w-9 h-9 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400 text-lg shrink-0">✓</div>
                      <div>
                        <p className="text-xs text-emerald-400/70 uppercase tracking-widest mb-0.5">Registered</p>
                        <p className="text-sm font-semibold text-emerald-300">{m.action_data.patient_name}</p>
                        <p className="text-xs font-mono text-emerald-400/60 mt-0.5">ID: {m.action_data.source_ref_id}</p>
                      </div>
                    </div>
                  )}

                  {/* Update success card */}
                  {m.action === "updated" && m.action_data && (
                    <div className="mt-2 rounded-2xl border border-blue-500/30 bg-blue-500/10 px-4 py-3 flex items-center gap-3">
                      <div className="w-9 h-9 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 text-lg shrink-0">↑</div>
                      <div>
                        <p className="text-xs text-blue-400/70 uppercase tracking-widest mb-0.5">Updated</p>
                        <p className="text-sm font-semibold text-blue-300">{m.action_data.patient_name}</p>
                        <p className="text-xs font-mono text-blue-400/60 mt-0.5">ID: {m.action_data.source_ref_id}</p>
                      </div>
                    </div>
                  )}

                  {/* DB update card */}
                  {m.action === "db_updated" && m.action_data && (
                    <div className="mt-2 rounded-2xl border border-cyan-500/30 bg-cyan-500/10 px-4 py-3 flex items-center gap-3">
                      <div className="w-9 h-9 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400 text-lg shrink-0">⟳</div>
                      <div>
                        <p className="text-xs text-cyan-400/70 uppercase tracking-widest mb-0.5">DB Updated</p>
                        <p className="text-sm font-semibold text-cyan-300">{m.action_data.source_ref_id}</p>
                        <p className="text-xs text-cyan-400/60 mt-0.5">{m.action_data.table} · {m.action_data.fields}</p>
                      </div>
                    </div>
                  )}

                  {/* Duplicate warning card */}
                  {m.action === "duplicate" && m.action_data && (
                    <div className="mt-2 rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 flex items-center gap-3">
                      <div className="w-9 h-9 rounded-full bg-amber-500/20 flex items-center justify-center text-amber-400 text-lg shrink-0">⚠</div>
                      <div>
                        <p className="text-xs text-amber-400/70 uppercase tracking-widest mb-0.5">Already Exists</p>
                        <p className="text-sm font-semibold text-amber-300">{m.action_data.patient_name}</p>
                        <p className="text-xs font-mono text-amber-400/60 mt-0.5">Existing ID: {m.action_data.existing_id}</p>
                      </div>
                    </div>
                  )}

                  {m.guidelines && m.guidelines.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {m.guidelines.map((g) => (
                        <span key={g} className="text-[10px] px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-400/70 font-mono border border-violet-500/20">
                          {g}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                {m.role === "user" && (
                  <div className="w-8 h-8 rounded-full bg-[#3a3b3c] flex items-center justify-center text-white/60 text-xs font-semibold shrink-0 mt-0.5">
                    You
                  </div>
                )}
              </div>
            ))}

            {/* Loading spinner */}
            {loading && (
              <div className="flex gap-3 justify-start">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center text-white text-xs font-bold shrink-0 mt-0.5">
                  C
                </div>
                <div className="flex items-center pt-1">
                  <SpinnerRing />
                </div>
              </div>
            )}

            <div ref={endRef} />
          </div>
        )}
      </div>

      {/* Input bar */}
      <div className="shrink-0 px-4 pb-6 pt-3">
        <div className="max-w-2xl mx-auto">
          <div className="flex items-center gap-2 bg-[#2d2f31] rounded-full px-5 py-3 border border-white/[0.08] focus-within:border-violet-500/40 transition-all shadow-xl">
            <input
              ref={inputRef}
              type="text"
              placeholder="Ask Concord anything…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              className="flex-1 bg-transparent text-sm text-white placeholder:text-white/30 focus:outline-none"
            />
            <button
              onClick={() => send()}
              disabled={loading || !input.trim()}
              className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center disabled:opacity-30 hover:opacity-90 transition-all active:scale-90 shrink-0"
            >
              <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </button>
          </div>
          <p className="text-center text-[11px] text-white/15 mt-2">
            Sample IDs: CLN-001 · CLN-002 · CLN-003 · LAB-001 · PHM-001
          </p>
        </div>
      </div>

      {/* Destructive operation confirmation modal */}
      {confirmPending && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-[#1e1f21] border border-red-500/30 rounded-2xl px-6 py-5 max-w-sm w-full mx-4 shadow-2xl">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-9 h-9 rounded-full bg-red-500/20 flex items-center justify-center text-red-400 text-lg shrink-0">⚠</div>
              <p className="text-sm font-semibold text-white">Confirm Destructive Action</p>
            </div>
            <p className="text-xs text-white/50 mb-1">You are about to run:</p>
            <p className="text-xs font-mono text-red-300/80 bg-red-500/10 rounded-lg px-3 py-2 mb-4 break-all">{confirmPending}</p>
            <p className="text-xs text-white/40 mb-4">This may permanently delete data. Are you sure?</p>
            <div className="flex gap-2">
              <button
                onClick={() => { const msg = confirmPending; setConfirmPending(null); setInput(""); send(msg, true); }}
                className="flex-1 py-2 rounded-xl bg-red-500/20 border border-red-500/40 text-red-300 text-sm font-semibold hover:bg-red-500/30 transition-all"
              >
                Yes, proceed
              </button>
              <button
                onClick={() => setConfirmPending(null)}
                className="flex-1 py-2 rounded-xl bg-white/5 border border-white/10 text-white/50 text-sm font-semibold hover:bg-white/10 transition-all"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
