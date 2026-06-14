"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAutoAnimate } from "@formkit/auto-animate/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API = "http://localhost:8080";

// ─── Types ────────────────────────────────────────────────────────────────────

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
  action_data?: {
    source_ref_id?: string;
    patient_name?: string;
    existing_id?: string;
    table?: string;
    fields?: string;
  };
};

type SourceRole = "CLN" | "LAB" | "PHM";

type Session = {
  id: string;
  title: string;
  messages: ChatMessage[];
  updatedAt: number;
};

const SESSIONS_KEY = "concord_sessions";

// ─── Constants ────────────────────────────────────────────────────────────────

const SOURCE_ROLES: { id: SourceRole; label: string; greeting: string; accent: string; accentBg: string; border: string }[] = [
  { id: "CLN", label: "Clinic",   greeting: "Clinic",   accent: "text-violet-400", accentBg: "bg-violet-500/20", border: "border-violet-500/40" },
  { id: "LAB", label: "Lab",      greeting: "Lab",      accent: "text-sky-400",    accentBg: "bg-sky-500/20",    border: "border-sky-500/40"    },
  { id: "PHM", label: "Pharmacy", greeting: "Pharmacy", accent: "text-emerald-400",accentBg: "bg-emerald-500/20",border: "border-emerald-500/40"},
];

const STARTER_CARDS = [
  { label: "Patient Lookup",    desc: "Search or view patient records",       color: "bg-sky-400/15 text-sky-300 border-sky-400/30"        },
  { label: "Drug Safety",       desc: "Prescribe with interaction checking",   color: "bg-pink-400/15 text-pink-300 border-pink-400/30"      },
  { label: "Reconcile Records", desc: "Compare records across locations",      color: "bg-emerald-400/15 text-emerald-300 border-emerald-400/30" },
  { label: "Register Patient",  desc: "Add a new clinic, lab or pharmacy patient", color: "bg-amber-400/15 text-amber-300 border-amber-400/30"  },
  { label: "Escalations",       desc: "List and resolve urgent conflicts",     color: "bg-red-400/15 text-red-300 border-red-400/30"         },
  { label: "Clinical Guidelines", desc: "Ask about drugs, allergies, dosing", color: "bg-violet-400/15 text-violet-300 border-violet-400/30" },
];

const STARTER_MESSAGES: Record<string, string> = {
  "Patient Lookup":       "Show all patients",
  "Drug Safety":          "Prescribe metformin for CLN-001",
  "Reconcile Records":    "Compare CLN-001 and LAB-001",
  "Register Patient":     "Add clinic patient: ",
  "Escalations":          "List unresolved escalations",
  "Clinical Guidelines":  "What are the dengue NSAID guidelines?",
};

const URGENCY_COLOR: Record<string, string> = {
  critical:            "bg-red-500/20 border-red-500/40 text-red-300",
  high:                "bg-orange-500/20 border-orange-500/40 text-orange-300",
  medium:              "bg-yellow-500/20 border-yellow-500/40 text-yellow-300",
  low:                 "bg-blue-500/20 border-blue-500/40 text-blue-300",
  prescription_issued: "bg-emerald-500/20 border-emerald-500/40 text-emerald-300",
};

const DESTRUCTIVE_KEYWORDS = ["delete", "remove patient", "remove record", "drop", "wipe", "erase"];
function isDestructive(msg: string) {
  return DESTRUCTIVE_KEYWORDS.some((k) => msg.toLowerCase().includes(k));
}

// ─── Nav icons ────────────────────────────────────────────────────────────────

function NavIcon({ children, active, onClick }: { children: React.ReactNode; active?: boolean; onClick?: () => void }) {
  return (
    <motion.button
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.93 }}
      onClick={onClick}
      className={`w-9 h-9 rounded-xl flex items-center justify-center transition-colors
        ${active ? "bg-white/10 text-white" : "text-white/30 hover:text-white/60 hover:bg-white/5"}`}
    >
      {children}
    </motion.button>
  );
}

// ─── Glowing orb ─────────────────────────────────────────────────────────────

function GlowOrb() {
  return (
    <div className="relative w-64 h-52 flex items-center justify-center select-none pointer-events-none">
      {/* outer glow */}
      <motion.div
        className="absolute w-56 h-56 rounded-full"
        style={{ background: "radial-gradient(circle at 40% 35%, #2dd4bf55 0%, #06b6d420 40%, transparent 70%)", filter: "blur(18px)" }}
        animate={{ scale: [1, 1.08, 1], opacity: [0.7, 1, 0.7] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      />
      {/* main sphere */}
      <motion.div
        className="relative w-44 h-44 rounded-full"
        style={{
          background: "radial-gradient(circle at 38% 32%, #5eead455 0%, #0891b222 45%, #0f172a88 100%)",
          boxShadow: "0 0 60px 10px #0d9488aa, inset 0 0 40px 6px #0e7490aa",
          backdropFilter: "blur(2px)",
          border: "1px solid rgba(94,234,212,0.18)",
        }}
        animate={{ y: [0, -8, 0], rotateZ: [0, 3, 0] }}
        transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
      >
        {/* inner highlight */}
        <div
          className="absolute top-5 left-7 w-16 h-10 rounded-full opacity-50"
          style={{ background: "radial-gradient(ellipse, #99f6e4cc 0%, transparent 80%)", filter: "blur(8px)" }}
        />
        {/* grid reflection lines */}
        <div className="absolute inset-0 rounded-full overflow-hidden opacity-20"
          style={{ background: "repeating-linear-gradient(0deg, transparent, transparent 10px, rgba(255,255,255,0.06) 10px, rgba(255,255,255,0.06) 11px)" }}
        />
      </motion.div>
    </div>
  );
}

// ─── AI thinking animation ────────────────────────────────────────────────────

function AIThinkingAnimation() {
  const bars = [0.5, 0.9, 0.65, 1.0, 0.75, 0.55, 0.85, 0.6];
  return (
    <div className="flex items-center gap-3 px-4 py-3">
      {/* Waveform bars */}
      <div className="flex items-end gap-0.75 h-5">
        {bars.map((maxH, i) => (
          <motion.span
            key={i}
            className="w-0.75 rounded-full bg-teal-400"
            animate={{ scaleY: [0.15, maxH, 0.15], opacity: [0.4, 1, 0.4] }}
            transition={{
              duration: 0.9 + i * 0.07,
              repeat: Infinity,
              delay: i * 0.08,
              ease: "easeInOut",
            }}
            style={{ height: 20, transformOrigin: "bottom" }}
          />
        ))}
      </div>
      {/* Scanning pulse ring */}
      <div className="relative w-5 h-5 shrink-0">
        <motion.span
          className="absolute inset-0 rounded-full border border-teal-400/60"
          animate={{ scale: [1, 1.9], opacity: [0.8, 0] }}
          transition={{ duration: 1.2, repeat: Infinity, ease: "easeOut" }}
        />
        <motion.span
          className="absolute inset-0 rounded-full border border-cyan-400/40"
          animate={{ scale: [1, 1.5], opacity: [0.6, 0] }}
          transition={{ duration: 1.2, repeat: Infinity, delay: 0.4, ease: "easeOut" }}
        />
        <span className="absolute inset-1.25 rounded-full bg-teal-400" />
      </div>
      {/* Label */}
      <motion.span
        className="text-[11px] text-teal-400/70 font-mono tracking-widest uppercase"
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
      >
        Analyzing
      </motion.span>
    </div>
  );
}

// ─── Action card ─────────────────────────────────────────────────────────────

function ActionCard({ action, action_data }: { action?: string; action_data?: ChatMessage["action_data"] }) {
  if (!action || !action_data) return null;
  const configs: Record<string, { border: string; bg: string; icon: string; label: string; text: string; iconBg: string }> = {
    registered: { border:"border-emerald-500/30", bg:"bg-emerald-500/10", icon:"✓", label:"Registered", text:"text-emerald-300", iconBg:"bg-emerald-500/20 text-emerald-400" },
    updated:    { border:"border-blue-500/30",    bg:"bg-blue-500/10",    icon:"↑", label:"Updated",    text:"text-blue-300",    iconBg:"bg-blue-500/20 text-blue-400"     },
    db_updated: { border:"border-cyan-500/30",    bg:"bg-cyan-500/10",    icon:"⟳", label:"DB Updated", text:"text-cyan-300",    iconBg:"bg-cyan-500/20 text-cyan-400"     },
    duplicate:  { border:"border-amber-500/30",   bg:"bg-amber-500/10",   icon:"⚠", label:"Already Exists", text:"text-amber-300", iconBg:"bg-amber-500/20 text-amber-400" },
    issued:     { border:"border-emerald-500/30", bg:"bg-emerald-500/10", icon:"💊",label:"Prescribed", text:"text-emerald-300", iconBg:"bg-emerald-500/20 text-emerald-400"},
    blocked:    { border:"border-red-500/30",     bg:"bg-red-500/10",     icon:"🚫",label:"Blocked",    text:"text-red-300",     iconBg:"bg-red-500/20 text-red-400"       },
  };
  const cfg = configs[action];
  if (!cfg) return null;
  const isBlocked = action === "blocked";
  return (
    <motion.div
      initial={{ opacity:0, scale:0.95, y:8 }}
      animate={isBlocked
        ? { opacity:1, scale:1, y:0, x:[0,-6,6,-4,4,0] }
        : { opacity:1, scale:1, y:0 }}
      transition={isBlocked
        ? { opacity:{duration:0.3}, scale:{duration:0.3}, x:{delay:0.3,duration:0.4} }
        : { type:"spring", stiffness:300, damping:22 }}
      className={`mt-2 rounded-2xl border ${cfg.border} ${cfg.bg} px-4 py-3 flex items-center gap-3`}
    >
      <div className={`w-9 h-9 rounded-full flex items-center justify-center text-lg shrink-0 ${cfg.iconBg}`}>{cfg.icon}</div>
      <div>
        <p className={`text-xs uppercase tracking-widest mb-0.5 opacity-70 ${cfg.text}`}>{cfg.label}</p>
        <p className={`text-sm font-semibold ${cfg.text}`}>{action_data.patient_name ?? action_data.source_ref_id}</p>
        {action_data.source_ref_id && <p className={`text-xs font-mono mt-0.5 opacity-60 ${cfg.text}`}>ID: {action_data.source_ref_id}</p>}
        {action_data.table && <p className={`text-xs mt-0.5 opacity-60 ${cfg.text}`}>{action_data.table} · {action_data.fields}</p>}
        {action_data.existing_id && <p className={`text-xs font-mono mt-0.5 opacity-60 ${cfg.text}`}>Existing: {action_data.existing_id}</p>}
      </div>
    </motion.div>
  );
}

// ─── Bell SVG ────────────────────────────────────────────────────────────────

function BellSVG({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
    </svg>
  );
}

// ─── Aurora background ────────────────────────────────────────────────────────

function AuroraBackground() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <style>{`
        @keyframes auroraMove1 {
          0%   { transform: translate(0%, 0%) scale(1); }
          33%  { transform: translate(8%, -12%) scale(1.1); }
          66%  { transform: translate(-6%, 8%) scale(0.95); }
          100% { transform: translate(0%, 0%) scale(1); }
        }
        @keyframes auroraMove2 {
          0%   { transform: translate(0%, 0%) scale(1.05); }
          33%  { transform: translate(-10%, 6%) scale(0.9); }
          66%  { transform: translate(12%, -8%) scale(1.1); }
          100% { transform: translate(0%, 0%) scale(1.05); }
        }
        @keyframes auroraMove3 {
          0%   { transform: translate(0%, 0%) scale(0.95); }
          50%  { transform: translate(6%, 10%) scale(1.1); }
          100% { transform: translate(0%, 0%) scale(0.95); }
        }
        @keyframes auroraMove4 {
          0%   { transform: translate(0%, 0%) scale(1); }
          40%  { transform: translate(-8%, -6%) scale(1.05); }
          100% { transform: translate(0%, 0%) scale(1); }
        }
      `}</style>
      {/* Orb 1 — teal, top-left */}
      <div style={{
        position:"absolute", top:"-10%", left:"-5%",
        width:"55%", height:"55%",
        background:"radial-gradient(ellipse, rgba(20,184,166,0.22) 0%, transparent 70%)",
        filter:"blur(72px)",
        animation:"auroraMove1 18s ease-in-out infinite",
      }} />
      {/* Orb 2 — cyan, top-right */}
      <div style={{
        position:"absolute", top:"-15%", right:"-10%",
        width:"60%", height:"60%",
        background:"radial-gradient(ellipse, rgba(6,182,212,0.18) 0%, transparent 70%)",
        filter:"blur(90px)",
        animation:"auroraMove2 23s ease-in-out infinite",
      }} />
      {/* Orb 3 — violet, bottom-right */}
      <div style={{
        position:"absolute", bottom:"-5%", right:"5%",
        width:"50%", height:"50%",
        background:"radial-gradient(ellipse, rgba(139,92,246,0.12) 0%, transparent 70%)",
        filter:"blur(80px)",
        animation:"auroraMove3 27s ease-in-out infinite",
      }} />
      {/* Orb 4 — teal-dark, bottom-left */}
      <div style={{
        position:"absolute", bottom:"10%", left:"-8%",
        width:"45%", height:"45%",
        background:"radial-gradient(ellipse, rgba(20,184,166,0.1) 0%, transparent 70%)",
        filter:"blur(64px)",
        animation:"auroraMove4 21s ease-in-out infinite",
      }} />
      {/* Subtle dot grid on top */}
      <div style={{
        position:"absolute", inset:0,
        backgroundImage:"radial-gradient(circle, rgba(255,255,255,0.025) 1px, transparent 1px)",
        backgroundSize:"28px 28px",
      }} />
    </div>
  );
}

// ─── Canvas particle network ──────────────────────────────────────────────────

function ParticleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    const PARTICLE_COUNT = 55;
    const MAX_DIST = 140;

    function resize() {
      if (!canvas) return;
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    type Particle = { x: number; y: number; vx: number; vy: number; r: number };
    const particles: Particle[] = Array.from({ length: PARTICLE_COUNT }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      r: Math.random() * 1.5 + 0.5,
    }));

    function draw() {
      if (!canvas || !ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < MAX_DIST) {
            const alpha = (1 - dist / MAX_DIST) * 0.18;
            ctx.beginPath();
            ctx.strokeStyle = `rgba(20,184,166,${alpha})`;
            ctx.lineWidth = 0.8;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      for (const p of particles) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(94,234,212,0.45)";
        ctx.fill();

        // Move
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
      }

      animId = requestAnimationFrame(draw);
    }

    draw();
    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ position:"absolute", inset:0, width:"100%", height:"100%", opacity:0.6, pointerEvents:"none" }}
    />
  );
}

// ─── Agent switcher ──────────────────────────────────────────────────────────

const AGENTS = [
  { id: null,        label: "Auto",      desc: "Router decides"         },
  { id: "register",  label: "Register",  desc: "Add / update patient"   },
  { id: "prescribe", label: "Prescribe", desc: "Drug safety check"      },
  { id: "query",     label: "Query",     desc: "Search & list records"  },
  { id: "reconcile", label: "Reconcile", desc: "Conflict analysis"      },
  { id: "db_update", label: "DB Update", desc: "Modify clinical data"   },
  { id: "chat",      label: "Chat",      desc: "General clinical Q&A"   },
] as const;

function AgentSwitcher({
  pinnedAgent,
  setPinnedAgent,
}: {
  pinnedAgent: string | null;
  setPinnedAgent: (v: string | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const current = AGENTS.find((a) => a.id === pinnedAgent) ?? AGENTS[0];

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-[11px] text-white/40 hover:text-white/70 transition-colors group"
      >
        <span>
          {pinnedAgent ? current.label : "Switch agent…"}
        </span>
        {pinnedAgent && (
          <span className="text-white/20 font-light">· {current.desc}</span>
        )}
        <svg className={`w-3 h-3 transition-transform ${open ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.97 }}
            transition={{ duration: 0.15 }}
            className="absolute bottom-full mb-2 left-0 w-52 rounded-xl bg-[#1a1a1e] border border-white/10 shadow-2xl overflow-hidden z-50"
          >
            {AGENTS.map((agent) => {
              const active = pinnedAgent === agent.id;
              return (
                <button
                  key={String(agent.id)}
                  onClick={() => { setPinnedAgent(agent.id); setOpen(false); }}
                  className={`w-full flex items-center justify-between px-3 py-2.5 text-left hover:bg-white/5 transition-colors ${active ? "bg-white/5" : ""}`}
                >
                  <div className="flex items-center gap-2.5">
                    <div>
                      <div className={`text-[12px] font-medium ${active ? "text-white" : "text-white/70"}`}>{agent.label}</div>
                      <div className="text-[10px] text-white/30">{agent.desc}</div>
                    </div>
                  </div>
                  {active && (
                    <svg className="w-3.5 h-3.5 text-teal-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </button>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages]           = useState<ChatMessage[]>([]);
  const [input, setInput]                 = useState("");
  const [loading, setLoading]             = useState(false);
  const endRef                            = useRef<HTMLDivElement>(null);
  const inputRef                          = useRef<HTMLTextAreaElement>(null);
  const [confirmPending, setConfirmPending] = useState<string | null>(null);
  const [activeRole, setActiveRole]       = useState<SourceRole>("CLN");
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [notifOpen, setNotifOpen]         = useState(false);
  const [prevUnread, setPrevUnread]       = useState(0);
  const [bellShake, setBellShake]         = useState(false);
  const [activeNav, setActiveNav]         = useState("home");
  const [pinnedAgent, setPinnedAgent]     = useState<string | null>(null);
  const [sessions, setSessions]           = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>(() => crypto.randomUUID());
  const [historyOpen, setHistoryOpen]     = useState(false);

  const [messageListRef] = useAutoAnimate<HTMLDivElement>();
  const [notifListRef]   = useAutoAnimate<HTMLDivElement>();

  const unreadCount = notifications.filter((n) => !n.is_read).length;
  const activeRoleData = SOURCE_ROLES.find((r) => r.id === activeRole)!;

  useEffect(() => {
    if (unreadCount > prevUnread) { setBellShake(true); setTimeout(() => setBellShake(false), 600); }
    setPrevUnread(unreadCount);
  }, [unreadCount]);

  useEffect(() => {
    async function fetchNotifs() {
      try {
        const res = await fetch(`${API}/notifications?limit=30&source=${activeRole}`);
        if (res.ok) setNotifications(await res.json());
      } catch { /* backend warming up */ }
    }
    fetchNotifs();
    const t = setInterval(fetchNotifs, 10000);
    return () => clearInterval(t);
  }, [activeRole]);

  async function markAllRead() {
    const ids = notifications.filter((n) => !n.is_read).map((n) => n.id);
    if (!ids.length) return;
    await fetch(`${API}/notifications/read`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(ids) });
    setNotifications((p) => p.map((n) => ({ ...n, is_read: true })));
  }
  async function clearAllNotifications() {
    await fetch(`${API}/notifications?source=${activeRole}`, { method:"DELETE" });
    setNotifications([]);
  }
  async function deleteNotification(id: string) {
    await fetch(`${API}/notifications/${id}`, { method:"DELETE" });
    setNotifications((p) => p.filter((n) => n.id !== id));
  }

  useEffect(() => { endRef.current?.scrollIntoView({ behavior:"smooth" }); }, [messages, loading]);

  // Load sessions from localStorage on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(SESSIONS_KEY);
      if (raw) setSessions(JSON.parse(raw));
    } catch { /* ignore */ }
  }, []);

  // Auto-save current session whenever messages change (only if there are messages)
  useEffect(() => {
    if (messages.length === 0) return;
    const title = messages.find((m) => m.role === "user")?.content.slice(0, 60) ?? "Untitled";
    const session: Session = { id: currentSessionId, title, messages, updatedAt: Date.now() };
    setSessions((prev) => {
      const rest = prev.filter((s) => s.id !== currentSessionId);
      const updated = [session, ...rest].slice(0, 50); // keep last 50
      try { localStorage.setItem(SESSIONS_KEY, JSON.stringify(updated)); } catch { /* ignore */ }
      return updated;
    });
  }, [messages, currentSessionId]);

  // Auto-grow textarea
  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
  }

  async function send(text?: string, confirmed = false) {
    const msg = (text ?? input).trim();
    if (!msg || loading) return;
    if (!confirmed && isDestructive(msg)) { setConfirmPending(msg); return; }

    const userMsg: ChatMessage = { role:"user", content:msg };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput("");
    if (inputRef.current) { inputRef.current.style.height = "auto"; }
    setLoading(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          message: msg,
          history: updated.slice(-12).map((m) => ({ role:m.role, content:m.content })),
          source_ref_id: "", reconciliation_context: null,
          forced_intent: pinnedAgent ?? "",
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setMessages((p) => [...p, { role:"assistant", content:data.reply, guidelines:data.guidelines_used, action:data.action, action_data:data.action_data }]);
    } catch (e: unknown) {
      setMessages((p) => [...p, { role:"assistant", content:`Something went wrong: ${e instanceof Error ? e.message : "Unknown error"}` }]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  }

  function loadSession(session: Session) {
    setMessages(session.messages);
    setCurrentSessionId(session.id);
    setHistoryOpen(false);
    setActiveNav("home");
  }

  function newChat() {
    setMessages([]);
    setCurrentSessionId(crypto.randomUUID());
    setActiveNav("home");
  }

  function deleteSession(id: string) {
    setSessions((prev) => {
      const updated = prev.filter((s) => s.id !== id);
      try { localStorage.setItem(SESSIONS_KEY, JSON.stringify(updated)); } catch { /* ignore */ }
      return updated;
    });
    if (id === currentSessionId) newChat();
  }

  const showWelcome = messages.length === 0 && !loading;

  return (
    <div className="flex h-screen bg-[#0a0a0b] text-white font-sans overflow-hidden">

      {/* ── Left sidebar ── */}
      <motion.aside
        initial={{ x: -60, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="w-14 shrink-0 flex flex-col items-center py-4 gap-2 border-r border-white/6 bg-[#0d0d0f] z-20"
      >
        {/* Logo */}
        <div className="w-9 h-9 rounded-xl bg-linear-to-br from-teal-400 to-cyan-600 flex items-center justify-center font-black text-sm text-white mb-3 shadow-lg shadow-teal-500/20">
          C
        </div>

        <div className="w-full px-2 flex flex-col gap-1">
          {/* Home */}
          <NavIcon active={activeNav === "home"} onClick={() => { setActiveNav("home"); setMessages([]); }}>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
          </NavIcon>
          {/* Patients */}
          <NavIcon active={activeNav === "patients"} onClick={() => { setActiveNav("patients"); send("Show all patients"); }}>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </NavIcon>
          {/* Escalations */}
          <NavIcon active={activeNav === "escal"} onClick={() => { setActiveNav("escal"); send("List unresolved escalations"); }}>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </NavIcon>
          {/* Prescriptions */}
          <NavIcon active={activeNav === "rx"} onClick={() => { setActiveNav("rx"); send("Show all prescriptions"); }}>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          </NavIcon>
          {/* Chat history */}
          <NavIcon active={historyOpen} onClick={() => setHistoryOpen((o) => !o)}>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </NavIcon>
        </div>

        {/* Bottom: settings + role toggle */}
        <div className="mt-auto w-full px-2 flex flex-col gap-1">
          {/* Settings placeholder */}
          <NavIcon>
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </NavIcon>
          {/* Notification bell in sidebar */}
          <div className="relative">
            <motion.button
              animate={bellShake ? { rotate:[0,-18,18,-12,12,-6,6,0] } : {}}
              transition={{ duration: 0.55 }}
              onClick={() => { setNotifOpen((o) => !o); if (!notifOpen) markAllRead(); }}
              className="w-9 h-9 rounded-xl flex items-center justify-center text-white/30 hover:text-white/60 hover:bg-white/5 transition-colors"
            >
              <BellSVG className="w-4 h-4" />
              <AnimatePresence>
                {unreadCount > 0 && (
                  <motion.span
                    key="badge"
                    initial={{ scale:0 }} animate={{ scale:1 }} exit={{ scale:0 }}
                    transition={{ type:"spring", stiffness:500, damping:20 }}
                    className="absolute top-1 right-1 w-3.5 h-3.5 rounded-full bg-red-500 text-white text-[8px] font-bold flex items-center justify-center"
                  >
                    {unreadCount > 9 ? "9+" : unreadCount}
                  </motion.span>
                )}
              </AnimatePresence>
            </motion.button>

            {/* Notification panel */}
            <AnimatePresence>
              {notifOpen && (
                <motion.div
                  initial={{ opacity:0, scale:0.93, x:-8 }}
                  animate={{ opacity:1, scale:1, x:0 }}
                  exit={{ opacity:0, scale:0.93, x:-8 }}
                  transition={{ type:"spring", stiffness:400, damping:28 }}
                  className="absolute left-12 bottom-0 w-80 max-h-96 overflow-y-auto rounded-2xl border border-white/10 bg-[#141416] shadow-2xl z-50"
                >
                  <div className="flex items-center justify-between px-4 py-3 border-b border-white/6">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-white">Notifications</span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded-full border font-semibold ${activeRoleData.accentBg} ${activeRoleData.border} ${activeRoleData.accent}`}>
                        {activeRoleData.label}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <button onClick={markAllRead} className="text-xs text-white/30 hover:text-white/60 transition-colors">Read all</button>
                      <button onClick={clearAllNotifications} className="text-xs text-red-400/50 hover:text-red-400 transition-colors">Clear</button>
                    </div>
                  </div>
                  {notifications.length === 0 ? (
                    <div className="px-4 py-8 text-center text-sm text-white/25">No notifications for {activeRoleData.label}</div>
                  ) : (
                    <div ref={notifListRef} className="divide-y divide-white/4">
                      {notifications.map((n) => (
                        <div key={n.id} className={`px-4 py-3 ${!n.is_read ? "bg-white/2" : ""}`}>
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
                              {!n.is_read && <motion.span initial={{scale:0}} animate={{scale:1}} className="w-1.5 h-1.5 rounded-full bg-teal-400" />}
                              <button onClick={() => deleteNotification(n.id)} className="text-white/20 hover:text-red-400 transition-colors text-xs">✕</button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

          </div>
        </div>
      </motion.aside>

      {/* ── History sidebar ── */}
      <AnimatePresence>
        {historyOpen && (
          <motion.div
            key="history-sidebar"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 220, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 340, damping: 32 }}
            className="shrink-0 flex flex-col bg-[#0f0f11] border-r border-white/6 overflow-hidden z-10"
            style={{ minWidth: 0 }}
          >
            <div className="px-4 pt-5 pb-3 flex items-center justify-between shrink-0">
              <span className="text-[11px] font-semibold text-white/35 uppercase tracking-widest">Chats</span>
              <button
                onClick={newChat}
                className="text-[11px] text-white/30 hover:text-teal-400 transition-colors"
              >
                + New
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-2 pb-4">
              {sessions.length === 0 ? (
                <p className="text-[12px] text-white/20 px-2 pt-4">No history yet</p>
              ) : (
                sessions.map((s) => (
                  <div
                    key={s.id}
                    onClick={() => loadSession(s)}
                    className={`group relative flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors mb-0.5 ${
                      s.id === currentSessionId
                        ? "bg-white/8 text-white"
                        : "text-white/50 hover:bg-white/5 hover:text-white/80"
                    }`}
                  >
                    <span className="text-[13px] truncate flex-1 leading-snug">{s.title}</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteSession(s.id); }}
                      className="shrink-0 opacity-0 group-hover:opacity-100 text-white/20 hover:text-red-400 transition-all text-[11px]"
                    >
                      ✕
                    </button>
                  </div>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Main area ── */}
      <div className="flex-1 flex flex-col min-w-0 relative">

        {/* Animated background */}
        <AuroraBackground />
        <ParticleCanvas />

        {/* Top bar */}
        <motion.header
          initial={{ opacity:0, y:-10 }}
          animate={{ opacity:1, y:0 }}
          transition={{ duration:0.35 }}
          className="relative z-10 shrink-0 flex items-center justify-between px-6 py-3 border-b border-white/5"
        >
          {/* Role selector */}
          <div className="flex items-center gap-1 bg-white/4 border border-white/7 rounded-full px-1 py-1">
            {SOURCE_ROLES.map((r) => (
              <button key={r.id}
                onClick={() => { setActiveRole(r.id); setNotifications([]); }}
                className="relative text-xs px-3 py-1 rounded-full font-semibold transition-colors duration-200"
              >
                {activeRole === r.id && (
                  <motion.div layoutId="role-pill"
                    className={`absolute inset-0 rounded-full border ${r.accentBg} ${r.border}`}
                    transition={{ type:"spring", stiffness:420, damping:32 }}
                  />
                )}
                <span className={`relative z-10 transition-colors ${activeRole === r.id ? r.accent : "text-white/30 hover:text-white/55"}`}>
                  {r.label}
                </span>
              </button>
            ))}
          </div>

          {/* Right: online + role badge */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <motion.span
                className="w-1.5 h-1.5 rounded-full bg-teal-400"
                animate={{ opacity:[1,0.3,1] }}
                transition={{ duration:2, repeat:Infinity }}
              />
              <span className="text-xs text-white/30">Online</span>
            </div>
            <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-semibold ${activeRoleData.accentBg} ${activeRoleData.border} ${activeRoleData.accent}`}>
              <span className="w-1.5 h-1.5 rounded-full bg-current opacity-70" />
              {activeRoleData.label}
            </div>
          </div>
        </motion.header>

        {/* ── Content ── */}
        <div className="flex-1 overflow-y-auto relative">
          <AnimatePresence mode="wait">

            {/* Welcome screen */}
            {showWelcome && (
              <motion.div
                key="welcome"
                initial={{ opacity:0 }} animate={{ opacity:1 }} exit={{ opacity:0, y:-16 }}
                transition={{ duration:0.3 }}
                className="flex flex-col items-center pt-6 pb-8 px-8 min-h-full"
              >
                {/* Orb */}
                <motion.div
                  initial={{ scale:0.7, opacity:0 }}
                  animate={{ scale:1, opacity:1 }}
                  transition={{ type:"spring", stiffness:200, damping:20, delay:0.05 }}
                >
                  <GlowOrb />
                </motion.div>

                {/* Greeting */}
                <motion.div
                  className="w-full max-w-4xl"
                  initial={{ opacity:0, y:14 }}
                  animate={{ opacity:1, y:0 }}
                  transition={{ delay:0.18, duration:0.4 }}
                >
                  <h1 className="text-4xl font-bold leading-tight text-white/90">
                    Hey! <span className={activeRoleData.accent}>{activeRoleData.label}</span>
                  </h1>
                  <h1 className="text-4xl font-bold leading-tight text-white/40 mt-0.5">
                    What can I help with?
                  </h1>
                </motion.div>

                {/* Starter cards */}
                <div className="w-full max-w-4xl grid grid-cols-3 gap-3 mt-6">
                  {STARTER_CARDS.map((card, i) => (
                    <motion.button
                      key={card.label}
                      initial={{ opacity:0, y:16 }}
                      animate={{ opacity:1, y:0 }}
                      transition={{ delay:0.24 + i * 0.06 }}
                      whileHover={{ scale:1.03, y:-2 }}
                      whileTap={{ scale:0.97 }}
                      onClick={() => {
                        const msg = STARTER_MESSAGES[card.label];
                        if (msg.endsWith(": ")) { setInput(msg); setTimeout(() => inputRef.current?.focus(), 50); }
                        else send(msg);
                      }}
                      className="text-left p-4 rounded-2xl bg-white/3 border border-white/7 hover:bg-white/6 hover:border-white/12 transition-colors group"
                    >
                      <span className={`inline-block text-[11px] font-semibold px-2 py-0.5 rounded-full border mb-2.5 ${card.color}`}>
                        {card.label}
                      </span>
                      <p className="text-xs text-white/35 group-hover:text-white/55 transition-colors leading-relaxed">
                        {card.desc}
                      </p>
                    </motion.button>
                  ))}
                </div>

                {/* Sample IDs hint */}
                <motion.p
                  initial={{ opacity:0 }}
                  animate={{ opacity:1 }}
                  transition={{ delay:0.7 }}
                  className="text-[11px] text-white/15 mt-6"
                >
                  Sample IDs: CLN-001 · CLN-002 · CLN-003 · LAB-001 · PHM-001
                </motion.p>
              </motion.div>
            )}

            {/* Conversation */}
            {!showWelcome && (
              <motion.div
                key="chat"
                initial={{ opacity:0 }}
                animate={{ opacity:1 }}
                className="max-w-4xl mx-auto px-6 py-8"
              >
                <div ref={messageListRef} className="space-y-7">
                  {messages.map((m, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity:0, y:16, scale:0.98 }}
                      animate={{ opacity:1, y:0, scale:1 }}
                      transition={{ type:"spring", stiffness:320, damping:26 }}
                      className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      {m.role === "assistant" && (
                        <motion.div
                          initial={{ scale:0 }} animate={{ scale:1 }}
                          transition={{ type:"spring", stiffness:400, damping:22, delay:0.05 }}
                          className="w-8 h-8 rounded-xl bg-linear-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white text-xs font-black shrink-0 mt-0.5 shadow-md shadow-teal-500/20"
                        >
                          C
                        </motion.div>
                      )}

                      <div className={`max-w-[78%] flex flex-col gap-1 ${m.role === "user" ? "items-end" : "items-start"}`}>
                        <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                          m.role === "user"
                            ? "bg-white/7 border border-white/9 text-white/90 rounded-tr-sm"
                            : "bg-transparent text-white/80"
                        }`}>
                          {m.role === "assistant" ? (
                            <div className="prose prose-invert prose-sm max-w-none
                              prose-p:leading-relaxed prose-p:my-1
                              prose-strong:text-white/90
                              prose-code:text-teal-300 prose-code:bg-teal-500/10 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:font-mono prose-code:before:content-none prose-code:after:content-none
                              prose-pre:bg-white/4 prose-pre:border prose-pre:border-white/7 prose-pre:rounded-xl
                              prose-ul:my-1 prose-li:my-0.5
                              prose-table:text-xs prose-th:text-white/60 prose-td:text-white/50
                              prose-h3:text-white/90 prose-h3:text-sm prose-h3:font-semibold prose-h3:mt-3 prose-h3:mb-1">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                            </div>
                          ) : (
                            <p className="whitespace-pre-wrap">{m.content}</p>
                          )}
                        </div>

                        <ActionCard action={m.action} action_data={m.action_data} />

                        {m.guidelines && m.guidelines.length > 0 && (
                          <motion.div initial={{opacity:0}} animate={{opacity:1}} transition={{delay:0.2}} className="flex flex-wrap gap-1 mt-1">
                            {m.guidelines.map((g) => (
                              <motion.span key={g}
                                initial={{scale:0.8,opacity:0}} animate={{scale:1,opacity:1}}
                                transition={{type:"spring",stiffness:300}}
                                className="text-[10px] px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-400/70 font-mono border border-teal-500/20"
                              >
                                {g}
                              </motion.span>
                            ))}
                          </motion.div>
                        )}
                      </div>

                      {m.role === "user" && (
                        <motion.div
                          initial={{scale:0}} animate={{scale:1}}
                          transition={{type:"spring",stiffness:400,damping:22}}
                          className="w-8 h-8 rounded-xl bg-white/7 border border-white/9 flex items-center justify-center text-white/50 text-[10px] font-semibold shrink-0 mt-0.5"
                        >
                          {activeRole}
                        </motion.div>
                      )}
                    </motion.div>
                  ))}
                </div>

                {/* Typing indicator */}
                <AnimatePresence>
                  {loading && (
                    <motion.div
                      key="typing"
                      initial={{opacity:0,y:12}} animate={{opacity:1,y:0}} exit={{opacity:0,y:8}}
                      transition={{duration:0.22}}
                      className="flex gap-3 mt-7"
                    >
                      <div className="w-8 h-8 rounded-xl bg-linear-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white text-xs font-black shrink-0 shadow-md shadow-teal-500/20">
                        C
                      </div>
                      <div className="bg-white/4 border border-white/6 rounded-2xl rounded-tl-sm">
                        <AIThinkingAnimation />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                <div ref={endRef} className="h-4" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── Input box ── */}
        <motion.div
          initial={{ opacity:0, y:24 }}
          animate={{ opacity:1, y:0 }}
          transition={{ delay:0.3, duration:0.4 }}
          className="relative z-10 shrink-0 px-6 pb-6 pt-3 max-w-4xl w-full mx-auto"
        >
          <div className="rounded-2xl bg-[#141416] border border-white/8 shadow-2xl focus-within:border-teal-500/30 transition-colors">

            {/* Sparkle + textarea */}
            <div className="flex items-start gap-3 px-4 pt-4 pb-2">
              <motion.div
                animate={{ rotate:[0,15,-15,0], scale:[1,1.1,1] }}
                transition={{ duration:3, repeat:Infinity, repeatDelay:4 }}
                className="text-teal-400 mt-0.5 shrink-0 text-base"
              >
                ✦
              </motion.div>
              <textarea
                ref={inputRef}
                rows={1}
                placeholder="Ask me anything……"
                value={input}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                className="flex-1 bg-transparent text-sm text-white placeholder:text-white/25 focus:outline-none resize-none leading-relaxed"
                style={{ minHeight:"28px", maxHeight:"160px" }}
              />
            </div>

            {/* Bottom bar */}
            <div className="flex items-center justify-between px-4 py-2.5 border-t border-white/5">
              <AgentSwitcher pinnedAgent={pinnedAgent} setPinnedAgent={setPinnedAgent} />
              <motion.button
                onClick={() => send()}
                disabled={loading || !input.trim()}
                whileTap={{ scale:0.88 }}
                whileHover={{ scale:1.05 }}
                className="w-8 h-8 rounded-xl bg-linear-to-br from-teal-400 to-cyan-600 flex items-center justify-center disabled:opacity-25 shadow-md shadow-teal-500/20 transition-opacity"
              >
                <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </motion.button>
            </div>
          </div>
        </motion.div>
      </div>

      {/* ── Destructive confirm modal ── */}
      <AnimatePresence>
        {confirmPending && (
          <motion.div
            key="backdrop"
            initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={() => setConfirmPending(null)}
          >
            <motion.div
              key="modal"
              initial={{opacity:0,scale:0.88,y:20}} animate={{opacity:1,scale:1,y:0}} exit={{opacity:0,scale:0.88,y:16}}
              transition={{type:"spring",stiffness:380,damping:28}}
              onClick={(e) => e.stopPropagation()}
              className="bg-[#141416] border border-red-500/25 rounded-2xl px-6 py-5 max-w-sm w-full mx-4 shadow-2xl"
            >
              <div className="flex items-center gap-3 mb-3">
                <motion.div
                  animate={{rotate:[0,-8,8,-5,5,0]}}
                  transition={{duration:0.5,delay:0.2}}
                  className="w-9 h-9 rounded-xl bg-red-500/15 flex items-center justify-center text-red-400 text-lg shrink-0"
                >⚠</motion.div>
                <p className="text-sm font-semibold text-white">Confirm Destructive Action</p>
              </div>
              <p className="text-xs text-white/40 mb-1">You are about to run:</p>
              <p className="text-xs font-mono text-red-300/80 bg-red-500/10 rounded-xl px-3 py-2 mb-4 break-all">{confirmPending}</p>
              <p className="text-xs text-white/30 mb-4">This may permanently delete data. Are you sure?</p>
              <div className="flex gap-2">
                <motion.button whileTap={{scale:0.96}}
                  onClick={() => { const m = confirmPending; setConfirmPending(null); send(m, true); }}
                  className="flex-1 py-2.5 rounded-xl bg-red-500/15 border border-red-500/30 text-red-300 text-sm font-semibold hover:bg-red-500/25 transition-colors"
                >Yes, proceed</motion.button>
                <motion.button whileTap={{scale:0.96}}
                  onClick={() => setConfirmPending(null)}
                  className="flex-1 py-2.5 rounded-xl bg-white/4 border border-white/8 text-white/40 text-sm font-semibold hover:bg-white/8 transition-colors"
                >Cancel</motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
