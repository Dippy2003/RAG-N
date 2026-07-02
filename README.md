# Concord — Clinical Record Reconciliation System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.105.0-lightgrey?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=nextdotjs)](https://nextjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-cyan?style=for-the-badge&logo=tailwindcss)](https://tailwindcss.com/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-blue?style=for-the-badge&logo=supabase)](https://supabase.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama-green?style=for-the-badge)](https://www.groq.com/)
[![Vapi AI](https://img.shields.io/badge/Vapi-AI-purple?style=for-the-badge)](https://vapi.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)](https://www.docker.com/)

> Built for **AgenTrix 2026 Hackathon** by Dippy2003  
> Multi-agent AI platform that reconciles fragmented patient records across Sri Lankan clinics, labs, and pharmacies using RAG-augmented LLMs, real-time conflict detection, and a full natural-language chat interface.

---

## The Problem

In Sri Lanka, a single patient's records are scattered across three disconnected systems:

| Location | What they record |
|---|---|
| **Clinic** | Diagnoses, prescriptions, allergies |
| **Laboratory** | Blood tests, blood type results |
| **Pharmacy** | Dispensed medications |

These systems never talk to each other. A patient prescribed **warfarin** at the clinic and **aspirin** at the pharmacy has a dangerous drug interaction — and no one catches it. A dengue patient given **ibuprofen** at the pharmacy risks severe haemorrhage. Concord fixes this automatically.

---

## How It Works

```
User types in chat
        │
        ▼
┌─────────────────────┐
│    Router Agent     │  ← classifies intent (7 types)
└────────┬────────────┘
         │
   ┌─────┴──────────────────────────────────────┐
   │                                             │
   ▼                                             ▼
register / update                          reconcile
   │                                             │
   ▼                                             ▼
Registration Agent                    Reconciliation Agent
• create patient record               • compare fields across locations
• link to cluster                     • detect conflicts
• embed for matching                  • RAG: retrieve guidelines per conflict
                                      • LLM Call 1: adjudicate each conflict
   ▼                                  • LLM Call 2: safety review
prescribe                             • notify ALL locations in cluster
   │
   ▼
Prescription Agent (tool loop)
• get_patient_record
• check_interaction → RAG (drug_interaction + allergy)
• issue_prescription OR block_prescription

   ▼
db_update
   │
   ▼
Database Agent (tool loop)
• 14 tools: update, add, remove, delete any Supabase table
• auto-resolves cluster_id
• re-embeds on identity changes

   ▼
query
   │
   ▼
Query Agent → LLM formats with RAG context
• 7 query types: patients, prescriptions, escalations, search...
```

---

## Architecture

### Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS | Chat UI, role selector, notification bell, voice chat |
| API | FastAPI + Uvicorn (Python 3.11+) | REST endpoints, intent routing, Vapi Custom LLM |
| Voice AI | Vapi AI + `@vapi-ai/web` SDK | Browser voice pipeline (STT → LLM → TTS) |
| STT | Deepgram (via Vapi) | Speech-to-text transcription |
| TTS | PlayHT (via Vapi) | Text-to-speech for assistant responses |
| Tunnel | Cloudflare Tunnel (`cloudflared`) | Exposes local backend to Vapi over HTTPS |
| LLM Primary | Gemini 2.0 Flash | Adjudication, conflict safety review |
| LLM Secondary | Groq Llama 3.3-70B | All agent tool loops + voice chat responses |
| LLM Tertiary | Groq Llama 3.1-8B | Rate-limit fallback |
| Embeddings | all-MiniLM-L6-v2 (offline, 384-dim) | Patient matching + RAG search |
| Database | Supabase (PostgreSQL + pgvector) | All data + vector search |
| Package Manager | `uv` (Python), `npm` (Node) |  |

### Why not LangChain?

Concord uses a **custom agent architecture** for full control:

| LangChain | Concord |
|---|---|
| LangChain agent loops | Custom `while turns < MAX_TURNS` tool loop |
| LangChain tools | Plain Python functions as Groq JSON tool schemas |
| LangChain RAG chains | Custom `rag_retriever.py` with pgvector |
| LangChain memory | Python dict with 5-min TTL cache |
| LLM wrappers | Direct `google.generativeai` + `groq` SDK calls |

---

## Project Structure

```
concord/
├── backend/
│   ├── api.py                  ← FastAPI routes + /chat/completions Vapi endpoint
│   ├── router_agent.py         ← Intent classification (7 intents)
│   ├── registration_agent.py   ← Patient register + update + re-embed
│   ├── agent.py                ← Reconciliation pipeline + LLM adjudication
│   ├── prescription_agent.py   ← Drug safety agentic tool loop (max 8 turns)
│   ├── database_agent.py       ← Full CRUD agentic tool loop (max 10 turns)
│   ├── query_agent.py          ← Non-agentic DB queries (7 query types)
│   ├── llm_interface.py        ← Gemini/Groq wrappers + Pydantic schemas
│   ├── rag_retriever.py        ← Category-filtered RAG with severity reranking
│   ├── knowledge_base.py       ← Seeds 32 clinical guidelines into Supabase
│   ├── embed_records.py        ← Patient embedding + re_embed_record()
│   ├── pyproject.toml
│   └── .env
│
├── db/
│   ├── 01_schema.sql           ← source_records, patient_clusters, conflicts
│   ├── 02_seed.sql             ← Sample patients (CLN-001, LAB-001, PHM-001)
│   ├── 03_match_function.sql   ← pgvector RPC for patient similarity search
│   ├── 04_knowledge_base.sql   ← pgvector RPC: match_guidelines()
│   └── 05_notifications.sql    ← notifications + prescriptions tables
│
├── frontend/
│   └── app/
│       └── page.tsx            ← Full chat UI + Vapi voice chat (single page)
│
├── SYSTEM_DOCUMENTATION.html   ← Full written docs (open in browser → Save as PDF)
└── SYSTEM_DIAGRAMS.html        ← Visual diagrams (open in browser → Save as PDF)
```

---

## The 7 Chat Intents

| Intent | Triggered by | Agent | Uses RAG? |
|---|---|---|---|
| `register` | "Add clinic patient: …" | Registration Agent | No |
| `update` | "Update CLN-001 name to …" | Registration Agent | No |
| `prescribe` | "**Prescribe** ibuprofen for CLN-001" | Prescription Agent | Yes — drug_interaction + allergy |
| `db_update` | "**Add medication** X", "Remove allergy Y", "Delete …" | Database Agent | No |
| `query` | "Show all patients", "List escalations", "Search …" | Query Agent + LLM | Yes — chat (0.35 threshold) |
| `reconcile` | "Compare CLN-001 and LAB-001" | Reconciliation Agent | Yes — per conflict type |
| `chat` | General clinical questions | LLM | Yes — all categories |

> **Critical distinction:** "Add medication warfarin" = `db_update` (no safety check). "**Prescribe** warfarin" = `prescribe` (full RAG interaction check).

---

## RAG System — Clinical Knowledge Base

32 guidelines embedded in Supabase as 384-dim vectors. Retrieved dynamically at decision time — no LLM retraining needed when you add a new guideline.

### Retrieval Strategy per Agent

| Agent | Function | Threshold | Categories |
|---|---|---|---|
| Prescription Agent | `retrieve_for_prescription()` | 0.30 | drug_interaction + allergy_mismatch |
| Reconciliation Agent | `retrieve_for_conflict()` | 0.25 | mapped from conflict_type |
| General Chat | `retrieve_for_chat()` | 0.35 | all |

**Severity re-ranking:** `score = similarity × weight` where critical=2.0×, high=1.5×, medium=1.0×, low=0.7×. Critical guidelines always surface first even at slightly lower cosine similarity.

### Guidelines Coverage

| Category | Count | Examples |
|---|---|---|
| Drug Interactions (DI-) | 12 | Warfarin+Aspirin (CRITICAL), Opioids+Benzos (CRITICAL), Metformin+Contrast, Statins+Macrolides |
| Sri Lanka Specific (LK-) | 6 | **Dengue+NSAIDs (CRITICAL)**, Malaria+G6PD (CRITICAL), Ayurvedic interactions |
| Allergy Protocols (AL-) | 8 | Penicillin (CRITICAL), Sulfonamide, NSAID hypersensitivity, Latex-Fruit |
| Data Integrity (DT-) | 4 | Blood type conflict, Patient identity matching, Renal dosing, Hepatic impairment |
| Paediatric (PD-) | 2 | Weight-based dosing, Age restrictions (aspirin/codeine — CRITICAL) |
| Pregnancy (PG-) | 2 | Category D/X contraindications (CRITICAL), Safe medications |
| Renal/Hepatic | 2 | eGFR-based dosing, Child-Pugh classification |

---

## Role-Based Notifications

Staff at each location see only their own notifications in the bell icon. When a cross-location conflict is detected, **all locations in the same patient cluster** are notified:

```
CLN-001 and LAB-001 are the same patient (same cluster)
→ Conflict detected: blood_type A+ (clinic) vs B+ (lab)
→ CLN role bell: 🔔 1 unread
→ LAB role bell: 🔔 1 unread
→ PHM role bell: (silent — not involved in this conflict)
```

Role selector in the header filters to: ALL | CLN | LAB | PHM

---

## Database — 7 Supabase Tables

| Table | Purpose |
|---|---|
| `source_records` | Patient records per location (with embedding vector) |
| `patient_clusters` | Groups the same patient across locations |
| `detected_conflicts` | Per-field conflicts found during reconciliation |
| `adjudications` | LLM resolutions for each conflict |
| `escalations` | Conflicts flagged for human clinician review |
| `notifications` | Per-location alerts (role-filtered) |
| `prescriptions` | Prescription history (active / blocked / cancelled) |
| `medical_guidelines` | RAG knowledge base (with embedding vector) |

---

## Voice Chat (Vapi AI)

Concord supports full voice conversation via [Vapi AI](https://vapi.ai). Click the microphone button in the chat header to speak to the assistant — no typing needed.

### Voice Pipeline

```
User speaks into mic
       │
       ▼
Vapi STT (Deepgram) — transcribes speech to text
       │
       ▼
POST /chat/completions → FastAPI backend (via Cloudflare Tunnel)
       │
       ▼
Same router + agents + RAG as text chat
(Groq Llama 3.3-70B, max 3 sentences, no markdown)
       │
       ▼
OpenAI-compatible SSE streaming response
       │
       ▼
Vapi TTS (PlayHT) — speaks the response aloud
```

### Voice Setup Requirements

1. **Vapi account** — create a free assistant ("Riley") at [vapi.ai](https://vapi.ai)
2. **Riley assistant config:**
   - Provider: **Custom LLM**
   - Custom LLM URL: your Cloudflare tunnel base URL (e.g. `https://xxxx.trycloudflare.com`)
   - Vapi appends `/chat/completions` automatically — do NOT add it to the URL
3. **Cloudflare Tunnel** — exposes local port 8080 to Vapi over HTTPS:
   ```bash
   cloudflared tunnel --url http://localhost:8080
   ```
   Copy the `trycloudflare.com` URL and paste it as the Custom LLM URL in Vapi.
4. **Public key** — add your Vapi public key (not private key) to the frontend `Vapi("your-public-key")` call.

> **Note:** The Cloudflare Tunnel URL changes on every `cloudflared` restart. Update the Vapi Custom LLM URL when this happens.

### Voice in the Chat UI

- Mic button in the top-right of the header starts/stops voice
- Status badge shows: **Connecting → Listening → Speaking**
- Voice transcript appears in the chat as user messages (prefixed with 🎙️)
- Assistant replies are both spoken aloud AND shown in chat
- Full turn transcripts use `conversation-update` event (not `transcript`) to avoid word-by-word splitting

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/chat` | Main chat endpoint — all intents handled here |
| POST | `/chat/completions` | Vapi Custom LLM endpoint (OpenAI SSE format) |
| GET | `/notifications` | List notifications (`?source=CLN\|LAB\|PHM`) |
| DELETE | `/notifications/{id}` | Delete one notification |
| DELETE | `/notifications` | Clear all for a location (`?source=CLN`) |

### POST /chat — Request

```json
{
  "message": "Prescribe metformin for CLN-001",
  "source_ref_id": "CLN-001"
}
```

### POST /chat — Response examples

```json
{ "action": "issued",   "drug": "metformin", "patient_name": "Perera Sunil" }
{ "action": "blocked",  "drug": "ibuprofen", "reason": "Dengue fever — NSAIDs contraindicated (LK-002)" }
{ "action": "db_updated","message": "Medication paracetamol added to CLN-001" }
{ "action": "reconciled","conflicts": [...], "resolutions": [...] }
{ "action": "query_result","data": [...], "total": 5 }
```

---

## Getting Started

### Prerequisites

- Python 3.11+ with `uv`: `pip install uv`
- Node.js 18+
- Supabase project with pgvector enabled
- Gemini API key — [aistudio.google.com](https://aistudio.google.com)
- Groq API key (free) — [console.groq.com](https://console.groq.com)
- Vapi account (free) — [vapi.ai](https://vapi.ai) *(for voice chat)*
- `cloudflared` binary — download from [developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) *(for voice chat)*

### 1. Environment Variables

Create `concord/backend/.env`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
VAPI_API_KEY=your_vapi_private_key
```

### 2. Supabase Setup

Run these in the Supabase SQL editor **in order**:

```
db/01_schema.sql          ← creates all tables
db/02_seed.sql            ← inserts sample patients
db/03_match_function.sql  ← creates patient vector search RPC
db/04_knowledge_base.sql  ← creates match_guidelines() RPC
db/05_notifications.sql   ← creates notifications + prescriptions tables
```

Also enable pgvector:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Backend Setup

```bash
cd concord/backend

uv sync                          # install Python dependencies

uv run python embed_records.py   # embed sample patient records (one-time)
uv run python knowledge_base.py  # seed 32 guidelines into Supabase (one-time)

uv run uvicorn api:app --reload --port 8080
```

API running at `http://localhost:8080` — Swagger docs at `/docs`.

### 4. Frontend Setup

```bash
cd concord/frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

### 5. Voice Chat Setup (optional)

```bash
# Start Cloudflare Tunnel to expose port 8080
cloudflared tunnel --url http://localhost:8080
# Copy the https://xxxx.trycloudflare.com URL
```

Then in your Vapi dashboard:
- Open your Riley assistant → **Model** tab → set **Provider** to **Custom LLM**
- Set **Custom LLM URL** to the `trycloudflare.com` URL (no path suffix)
- Save. Click the mic button in Concord to start talking.

### 6. Smoke Tests

```bash
# Test RAG retrieval (should print matched guidelines for 6 queries)
uv run python rag_retriever.py

# Test LLM adjudication (2 LLM calls, JSON output)
uv run python llm_interface.py
```

---

## Sample Patients (from db/02_seed.sql)

| ID | Location | Patient | Pre-loaded conflict |
|---|---|---|---|
| CLN-001 | Clinic | Nimal Perera | Has warfarin + aspirin interaction |
| LAB-001 | Lab | N. Perera | Blood type B+ (conflicts with CLN-001's A+) |
| PHM-001 | Pharmacy | Nimal P. | In same cluster as CLN-001 and LAB-001 |
| CLN-002 | Clinic | Kamala Silva | Clean record |
| CLN-003 | Clinic | Sunil Fernando | Clean record |

---

## Try These in Chat

```
Show all patients
List unresolved escalations
Compare CLN-001 and LAB-001
Search patients with warfarin
Prescribe ibuprofen for CLN-001        ← triggers dengue/NSAID check
Add medication paracetamol for CLN-001 ← no safety check (db_update)
Update CLN-001 phone to 0771234567
Add allergy penicillin for CLN-001
Prescribe amoxicillin for CLN-001      ← blocked: penicillin allergy
What are the dengue NSAID guidelines?
```

### Try These by Voice (click the mic button)

```
"Tell me about CLN-001"              ← patient lookup via voice
"What are the dengue guidelines?"    ← RAG-powered clinical Q&A
"Search patients with warfarin"      ← query intent via speech
"What drug interactions should I know about?"
```

> Voice responses are limited to 3 sentences, spoken aloud in plain English (no markdown).
> For actions that modify data (prescribe, register, db_update), use text chat — voice is read-only by design.

---

## What Gets Escalated vs Resolved

| Resolved automatically | Escalated to clinician |
|---|---|
| Minor name/address discrepancies | Drug-drug interactions (critical severity) |
| Duplicate records with consistent data | Allergy conflicts with active medications |
| Phone/address mismatches | Blood type mismatches across sources |
| Low-confidence lab value differences | Any conflict where LLM confidence < 0.7 |

Escalations stored with urgency: `critical` / `urgent` / `routine`.

---

## Key Design Decisions

**Gemini + Groq instead of just one provider**  
Gemini 2.0 Flash handles adjudication (structured JSON output). Groq handles all agentic tool loops (faster for multi-turn). Automatic failover: Gemini fails → Groq 3.3-70B → Groq 3.1-8B.

**Custom agents instead of LangChain**  
Full control over tool schemas, thresholds, and failover. No framework overhead. Each agent has a purpose-built tool set.

**Local embeddings (all-MiniLM-L6-v2)**  
Runs offline, no API cost, 384 dimensions is sufficient for clinical text. Model is cached after first load.

**Supabase + pgvector instead of dedicated vector DB**  
One database for both relational data and vector search. The `match_guidelines` SQL RPC gives sub-10ms similarity search.

**Severity re-ranking in RAG**  
A CRITICAL guideline at 0.60 similarity (score 1.20) ranks above a MEDIUM guideline at 0.70 similarity (score 0.70). Patient safety always surfaces first.

**Cluster-wide notifications**  
When a conflict involves CLN-001 and LAB-001, both locations are notified — not just the one whose chat triggered the reconciliation.
