# Concord — Autonomous Clinical Record Reconciliation

> Built for **AgenTrix 2026 Hackathon** by Dippy2003  
> Autonomous AI system that reconciles fragmented patient records across Sri Lankan clinics, labs, and pharmacies — using RAG + multi-agent architecture.

---

## What Problem Does This Solve?

In Sri Lanka, a single patient's health records are scattered across:
- **Clinics** — diagnoses, prescriptions
- **Laboratories** — blood tests, blood type
- **Pharmacies** — dispensed medications

These systems don't talk to each other. A patient might be prescribed **warfarin** at the clinic and **aspirin** at the pharmacy — a dangerous drug interaction that no one catches because the records are never compared.

**Concord** automatically pulls all records for a patient, detects conflicts, and uses AI to resolve or escalate them — without any human having to manually compare files.

---

## How It Works — The Full Flow

```
Patient gives their ID
        │
        ▼
┌───────────────────┐      ┌──────────────────────┐
│  Identity Agent   │      │  Reconciliation Agent │
│                   │      │                       │
│ 1. Look up the    │◄────►│ 1. Match records from │
│    record for     │      │    clinic, lab, pharma│
│    given ID       │      │                       │
│                   │      │ 2. Detect conflicts:  │
│ 2. Compare every  │      │    drug interactions, │
│    detail patient │      │    allergy mismatches,│
│    stated vs what │      │    data inconsistency │
│    is on file     │      │                       │
│    (name, DOB,    │      │ 3. Retrieve clinical  │
│     NIC, phone,   │      │    guidelines via RAG │
│     address)      │      │    (vector search)    │
│                   │      │                       │
│ 3. If ID wrong →  │      │ 4. LLM decides:       │
│    find correct   │      │    resolve or escalate│
│    ID             │      │    each conflict      │
└───────────────────┘      └──────────────────────┘
        │                          │
        └──────────┬───────────────┘
                   ▼
        ┌─────────────────────┐
        │   Combined Result   │
        │                     │
        │  • Identity status  │
        │  • Per-field match  │
        │  • All conflicts    │
        │  • AI resolutions   │
        │  • Escalations      │
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Concord Assistant │
        │   (Chat with AI)    │
        │                     │
        │  Ask anything about │
        │  the results, drug  │
        │  risks, guidelines  │
        └─────────────────────┘
```

Both agents run **at the same time** (parallel threads). If the Identity Agent finds the patient used the wrong ID, the reconciliation re-runs automatically with the correct one.

---

## Architecture

### Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| LLM | Groq — `llama-3.3-70b-versatile` (free tier) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs offline, 384-dim) |
| Vector DB | Supabase + pgvector (cosine similarity search) |
| Database | Supabase (PostgreSQL) |
| Frontend | Next.js 15 + TypeScript + Tailwind CSS |
| Package Manager | `uv` (Python), `npm` (Node) |

### What is RAG?

**RAG = Retrieval-Augmented Generation**

Instead of the LLM relying only on what it learned during training, we:
1. Store 16 curated clinical guidelines in Supabase as vector embeddings
2. When a conflict is detected (e.g. warfarin + aspirin), we embed the conflict description and search for the most similar guidelines
3. Those guidelines are injected into the LLM prompt as context
4. The LLM makes its decision **grounded in real clinical evidence**, not just pattern matching

This means if you add a new drug interaction guideline to the database, the AI immediately knows about it — no retraining needed.

### What is Agentic?

Instead of a fixed pipeline ("step 1 → step 2 → step 3"), the **LLM itself decides** what to do next using tools:

```
LLM thinks → calls tool → sees result → thinks again → calls another tool → ...
```

The reconciliation agent has 6 tools:
- `match_patient_records` — find all records for this patient
- `detect_record_conflicts` — compare records and list conflicts
- `retrieve_medical_guidelines` — RAG search for relevant guidelines
- `resolve_conflict` — mark a conflict as resolved with rationale
- `escalate_conflict` — flag for human clinician review
- `complete_reconciliation` — finish and return results

The LLM decides which tool to call and when. It might retrieve guidelines for one conflict, resolve it, then escalate a different one — all in one session.

---

## Project Structure

```
concord/
├── backend/
│   ├── api.py                # FastAPI server — all HTTP endpoints
│   ├── agent.py              # Reconciliation agent (Groq + tool loop)
│   ├── identity_agent.py     # Identity validation agent
│   ├── rag_retriever.py      # Vector search over medical guidelines
│   ├── knowledge_base.py     # Seeds the 16 clinical guidelines into DB
│   ├── identity_matcher.py   # Cross-source patient identity matching
│   ├── conflict_detector.py  # Rule-based conflict detection
│   ├── adjudicator.py        # LLM-based conflict adjudication (pipeline mode)
│   ├── action_executor.py    # Applies resolved changes to DB
│   ├── escalation_reviewer.py# LLM escalation review (pipeline mode)
│   ├── embed_records.py      # Embeds patient records for similarity search
│   ├── llm_interface.py      # Shared LLM utilities
│   ├── pyproject.toml        # Python dependencies
│   └── .env.example          # Environment variable template
│
├── db/
│   ├── 01_schema.sql         # All table definitions
│   ├── 02_seed.sql           # Sample patient data (CLN-001, LAB-001, etc.)
│   ├── 03_match_function.sql # pgvector RPC for patient record search
│   └── 04_knowledge_base.sql # pgvector RPC for guideline search
│
└── frontend/
    └── app/
        └── page.tsx          # Entire UI — single-page React app
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/reconcile/{id}` | Pipeline mode — 5-step fixed reconciliation |
| POST | `/reconcile-agent/{id}` | Agent mode — LLM drives the tool loop with RAG |
| POST | `/reconcile-verified` | Verified mode — dual agents run in parallel |
| POST | `/chat` | Chat with AI about any clinical question |

### `/reconcile-verified` — the main endpoint

Accepts:
```json
{
  "source_ref_id": "CLN-001",
  "patient_name": "Nimal Perera",
  "dob": "1975-03-12",
  "nic": "750312123V",
  "phone": "0771234567",
  "address": "Colombo 7"
}
```

Returns:
```json
{
  "identity": {
    "is_correct": true,
    "field_details": [
      { "field": "name", "provided": "Nimal Perera", "stored": "Nimal Perera", "match": true },
      { "field": "dob",  "provided": "1975-03-12",   "stored": "1975-03-12",   "match": true }
    ],
    "confidence": 0.97
  },
  "reconciliation": {
    "conflicts_detected": 1,
    "conflicts": [...],
    "resolutions": [...],
    "overall_safe": false,
    "adjudication_summary": "..."
  },
  "id_was_corrected": false
}
```

### `/chat` — conversational AI

```json
{
  "message": "Why was this conflict escalated?",
  "source_ref_id": "CLN-001",
  "reconciliation_context": { ...full result from previous reconciliation... },
  "history": []
}
```

The chat is context-aware — if you've reconciled a patient first, the assistant knows everything about that patient's conflicts and can answer specific questions about them.

---

## UI Modes

### Pipeline Mode
Classic deterministic reconciliation. Uses rule-based conflict detection + 2 LLM calls (adjudication + escalation review). Fast.

### RAG + Agent Mode
The LLM drives the full loop. Retrieves clinical guidelines per conflict via vector search. More thorough, takes more turns.

### Verified Mode (Recommended)
Combines both agents:
1. **Identity Agent** checks every detail the patient stated (name, DOB, NIC, phone, address) against what's on file, field by field
2. **Reconciliation Agent** runs the full conflict analysis in parallel
3. If the ID was wrong, reconciliation re-runs automatically with the corrected ID

Shows a per-field match/mismatch table so you can see exactly which details the patient got wrong.

### Ask AI (Chat)
Chat panel in the bottom-right corner. Asks questions grounded in clinical guidelines retrieved via RAG. When a patient has been reconciled, the assistant has full context of their conflicts, resolutions, and escalations.

---

## Getting Started

### Prerequisites

- Python 3.13+
- Node.js 18+
- `uv` (Python package manager): `pip install uv`
- A Supabase project with pgvector enabled
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Database Setup

Run these SQL files **in order** in the Supabase SQL editor:

```
db/01_schema.sql          — creates all tables
db/02_seed.sql            — inserts sample patients
db/03_match_function.sql  — creates patient vector search RPC
db/04_knowledge_base.sql  — creates guideline vector search RPC
```

Also run this to disable RLS on the guidelines table:
```sql
alter table medical_guidelines disable row level security;
```

### 2. Backend Setup

```bash
cd concord/backend

# Copy and fill in your API keys
cp .env.example .env
# Edit .env — add your SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY

# Install dependencies
uv sync

# Embed the patient records (one-time setup)
uv run python embed_records.py

# Seed the medical guidelines knowledge base (one-time setup)
uv run python knowledge_base.py

# Start the API server
uv run uvicorn api:app --reload --port 8080
```

The API is now running at `http://localhost:8080`.  
Swagger docs: `http://localhost:8080/docs`

### 3. Frontend Setup

```bash
cd concord/frontend

npm install
npm run dev
```

Open `http://localhost:3000`.

---

## Sample Patient IDs

These are seeded in the database (from `db/02_seed.sql`):

| ID | Source | Patient |
|---|---|---|
| CLN-001 | Clinic | Nimal Perera — has drug interaction conflict (warfarin + aspirin) |
| CLN-002 | Clinic | Kamala Silva |
| CLN-003 | Clinic | Sunil Fernando |
| LAB-001 | Laboratory | N. Perera (same patient as CLN-001, matched by AI) |
| PHM-001 | Pharmacy | Nimal P. (same patient as CLN-001, matched by AI) |

Entering `CLN-001` in Agent or Verified mode will detect the warfarin/aspirin conflict and escalate it for clinician review.

---

## The 16 Clinical Guidelines (Knowledge Base)

Stored as vector embeddings in `medical_guidelines`. Covers:

**Drug Interactions**
- Warfarin + Aspirin (critical bleeding risk)
- Methotrexate + Aspirin (toxicity)
- ACE Inhibitor + Potassium-Sparing Diuretic (hyperkalemia)
- Fluoxetine + Tramadol (serotonin syndrome)

**Allergy Protocols**
- Penicillin cross-reactivity with cephalosporins
- Sulfonamide allergy conflicts
- NSAID hypersensitivity
- Contrast media allergy

**Data Integrity Rules**
- Blood type mismatch across sources
- Date of birth discrepancy
- Medication reconciliation on admission
- Allergy documentation completeness
- Renal dosing adjustments

**Sri Lanka-Specific Rules**
- NIC number format normalisation (old 9+V vs new 12-digit)
- Herbal and Ayurvedic medicine interactions

---

## Environment Variables

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key
GROQ_API_KEY=your_groq_api_key
```

Get Groq API key free at: [console.groq.com](https://console.groq.com)  
Get Supabase credentials from: Project Settings → API

---

## Key Design Decisions

**Why Groq instead of OpenAI/Gemini?**  
Free tier with generous limits. Uses the OpenAI-compatible API format so tool calling works the same way. Gemini's free tier was exhausted by the multi-turn agent loops.

**Why run agents in parallel?**  
Identity validation and record reconciliation are independent — they don't need each other's results to start. Running them concurrently with `ThreadPoolExecutor` cuts the total wait time roughly in half.

**Why local embeddings (`all-MiniLM-L6-v2`)?**  
It runs fully offline, no API cost, and 384 dimensions is more than sufficient for clinical text similarity. The model is cached after first load.

**Why Supabase + pgvector instead of a dedicated vector DB?**  
Simplifies infrastructure — one database for both relational data (patient records, conflicts) and vector search (embeddings). The `match_records` and `match_guidelines` SQL RPCs give sub-10ms similarity search.

---

## What Gets Escalated vs Resolved

The LLM agent makes this call per conflict based on retrieved guidelines:

| Resolved automatically | Escalated to clinician |
|---|---|
| Minor data discrepancies (name spelling, address format) | Drug-drug interactions (especially critical severity) |
| Duplicate records with consistent data | Allergy conflicts with current medications |
| Lab value differences within normal range | Blood type mismatches |
| Address/phone mismatches | Any conflict where the LLM confidence is < 0.7 |

Escalations are stored in the `escalations` table with a UUID and urgency level (critical / high / medium / low).
