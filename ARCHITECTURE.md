# AxonIQ — Refactored Architecture

## Design Pattern: Layered Clean Architecture

```
axoniq_refactored/
├── api/
│   ├── core/                    # Shared primitives — no business logic
│   │   ├── __init__.py
│   │   ├── config.py            # All env vars & constants in one place
│   │   └── schemas.py           # All Pydantic models shared across layers
│   │
│   ├── classifier/              # MS symptom classification engine
│   │   ├── __init__.py
│   │   ├── patterns.py          # SYMPTOM_PATTERNS, LOW_PRIORITY_PATTERNS, etc.
│   │   ├── engine.py            # classify() — pure function, no I/O
│   │   └── nlp.py               # spaCy / NegEx loader & helpers
│   │
│   ├── rag/                     # Knowledge base (vector store)
│   │   ├── __init__.py
│   │   ├── knowledge.py         # Hard-coded MS guidelines
│   │   ├── store.py             # ChromaDB init, build, query
│   │   └── extractor.py         # Docx/document text extraction
│   │
│   ├── agent/                   # Agentic LLM chat layer
│   │   ├── __init__.py
│   │   ├── llm.py               # LLM provider setup (Ollama / HuggingFace)
│   │   ├── prompts.py           # SYSTEM_PROMPT and context builders
│   │   ├── greeting.py          # Fast-path greeting detection & responses
│   │   └── chat.py              # Main chat() orchestrator
│   │
│   ├── routes/                  # FastAPI routers — HTTP boundary only
│   │   ├── __init__.py
│   │   ├── classify.py          # /classify, /analyze
│   │   ├── chat.py              # /chat, /session/*
│   │   ├── admin.py             # /admin/*
│   │   └── frontend.py          # Static HTML serving
│   │
│   ├── database.py              # SQLite data layer (unchanged interface)
│   ├── auth.py                  # JWT auth (unchanged interface)
│   ├── memory.py                # File-based session memory (unchanged)
│   ├── email_utils.py           # Email utilities (unchanged)
│   └── main.py                  # App factory — wires everything together
```

## Why This Design?

### Problem with original `main.py` (1,534 lines)
- **Zero cohesion**: symptom patterns (domain data), NLP helpers, scoring logic,
  API routes, admin endpoints, and frontend serving all in one file
- **High coupling**: agent.py imports `from api.main import classify` — a route
  file depended on a sibling module, creating a circular-import risk
- **Untestable in isolation**: can't unit-test the classifier without loading
  FastAPI, spaCy, ChromaDB, and all routes

### How the refactor fixes it
| Concern            | Old location            | New location                     |
|--------------------|-------------------------|----------------------------------|
| Pattern data        | main.py (800+ lines)   | classifier/patterns.py           |
| NLP helpers         | main.py                | classifier/nlp.py                |
| Scoring / classify()| main.py                | classifier/engine.py             |
| LLM setup           | agent.py               | agent/llm.py                     |
| System prompt       | agent.py               | agent/prompts.py                 |
| Greeting detection  | agent.py               | agent/greeting.py                |
| Chat orchestration  | agent.py               | agent/chat.py                    |
| ChromaDB I/O        | rag.py                 | rag/store.py                     |
| MS guidelines data  | rag.py                 | rag/knowledge.py                 |
| Docx extraction     | rag.py                 | rag/extractor.py                 |
| HTTP classify routes| main.py                | routes/classify.py               |
| HTTP chat routes    | main.py                | routes/chat.py                   |
| HTTP admin routes   | main.py                | routes/admin.py                  |
| Frontend serving    | main.py                | routes/frontend.py               |
| App wiring          | main.py                | main.py (now ~60 lines)          |

### Coupling rules
- `core/` imports nothing from `api/`
- `classifier/` imports only `core/`
- `rag/` imports only `core/`
- `agent/` imports `classifier/`, `rag/`, `core/`
- `routes/` imports `agent/`, `classifier/`, `core/`, `database`, `auth`
- `main.py` imports `routes/` only

This is a strict **dependency inversion** — high-level policy (classifier engine)
never depends on low-level details (HTTP routes, databases).

---

## v11 Changes (AxonIQ Enhanced Build)

### New Features

#### LLM Provider (api/agent/llm.py)
Claude-first waterfall: `_ClaudeLLM → _HFChatLLM → Ollama`
- Set `ANTHROPIC_API_KEY` to use Claude (most reliable, zero parse fragility)
- If absent/invalid, auto-falls through to HuggingFace model probe
- Zero changes needed in any consumer — same `.invoke()` interface

#### Output Format (api/agent/prompts.py)
Replaced fragile TIER:/FOUND: line format with structured JSON block.
LLMs (especially Claude) output this with near-zero parse errors.
Three-level fallback in `_parse()`: JSON → TIER/FOUND → XML tags.

#### McDonald 2017 DIS/DIT Criteria (state.py, node_goal_setter.py, database.py)
New state fields: `dis_regions`, `dit_episodes`, `symptom_timeline`
Persisted per turn in DB. Goal setter applies McDonald fast-track:
- 2+ CNS regions (DIS) → immediate HIGH escalation
- 1 region + 2+ prior episodes (DIS+DIT) → immediate HIGH escalation

#### Uhthoff / Lhermitte Probe (api/agent/prompts.py → gather())
Automatically injected when optic neuritis signals are present (turns ≥ 2).
Ensures the LLM asks about heat sensitivity, prior episode, and vision recovery.

#### Retry + Validation Loop (api/agent/graph/node_llm.py)
Up to 2 retries when a suspicious tier regression is detected
(non-first turn, had features, LLM returns LOW with nothing found).

#### Tier History Logging (api/database.py)
`tier_log` JSON column on `chat_sessions`. Every turn appends
`{turn, tier, features, timestamp}`. Available via `GET /session/{id}/tier-log`.

#### Clinical Summary Export (api/routes/chat.py, api/agent/chat.py)
`GET /session/{id}/export` — generates structured clinical pre-assessment
via LLM (SUMMARY_GOAL) including McDonald assessment and neurologist note.

#### DB-backed NIfTI Queue (api/database.py, api/routes/chat.py)
`nifti_queue` SQLite table replaces in-process dict.
Works correctly across multiple Uvicorn workers on HuggingFace Spaces.

### Environment Variables (all optional — system works without them)
```
ANTHROPIC_API_KEY   Claude API key (primary LLM — falls back to HF if absent)
ANTHROPIC_MODEL     Claude model string (default: claude-haiku-4-5-20251001)
HF_TOKEN            HuggingFace token (fallback LLM)
HF_MODEL            Preferred HF model (blank = auto-probe)
```

### Coupling rules — unchanged
- core/ imports nothing from api/
- classifier/ imports only core/
- rag/ imports only core/
- agent/ imports classifier/, rag/, core/
- routes/ imports agent/, classifier/, core/, database, auth
- main.py imports routes/ only
