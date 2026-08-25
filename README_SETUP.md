# NeuroCheck v9.1 — Local Setup Guide

## Prerequisites

| Software | Version | Download |
|----------|---------|----------|
| Python   | 3.10+   | https://python.org/downloads |
| Ollama   | Latest  | https://ollama.com/download |
| Git      | Any     | https://git-scm.com |

---

## One-Time Setup (run this once)

```bash
# 1. Clone / unzip the project, then enter it
cd neurocheck

# 2. Run the full setup script
python3 setup.py
```

The setup script will:
- Create a `.venv` virtual environment
- Install all Python packages from `requirements.txt`
- Build the spaCy English model (`en_model/`)
- Verify the NegEx negation pipeline
- Create and initialise the SQLite database (`neurocheck.db`)
- Check Ollama and pull `llama3.1:8b` if needed

---

## Before First Run — Set Your JWT Secret

Open `.env` and replace the placeholder:

```env
JWT_SECRET=CHANGE_THIS_TO_A_LONG_RANDOM_SECRET_BEFORE_RUNNING
```

Generate a real secret:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output into `.env`:
```env
JWT_SECRET=a1b2c3d4e5f6...  (your generated value)
```

---

## Start the Server

```bash
python3 start.py
```

Or manually:
```bash
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Open in Browser

| URL | Description |
|-----|-------------|
| http://localhost:8000 | MS Symptom Classifier |
| http://localhost:8000/chat-ui | AI Chat (login required) |
| http://localhost:8000/docs | Swagger API docs |

---

## Ollama Setup (for AI Chat)

Ollama runs the local LLM (Llama 3.1) that powers the chat assistant.

### Install Ollama

**Windows / macOS:** Download from https://ollama.com/download and run the installer.

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Start Ollama and Pull the Model

```bash
# Start the Ollama server (runs in background)
ollama serve

# Pull the model (4.7 GB — do this once)
ollama pull llama3.1:8b

# Verify it's working
ollama run llama3.1:8b "Hello"
```

> **Note:** The MS Symptom Classifier (`/classify`) works without Ollama.
> Only the `/chat` AI assistant requires Ollama to be running.

---

## Database

NeuroCheck uses SQLite — no separate database server needed.

**File:** `neurocheck.db` (created automatically by `setup.py`)

**Tables:**

| Table | What it stores |
|-------|---------------|
| `users` | Registered accounts (username, email, hashed password) |
| `chat_sessions` | Each conversation linked to a user (title, timestamps) |
| `messages` | Every message with role and UTC timestamp |

### Inspect the Database

```bash
# Open SQLite shell
sqlite3 neurocheck.db

# Inside the shell:
.tables
SELECT * FROM users;
SELECT session_uuid, title, updated_at FROM chat_sessions ORDER BY updated_at DESC;
SELECT role, content, created_at FROM messages WHERE session_uuid='<uuid>' ORDER BY id;
.quit
```

---

## Project Structure

```
neurocheck/
├── api/
│   ├── main.py        ← FastAPI app + all endpoints
│   ├── auth.py        ← JWT login / register (NEW)
│   ├── database.py    ← SQLite layer (NEW)
│   ├── memory.py      ← In-memory session store
│   ├── agent.py       ← LangGraph AI agent
│   └── rag.py         ← ChromaDB RAG knowledge base
├── frontend/
│   ├── chat.html      ← AI Chat UI (login + session sidebar)
│   └── index.html     ← MS Classifier UI
├── en_model/          ← Built by setup.py (spaCy)
├── neurocheck.db      ← Created by setup.py (SQLite)
├── chroma_db/         ← Created at runtime (ChromaDB)
├── .env               ← Your config (never commit this)
├── .env.example       ← Template
├── setup.py           ← One-time setup script
├── start.py           ← Start the server
└── requirements.txt
```

---

## Auth Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/auth/register` | Create account |
| `POST` | `/auth/login` | Get JWT token |
| `GET`  | `/auth/me` | Current user info |

---

## Common Issues

### `spacy.errors.E050` — Model not found
```bash
python3 setup.py   # rebuilds en_model/
```

### `Connection refused` on `/chat`
Ollama is not running. Start it:
```bash
ollama serve
```

### `401 Unauthorized` on `/chat`
Your JWT token expired or you are not logged in.
Open `http://localhost:8000/chat-ui` and log in again.

### Port 8000 already in use
```bash
python3 start.py --port 8080
```

### Database errors
```bash
python3 -c "
import os; os.environ['NEUROCHECK_DB']='neurocheck.db'
from api.database import init_db; init_db(); print('OK')
"
```
