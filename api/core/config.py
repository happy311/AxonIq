"""
AxonIQ — Central Configuration (HuggingFace Spaces build, classifier-free)

Storage strategy for HuggingFace Spaces:
  /data/ → persistent volume (DB, ChromaDB, backups) — add via Space Settings → Storage
  /tmp/  → ephemeral (session JSON cache only — rebuilt from DB on restart)
"""
from __future__ import annotations
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
DOCX_PATH    = BASE_DIR / "AGENTIC AI chatbots.docx"

_DATA_DIR  = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH    = Path(os.environ.get("NEUROCHECK_DB",  str(_DATA_DIR / "neurocheck.db")))
CHROMA_DIR = Path(os.environ.get("CHROMA_DB_PATH", str(_DATA_DIR / "chroma_db")))
BACKUP_DIR = _DATA_DIR / "backups"

# Ephemeral session cache
SESSION_DIR = Path("/tmp/neurocheck_sessions")

# ── LLM — Anthropic (primary) then HuggingFace (fallback) ────────────────────
# Set ANTHROPIC_API_KEY to use Claude. If absent/invalid, HF models are tried.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
USE_CLAUDE        = bool(ANTHROPIC_API_KEY) and ANTHROPIC_API_KEY not in ("", "none", "your-key-here")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = os.environ.get("HF_MODEL", "")   # leave blank → auto-probe
USE_HF   = bool(HF_TOKEN) and HF_TOKEN not in ("", "none", "local")

# ── Auth ──────────────────────────────────────────────────────────────────────
JWT_SECRET         = os.environ.get("JWT_SECRET", "axoniq-dev-secret-change-in-production")
JWT_ALGORITHM      = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "10080"))  # 7 days

# ── RAG ───────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
RAG_MIN_SCORE = 0.3
RAG_TOP_K     = 5

# ── Rate limits ───────────────────────────────────────────────────────────────
CHAT_RATE_LIMIT    = "20/minute"
ANALYZE_RATE_LIMIT = "30/minute"

# ── Misc ──────────────────────────────────────────────────────────────────────
ADMIN_USERNAME      = os.environ.get("ADMIN_USERNAME", "")
SMTP_EMAIL          = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD       = os.environ.get("SMTP_PASSWORD", "")
SESSION_MAX_HISTORY = 40

# ── MRI NIfTI Service ─────────────────────────────────────────────────────────
# POST target for NIfTI (.nii.gz) scans.
# Defaults to a placeholder — replace with your real service URL via env var.
# On connection failure / timeout the patient is asked to try again later.
# NOTE: ngrok domains are blocked at the network/TLS layer from within this Space
# (SSL: UNEXPECTED_EOF_WHILE_READING — SNI-based filtering), confirmed 2026-08-25.
# The Kaggle server now tunnels via Cloudflare instead — set MRI_SERVICE_URL to
# the printed *.trycloudflare.com (or your named-tunnel) URL + /predict.
MRI_SERVICE_URL = os.environ.get("MRI_SERVICE_URL", "")
# Timeout in seconds for the segmentation-ensemble inference call.
# CPU inference can take up to 20 min; default is 1500 s (25 min) to give headroom.
# Override via MRI_SERVICE_TIMEOUT env var if your server is faster or slower.
MRI_SERVICE_TIMEOUT = int(os.environ.get("MRI_SERVICE_TIMEOUT", "1500"))

# ── Ollama (local / remote) ───────────────────────────────────────────────────
# Set OLLAMA_BASE_URL to point at a remote Ollama server
# (e.g. http://192.168.1.5:11434 or your friend's public ngrok URL).
# OLLAMA_MODEL lets you override which model to pull.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL",    "llama3.1:8b")
