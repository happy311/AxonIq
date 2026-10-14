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
BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", str(_DATA_DIR.parent / "backups" if str(_DATA_DIR) != "/" else "/backups")))

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

# ── Environment ───────────────────────────────────────────────────────────────
ENV    = os.environ.get("ENV", "production").lower()      # "dev" enables /docs and non-Secure cookies
IS_DEV = ENV == "dev"

# ── Auth ──────────────────────────────────────────────────────────────────────
_PLACEHOLDER_MARKERS = ("change", "placeholder", "example", "replace_me", "replaceme",
                        "your_secret", "your-secret", "yoursecret")


def validate_jwt_secret(secret: str) -> None:
    """Raise RuntimeError if the secret is missing, short, or a placeholder.
    (start.py carries an equivalent standalone check — keep them in sync.)"""
    if not secret:
        raise RuntimeError("JWT_SECRET is not set. Generate one with: "
                           "python -c \"import secrets; print(secrets.token_urlsafe(48))\"")
    if len(secret) < 32:
        raise RuntimeError(f"JWT_SECRET is too short ({len(secret)} chars); at least 32 are required.")
    low = secret.lower()
    if any(m in low for m in _PLACEHOLDER_MARKERS) or len(set(secret)) < 8:
        raise RuntimeError("JWT_SECRET looks like a placeholder/low-entropy value. "
                           "Replace it with a random value.")


JWT_SECRET = os.environ.get("JWT_SECRET", "")
validate_jwt_secret(JWT_SECRET)
JWT_ALGORITHM      = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "480"))   # 8 h default
if not 5 <= JWT_EXPIRE_MINUTES <= 1440:
    raise RuntimeError("JWT_EXPIRE_MINUTES must be between 5 and 1440 (24 h).")
RESET_TOKEN_MINUTES       = 10
ADMIN_RESET_TOKEN_MINUTES = int(os.environ.get("ADMIN_RESET_TOKEN_MINUTES", "60"))
OTP_EXPIRY_MINUTES        = 15
OTP_MAX_ATTEMPTS          = 5

# Cookie-based session (HttpOnly) + CSRF double-submit
AUTH_COOKIE_NAME = "nc_session"
CSRF_COOKIE_NAME = "nc_csrf"
COOKIE_SECURE    = os.environ.get("COOKIE_SECURE", "0" if IS_DEV else "1") == "1"

# ── CORS ──────────────────────────────────────────────────────────────────────
# Comma-separated list of exact origins, e.g. "https://app.example.com".
# Empty = no cross-origin access (the UI is same-origin and does not need CORS).
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if "*" in ALLOWED_ORIGINS:
    raise RuntimeError("ALLOWED_ORIGINS must not contain '*'.")

# ── Auth rate limits ──────────────────────────────────────────────────────────
LOGIN_RATE_IP          = "5/minute"
LOGIN_RATE_USER        = (5, 60)          # (count, window seconds) per username
REGISTER_RATE_IP       = "5/hour"
CHANGE_PW_RATE         = "5/minute"
FORGOT_RATE_EMAIL      = (5, 3600)
FORGOT_RATE_IP         = (10, 3600)
VERIFY_RATE_EMAIL      = (10, 3600)
VERIFY_RATE_IP         = (10, 3600)
LOCKOUT_THRESHOLD      = 5                # consecutive failures before lockout starts
LOCKOUT_MAX_MINUTES    = 60

# ── Chat cost / abuse controls ────────────────────────────────────────────────
CHAT_USER_RATE         = (20, 60)         # per authenticated user per minute
CHAT_DAILY_QUOTA       = int(os.environ.get("CHAT_DAILY_QUOTA", "200"))
HISTORY_TOKEN_BUDGET   = int(os.environ.get("HISTORY_TOKEN_BUDGET", "6000"))   # ~chars/4
LLM_TIMEOUT_S          = float(os.environ.get("LLM_TIMEOUT_S", "60"))
LLM_MAX_TOKENS         = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

# ── Uploads / MRI job limits ──────────────────────────────────────────────────
MRI_MAX_UPLOAD_BYTES     = int(os.environ.get("MRI_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
MRI_MAX_DECOMPRESSED     = int(os.environ.get("MRI_MAX_DECOMPRESSED_BYTES", str(1024 * 1024 * 1024)))
MRI_MAX_DIM              = int(os.environ.get("MRI_MAX_DIM", "1024"))        # voxels per axis
MRI_MAX_VOXELS           = int(os.environ.get("MRI_MAX_VOXELS", str(512 * 512 * 400)))
MRI_VALIDATE_TIMEOUT_S   = int(os.environ.get("MRI_VALIDATE_TIMEOUT_S", "60"))
MRI_VALIDATE_MEM_MB      = int(os.environ.get("MRI_VALIDATE_MEM_MB", "2048"))
MRI_CONVERT_MEM_MB      = int(os.environ.get("MRI_CONVERT_MEM_MB", "4096"))   # DICOM-zip → NIfTI sandbox
MRI_CONVERT_TIMEOUT_S    = int(os.environ.get("MRI_CONVERT_TIMEOUT_S", "180"))
MRI_MAX_QUEUE            = int(os.environ.get("MRI_MAX_QUEUE", "5"))
MRI_DAILY_UPLOAD_QUOTA   = int(os.environ.get("MRI_DAILY_UPLOAD_QUOTA", "10"))
MRI_JOB_STALE_MINUTES    = int(os.environ.get("MRI_JOB_STALE_MINUTES", "60"))
UPLOAD_DIR_NAME          = "uploads"

# ── Roles: administrator (from env) / doctors (approved by the administrator) ─
# The administrator account is created from these variables at startup (see api/core/bootstrap.py).
# ADMIN_PASSWORD must satisfy the password policy (>= 12 chars). Leave ADMIN_USERNAME empty to
# disable env-based admin (then `python make_admin.py <user>` is the only way to grant admin).
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "").strip()
ADMIN_EMAIL    = os.environ.get("ADMIN_EMAIL", "").strip().lower()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
DOCTOR_SEARCH_MIN_CHARS = 3
DOCTOR_SEARCH_MAX_RESULTS = 20
DOCTOR_DOWNLOAD_RATE   = os.environ.get("DOCTOR_DOWNLOAD_RATE", "10/hour")

# ── Stored MRI scans (kept encrypted so an approved doctor can download them as DICOM) ──
MRI_STORE_DIR          = Path(os.environ.get("MRI_STORE_DIR", str(_DATA_DIR / "mri_store")))
MRI_STORE_MAX_PER_USER = int(os.environ.get("MRI_STORE_MAX_PER_USER", "20"))

# ── Data protection ───────────────────────────────────────────────────────────
# Fernet key (urlsafe base64, 32 bytes). Generate:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
DATA_ENCRYPTION_KEY    = os.environ.get("DATA_ENCRYPTION_KEY", "")
REQUIRE_DATA_ENCRYPTION = os.environ.get("REQUIRE_DATA_ENCRYPTION", "0" if IS_DEV else "1") == "1"
RETENTION_DAYS         = int(os.environ.get("RETENTION_DAYS", "0"))          # 0 = keep until user deletes
DELETED_SESSION_PURGE_DAYS = int(os.environ.get("DELETED_SESSION_PURGE_DAYS", "0"))

# ── Checkpoint integrity ──────────────────────────────────────────────────────
# Comma-separated "filename:sha256" pairs, e.g. "axial.pt:ab12...,coronal.pt:..."
# or set MRI_CHECKPOINT_SHA256_FILE to a file with "sha256  filename" lines.
MRI_CHECKPOINT_SHA256 = os.environ.get("MRI_CHECKPOINT_SHA256", "")
MRI_CHECKPOINT_SHA256_FILE = os.environ.get("MRI_CHECKPOINT_SHA256_FILE", str(BASE_DIR / "checkpoints" / "SHA256SUMS"))

# ── RAG ───────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
RAG_MIN_SCORE = 0.3
RAG_TOP_K     = 5

# ── Rate limits ───────────────────────────────────────────────────────────────
CHAT_RATE_LIMIT    = "20/minute"
ANALYZE_RATE_LIMIT = "30/minute"

# ── Misc ──────────────────────────────────────────────────────────────────────
SMTP_EMAIL          = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD       = os.environ.get("SMTP_PASSWORD", "")
SESSION_MAX_HISTORY = 40

# ── MRI segmentation engine ───────────────────────────────────────────────────
# v21: the segmentation ensemble no longer lives on a separate Kaggle server
# behind a tunnel — it runs in-process (api/mri_engine/), CPU-only, loaded
# from local checkpoint files. See api/mri_engine/config.py for
# MRI_CHECKPOINT_DIR, MRI_DEVICE, MRI_MAX_CONCURRENT_JOBS, MRI_INFERENCE_TIMEOUT.

# ── Ollama (local / remote) ───────────────────────────────────────────────────
# Set OLLAMA_BASE_URL to point at a remote Ollama server
# (e.g. http://192.168.1.5:11434 or your friend's public ngrok URL).
# OLLAMA_MODEL lets you override which model to pull.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL",    "llama3.1:8b")

# Public base URL used in e-mailed links (admin password reset). Never derived from the
# request Host header in production (host-header poisoning).
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
