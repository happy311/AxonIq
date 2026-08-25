#!/usr/bin/env python3
"""
NeuroCheck v9.1 — Start Server
================================
Run after setup.py:
    python3 start.py
    python3 start.py --port 8080
    python3 start.py --workers 2
"""
from __future__ import annotations
import argparse, os, pathlib, subprocess, sys

ROOT  = pathlib.Path(__file__).parent.resolve()
ENV   = ROOT / ".env"
VENV  = ROOT / ".venv"

# Load .env
if ENV.exists():
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# Warn about placeholder secret
if os.environ.get("JWT_SECRET", "CHANGE") in ("CHANGE_THIS_TO_A_LONG_RANDOM_SECRET_BEFORE_RUNNING", "CHANGE_THIS", "CHANGE"):
    print("\033[93m⚠  JWT_SECRET is still the placeholder! Edit .env before storing real data.\033[0m")

# Decide which python to use
def best_python():
    candidates = [VENV / "bin" / "python", VENV / "Scripts" / "python.exe", pathlib.Path(sys.executable)]
    for p in candidates:
        if p.exists():
            return str(p)
    return sys.executable

parser = argparse.ArgumentParser(description="Start NeuroCheck server")
parser.add_argument("--host",    default=os.environ.get("HOST", "0.0.0.0"))
parser.add_argument("--port",    default=int(os.environ.get("PORT", 8000)), type=int)  # HF uses 7860
parser.add_argument("--workers", default=int(os.environ.get("WORKERS", 1)), type=int)
parser.add_argument("--reload",  action="store_true", default=True, help="Auto-reload on code change (dev mode)")
parser.add_argument("--no-reload", dest="reload", action="store_false")
# MRI analysis (segmentation ensemble) can take 15-20 min on CPU.  Keep-alive must outlast the
# longest possible HTTP response so proxies (nginx, HF, Railway) don't close the
# connection mid-analysis.  1800 s = 30 min gives 10 min of headroom.
parser.add_argument("--timeout-keep-alive", default=int(os.environ.get("TIMEOUT_KEEP_ALIVE", 1800)), type=int)
args = parser.parse_args()

py = best_python()
cmd = [
    py, "-m", "uvicorn", "api.main:app",
    "--host", args.host,
    "--port", str(args.port),
    "--workers", str(args.workers),
    "--log-level", os.environ.get("LOG_LEVEL", "info").lower(),
    "--timeout-keep-alive", str(args.timeout_keep_alive),
]
if args.reload and args.workers == 1:
    cmd.append("--reload")

print(f"""
\033[96m══ NeuroCheck v9.1 ══════════════════════════\033[0m
  \033[1mClassifier :\033[0m  http://localhost:{args.port}
  \033[1mChat UI    :\033[0m  http://localhost:{args.port}/chat-ui
  \033[1mAPI Docs   :\033[0m  http://localhost:{args.port}/docs
  \033[1mWorkers    :\033[0m  {args.workers}
  \033[1mReload     :\033[0m  {args.reload and args.workers == 1}
\033[96m═════════════════════════════════════════════\033[0m
""")

os.environ["SPACY_MODEL"] = str(ROOT / os.environ.get("SPACY_MODEL", "en_model"))
os.environ["NEUROCHECK_DB"] = str(ROOT / os.environ.get("NEUROCHECK_DB", "neurocheck.db"))

os.chdir(ROOT)
os.execv(py, cmd)
