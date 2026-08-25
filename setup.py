#!/usr/bin/env python3
"""
NeuroCheck v9.1 — Full Local Setup Script
==========================================
Run once before first launch:
    python3 setup.py

What this does:
  1. Checks Python version (needs 3.10+)
  2. Creates / activates a virtual environment (.venv)
  3. Installs ALL pip requirements
  4. Builds the spaCy English model (en_model/)
  5. Verifies NegEx pipeline
  6. Initialises the SQLite database (neurocheck.db)
  7. Checks Ollama + pulls llama3.1:8b if missing
  8. Prints a ready summary + how to start the server
"""
from __future__ import annotations
import subprocess, sys, os, pathlib, json, textwrap, shutil, platform, urllib.request

ROOT   = pathlib.Path(__file__).parent.resolve()
VENV   = ROOT / ".venv"
DB     = ROOT / "neurocheck.db"
MODEL  = ROOT / "en_model"
ENV    = ROOT / ".env"

# ── colour helpers ────────────────────────────────────────────────────────────
def green(s):  return f"\033[92m{s}\033[0m"
def red(s):    return f"\033[91m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"
def cyan(s):   return f"\033[96m{s}\033[0m"

def header(msg):
    print(f"\n{bold(cyan('══'))} {bold(msg)}")

def ok(msg):    print(f"  {green('✔')}  {msg}")
def warn(msg):  print(f"  {yellow('⚠')}  {msg}")
def fail(msg):  print(f"  {red('✘')}  {msg}")
def step(msg):  print(f"  {cyan('→')}  {msg}")

def run(cmd, *, cwd=None, capture=False, check=True):
    r = subprocess.run(cmd, cwd=cwd or ROOT,
                       capture_output=capture,
                       text=True)
    if check and r.returncode != 0:
        fail(f"Command failed: {' '.join(str(c) for c in cmd)}")
        if capture:
            print(r.stderr or r.stdout)
        sys.exit(1)
    return r

# ── Resolve python inside venv ────────────────────────────────────────────────
def venv_python():
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"

def venv_pip():
    return [str(venv_python()), "-m", "pip"]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Python version check
# ─────────────────────────────────────────────────────────────────────────────
header("Step 0 · Python version check")
major, minor = sys.version_info.major, sys.version_info.minor
if major < 3 or (major == 3 and minor < 10):
    fail(f"Python 3.10+ required. You have {major}.{minor}.")
    fail("Download from https://python.org/downloads")
    sys.exit(1)
ok(f"Python {major}.{minor} ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Virtual environment
# ─────────────────────────────────────────────────────────────────────────────
header("Step 1 · Virtual environment")
if VENV.exists():
    ok(f".venv already exists at {VENV}")
else:
    step("Creating .venv …")
    run([sys.executable, "-m", "venv", str(VENV)])
    ok(".venv created")

PY = str(venv_python())
step(f"Using Python: {PY}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Install requirements
# ─────────────────────────────────────────────────────────────────────────────
header("Step 2 · Install Python dependencies")

req_file = ROOT / "requirements.txt"
if not req_file.exists():
    fail("requirements.txt not found"); sys.exit(1)

step("Upgrading pip …")
run(venv_pip() + ["install", "--quiet", "--upgrade", "pip"])
ok("pip upgraded")

step("Installing requirements (this may take 3-8 minutes first time) …")
run(venv_pip() + ["install", "--quiet", "-r", str(req_file)])
ok("All requirements installed ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build spaCy model
# ─────────────────────────────────────────────────────────────────────────────
header("Step 3 · spaCy English model")
if MODEL.exists() and (MODEL / "config.cfg").exists():
    ok(f"Model already built at {MODEL}")
else:
    step("Building blank spaCy English model …")
    script = textwrap.dedent(f"""
        import spacy
        from spacy.lang.en import English
        from pathlib import Path
        nlp = English()
        nlp.add_pipe('sentencizer')
        out = Path(r'{MODEL}')
        out.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(str(out))
        print('Model saved to', out)
    """)
    run([PY, "-c", script])
    ok("spaCy model built ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Verify NegEx
# ─────────────────────────────────────────────────────────────────────────────
header("Step 4 · Verify NegEx pipeline")
script = textwrap.dedent(f"""
    import spacy
    from negspacy.termsets import termset
    from negspacy.negation import Negex
    nlp = spacy.load(r'{MODEL}')
    ts = termset('en_clinical')
    nlp.add_pipe('negex', config={{'ent_types': ['SYMPTOM'], 'neg_termset': ts.get_patterns()}})
    print('NegEx OK')
""")
r = run([PY, "-c", script], capture=True, check=False)
if "NegEx OK" in (r.stdout or ""):
    ok("NegEx pipeline verified ✓")
else:
    warn(f"NegEx warning: {r.stderr.strip()[:200]}")
    warn("App will still run — NegEx may degrade negation detection")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — .env file
# ─────────────────────────────────────────────────────────────────────────────
header("Step 5 · Environment file (.env)")
if not ENV.exists():
    warn(".env not found — creating from defaults")
    import shutil as _sh
    example = ROOT / ".env.example"
    if example.exists():
        _sh.copy(example, ENV)
    else:
        ENV.write_text(
            "SPACY_MODEL=en_model\nNEUROCHECK_DB=neurocheck.db\n"
            "JWT_SECRET=CHANGE_THIS\nJWT_EXPIRE_MINUTES=10080\n"
            "HOST=0.0.0.0\nPORT=8000\nWORKERS=1\n",
            encoding="utf-8",
        )
    warn(f"Edit {ENV} and set a real JWT_SECRET!")
else:
    env_text = ENV.read_text(encoding="utf-8")
    if "CHANGE_THIS" in env_text:
        warn(f".env exists but JWT_SECRET is still the placeholder!")
        warn("  Open .env and set: JWT_SECRET=<your random string>")
        warn("  Generate one: python3 -c \"import secrets; print(secrets.token_hex(32))\"")
    else:
        ok(".env configured ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Initialise SQLite database
# ─────────────────────────────────────────────────────────────────────────────
header("Step 6 · SQLite database")

# Load .env vars so database.py sees NEUROCHECK_DB
for line in ENV.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

db_path = ROOT / os.environ.get("NEUROCHECK_DB", "neurocheck.db")
step(f"Database path: {db_path}")

script = textwrap.dedent(f"""
    import sys, os
    sys.path.insert(0, r'{ROOT}')
    os.environ['NEUROCHECK_DB'] = r'{db_path}'
    from api.database import init_db
    init_db()
    print('DB_OK')
""")
r = run([PY, "-c", script], capture=True, check=False)
if "DB_OK" in (r.stdout or ""):
    ok(f"Database initialised: {db_path} ✓")
    # Show table info
    import sqlite3
    if db_path.exists():
        with sqlite3.connect(str(db_path)) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            for (t,) in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                ok(f"  Table '{t}': {count} rows")
else:
    fail(f"Database init failed:\n{r.stderr.strip()[:500]}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Ollama check
# ─────────────────────────────────────────────────────────────────────────────
header("Step 7 · Ollama & LLM model")

ollama_bin = shutil.which("ollama")
if not ollama_bin:
    warn("Ollama not found in PATH.")
    warn("Download + install from: https://ollama.com/download")
    warn("Then run:  ollama pull llama3.1:8b")
    warn("The app will start but /chat will fail until Ollama is running.")
else:
    ok(f"Ollama found: {ollama_bin}")
    # Check if server running
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            tags = json.loads(resp.read())
            models = [m["name"] for m in tags.get("models", [])]
            if any("llama3.1" in m for m in models):
                ok(f"llama3.1 model present ✓")
            else:
                warn("llama3.1:8b not pulled yet.")
                step("Pulling llama3.1:8b (≈4.7 GB) — this runs in background …")
                subprocess.Popen(["ollama", "pull", "llama3.1:8b"])
                warn("Pull started in background. Wait for it before using /chat.")
    except Exception:
        warn("Ollama is installed but not running.")
        warn("Start it with:  ollama serve")
        warn("Then pull model: ollama pull llama3.1:8b")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
{bold(green('══════════════════════════════════════════════════'))}
{bold(green('  NeuroCheck setup complete!'))}
{bold(green('══════════════════════════════════════════════════'))}

{bold('To start the server:')}

  {cyan('# Option A — using the venv directly (recommended)')}
  {yellow('.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload')}

  {cyan('# Option B — using the setup script')}
  {yellow('.venv/bin/python scripts/setup_and_run.py')}

{bold('Once running, open:')}
  🌐  http://localhost:8000          →  MS Symptom Classifier
  💬  http://localhost:8000/chat-ui  →  AI Chat (login required)
  📖  http://localhost:8000/docs     →  API docs (Swagger UI)

{bold('Database:')}
  📁  {db_path}
  Tables: users · chat_sessions · messages

{bold('Before first user login:')}
  Open .env and change JWT_SECRET to a real secret.
  Generate one:
    {yellow("python3 -c \"import secrets; print(secrets.token_hex(32))\"")}
""")
