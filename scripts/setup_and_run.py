#!/usr/bin/env python3
"""
NeuroCheck — One-command setup and launch script.
Run:  python3 scripts/setup_and_run.py
"""
import subprocess, sys, os, pathlib

ROOT = pathlib.Path(__file__).parent.parent
MODEL_PATH = ROOT / "en_model"

def run(cmd, **kw):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        print(f"  ERROR (exit {r.returncode})")
        sys.exit(r.returncode)
    return r

print("\n── NeuroCheck v9.0 Setup ─────────────────────────────")

# 1. Build spaCy blank English model if not present
if not MODEL_PATH.exists():
    print("\n[1/3] Building spaCy English model...")
    script = f"""
import spacy
from spacy.lang.en import English
nlp = English()
nlp.add_pipe('sentencizer')
nlp.to_disk('{MODEL_PATH}')
print('  Model saved to {MODEL_PATH}')
"""
    run([sys.executable, "-c", script])
else:
    print(f"\n[1/3] spaCy model already exists at {MODEL_PATH} ✓")

# 2. Verify negspacy
print("\n[2/3] Verifying NegEx...")
run([sys.executable, "-c", """
import spacy
from negspacy.termsets import termset
from negspacy.negation import Negex
from pathlib import Path
root = Path.cwd()
model = root / 'en_model'
nlp = spacy.load(str(model) if model.exists() else 'en_core_web_sm')
ts = termset('en_clinical')
nlp.add_pipe('negex', config={'ent_types': ['SYMPTOM'], 'neg_termset': ts.get_patterns()})
print('  NegEx pipeline ready ✓')
"""], cwd=str(ROOT))

# 3. Launch FastAPI
print("\n[3/3] Starting FastAPI server on http://0.0.0.0:8000 ...")
print("  → Open http://localhost:8000 in your browser")
print("  → API docs: http://localhost:8000/docs")
print("  → Press Ctrl+C to stop\n")

os.environ["SPACY_MODEL"] = str(MODEL_PATH)
run([
    sys.executable, "-m", "uvicorn", "api.main:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload",
], cwd=str(ROOT))
