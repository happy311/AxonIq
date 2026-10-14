#!/usr/bin/env python3
"""One-off: encrypt rows written BEFORE DATA_ENCRYPTION_KEY was enabled
(messages.content, chat_sessions.title, mri_results.findings_json). Idempotent; run with the app stopped
(or at a quiet time) and AFTER taking a backup:

    docker compose exec api python scripts/encrypt_existing.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.core import crypto
from api.database import get_conn, init_all

if not crypto.enabled():
    sys.exit("DATA_ENCRYPTION_KEY is not set — nothing to do.")
init_all()
n = 0
with get_conn() as c:
    for table, col, key in (("messages", "content", "id"), ("chat_sessions", "title", "session_uuid"),
                            ("mri_results", "findings_json", "id")):
        try:
            rows = c.execute(f"SELECT {key} AS k, {col} AS v FROM {table}").fetchall()
        except Exception:
            continue
        for r in rows:
            if r["v"] is not None and not crypto.is_encrypted(r["v"]):
                c.execute(f"UPDATE {table} SET {col}=? WHERE {key}=?", (crypto.encrypt_text(r["v"]), r["k"]))
                n += 1
print(f"encrypted {n} value(s)")
