"""
AxonIQ — Persistent Database Layer
SQLite-backed. On HuggingFace Spaces, DB lives at /data/neurocheck.db
(persistent volume). Falls back gracefully to /tmp if /data is unavailable.

Tables: users, chat_sessions, messages, password_reset_otps, user_logs
"""
from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from loguru import logger

from api.core.config import DB_PATH, BACKUP_DIR, IS_DEV
from api.core.crypto import encrypt_text, decrypt_text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    """Make sure the DB parent directory exists. Handles /data not mounted."""
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except PermissionError:
        if not IS_DEV:
            # Never silently put patient data on an ephemeral/world-readable path.
            raise RuntimeError(f"Cannot create database directory {DB_PATH.parent} — is the data volume mounted and writable?")
        # dev only: fall back to /tmp
        from api.core import config as _cfg
        import os
        fallback = Path("/tmp/neurocheck.db")
        os.environ["NEUROCHECK_DB"] = str(fallback)
        _cfg.DB_PATH = fallback  # type: ignore[attr-defined]
        fallback.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_conn():
    """Thread-safe SQLite connection with WAL mode."""
    _ensure_dirs()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    _ensure_dirs()
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
            email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    NOT NULL,
            is_admin      INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS chat_sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_uuid  TEXT    NOT NULL UNIQUE,
            user_id       INTEGER NOT NULL REFERENCES users(id),
            title         TEXT    NOT NULL DEFAULT 'New Chat',
            created_at    TEXT    NOT NULL,
            updated_at    TEXT    NOT NULL,
            deleted_at    TEXT    DEFAULT NULL,
            ms_tier       TEXT    DEFAULT NULL,
            ms_tier_label TEXT    DEFAULT NULL,
            ms_symptoms   TEXT    DEFAULT NULL,
            concluded_at  TEXT    DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_uuid TEXT    NOT NULL REFERENCES chat_sessions(session_uuid),
            role         TEXT    NOT NULL CHECK(role IN ('human','assistant','system')),
            content      TEXT    NOT NULL,
            created_at   TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_user    ON chat_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_uuid);

        CREATE TABLE IF NOT EXISTS password_reset_otps (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            email      TEXT    NOT NULL,
            otp_hash   TEXT    NOT NULL,
            expires_at TEXT    NOT NULL,
            used       INTEGER NOT NULL DEFAULT 0,
            created_at TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER REFERENCES users(id),
            username     TEXT,
            action       TEXT    NOT NULL,
            detail       TEXT    DEFAULT NULL,
            ip_address   TEXT    DEFAULT NULL,
            session_uuid TEXT    DEFAULT NULL,
            created_at   TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_logs_user ON user_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_logs_time ON user_logs(created_at);
        """)
    migrate_security_v1()
    migrate_roles_v1()
    try:
        os.chmod(DB_PATH, 0o600)
    except OSError:
        pass
    _backup_db()


def _backup_db() -> None:
    """Consistent online backup (sqlite backup API) → BACKUP_DIR (separate from the
    data volume), mode 0600, Fernet-encrypted when DATA_ENCRYPTION_KEY is set.
    Keeps the 7 most recent. Failures are logged, never swallowed silently."""
    import glob, tempfile
    from api.core.crypto import encrypt_bytes, enabled
    try:
        if not DB_PATH.exists():
            return
        BACKUP_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d")
        dest = BACKUP_DIR / (f"neurocheck_{ts}.db.enc" if enabled() else f"neurocheck_{ts}.db")
        if dest.exists():
            return
        fd, tmp = tempfile.mkstemp(dir=str(BACKUP_DIR), suffix=".tmp")
        os.close(fd)
        try:
            src = sqlite3.connect(str(DB_PATH))
            dst = sqlite3.connect(tmp)
            with dst:
                src.backup(dst)
            src.close(); dst.close()
            data = Path(tmp).read_bytes()
            dest.write_bytes(encrypt_bytes(data))
            os.chmod(dest, 0o600)
        finally:
            Path(tmp).unlink(missing_ok=True)
        for old in sorted(glob.glob(str(BACKUP_DIR / "neurocheck_*")))[:-7]:
            Path(old).unlink(missing_ok=True)
    except Exception as e:
        logger.error("[Backup] failed: {}", type(e).__name__)


# ── Safe migrations (idempotent) ──────────────────────────────────────────────

def _safe_alter(sql: str) -> None:
    try:
        with get_conn() as conn:
            conn.execute(sql)
    except Exception:
        pass  # column already exists


def migrate_add_admin_column() -> None:
    for sql in [
        "ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE chat_sessions ADD COLUMN ms_tier TEXT DEFAULT NULL",
        "ALTER TABLE chat_sessions ADD COLUMN ms_tier_label TEXT DEFAULT NULL",
        "ALTER TABLE chat_sessions ADD COLUMN ms_symptoms TEXT DEFAULT NULL",
        "ALTER TABLE chat_sessions ADD COLUMN concluded_at TEXT DEFAULT NULL",
    ]:
        _safe_alter(sql)


def migrate_add_logs_table() -> None:
    try:
        with get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id),
                    username TEXT, action TEXT NOT NULL,
                    detail TEXT, ip_address TEXT, session_uuid TEXT,
                    created_at TEXT NOT NULL
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_user ON user_logs(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_time ON user_logs(created_at)")
    except Exception:
        pass


def migrate_add_otp_table() -> None:
    try:
        with get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS password_reset_otps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    email TEXT NOT NULL, otp_hash TEXT NOT NULL,
                    expires_at TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )""")
    except Exception:
        pass


# ── User CRUD ─────────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password_hash: str, consent_version: Optional[str] = None,
                role: str = "patient", doctor_license: Optional[str] = None,
                doctor_institution: Optional[str] = None) -> int:
    """role='patient' (default) or 'doctor'. Doctors start with doctor_status='pending' and cannot
    sign in until an administrator approves them. The 'admin' role is never created here."""
    if role not in ("patient", "doctor"):
        raise ValueError("invalid role")
    now = _now()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, created_at, consent_at, consent_version, "
            "role, doctor_status, doctor_license, doctor_institution) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (username.strip(), email.strip().lower(), password_hash, now,
             now if consent_version else None, consent_version,
             role, "pending" if role == "doctor" else None, doctor_license, doctor_institution),
        )
        return cur.lastrowid


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email=? COLLATE NOCASE", (email.strip().lower(),)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None


def set_admin(username: str, is_admin: bool = True) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE users SET is_admin=?, role=CASE WHEN ?=1 THEN 'admin' "
            "WHEN role='admin' THEN 'patient' ELSE role END WHERE username=? COLLATE NOCASE",
            (1 if is_admin else 0, 1 if is_admin else 0, username),
        )
        return cur.rowcount > 0


def update_user_password(user_id: int, new_hash: str) -> int:
    """Set a new password; bumps token_version (invalidating every issued token),
    stamps password_changed_at and clears must_change_password. Returns new version."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash=?, token_version=token_version+1, "
            "password_changed_at=?, must_change_password=0 WHERE id=?",
            (new_hash, _now(), user_id),
        )
        row = conn.execute("SELECT token_version FROM users WHERE id=?", (user_id,)).fetchone()
        return int(row["token_version"]) if row else 0


def admin_force_password_reset(user_id: int, unusable_hash: str) -> None:
    """Invalidate the current password + all sessions; user must use a reset link."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash=?, token_version=token_version+1, "
            "password_changed_at=?, must_change_password=1 WHERE id=?",
            (unusable_hash, _now(), user_id),
        )


def set_password_hash_only(user_id: int, new_hash: str) -> None:
    """Transparent hash upgrade at login (does NOT bump token_version)."""
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?", (new_hash, user_id))


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_db_session(session_uuid: str, user_id: int, title: str = "New Chat") -> None:
    now = _now()
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO chat_sessions "
            "(session_uuid, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
            (session_uuid, user_id, title, now, now),
        )


def get_user_sessions(user_id: int) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_sessions WHERE user_id=? AND deleted_at IS NULL "
            "ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        out = [dict(r) for r in rows]
        for r in out:
            r["title"] = decrypt_text(r["title"])
        return out


def update_session_title(session_uuid: str, title: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE chat_sessions SET title=?, updated_at=? WHERE session_uuid=?",
            (encrypt_text(title[:80]), _now(), session_uuid),
        )


def touch_session(session_uuid: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE chat_sessions SET updated_at=? WHERE session_uuid=?",
            (_now(), session_uuid),
        )


def delete_db_session(session_uuid: str) -> list:
    """HARD delete a session and everything hanging off it (messages, MRI results,
    job rows, queued upload rows). Returns queued file paths the caller must unlink."""
    with get_conn() as conn:
        return _hard_delete_sessions(conn, [session_uuid])


def _hard_delete_sessions(conn, uuids: list) -> list:
    files: list = []
    for u in uuids:
        for r in conn.execute("SELECT file_path FROM nifti_queue WHERE session_uuid=?", (u,)).fetchall() \
                if _table_exists(conn, "nifti_queue") else []:
            try:
                parsed = json.loads(r["file_path"])
                files += [v for v in (parsed.get("flair"), parsed.get("t1")) if v] if isinstance(parsed, dict) else [r["file_path"]]
            except Exception:
                files.append(r["file_path"])
        if _table_exists(conn, "nifti_queue"):
            conn.execute("DELETE FROM nifti_queue WHERE session_uuid=?", (u,))
        if _table_exists(conn, "mri_scans"):
            files += [r["file_path"] for r in conn.execute(
                "SELECT file_path FROM mri_scans WHERE session_uuid=?", (u,)).fetchall()]
            conn.execute("DELETE FROM mri_scans WHERE session_uuid=?", (u,))
        if _table_exists(conn, "mri_results"):
            conn.execute("DELETE FROM mri_results WHERE session_uuid=?", (u,))
        if _table_exists(conn, "mri_jobs"):
            conn.execute("DELETE FROM mri_jobs WHERE session_uuid=?", (u,))
        conn.execute("DELETE FROM user_logs WHERE session_uuid=?", (u,))
        conn.execute("DELETE FROM messages WHERE session_uuid=?", (u,))
        conn.execute("DELETE FROM chat_sessions WHERE session_uuid=?", (u,))
    return files


def _table_exists(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def delete_user_everything(user_id: int) -> list:
    """Right-to-erasure: hard-delete the user and ALL their data. Returns files to unlink."""
    with get_conn() as conn:
        uuids = [r["session_uuid"] for r in conn.execute(
            "SELECT session_uuid FROM chat_sessions WHERE user_id=?", (user_id,)).fetchall()]
        files = _hard_delete_sessions(conn, uuids)
        if _table_exists(conn, "mri_scans"):          # any scan not tied to a listed session
            files += [r["file_path"] for r in conn.execute(
                "SELECT file_path FROM mri_scans WHERE user_id=?", (user_id,)).fetchall()]
            conn.execute("DELETE FROM mri_scans WHERE user_id=?", (user_id,))
        if _table_exists(conn, "assignments"):
            conn.execute("DELETE FROM assignments WHERE patient_id=? OR doctor_id=?", (user_id, user_id))
        for tbl in ("mri_results", "usage_counters", "password_reset_otps", "user_logs"):
            if _table_exists(conn, tbl):
                conn.execute(f"DELETE FROM {tbl} WHERE user_id=?", (user_id,))
        if _table_exists(conn, "mri_jobs"):
            conn.execute("DELETE FROM mri_jobs WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        return files


def purge_old_data(retention_days: int, soft_deleted_days: int = 0) -> tuple:
    """Retention policy. Returns (sessions_purged, files_to_unlink).
    - sessions untouched for > retention_days (0 = disabled)
    - legacy soft-deleted sessions older than soft_deleted_days (always purged)"""
    from datetime import timedelta
    files: list = []
    n = 0
    with get_conn() as conn:
        uuids: list = []
        if retention_days > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
            uuids += [r["session_uuid"] for r in conn.execute(
                "SELECT session_uuid FROM chat_sessions WHERE updated_at < ?", (cutoff,)).fetchall()]
        cutoff2 = (datetime.now(timezone.utc) - timedelta(days=soft_deleted_days)).isoformat()
        uuids += [r["session_uuid"] for r in conn.execute(
            "SELECT session_uuid FROM chat_sessions WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff2,)).fetchall()]
        uuids = list(dict.fromkeys(uuids))
        if uuids:
            files = _hard_delete_sessions(conn, uuids)
            n = len(uuids)
    return n, files


def export_user_data(user_id: int) -> dict:
    """GDPR-style export of everything we hold about the user (no secrets)."""
    with get_conn() as conn:
        u = conn.execute("SELECT id, username, email, created_at FROM users WHERE id=?", (user_id,)).fetchone()
        if not u:
            return {}
        sessions = []
        for sr in conn.execute("SELECT * FROM chat_sessions WHERE user_id=? ORDER BY created_at", (user_id,)).fetchall():
            sd = dict(sr)
            sd["title"] = decrypt_text(sd["title"])
            sd["messages"] = [
                {"role": m["role"], "content": decrypt_text(m["content"]), "created_at": m["created_at"]}
                for m in conn.execute("SELECT role, content, created_at FROM messages WHERE session_uuid=? ORDER BY id",
                                      (sd["session_uuid"],)).fetchall()]
            sessions.append(sd)
        mri = []
        if _table_exists(conn, "mri_results"):
            for r in conn.execute("SELECT * FROM mri_results WHERE user_id=? ORDER BY id", (user_id,)).fetchall():
                d = dict(r)
                d["findings_json"] = json.loads(decrypt_text(d["findings_json"]) or "{}")
                mri.append(d)
        logs = [dict(r) for r in conn.execute(
            "SELECT action, detail, ip_address, session_uuid, created_at FROM user_logs WHERE user_id=? ORDER BY id",
            (user_id,)).fetchall()]
    return {"exported_at": _now(), "user": dict(u), "sessions": sessions, "mri_results": mri, "activity_log": logs}


def get_session_owner(session_uuid: str) -> Optional[int]:
    """
    Return the owner user_id for a session, or None if not found / soft-deleted.
    Filtering deleted_at IS NULL ensures deleted sessions cannot be reanimated
    by sending a message with an old session_uuid.  [v11.3 fix]
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT user_id FROM chat_sessions WHERE session_uuid=? AND deleted_at IS NULL",
            (session_uuid,),
        ).fetchone()
        return row["user_id"] if row else None


# ── Message CRUD ──────────────────────────────────────────────────────────────

def save_message(session_uuid: str, role: str, content: str) -> int:
    now = _now()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (session_uuid, role, content, created_at) VALUES (?,?,?,?)",
            (session_uuid, role, encrypt_text(content), now),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at=? WHERE session_uuid=?",
            (now, session_uuid),
        )
        return cur.lastrowid


def get_messages(session_uuid: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_uuid=? ORDER BY id ASC",
            (session_uuid,),
        ).fetchall()
        return [{**dict(r), "content": decrypt_text(r["content"])} for r in rows]


def get_message_count(session_uuid: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM messages WHERE session_uuid=?", (session_uuid,)
        ).fetchone()
        return row["n"] if row else 0


# ── Conclusion ────────────────────────────────────────────────────────────────

def save_conclusion(session_uuid: str, tier: str, tier_label: str,
                    symptoms: list) -> None:
    now = _now()
    with get_conn() as conn:
        conn.execute("""
            UPDATE chat_sessions
            SET ms_tier=?, ms_tier_label=?, ms_symptoms=?,
                concluded_at=?, updated_at=?
            WHERE session_uuid=?
        """, (tier, tier_label, json.dumps(symptoms), now, now, session_uuid))



def migrate_add_session_state() -> None:
    """Add phase/tier/features columns to chat_sessions for stateful agent."""
    with get_conn() as conn:
        for col, default in [
            ("phase",    "'gathering'"),
            ("tier",     "'LOW'"),
            ("features", "'[]'"),
        ]:
            try:
                conn.execute(f"ALTER TABLE chat_sessions ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}")
            except Exception:
                pass  # already exists


def migrate_add_clinical_state() -> None:
    """Add McDonald criteria + timeline columns (v11 upgrade)."""
    with get_conn() as conn:
        for col, default in [
            ("dis_regions",      "'[]'"),
            ("dit_episodes",     "0"),
            ("symptom_timeline", "'[]'"),
        ]:
            try:
                conn.execute(f"ALTER TABLE chat_sessions ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}")
            except Exception:
                pass  # already exists


def migrate_add_tier_log() -> None:
    """Add tier_log column for risk trajectory tracking (v11 upgrade)."""
    with get_conn() as conn:
        try:
            conn.execute("ALTER TABLE chat_sessions ADD COLUMN tier_log TEXT NOT NULL DEFAULT '[]'")
        except Exception:
            pass  # already exists


def migrate_add_nifti_queue() -> None:
    """
    Add persistent nifti_queue table so NIfTI uploads survive across
    multiple Uvicorn workers on HuggingFace Spaces (replaces in-process dict).
    """
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS nifti_queue (
            session_uuid TEXT NOT NULL PRIMARY KEY,
            file_path    TEXT NOT NULL,
            created_at   TEXT NOT NULL
        );
        """)


# ── NIfTI queue (DB-backed, works across multiple workers) ────────────────────

def store_nifti_paths(session_uuid: str, flair_path: str, t1_path: Optional[str] = None) -> None:
    """Queue the FLAIR NIfTI path (as JSON) for the next chat turn in this session.

    t1_path is accepted for backward compatibility with older callers but is no
    longer required — the MRI analysis backend is FLAIR-only now (see
    server-ensemble.ipynb). It is stored as None/absent when not provided.
    """
    import json as _json
    payload = _json.dumps({"flair": flair_path, "t1": t1_path})
    now = _now()
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO nifti_queue (session_uuid, file_path, created_at) VALUES (?, ?, ?)",
            (session_uuid, payload, now),
        )


def pop_nifti_paths(session_uuid: str) -> Optional[dict]:
    """Return and delete the queued NIfTI paths dict ({"flair": ..., "t1": None}).
    Returns None if nothing queued."""
    import json as _json
    with get_conn() as conn:
        row = conn.execute(
            "SELECT file_path FROM nifti_queue WHERE session_uuid=?",
            (session_uuid,),
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM nifti_queue WHERE session_uuid=?", (session_uuid,))
        raw = row["file_path"]
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict) and "flair" in parsed:
                return parsed
        except Exception:
            pass
        # Legacy single-file format — wrap as flair only
        return {"flair": raw, "t1": None}


# Backward-compat shims (used by any code still calling the old single-file API)
def store_nifti_path(session_uuid: str, path: str) -> None:
    store_nifti_paths(session_uuid, path, "")


def pop_nifti_path(session_uuid: str) -> Optional[str]:
    paths = pop_nifti_paths(session_uuid)
    return paths["flair"] if paths else None

def has_queued_nifti_paths(session_uuid: str) -> bool:
    """
    Non-destructive peek: True if NIfTI files are queued for this session.

    BUG FIX (Bug 4): used by chat route to decide whether to skip the
    greeting fast-path.  Unlike pop_nifti_paths() this does NOT delete the
    row, so the files remain available for the subsequent agent call.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM nifti_queue WHERE session_uuid=? LIMIT 1",
            (session_uuid,),
        ).fetchone()
    return row is not None


# ── MRI background job tracking (async upload → poll → result) ───────────────
# Added so /mri/upload can kick off the (multi-minute) MRI analysis in a
# background task and return immediately, instead of the frontend holding a
# single HTTP request open for up to 30 minutes. The frontend instead polls
# GET /mri/status/{session_id} (cheap, fast) and only calls
# GET /mri/result/{session_id} once status flips to "done".

def migrate_add_mri_jobs() -> None:
    """Add mri_jobs table for tracking background MRI analysis jobs."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS mri_jobs (
            session_uuid TEXT NOT NULL PRIMARY KEY,
            status       TEXT NOT NULL DEFAULT 'processing',
            error        TEXT DEFAULT NULL,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );
        """)


def create_mri_job(session_uuid: str, user_id: Optional[int] = None) -> None:
    """Create/reset the job row for a session (always starts 'processing')."""
    now = _now()
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO mri_jobs (session_uuid, status, error, created_at, updated_at, user_id) "
            "VALUES (?, 'processing', NULL, ?, ?, ?)",
            (session_uuid, now, now, user_id),
        )


def try_start_mri_job(session_uuid: str, user_id: int, max_queue: int) -> str:
    """Atomically enforce: 1 active job per user + bounded global queue.
    Returns 'ok' | 'user_busy' | 'queue_full'."""
    now = _now()
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM mri_jobs WHERE user_id=? AND status='processing'", (user_id,)).fetchone():
            return "user_busy"
        n = conn.execute("SELECT COUNT(*) FROM mri_jobs WHERE status='processing'").fetchone()[0]
        if n >= max_queue:
            return "queue_full"
        conn.execute(
            "INSERT OR REPLACE INTO mri_jobs (session_uuid, status, error, created_at, updated_at, user_id) "
            "VALUES (?, 'processing', NULL, ?, ?, ?)", (session_uuid, now, now, user_id))
        return "ok"


def fail_stale_mri_jobs(stale_minutes: int) -> int:
    """Any job still 'processing' after stale_minutes (or left over from a crash) → error."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE mri_jobs SET status='error', error='Analysis timed out or was interrupted.', updated_at=? "
            "WHERE status='processing' AND updated_at < ?", (_now(), cutoff))
        return cur.rowcount


def fail_all_processing_mri_jobs() -> int:
    """Startup: in-memory engine state is gone, so nothing can still be processing."""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE mri_jobs SET status='error', error='Analysis was interrupted by a server restart.', updated_at=? "
            "WHERE status='processing'", (_now(),))
        return cur.rowcount


def delete_old_mri_jobs(days: int = 7) -> int:
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with get_conn() as conn:
        return conn.execute("DELETE FROM mri_jobs WHERE status!='processing' AND updated_at < ?", (cutoff,)).rowcount


def set_mri_job_status(session_uuid: str, status: str, error: Optional[str] = None) -> None:
    """Update job status. status is one of: processing | done | error."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE mri_jobs SET status=?, error=?, updated_at=? WHERE session_uuid=?",
            (status, error, _now(), session_uuid),
        )


def migrate_add_mri_results() -> None:
    """
    Add mri_results table — one row per completed MRI analysis (NIfTI ensemble
    or pasted radiologist text), independent of chat_sessions/mri_jobs which
    only track the *current* in-progress state. This is what makes disease
    progression trackable: a patient's scans accumulate here across sessions
    and can be queried in time order (see save_mri_result / get_latest_mri_result
    / get_mri_history_for_user).
    """
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS mri_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            session_uuid      TEXT    NOT NULL REFERENCES chat_sessions(session_uuid),
            user_id           INTEGER NOT NULL REFERENCES users(id),
            source            TEXT    NOT NULL,               -- 'nifti' or 'text'
            tier              TEXT    NOT NULL,
            lesion_count      INTEGER DEFAULT NULL,
            lesion_volume_mm3 REAL    DEFAULT NULL,
            lesion_locations  TEXT    NOT NULL DEFAULT '[]',
            dis_regions       TEXT    NOT NULL DEFAULT '[]',
            dis_met           INTEGER NOT NULL DEFAULT 0,
            dit_met           INTEGER NOT NULL DEFAULT 0,
            dit_episodes      INTEGER NOT NULL DEFAULT 0,
            findings_json     TEXT    NOT NULL,
            created_at        TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_mri_results_user    ON mri_results(user_id);
        CREATE INDEX IF NOT EXISTS idx_mri_results_session ON mri_results(session_uuid);
        CREATE INDEX IF NOT EXISTS idx_mri_results_time    ON mri_results(created_at);
        """)


def save_mri_result(
    session_uuid:  str,
    user_id:       int,
    source:        str,
    tier:          str,
    findings:      dict,
    lesion_count:  Optional[int]   = None,
    lesion_volume: Optional[float] = None,
    dis_regions:   list | None     = None,
    dit_episodes:  int             = 0,
) -> int:
    """
    Persist one completed MRI analysis as a longitudinal data point.
    `lesion_count` / `lesion_volume` should be pre-computed by the caller via
    api.agent.tools.results_summary.extract_lesion_metrics(findings), so the
    same numbers shown to the patient are the ones stored here.
    Returns the new row id.
    """
    dis_regions = dis_regions or []
    now = _now()
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO mri_results
               (session_uuid, user_id, source, tier, lesion_count, lesion_volume_mm3,
                lesion_locations, dis_regions, dis_met, dit_met, dit_episodes,
                findings_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_uuid, user_id, source, tier,
                lesion_count, lesion_volume,
                json.dumps(findings.get("lesion_locations", [])),
                json.dumps(dis_regions),
                1 if findings.get("dis_met") else 0,
                1 if findings.get("dit_met") else 0,
                dit_episodes,
                encrypt_text(json.dumps(findings)),
                now,
            ),
        )
        return cur.lastrowid


def _row_to_mri_result(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["lesion_locations"] = json.loads(d.get("lesion_locations") or "[]")
    d["dis_regions"]      = json.loads(d.get("dis_regions") or "[]")
    d["dis_met"]          = bool(d.get("dis_met"))
    d["dit_met"]          = bool(d.get("dit_met"))
    return d


def get_latest_mri_result(user_id: int) -> Optional[dict]:
    """
    Most recently saved MRI result for this user, across all their sessions —
    used to build the "compared to your last scan" progression note the
    moment a NEW result comes in. Call this BEFORE save_mri_result() for the
    new scan, or it will just return the new scan itself.
    """
    with get_conn() as conn:
        row = conn.execute(
            """SELECT id, session_uuid, source, tier, lesion_count, lesion_volume_mm3,
                      lesion_locations, dis_regions, dis_met, dit_met, dit_episodes, created_at
               FROM mri_results WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
    return _row_to_mri_result(row) if row else None


def get_mri_history_for_user(user_id: int, limit: int = 100) -> list:
    """Full MRI result history for a user across all sessions, oldest → newest —
    for a progression view / trend chart."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, session_uuid, source, tier, lesion_count, lesion_volume_mm3,
                      lesion_locations, dis_regions, dis_met, dit_met, dit_episodes, created_at
               FROM mri_results WHERE user_id=? ORDER BY created_at ASC, id ASC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [_row_to_mri_result(r) for r in rows]


def get_mri_history_for_session(session_uuid: str) -> list:
    """MRI results saved within a single session (usually 0 or 1 — a patient
    typically uploads once per session — but supports rechecks)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, session_uuid, source, tier, lesion_count, lesion_volume_mm3,
                      lesion_locations, dis_regions, dis_met, dit_met, dit_episodes, created_at
               FROM mri_results WHERE session_uuid=? ORDER BY created_at ASC, id ASC""",
            (session_uuid,),
        ).fetchall()
    return [_row_to_mri_result(r) for r in rows]


def migrate_add_mri_case_id() -> None:
    """Add case_id column to mri_jobs (v17) so a later 'check the mri again' can
    re-poll the ensemble server's /result/<case_id> for THIS session's actual
    submission, instead of the LLM having no grounded way to answer and
    fabricating a result (see: hallucinated case-ID retrieval bug)."""
    with get_conn() as conn:
        try:
            conn.execute("ALTER TABLE mri_jobs ADD COLUMN case_id TEXT DEFAULT NULL")
        except Exception:
            pass  # already exists


def set_mri_job_case_id(session_uuid: str, case_id: str) -> None:
    """Record the ensemble server's case_id for this session's submission, as
    soon as POST /predict acknowledges it — independent of whether our own
    polling later times out. This is what makes a real 'recheck via case id'
    possible instead of a hallucinated one."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE mri_jobs SET case_id=?, updated_at=? WHERE session_uuid=?",
            (case_id, _now(), session_uuid),
        )


def get_mri_job(session_uuid: str) -> Optional[Dict[str, Any]]:
    """Return {status, error, case_id, created_at, updated_at} for a session's
    MRI job, or None if no job has ever been started for this session."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status, error, case_id, created_at, updated_at FROM mri_jobs WHERE session_uuid=?",
            (session_uuid,),
        ).fetchone()
        return dict(row) if row else None


def get_session_state(session_uuid: str) -> dict:
    """Get phase, tier, features + clinical state for a session."""
    import json
    with get_conn() as conn:
        row = conn.execute(
            """SELECT phase, tier, features,
                      dis_regions, dit_episodes, symptom_timeline
               FROM chat_sessions WHERE session_uuid=?""",
            (session_uuid,)
        ).fetchone()
    if not row:
        return {
            "phase": "gathering", "tier": "LOW", "features": [],
            "dis_regions": [], "dit_episodes": 0, "symptom_timeline": [],
        }

    def _parse_json(val, default):
        try:
            return json.loads(val or json.dumps(default))
        except Exception:
            return default

    return {
        "phase":            row["phase"]        or "gathering",
        "tier":             row["tier"]         or "LOW",
        "features":         _parse_json(row["features"], []),
        "dis_regions":      _parse_json(row["dis_regions"], []),
        "dit_episodes":     int(row["dit_episodes"] or 0),
        "symptom_timeline": _parse_json(row["symptom_timeline"], []),
    }


def update_session_state(
    session_uuid:    str,
    phase:           str,
    tier:            str,
    features:        list,
    dis_regions:     list | None = None,
    dit_episodes:    int         = 0,
    symptom_timeline: list | None = None,
    human_turn:      int         = 0,
) -> None:
    """
    Persist phase, tier, features, and McDonald clinical fields after each turn.
    Appends a tier_log entry for risk trajectory analytics.
    """
    import json
    now = _now()

    dis_regions      = dis_regions      or []
    symptom_timeline = symptom_timeline or []

    # Fetch current tier_log and append
    with get_conn() as conn:
        row = conn.execute(
            "SELECT tier_log FROM chat_sessions WHERE session_uuid=?",
            (session_uuid,),
        ).fetchone()
        try:
            tier_log = json.loads(row["tier_log"] or "[]") if row else []
        except Exception:
            tier_log = []

        tier_log.append({
            "turn":      human_turn,
            "tier":      tier,
            "features":  features,
            "timestamp": now,
        })

        conn.execute(
            """UPDATE chat_sessions
               SET phase=?, tier=?, features=?,
                   dis_regions=?, dit_episodes=?, symptom_timeline=?,
                   tier_log=?
               WHERE session_uuid=?""",
            (
                phase, tier, json.dumps(features),
                json.dumps(dis_regions), dit_episodes, json.dumps(symptom_timeline),
                json.dumps(tier_log),
                session_uuid,
            ),
        )


def get_tier_log(session_uuid: str) -> list:
    """Return the tier progression log for a session (for clinician view / analytics)."""
    import json
    with get_conn() as conn:
        row = conn.execute(
            "SELECT tier_log FROM chat_sessions WHERE session_uuid=?",
            (session_uuid,),
        ).fetchone()
    if not row:
        return []
    try:
        return json.loads(row["tier_log"] or "[]")
    except Exception:
        return []


def get_session_export_data(session_uuid: str) -> dict:
    """Aggregate all session data needed for the clinical summary export."""
    import json
    with get_conn() as conn:
        row = conn.execute(
            """SELECT phase, tier, features, dis_regions, dit_episodes,
                      symptom_timeline, tier_log, ms_tier, ms_symptoms,
                      created_at, updated_at, title
               FROM chat_sessions WHERE session_uuid=?""",
            (session_uuid,),
        ).fetchone()

    if not row:
        return {}

    def _safe_json(val, default):
        try:
            return json.loads(val or json.dumps(default))
        except Exception:
            return default

    return {
        "session_uuid":     session_uuid,
        "title":            decrypt_text(row["title"]),
        "phase":            row["phase"],
        "tier":             row["tier"],
        "features":         _safe_json(row["features"], []),
        "dis_regions":      _safe_json(row["dis_regions"], []),
        "dit_episodes":     int(row["dit_episodes"] or 0),
        "symptom_timeline": _safe_json(row["symptom_timeline"], []),
        "tier_log":         _safe_json(row["tier_log"], []),
        "created_at":       row["created_at"],
        "updated_at":       row["updated_at"],
    }


def get_conclusion(session_uuid: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT ms_tier, ms_tier_label, ms_symptoms, concluded_at "
            "FROM chat_sessions WHERE session_uuid=?",
            (session_uuid,),
        ).fetchone()
        if not row or not row["ms_tier"]:
            return None
        return {
            "tier":         row["ms_tier"],
            "tier_label":   row["ms_tier_label"],
            "symptoms":     json.loads(row["ms_symptoms"] or "[]"),
            "concluded_at": row["concluded_at"],
        }


# ── Admin ─────────────────────────────────────────────────────────────────────

def get_all_users_admin() -> list:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT u.id, u.username, u.email, u.created_at, u.is_admin, u.role, u.doctor_status,
                COUNT(DISTINCT cs.id)  AS session_count,
                COUNT(DISTINCT m.id)   AS message_count,
                MAX(CASE WHEN cs.ms_tier='CRITICAL_EMERGENCY' THEN 5
                         WHEN cs.ms_tier='HIGH'     THEN 4
                         WHEN cs.ms_tier='MODERATE' THEN 3
                         WHEN cs.ms_tier='WATCH'    THEN 2
                         WHEN cs.ms_tier='LOW'      THEN 1
                         ELSE 0 END) AS highest_risk_score,
                (SELECT cs2.ms_tier FROM chat_sessions cs2
                 WHERE cs2.user_id=u.id AND cs2.ms_tier IS NOT NULL
                 ORDER BY cs2.concluded_at DESC LIMIT 1) AS latest_tier
            FROM users u
            LEFT JOIN chat_sessions cs ON cs.user_id=u.id AND cs.deleted_at IS NULL
            LEFT JOIN messages m       ON m.session_uuid=cs.session_uuid
            GROUP BY u.id ORDER BY u.created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]


def get_all_sessions_admin() -> tuple:
    with get_conn() as conn:
        s = conn.execute("SELECT COUNT(*) FROM chat_sessions WHERE deleted_at IS NULL").fetchone()[0]
        m = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return s, m


def get_user_sessions_with_messages(user_id: int) -> list:
    with get_conn() as conn:
        sessions = conn.execute(
            "SELECT * FROM chat_sessions WHERE user_id=? AND deleted_at IS NULL ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        result = []
        for s in sessions:
            s = dict(s)
            s["title"] = decrypt_text(s["title"])
            msgs = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_uuid=? ORDER BY id ASC",
                (s["session_uuid"],),
            ).fetchall()
            s["messages"] = [{**dict(m), "content": decrypt_text(m["content"])} for m in msgs]
            s["conclusion"] = get_conclusion(s["session_uuid"])
            result.append(s)
        return result


# ── Logging ───────────────────────────────────────────────────────────────────

def log_action(action: str, user_id=None, username=None,
               detail=None, ip_address=None, session_uuid=None) -> None:
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO user_logs "
                "(user_id, username, action, detail, ip_address, session_uuid, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (user_id, username, action, detail, ip_address, session_uuid, _now()),
            )
    except Exception:
        pass


def get_user_logs(user_id: int, limit: int = 200) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT action, detail, ip_address, session_uuid, created_at "
            "FROM user_logs WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_logs(limit: int = 500) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT created_at, username, action, detail, ip_address, session_uuid "
            "FROM user_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── OTP ───────────────────────────────────────────────────────────────────────
# otp_hash column now holds HMAC-SHA256(JWT_SECRET, salt|email|otp) — see api/routes/password.py

def save_otp(user_id: int, email: str, otp_mac: str, expires_at: str, salt: str) -> None:
    now = _now()
    with get_conn() as conn:
        conn.execute("UPDATE password_reset_otps SET used=1 WHERE user_id=?", (user_id,))
        conn.execute(
            "INSERT INTO password_reset_otps "
            "(user_id, email, otp_hash, otp_salt, attempts, locked, expires_at, used, created_at) "
            "VALUES (?,?,?,?,0,0,?,0,?)",
            (user_id, email.lower(), otp_mac, salt, expires_at, now),
        )


def get_active_otp(email: str) -> Optional[dict]:
    """Latest unused, unlocked, unexpired OTP for the email (or None)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM password_reset_otps WHERE email=? AND used=0 AND locked=0 "
            "AND expires_at>? AND otp_salt!='' ORDER BY id DESC LIMIT 1",
            (email.lower(), _now()),
        ).fetchone()
        return dict(row) if row else None


def burn_otp_attempt(otp_id: int, max_attempts: int) -> bool:
    """Atomically spend one attempt BEFORE comparing (parallel guesses can't exceed the cap).
    Locks the OTP when the cap is reached. Returns False if no attempt was available."""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE password_reset_otps SET attempts=attempts+1, "
            "locked=CASE WHEN attempts+1>=? THEN 1 ELSE 0 END "
            "WHERE id=? AND used=0 AND locked=0 AND attempts<?",
            (max_attempts, otp_id, max_attempts),
        )
        return cur.rowcount == 1


def mark_otp_used(otp_id: int) -> bool:
    """Invalidate after success. Returns False if it was already used (race)."""
    with get_conn() as conn:
        return conn.execute("UPDATE password_reset_otps SET used=1 WHERE id=? AND used=0", (otp_id,)).rowcount == 1


# ── Single-use token ids, login lockout, quotas, audit ───────────────────────

def consume_jti(jti: str, expires_at: str) -> bool:
    """Mark a token id as consumed. False if it was already consumed (replay)."""
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO consumed_tokens (jti, expires_at) VALUES (?,?)", (jti, expires_at))
        return True
    except sqlite3.IntegrityError:
        return False


def get_lockout_until(identifier: str) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT locked_until FROM login_attempts WHERE identifier=?",
                           (identifier.lower(),)).fetchone()
    if row and row["locked_until"] and row["locked_until"] > _now():
        return row["locked_until"]
    return None


def record_login_failure(identifier: str, threshold: int, max_minutes: int) -> None:
    """Exponential backoff: after `threshold` consecutive failures lock for 1,2,4,… min (capped)."""
    from datetime import timedelta
    ident = identifier.lower()[:200]
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT failures FROM login_attempts WHERE identifier=?", (ident,)).fetchone()
        n = (row["failures"] if row else 0) + 1
        locked = None
        if n >= threshold:
            minutes = min(max_minutes, 2 ** (n - threshold))
            locked = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
        conn.execute(
            "INSERT INTO login_attempts (identifier, failures, locked_until, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(identifier) DO UPDATE SET failures=?, locked_until=?, updated_at=?",
            (ident, n, locked, _now(), n, locked, _now()),
        )


def clear_login_failures(identifier: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM login_attempts WHERE identifier=?", (identifier.lower()[:200],))


def consume_quota(user_id: int, kind: str, limit: int) -> bool:
    """Atomic daily quota (UTC day). Survives session deletion. False → quota exhausted."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT count FROM usage_counters WHERE user_id=? AND day=? AND kind=?",
                           (user_id, day, kind)).fetchone()
        used = row["count"] if row else 0
        if used >= limit:
            return False
        conn.execute(
            "INSERT INTO usage_counters (user_id, day, kind, count) VALUES (?,?,?,1) "
            "ON CONFLICT(user_id, day, kind) DO UPDATE SET count=count+1", (user_id, day, kind))
        return True


def audit(admin_id: Optional[int], admin_username: Optional[str], action: str,
          target: Optional[str] = None, ip: Optional[str] = None) -> None:
    """Append-only admin audit trail. Raises on failure — an admin action that can't be
    audited must not proceed."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO admin_audit (admin_id, admin_username, action, target, ip, created_at) VALUES (?,?,?,?,?,?)",
            (admin_id, admin_username, action, target, ip, _now()))


def get_audit_entries(limit: int = 500) -> list:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT id, admin_id, admin_username, action, target, ip, created_at FROM admin_audit "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]


def housekeeping_db() -> None:
    """Prune expired security rows."""
    from datetime import timedelta
    now = _now()
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    with get_conn() as conn:
        conn.execute("DELETE FROM consumed_tokens WHERE expires_at < ?", (now,))
        conn.execute("DELETE FROM password_reset_otps WHERE expires_at < ?", (old,))
        conn.execute("DELETE FROM login_attempts WHERE updated_at < ? AND (locked_until IS NULL OR locked_until < ?)", (old, now))
        conn.execute("DELETE FROM usage_counters WHERE day < date('now','-7 day')")


def list_queued_upload_files() -> list:
    with get_conn() as conn:
        if not _table_exists(conn, "nifti_queue"):
            return []
        out = []
        for r in conn.execute("SELECT file_path FROM nifti_queue").fetchall():
            try:
                d = json.loads(r["file_path"])
                out += [v for v in (d.get("flair"), d.get("t1")) if v]
            except Exception:
                out.append(r["file_path"])
        return out


# ── Security schema (idempotent; called from init_db) ────────────────────────

def migrate_security_v1() -> None:
    for sql in [
        "ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN password_changed_at TEXT DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE users ADD COLUMN consent_at TEXT DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN consent_version TEXT DEFAULT NULL",
        "ALTER TABLE password_reset_otps ADD COLUMN otp_salt TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE password_reset_otps ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE password_reset_otps ADD COLUMN locked INTEGER NOT NULL DEFAULT 0",
    ]:
        _safe_alter(sql)
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS consumed_tokens (
            jti TEXT PRIMARY KEY, expires_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS login_attempts (
            identifier TEXT PRIMARY KEY, failures INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS usage_counters (
            user_id INTEGER NOT NULL, day TEXT NOT NULL, kind TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (user_id, day, kind));
        CREATE TABLE IF NOT EXISTS admin_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER, admin_username TEXT,
            action TEXT NOT NULL, target TEXT, ip TEXT, created_at TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS admin_audit_no_update BEFORE UPDATE ON admin_audit
            BEGIN SELECT RAISE(ABORT, 'admin_audit is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS admin_audit_no_delete BEFORE DELETE ON admin_audit
            BEGIN SELECT RAISE(ABORT, 'admin_audit is append-only'); END;
        """)
    # mri_jobs.user_id (table is created by migrate_add_mri_jobs; make sure both orders work)
    migrate_add_mri_jobs()
    _safe_alter("ALTER TABLE mri_jobs ADD COLUMN user_id INTEGER DEFAULT NULL")
    migrate_add_otp_table()
    for sql in [
        "ALTER TABLE password_reset_otps ADD COLUMN otp_salt TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE password_reset_otps ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE password_reset_otps ADD COLUMN locked INTEGER NOT NULL DEFAULT 0",
    ]:
        _safe_alter(sql)


# ── Roles (patient / doctor / admin) ──────────────────────────────────────────

def migrate_roles_v1() -> None:
    """users.role ('patient'|'doctor'|'admin'), doctor approval state, env-managed flag,
    and the mri_scans table (encrypted stored copies a doctor may download)."""
    for sql in [
        "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'patient'",
        "ALTER TABLE users ADD COLUMN doctor_status TEXT DEFAULT NULL",      # pending|approved|removed
        "ALTER TABLE users ADD COLUMN doctor_license TEXT DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN doctor_institution TEXT DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN doctor_reviewed_at TEXT DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN doctor_reviewed_by TEXT DEFAULT NULL",
        "ALTER TABLE users ADD COLUMN env_managed INTEGER NOT NULL DEFAULT 0",
    ]:
        _safe_alter(sql)
    with get_conn() as conn:
        conn.execute("UPDATE users SET role='admin' WHERE is_admin=1 AND role='patient'")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS mri_scans (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL REFERENCES users(id),
            session_uuid TEXT    NOT NULL,
            file_path    TEXT    NOT NULL,
            size_bytes   INTEGER NOT NULL DEFAULT 0,
            suffix       TEXT    NOT NULL DEFAULT '.nii.gz',
            created_at   TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_mri_scans_user    ON mri_scans(user_id);
        CREATE INDEX IF NOT EXISTS idx_mri_scans_session ON mri_scans(session_uuid);
        CREATE TABLE IF NOT EXISTS assignments (
            patient_id  INTEGER PRIMARY KEY REFERENCES users(id),   -- one doctor per patient
            doctor_id   INTEGER NOT NULL REFERENCES users(id),
            assigned_at TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_assignments_doctor ON assignments(doctor_id);
        """)


def set_role_admin_from_env(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET is_admin=1, role='admin', env_managed=1, doctor_status=NULL WHERE id=?",
                     (user_id,))


def create_env_admin(username: str, email: str, password_hash: str) -> int:
    now = _now()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, created_at, is_admin, role, env_managed) "
            "VALUES (?,?,?,?,1,'admin',1)", (username, email, password_hash, now))
        return cur.lastrowid


def demote_env_admins_except(username: str) -> int:
    """Rotation safety: an env-managed admin whose name is no longer in ADMIN_USERNAME loses admin."""
    with get_conn() as conn:
        rows = conn.execute("SELECT id FROM users WHERE env_managed=1 AND is_admin=1 "
                            "AND username <> ? COLLATE NOCASE", (username,)).fetchall()
        for r in rows:
            conn.execute("UPDATE users SET is_admin=0, role='patient', token_version=token_version+1 WHERE id=?",
                         (r["id"],))
        return len(rows)


def list_doctors(status: Optional[str] = None) -> list:
    q = ("SELECT id, username, email, created_at, doctor_status, doctor_license, doctor_institution, "
         "doctor_reviewed_at, doctor_reviewed_by, "
         "(SELECT COUNT(*) FROM assignments a WHERE a.doctor_id=users.id) AS assigned_count "
         "FROM users WHERE role='doctor'")
    args: tuple = ()
    if status:
        q += " AND doctor_status=?"
        args = (status,)
    q += " ORDER BY CASE doctor_status WHEN 'pending' THEN 0 WHEN 'approved' THEN 1 ELSE 2 END, created_at DESC"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def set_doctor_status(user_id: int, status: str, reviewer: str) -> bool:
    """approved | removed. Always bumps token_version so removal kills live sessions at once.
    Only touches rows whose role is 'doctor'."""
    if status not in ("approved", "removed"):
        raise ValueError("invalid status")
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE users SET doctor_status=?, doctor_reviewed_at=?, doctor_reviewed_by=?, "
            "token_version=token_version+1 WHERE id=? AND role='doctor'",
            (status, _now(), reviewer, user_id))
        return cur.rowcount > 0


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_patients_by_email(query: str, limit: int = 20) -> list:
    """Patients (role='patient' only — never doctors/admins) whose e-mail contains `query`."""
    q = f"%{_like_escape(query.strip().lower())}%"
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT u.id, u.username, u.email, u.created_at,
                      (SELECT COUNT(*) FROM chat_sessions cs WHERE cs.user_id=u.id AND cs.deleted_at IS NULL) AS session_count,
                      (SELECT COUNT(*) FROM mri_scans ms WHERE ms.user_id=u.id) AS scan_count,
                      (SELECT cs2.ms_tier FROM chat_sessions cs2 WHERE cs2.user_id=u.id AND cs2.ms_tier IS NOT NULL
                       ORDER BY cs2.concluded_at DESC LIMIT 1) AS latest_tier
               FROM users u WHERE u.role='patient' AND u.email LIKE ? ESCAPE '\\'
               ORDER BY u.email LIMIT ?""", (q, limit)).fetchall()
        return [dict(r) for r in rows]


def get_patient(user_id: int) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute("SELECT id, username, email, created_at FROM users WHERE id=? AND role='patient'",
                         (user_id,)).fetchone()
        return dict(r) if r else None


def get_patient_clinical_overview(user_id: int) -> list:
    """Per-session symptom picture for a clinician: tier, detected symptoms, features,
    McDonald fields, risk trajectory, MRI summaries. Deliberately NO chat transcript."""
    def _j(v, d):
        try:
            return json.loads(v) if v else d
        except Exception:
            return d
    out = []
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT session_uuid, created_at, updated_at, tier, features, dis_regions, dit_episodes,
                      symptom_timeline, tier_log, ms_tier, ms_tier_label, ms_symptoms, concluded_at
               FROM chat_sessions WHERE user_id=? AND deleted_at IS NULL ORDER BY created_at DESC""",
            (user_id,)).fetchall()
        for r in rows:
            mri = conn.execute(
                """SELECT id, source, tier, lesion_count, lesion_volume_mm3, lesion_locations,
                          dis_met, dit_met, created_at FROM mri_results
                   WHERE session_uuid=? ORDER BY id""", (r["session_uuid"],)).fetchall()
            out.append({
                "session_uuid": r["session_uuid"], "created_at": r["created_at"], "updated_at": r["updated_at"],
                "current_tier": r["tier"], "concluded_tier": r["ms_tier"], "concluded_label": r["ms_tier_label"],
                "concluded_at": r["concluded_at"],
                "symptoms": _j(r["ms_symptoms"], []), "features": _j(r["features"], []),
                "dis_regions": _j(r["dis_regions"], []), "dit_episodes": int(r["dit_episodes"] or 0),
                "symptom_timeline": _j(r["symptom_timeline"], []), "tier_log": _j(r["tier_log"], []),
                "mri_results": [{**dict(m), "lesion_locations": _j(m["lesion_locations"], []),
                                 "dis_met": bool(m["dis_met"]), "dit_met": bool(m["dit_met"])} for m in mri],
            })
    return out


# ── Stored MRI scans ──────────────────────────────────────────────────────────

def count_scans(user_id: int) -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM mri_scans WHERE user_id=?", (user_id,)).fetchone()[0]


def add_scan(user_id: int, session_uuid: str, file_path: str, size_bytes: int, suffix: str) -> int:
    with get_conn() as conn:
        return conn.execute(
            "INSERT INTO mri_scans (user_id, session_uuid, file_path, size_bytes, suffix, created_at) "
            "VALUES (?,?,?,?,?,?)", (user_id, session_uuid, file_path, size_bytes, suffix, _now())).lastrowid


def list_scans_for_user(user_id: int) -> list:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT s.id, s.session_uuid, s.size_bytes, s.created_at,
                      (SELECT r.tier FROM mri_results r WHERE r.session_uuid=s.session_uuid ORDER BY r.id DESC LIMIT 1) AS tier,
                      (SELECT r.lesion_count FROM mri_results r WHERE r.session_uuid=s.session_uuid ORDER BY r.id DESC LIMIT 1) AS lesion_count,
                      (SELECT r.lesion_volume_mm3 FROM mri_results r WHERE r.session_uuid=s.session_uuid ORDER BY r.id DESC LIMIT 1) AS lesion_volume_mm3
               FROM mri_scans s WHERE s.user_id=? ORDER BY s.created_at DESC""", (user_id,)).fetchall()]


def get_scan(scan_id: int) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute("SELECT * FROM mri_scans WHERE id=?", (scan_id,)).fetchone()
        return dict(r) if r else None


# ── Patient → doctor assignment (least-loaded) ────────────────────────────────

def _least_loaded_doctor(conn) -> Optional[int]:
    """Approved doctor with the FEWEST currently assigned patients.
    Ties → the doctor who was assigned a patient longest ago (never-assigned first), then lowest id."""
    row = conn.execute(
        """SELECT u.id FROM users u
           LEFT JOIN assignments a ON a.doctor_id = u.id
           WHERE u.role='doctor' AND u.doctor_status='approved'
           GROUP BY u.id
           ORDER BY COUNT(a.patient_id) ASC,
                    COALESCE(MAX(a.assigned_at), '') ASC,
                    u.id ASC
           LIMIT 1""").fetchone()
    return row["id"] if row else None


def assign_patient(patient_id: int) -> Optional[int]:
    """Give the patient a doctor and return the doctor id (None if no approved doctor exists yet).
    Sticky: a patient who already has an approved doctor keeps them. Runs in one write
    transaction so two simultaneous uploads can't both see the same 'least loaded' count."""
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            """SELECT a.doctor_id FROM assignments a JOIN users d ON d.id=a.doctor_id
               WHERE a.patient_id=? AND d.role='doctor' AND d.doctor_status='approved'""",
            (patient_id,)).fetchone()
        if cur:
            return cur["doctor_id"]
        doc = _least_loaded_doctor(conn)
        if doc is None:
            return None
        conn.execute(
            "INSERT INTO assignments (patient_id, doctor_id, assigned_at) VALUES (?,?,?) "
            "ON CONFLICT(patient_id) DO UPDATE SET doctor_id=excluded.doctor_id, assigned_at=excluded.assigned_at",
            (patient_id, doc, _now()))
        return doc


def assign_unassigned() -> int:
    """Assign every patient who has an MRI scan but no (valid) doctor — oldest scan first.
    Called when a doctor is approved/removed and at startup. Returns number assigned."""
    with get_conn() as conn:
        ids = [r["user_id"] for r in conn.execute(
            """SELECT s.user_id FROM mri_scans s JOIN users p ON p.id=s.user_id AND p.role='patient'
               LEFT JOIN assignments a ON a.patient_id=s.user_id
               LEFT JOIN users d ON d.id=a.doctor_id AND d.role='doctor' AND d.doctor_status='approved'
               WHERE d.id IS NULL GROUP BY s.user_id ORDER BY MIN(s.created_at)""").fetchall()]
    return sum(1 for pid in ids if assign_patient(pid) is not None)


def release_doctor_patients(doctor_id: int) -> int:
    """Drop all of a doctor's assignments (used when the doctor is removed); caller re-runs assign_unassigned()."""
    with get_conn() as conn:
        return conn.execute("DELETE FROM assignments WHERE doctor_id=?", (doctor_id,)).rowcount


def list_assigned_patients(doctor_id: int) -> list:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT u.id, u.username, u.email, u.created_at, a.assigned_at,
                      (SELECT COUNT(*) FROM chat_sessions cs WHERE cs.user_id=u.id AND cs.deleted_at IS NULL) AS session_count,
                      (SELECT COUNT(*) FROM mri_scans ms WHERE ms.user_id=u.id) AS scan_count,
                      (SELECT cs2.ms_tier FROM chat_sessions cs2 WHERE cs2.user_id=u.id AND cs2.ms_tier IS NOT NULL
                       ORDER BY cs2.concluded_at DESC LIMIT 1) AS latest_tier
               FROM assignments a JOIN users u ON u.id=a.patient_id
               WHERE a.doctor_id=? AND u.role='patient' ORDER BY a.assigned_at DESC""", (doctor_id,)).fetchall()]


def init_all() -> None:
    """Schema + every migration, in the right order (used by startup and tests)."""
    init_db()
    migrate_add_admin_column()
    migrate_add_logs_table()
    migrate_add_otp_table()
    migrate_add_session_state()
    migrate_add_clinical_state()
    migrate_add_tier_log()
    migrate_add_nifti_queue()
    migrate_add_mri_jobs()
    migrate_add_mri_case_id()
    migrate_add_mri_results()
    migrate_security_v1()
    migrate_roles_v1()
