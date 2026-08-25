"""
AxonIQ — Persistent Database Layer
SQLite-backed. On HuggingFace Spaces, DB lives at /data/neurocheck.db
(persistent volume). Falls back gracefully to /tmp if /data is unavailable.

Tables: users, chat_sessions, messages, password_reset_otps, user_logs
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from api.core.config import DB_PATH, BACKUP_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    """Make sure the DB parent directory exists. Handles /data not mounted."""
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # /data not mounted — fall back to /tmp silently
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
    _backup_db()


def _backup_db() -> None:
    import shutil, glob
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        if not DB_PATH.exists():
            return
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d")
        dest = BACKUP_DIR / f"neurocheck_{ts}.db"
        if not dest.exists():
            shutil.copy2(str(DB_PATH), str(dest))
        for old in sorted(glob.glob(str(BACKUP_DIR / "neurocheck_*.db")))[:-7]:
            Path(old).unlink(missing_ok=True)
    except Exception:
        pass


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

def create_user(username: str, email: str, password_hash: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?,?,?,?)",
            (username.strip(), email.strip().lower(), password_hash, _now()),
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
            "UPDATE users SET is_admin=? WHERE username=? COLLATE NOCASE",
            (1 if is_admin else 0, username),
        )
        return cur.rowcount > 0


def update_user_password(user_id: int, new_hash: str) -> None:
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
        return [dict(r) for r in rows]


def update_session_title(session_uuid: str, title: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE chat_sessions SET title=?, updated_at=? WHERE session_uuid=?",
            (title[:80], _now(), session_uuid),
        )


def touch_session(session_uuid: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE chat_sessions SET updated_at=? WHERE session_uuid=?",
            (_now(), session_uuid),
        )


def delete_db_session(session_uuid: str) -> None:
    """Soft-delete — never removes data."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE chat_sessions SET deleted_at=? WHERE session_uuid=?",
            (_now(), session_uuid),
        )


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
            (session_uuid, role, content, now),
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
        return [dict(r) for r in rows]


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


def create_mri_job(session_uuid: str) -> None:
    """Create/reset the job row for a session — call right before launching
    the background task. Always starts in 'processing' status."""
    now = _now()
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO mri_jobs (session_uuid, status, error, created_at, updated_at) "
            "VALUES (?, 'processing', NULL, ?, ?)",
            (session_uuid, now, now),
        )


def set_mri_job_status(session_uuid: str, status: str, error: Optional[str] = None) -> None:
    """Update job status. status is one of: processing | done | error."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE mri_jobs SET status=?, error=?, updated_at=? WHERE session_uuid=?",
            (status, error, _now(), session_uuid),
        )


def get_mri_job(session_uuid: str) -> Optional[Dict[str, Any]]:
    """Return {status, error, created_at, updated_at} for a session's MRI job,
    or None if no job has ever been started for this session."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status, error, created_at, updated_at FROM mri_jobs WHERE session_uuid=?",
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
        "title":            row["title"],
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
            SELECT u.id, u.username, u.email, u.created_at, u.is_admin,
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
            msgs = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_uuid=? ORDER BY id ASC",
                (s["session_uuid"],),
            ).fetchall()
            s["messages"] = [dict(m) for m in msgs]
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

def save_otp(user_id: int, email: str, otp_hash: str, expires_at: str) -> None:
    now = _now()
    with get_conn() as conn:
        conn.execute("UPDATE password_reset_otps SET used=1 WHERE user_id=?", (user_id,))
        conn.execute(
            "INSERT INTO password_reset_otps "
            "(user_id, email, otp_hash, expires_at, used, created_at) VALUES (?,?,?,?,0,?)",
            (user_id, email, otp_hash, expires_at, now),
        )


def get_valid_otp(email: str) -> Optional[dict]:
    now = _now()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM password_reset_otps "
            "WHERE email=? AND used=0 AND expires_at>? ORDER BY id DESC LIMIT 1",
            (email.lower(), now),
        ).fetchone()
        return dict(row) if row else None


def mark_otp_used(otp_id: int) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE password_reset_otps SET used=1 WHERE id=?", (otp_id,))
