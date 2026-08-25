"""
NeuroCheck Agentic AI — Session Memory
File-based session store: no Redis required for local deployment.
Each session is a JSON file in /tmp/neurocheck_sessions/.
"""
from __future__ import annotations
import json, uuid, os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from api.core.config import SESSION_DIR, SESSION_MAX_HISTORY as MAX_HISTORY
SESSION_DIR.mkdir(parents=True, exist_ok=True)


def new_session() -> str:
    """Create a new session and return its ID."""
    sid = str(uuid.uuid4())
    _write(sid, {
        "session_id": sid,
        "created_at": _now(),
        "messages": [],
        "classification": None,
        "symptoms_collected": [],
    })
    return sid


def get_session(sid: str) -> Dict[str, Any] | None:
    p = _path(sid)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def append_message(sid: str, role: str, content: str) -> None:
    """Append a human/assistant message to session history."""
    s = get_session(sid)
    if s is None:
        return
    s["messages"].append({"role": role, "content": content, "ts": _now()})
    # Keep last MAX_HISTORY messages
    if len(s["messages"]) > MAX_HISTORY:
        s["messages"] = s["messages"][-MAX_HISTORY:]
    _write(sid, s)


def set_classification(sid: str, classification: Dict[str, Any]) -> None:
    s = get_session(sid)
    if s is None:
        return
    s["classification"] = classification
    _write(sid, s)


def get_history(sid: str) -> List[Dict[str, str]]:
    s = get_session(sid)
    if s is None:
        return []
    return s.get("messages", [])


def get_lc_messages(sid: str):
    """Return LangChain-compatible message list for context injection."""
    from langchain_core.messages import HumanMessage, AIMessage
    msgs = get_history(sid)
    lc = []
    for m in msgs:
        if m["role"] == "human":
            lc.append(HumanMessage(content=m["content"]))
        else:
            lc.append(AIMessage(content=m["content"]))
    return lc


def list_sessions() -> List[str]:
    return [p.stem for p in SESSION_DIR.glob("*.json")]


# ── Internal helpers ──────────────────────────────────────────────────────────
def _path(sid: str) -> Path:
    return SESSION_DIR / f"{sid}.json"


def _write(sid: str, data: Dict) -> None:
    _path(sid).write_text(json.dumps(data, indent=2))


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"
