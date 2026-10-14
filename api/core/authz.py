"""AxonIQ — authorization helpers."""
import re

from fastapi import HTTPException

_SID_RE = re.compile(r"^[A-Za-z0-9\-]{8,64}$")


def require_session_owner(session_id: str, user: dict) -> None:
    """404 if the session is missing / deleted / malformed; 403 if it belongs to someone else.
    Use on every route that takes a session_id."""
    from api.database import get_session_owner
    if not session_id or not _SID_RE.match(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    owner = get_session_owner(session_id)          # None when missing or soft-deleted
    if owner is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if owner != user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
