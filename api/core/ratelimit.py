"""
AxonIQ — keyed sliding-window rate limiter (per username / email / user-id).

slowapi (api/core/limiter.py) handles per-IP limits; this complements it for
limits keyed on something other than the IP. State is in-memory, which is
correct for this app's single-worker constraint; lockouts that must survive a
restart live in the database (see api/database.py: login_attempts).
"""
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException

_lock = threading.Lock()
_hits: dict = defaultdict(deque)
_MAX_KEYS = 50_000


def hit(key: str, limit: int, window_s: int) -> bool:
    """Record an attempt. Returns False (and does not record) if over the limit."""
    now = time.monotonic()
    with _lock:
        if len(_hits) > _MAX_KEYS:
            for k in [k for k, d in _hits.items() if not d or now - d[-1] > 3600]:
                _hits.pop(k, None)
        dq = _hits[key]
        while dq and now - dq[0] > window_s:
            dq.popleft()
        if len(dq) >= limit:
            return False
        dq.append(now)
        return True


def enforce(key: str, limit: int, window_s: int, detail: str = "Too many requests — please try again later.") -> None:
    if not hit(key, limit, window_s):
        raise HTTPException(status_code=429, detail=detail, headers={"Retry-After": str(window_s)})


def reset(key: str | None = None) -> None:
    with _lock:
        if key is None:
            _hits.clear()
        else:
            _hits.pop(key, None)
