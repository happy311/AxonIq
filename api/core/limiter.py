"""
AxonIQ — Shared Rate Limiter

Single Limiter instance shared across the entire app.
main.py attaches this to app.state.limiter so slowapi can find it.
Route files import from here — never create a local Limiter().

Why this matters:
  slowapi's @limiter.limit() decorator resolves rate-limit state via
  request.app.state.limiter at runtime. If a route creates its OWN
  Limiter() instance (different object), slowapi cannot find matching
  state → raises 422 Unprocessable Entity before the request body is
  even parsed.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
