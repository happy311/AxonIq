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

# KEY = request.client.host. Behind a reverse proxy this is the REAL client IP only because
# uvicorn runs with `--proxy-headers --forwarded-allow-ips=<trusted proxy IP/CIDR>`, which
# rewrites request.client from X-Forwarded-For *only* for connections from those proxies.
# We never parse X-Forwarded-For ourselves — a spoofed header from an untrusted peer is ignored,
# so an attacker cannot dodge the limiter (or frame another IP) by sending their own header.
limiter = Limiter(key_func=get_remote_address)
