"""
AxonIQ — pure-ASGI middleware: request-id + security headers on EVERY response
(API, static files, error pages). Pure ASGI (not BaseHTTPMiddleware) so streaming
uploads and BackgroundTasks are untouched.
"""
import re
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_RID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")

CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self'",                                   # no inline scripts, no eval
    "style-src 'self' https://fonts.googleapis.com",       # Google Fonts CSS only
    "style-src-attr 'unsafe-inline'",                      # style="" attributes only (see README tradeoff)
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

SECURITY_HEADERS = {
    b"content-security-policy": CSP.encode(),
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
    b"referrer-policy": b"no-referrer",
    b"strict-transport-security": b"max-age=63072000; includeSubDomains",
    b"permissions-policy": b"camera=(), microphone=(), geolocation=(), payment=(), usb=(), interest-cohort=()",
    b"cross-origin-opener-policy": b"same-origin",
}
_STATIC_PREFIXES = ("/js/", "/css/", "/static/")


class RequestContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _RID_RE.match(incoming) else uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        scope.setdefault("state", {})["request_id"] = rid
        path = scope.get("path", "")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", [])
                           if k.lower() not in SECURITY_HEADERS and k.lower() != b"x-request-id"]
                headers += list(SECURITY_HEADERS.items())
                headers.append((b"x-request-id", rid.encode()))
                if not path.startswith(_STATIC_PREFIXES) and not any(k.lower() == b"cache-control" for k, _ in headers):
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            request_id_var.reset(token)


class _BodyTooLarge(Exception):
    pass


class MaxBodySizeMiddleware:
    """Reject oversized request bodies BEFORE they are buffered/spooled.
    Checks Content-Length up front and also counts streamed bytes (chunked uploads).
    limits: {path_prefix: max_bytes}; `default` applies elsewhere."""

    def __init__(self, app, default: int = 256 * 1024, limits: dict | None = None):
        self.app, self.default, self.limits = app, default, limits or {}

    def _limit_for(self, path: str) -> int:
        for prefix, lim in self.limits.items():
            if path.startswith(prefix):
                return lim
        return self.default

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            return await self.app(scope, receive, send)
        limit = self._limit_for(scope.get("path", ""))
        headers = dict(scope.get("headers") or [])
        cl = headers.get(b"content-length")
        started = False

        async def reject():
            body = b'{"detail":"Request body too large"}'
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"connection", b"close")]})
            await send({"type": "http.response.body", "body": body})

        if cl and cl.isdigit() and int(cl) > limit:
            return await reject()

        seen = 0

        async def counting_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await reject()
