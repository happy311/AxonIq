"""
AxonIQ — loguru configuration with PHI/secret redaction + request-id.

Redacts e-mail addresses, JWTs, Bearer tokens, and password/token/otp key=value pairs
from every log line. (Defence in depth — call sites also avoid logging such data.)
"""
import re
import sys

from loguru import logger

from api.core.middleware import request_id_var

_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), "[jwt]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"), "Bearer [redacted]"),
    (re.compile(r"(?i)(password|passwd|pwd|secret|token|otp|api[_-]?key|authorization)(['\"]?\s*[:=]\s*['\"]?)[^\s,'\"}&]+"),
     r"\1\2[redacted]"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "[api-key]"),
]


def redact(text: str) -> str:
    for rx, rep in _PATTERNS:
        text = rx.sub(rep, text)
    return text


def _filter(record) -> bool:
    record["message"] = redact(record["message"])
    record["extra"]["request_id"] = request_id_var.get()
    return True


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr, level=level, filter=_filter, backtrace=False, diagnose=False,   # diagnose=False: no variable values in tracebacks
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | rid={extra[request_id]} | {name}:{function}:{line} - {message}",
    )
