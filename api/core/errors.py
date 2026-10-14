"""AxonIQ — never leak internals: generic client message + request-id, detail server-side only."""
from fastapi import HTTPException
from loguru import logger

from api.core.middleware import request_id_var

GENERIC_500 = "Something went wrong on our side. Please try again."


def current_request_id() -> str:
    return request_id_var.get()


def server_error(context: str, exc: BaseException | None = None, status_code: int = 500,
                 message: str = GENERIC_500) -> HTTPException:
    """Log full detail server-side; return an HTTPException that exposes only a request-id."""
    rid = current_request_id()
    if exc is not None:
        logger.opt(exception=exc).error("[{}] {}", context, type(exc).__name__)
    else:
        logger.error("[{}] failed", context)
    return HTTPException(status_code=status_code, detail=f"{message} (ref: {rid})")


def generic_job_error() -> str:
    """Safe text stored in mri_jobs.error / shown to the user."""
    return f"Analysis failed. Please try again or contact support (ref: {current_request_id()})."
