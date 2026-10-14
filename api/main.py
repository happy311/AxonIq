"""
AxonIQ — Application Factory (hardened)

Startup order:
  1. Validate security configuration (encryption key, JWT secret already checked at import)
  2. SQLite schema + migrations
  3. Crash recovery: fail orphaned MRI jobs, clear stale upload files (janitor)
  4. RAG knowledge base (ChromaDB)
  5. LLM probe, MRI engine warm-up (background thread)

There is NO startup-time admin grant. Use `python make_admin.py USERNAME` (see that file).
"""
from __future__ import annotations
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.core.config import (
    IS_DEV, ALLOWED_ORIGINS, REQUIRE_DATA_ENCRYPTION, MRI_MAX_UPLOAD_BYTES, DB_PATH,
)
from api.core.logging_setup import setup_logging
from api.core.middleware import RequestContextMiddleware, MaxBodySizeMiddleware
from api.core.errors import current_request_id, GENERIC_500

setup_logging(os.environ.get("LOG_LEVEL", "INFO").upper())


async def _janitor_loop():
    from api.mri_engine import janitor
    while True:
        await asyncio.sleep(600)
        try:
            await asyncio.to_thread(janitor.run_once)
        except Exception as e:                      # never let the janitor die
            logger.error("[janitor] run failed: {}", type(e).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Security preconditions
    from api.core import crypto
    crypto.require_key_if_configured_strict(REQUIRE_DATA_ENCRYPTION)

    # 1b. Pinned checkpoint integrity — refuse to start on any mismatch / unpinned file
    from api.mri_engine.checkpoints import verify_checkpoint_hashes
    verify_checkpoint_hashes()

    # 2. Database
    from api.database import init_all
    init_all()
    logger.info("[AxonIQ] SQLite at: {} (encryption at rest: {})", DB_PATH, crypto.enabled())

    # 2b. Administrator account from ADMIN_USERNAME / ADMIN_PASSWORD (aborts startup if unsafe)
    from api.core.bootstrap import ensure_env_admin
    ensure_env_admin()
    from api.database import assign_unassigned
    assign_unassigned()                       # patients with an MRI but no doctor yet

    # 3. Crash recovery / cleanup
    from api.mri_engine import janitor
    janitor.run_startup()

    # 4. RAG knowledge base
    try:
        from api.rag.store import build_knowledge_base
        build_knowledge_base()
    except Exception as e:
        logger.warning("[RAG] Could not build knowledge base: {}", type(e).__name__)

    # 5. Probe LLM
    try:
        from api.agent.llm import llm
        logger.info("[LLM] Active model: {}", getattr(llm, "_model", "ollama"))
    except Exception as e:
        logger.error("[LLM] Startup probe failed: {}", type(e).__name__)

    # 6. MRI engine (verifies pinned checkpoint hashes; refuses to load on mismatch)
    import threading

    def _init_mri_engine():
        try:
            from api.mri_engine import pipeline as mri_pipeline
            mri_pipeline.initialize()
        except Exception as e:
            logger.error("[MRI Engine] Startup thread crashed: {}", type(e).__name__)

    threading.Thread(target=_init_mri_engine, daemon=True, name="mri-engine-init").start()
    janitor_task = asyncio.create_task(_janitor_loop())

    yield
    janitor_task.cancel()
    logger.info("[AxonIQ] Shutdown complete.")


def create_app() -> FastAPI:
    from api.core.limiter import limiter

    app = FastAPI(
        title="AxonIQ — Agentic MS Clinical AI",
        version="11.0.0",
        lifespan=lifespan,
        # API docs only in dev — they enumerate every route/parameter.
        docs_url="/docs" if IS_DEV else None,
        redoc_url="/redoc" if IS_DEV else None,
        openapi_url="/openapi.json" if IS_DEV else None,
    )
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(status_code=429, headers={"Retry-After": "60"},
                            content={"detail": "Too many requests — please slow down and try again in a moment."})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        # Never log (or echo) request bodies — they may hold passwords or PHI.
        errs = exc.errors()
        fields = [".".join(str(p) for p in e.get("loc", ()) if p != "body") for e in errs]
        logger.warning("[422] {} {} fields={}", request.method, request.url.path, fields)
        safe = [{"loc": [p for p in e.get("loc", ())],
                 "msg": str(e.get("msg", "Invalid value")).removeprefix("Value error, ")} for e in errs]
        return JSONResponse(status_code=422, content={"detail": safe})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        rid = current_request_id()
        logger.opt(exception=exc).error("[500] {} {}", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"{GENERIC_500} (ref: {rid})"})

    # CORS: exact allow-list from ALLOWED_ORIGINS; none configured → no CORS headers at all
    # (the bundled UI is same-origin and needs none).
    if ALLOWED_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
        )
    # Body caps (outermost-but-one): 256 KB everywhere, large only for the MRI upload.
    app.add_middleware(MaxBodySizeMiddleware, default=256 * 1024,
                       limits={"/mri/upload": MRI_MAX_UPLOAD_BYTES + 2 * 1024 * 1024})
    app.add_middleware(RequestContextMiddleware)      # added last = outermost: headers on ALL responses

    from api.auth import router as auth_router
    from api.routes.classify import router as analyze_router
    from api.routes.chat import router as chat_router
    from api.routes.admin import router as admin_router
    from api.routes.password import router as password_router
    from api.routes.account import router as account_router
    from api.routes.doctor import router as doctor_router
    from api.routes.frontend import router as frontend_router, mount_static

    app.include_router(auth_router)
    app.include_router(analyze_router)
    app.include_router(chat_router)
    app.include_router(admin_router)
    app.include_router(password_router)
    app.include_router(account_router)
    app.include_router(doctor_router)
    app.include_router(frontend_router)
    mount_static(app)

    @app.get("/ping")
    async def ping():
        return "pong"

    @app.get("/health")
    async def health():
        # Public: liveness only. Detailed LLM/MRI info is at GET /admin/health (admin + audited).
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
