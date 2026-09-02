"""
AxonIQ — Application Factory (classifier-free build)
Startup order:
  1. Ensure /data directory exists (persistent storage on HF Spaces)
  2. SQLite schema + migrations
  3. Auto-grant admin from env var
  4. RAG knowledge base (ChromaDB)
"""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.core.config import ADMIN_USERNAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Database
    from api.db import (
        init_db, migrate_add_admin_column,
        migrate_add_logs_table, migrate_add_otp_table, migrate_add_session_state,
        migrate_add_clinical_state, migrate_add_tier_log, migrate_add_nifti_queue,
        migrate_add_mri_jobs, migrate_add_mri_case_id,
        set_admin, get_user_by_username, DB_PATH,
    )
    init_db()
    migrate_add_admin_column()
    migrate_add_logs_table()
    migrate_add_otp_table()
    migrate_add_session_state()
    migrate_add_clinical_state()      # v11: dis_regions, dit_episodes, symptom_timeline
    migrate_add_tier_log()            # v11: risk trajectory log
    migrate_add_nifti_queue()         # v11: DB-backed NIfTI queue (multi-worker safe)
    migrate_add_mri_jobs()            # v12: async MRI job status (upload → poll → result)
    migrate_add_mri_case_id()         # v17: persist case_id for a real "check again" recheck
    logger.info("[AxonIQ] SQLite at: {}", DB_PATH)

    # 2. Auto-grant admin from env
    if ADMIN_USERNAME:
        u = get_user_by_username(ADMIN_USERNAME)
        if u and not u.get("is_admin"):
            set_admin(ADMIN_USERNAME, True)
            logger.info("[AxonIQ] Admin granted to '{}'", ADMIN_USERNAME)

    # 3. RAG knowledge base
    try:
        from api.rag.store import build_knowledge_base
        build_knowledge_base()
    except Exception as e:
        logger.warning("[RAG] Could not build knowledge base: {}", e)

    # 4. Probe LLM — log which model was selected
    try:
        from api.agent.llm import llm
        model = getattr(llm, "_model", "ollama")
        logger.info("[LLM] Active model: {}", model)
    except Exception as e:
        logger.error("[LLM] Startup probe failed: {}", e)

    yield
    logger.info("[AxonIQ] Shutdown complete.")


def create_app() -> FastAPI:
    from api.core.limiter import limiter  # shared instance used by all routes

    app = FastAPI(
        title="AxonIQ — Agentic MS Clinical AI",
        version="10.0.0",
        description=(
            "MS Clinical Decision Support — LLM-powered, RAG-augmented. "
            "50 validated cases + clinical guidelines. Not a diagnosis."
        ),
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # 429 user-friendly response
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests — please slow down and try again in a moment."},
        )

    # 422 diagnostic handler — logs exact validation failure so we can debug
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        raw_body = b""
        try:
            raw_body = await request.body()
        except Exception:
            pass
        logger.warning(
            "[422] {} {} | content-type={} | body={} | errors={}",
            request.method,
            request.url.path,
            request.headers.get("content-type", "MISSING"),
            raw_body[:300],
            exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Routers
    from api.auth import router as auth_router
    from api.routes.classify import router as analyze_router
    from api.routes.chat import router as chat_router
    from api.routes.admin import router as admin_router
    from api.routes.password import router as password_router
    from api.routes.frontend import router as frontend_router, mount_static

    app.include_router(auth_router)
    app.include_router(analyze_router)
    app.include_router(chat_router)
    app.include_router(admin_router)
    app.include_router(password_router)
    app.include_router(frontend_router)
    mount_static(app)

    @app.get("/ping")
    async def ping():
        return "pong"

    @app.get("/health")
    async def health():
        from api.agent.llm import llm
        provider = type(llm).__name__
        model    = getattr(llm, "_model", "ollama")
        return {
            "status":   "ok",
            "version":  "11.0.0",
            "llm_provider": provider,
            "llm_model": model,
            "rag": "ChromaDB — 10 guidelines + 50 validated MS cases",
            "clinical": "McDonald 2017 DIS/DIT criteria tracking enabled",
            "team": {
                "domain_expert":          "Dr. Avasarala, MD PhD — University of Kentucky",
                "lead_ai_architect":      "Raghu Gangolu — NIT Warangal",
                "principal_investigator": "Dr. Kadambari, PhD — NIT Warangal",
            },
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
