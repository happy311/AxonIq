"""
AxonIQ — Analyze Route (deprecated)

The /analyze endpoint previously powered the live risk panel in the UI.
The risk panel has been removed — classification now happens internally
in the agent and is stored in the session DB (session.tier / session.features).

This file is kept so the router registration in main.py doesn't break.
The endpoint returns 410 Gone so any stale frontend calls fail gracefully.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["analyze"])


@router.post("/analyze")
async def analyze_endpoint():
    """Deprecated — risk panel removed. Classification is now internal."""
    return JSONResponse(
        status_code=410,
        content={"detail": "The /analyze endpoint is deprecated. Classification is now handled internally by the agent."},
    )
