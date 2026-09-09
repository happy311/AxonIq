"""
AxonIQ — Frontend Routes (classifier-free build)

/ and /chat-ui → chat.html  (login page is the first screen inside chat.html)
/admin         → admin.html

Static file mounting:
  CSS: served at /css/* from frontend/css/
  JS:  served at /js/*  from frontend/js/
  (no /static prefix — chat.html links to /css/ and /js/ directly)
"""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.core.config import FRONTEND_DIR

router = APIRouter(tags=["frontend"])


def _read_html(filename: str) -> str:
    path = FRONTEND_DIR / filename
    if not path.exists():
        return f"<h1>File not found: {filename}</h1>"
    return path.read_text(encoding="utf-8")


# / → chat.html (login overlay is the first screen)
@router.get("/", response_class=HTMLResponse)
async def serve_root():
    return HTMLResponse(content=_read_html("chat.html"), status_code=200)


@router.get("/chat-ui",  response_class=HTMLResponse)
@router.get("/chat-ui/", response_class=HTMLResponse)
@router.get("/chat",     response_class=HTMLResponse)
async def serve_chat():
    return HTMLResponse(content=_read_html("chat.html"), status_code=200)


@router.get("/admin",  response_class=HTMLResponse)
@router.get("/admin/", response_class=HTMLResponse)
async def serve_admin():
    return HTMLResponse(content=_read_html("admin.html"), status_code=200)


def mount_static(app) -> None:
    """
    Mount frontend sub-directories so chat.html can load:
      /css/variables.css  → frontend/css/variables.css
      /js/auth.js         → frontend/js/auth.js
    """
    css_dir = FRONTEND_DIR / "css"
    js_dir  = FRONTEND_DIR / "js"

    if css_dir.is_dir():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.is_dir():
        app.mount("/js",  StaticFiles(directory=str(js_dir)),  name="js")

    # Also mount root frontend dir for any other static assets (favicon etc.)
    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
