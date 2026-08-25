"""
AxonIQ — Chat Routes (v11)

SYSTEM DESIGN:
- Route reads phase/tier/features + McDonald clinical fields from DB before calling agent
- Route writes all state back to DB after agent returns
- No state derived from message text in this layer

v11 changes:
  - NIfTI queue backed by SQLite (replaces in-process dict — works across
    multiple Uvicorn workers on HuggingFace Spaces)
  - chat() now returns a dict instead of a 4-tuple; route unpacks from dict
  - McDonald clinical fields (dis_regions, dit_episodes, symptom_timeline)
    are threaded through read → agent → write each turn
  - GET /session/{session_id}/export  — generates clinical summary for referral

No `from __future__ import annotations` — required for FastAPI + slowapi.
"""
import os
import asyncio
import uuid as _uuid
import tempfile

from fastapi import APIRouter, Body, Depends, HTTPException, Request, UploadFile, File, BackgroundTasks
from loguru import logger

from api.core.config import CHAT_RATE_LIMIT
from api.core.limiter import limiter
from api.core.schemas import ChatRequest, ChatResponse
from api.auth import get_current_user

router = APIRouter(tags=["chat"])


# ─────────────────────────────────────────────────────────────────────────────
# /chat  — main conversation endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
@limiter.limit(CHAT_RATE_LIMIT)
async def chat_endpoint(
    request: Request,
    req: ChatRequest = Body(...),
    current_user: dict = Depends(get_current_user),
):
    from api.agent.chat import chat as agent_chat
    from api.database import (
        create_db_session, save_message, get_session_owner,
        update_session_title, get_message_count, log_action,
        touch_session, get_session_state, update_session_state,
    )
    from api.database import get_messages as _get_msgs

    message_text = req.message.strip()
    if not message_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    user_id = current_user["id"]

    # ── Session management ────────────────────────────────────────────────────
    sid = req.session_id
    if not sid:
        sid = str(_uuid.uuid4())
        create_db_session(sid, user_id)
        logger.info("[chat] New session: {} user={}", sid, user_id)
    else:
        owner = get_session_owner(sid)
        if owner is not None and owner != user_id:
            raise HTTPException(status_code=403, detail="Session belongs to another user")
        if owner is None:
            # Session not found or was soft-deleted.
            # Always generate a fresh UUID — never reuse a deleted session_uuid
            # because the old row still exists in DB and would cause a UNIQUE violation.
            sid = str(_uuid.uuid4())
            create_db_session(sid, user_id)
            logger.info("[chat] New session (replaced missing/deleted): {} user={}", sid, user_id)

    # ── Read full clinical state from DB ──────────────────────────────────────
    db_state       = get_session_state(sid)
    phase          = db_state["phase"]
    tier           = db_state["tier"]
    features       = db_state["features"]
    dis_regions    = db_state.get("dis_regions", [])
    dit_episodes   = db_state.get("dit_episodes", 0)
    symptom_timeline = db_state.get("symptom_timeline", [])

    # ── Persist user message ─────────────────────────────────────────────────
    save_message(sid, "human", message_text)
    log_action(
        "CHAT_MESSAGE", user_id=user_id, username=current_user["username"],
        detail=message_text[:120], session_uuid=sid,
        ip_address=request.client.host if request.client else None,
    )
    if get_message_count(sid) <= 1:
        update_session_title(sid, message_text[:60].strip())

    # ── Load full history ────────────────────────────────────────────────────
    history = [{"role": m["role"], "content": m["content"]} for m in _get_msgs(sid)]

    # ── Greeting fast-path (runs for all phases, not just gathering) ─────────
    # Handles "hey", "thanks", meta questions etc. without invoking the LLM.
    # v11.2: passes has_clinical_context so mid-session greetings get a
    # contextual "I'm still here" response instead of falling to the LLM
    # which would produce a generic "Hello! How can I assist you?" reply.
    #
    # BUG FIX (Bug 4): Previously _pop_nifti_paths() was called BEFORE this
    # fast-path.  If the user sent a greeting ("thanks") while NIfTI files
    # were queued in DB, the files were popped and silently discarded when
    # the greeting path returned early without passing them to the agent.
    # Fix: peek (non-destructive) to check if files are queued; skip the
    # greeting fast-path when files are waiting; only pop AFTER the check.
    from api.agent.greeting import is_greeting, greeting_response
    from api.database import has_queued_nifti_paths
    has_history      = any(m["role"] == "assistant" for m in history)
    has_clinical_ctx = bool(features)
    has_nifti_queued = has_queued_nifti_paths(sid)

    if not has_nifti_queued and is_greeting(
        message_text, has_history=has_history, has_clinical_context=has_clinical_ctx
    ):
        is_first = not has_history
        fast = greeting_response(
            message_text.lower(),
            is_first_turn=is_first,
            has_clinical_context=has_clinical_ctx,
            phase=phase,
        )
        if fast:
            save_message(sid, "assistant", fast)
            touch_session(sid)
            logger.info("[chat] Greeting fast-path (phase={}) — sid={}", phase, sid)
            return ChatResponse(
                session_id=sid,
                response=fast,
                turn=len(history) + 1,
            )

    # ── Check if a NIfTI file pair was queued for this session ───────────────
    # Popped HERE (after greeting fast-path) so files are never consumed
    # and lost when a greeting short-circuits before the agent runs.
    nifti_paths = _pop_nifti_paths(sid)

    # ── Run agent ─────────────────────────────────────────────────────────────
    # agent_chat() calls synchronous blocking I/O (httpx for MRI, LangChain for
    # LLM). We must run it in a thread so the event loop stays free to handle
    # other requests and send keep-alive frames while MRI analysis runs
    # (which can take up to 20 min). Without to_thread the entire Uvicorn
    # worker hangs and the proxy / browser times out.
    human_turns = sum(1 for m in history if m["role"] == "human")
    try:
        result = await asyncio.to_thread(
            agent_chat,
            session_id=sid,
            user_message=message_text,
            history=history,
            phase=phase,
            tier=tier,
            features=features,
            nifti_paths=nifti_paths,
            dis_regions=dis_regions,
            dit_episodes=dit_episodes,
            symptom_timeline=symptom_timeline,
        )
    except Exception as e:
        logger.error("[chat] Agent error: {}", e)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    prose          = result["prose"]
    new_tier       = result["tier"]
    new_features   = result["features"]
    next_phase     = result["next_phase"]
    new_dis        = result["dis_regions"]
    new_dit        = result["dit_episodes"]
    new_timeline   = result["symptom_timeline"]

    # ── Persist response + updated clinical state ─────────────────────────────
    save_message(sid, "assistant", prose)
    update_session_state(
        sid, next_phase, new_tier, new_features,
        dis_regions=new_dis,
        dit_episodes=new_dit,
        symptom_timeline=new_timeline,
        human_turn=human_turns,
    )
    touch_session(sid)

    logger.info(
        "[chat] sid={} phase={}→{} tier={}→{} dis={}",
        sid, phase, next_phase, tier, new_tier, new_dis,
    )

    return ChatResponse(
        session_id=sid,
        response=prose,
        turn=len(history) + 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# /mri/upload  — FLAIR-only NIfTI file upload
#
# The MRI analysis backend now runs the trained fold×orientation segmentation
# ensemble (see server-ensemble.ipynb) instead of LST-AI, and that pipeline
# only needs a single FLAIR scan — T1 is no longer requested or sent anywhere
# downstream.
#
# v12 change (async upload → poll → result):
# Previously the frontend had to send a follow-up /chat message that blocked
# the HTTP request open for up to 30 minutes while the agent submitted the
# scan to the Kaggle service and polled it internally. That single long-held
# request is exactly what free tunnels / proxies are unreliable with.
#
# Now: this endpoint saves the file, immediately schedules the full agent
# turn (MRI submit+poll → RAG → LLM response) as a FastAPI BackgroundTask,
# and returns right away with {status: "processing"}. The frontend polls the
# cheap GET /mri/status/{session_id} endpoint every few seconds, and once
# that reports "done" it calls GET /mri/result/{session_id} once to fetch the
# assistant's response. No request is ever held open for more than a second
# or two — the only long-running work happens server-side, off the HTTP
# request/response cycle entirely.
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/mri/upload")
@limiter.limit("5/minute")
async def upload_nifti(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str,
    flair: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a FLAIR NIfTI file for MRI analysis. Must be .nii or .nii.gz.

    Frontend workflow:
      1. POST the file here → {status: "processing"} (analysis starts immediately
         in the background — this call returns in well under a second)
      2. Poll GET /mri/status/{session_id} every few seconds until status is
         "done" (or "error")
      3. Call GET /mri/result/{session_id} once to fetch the assistant's response
    """
    from api.database import get_session_owner, create_mri_job

    owner = get_session_owner(session_id)
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    MAX_BYTES = 500 * 1024 * 1024  # 500 MB per file

    def _validate_and_save(upload: UploadFile, label: str) -> str:
        fname = upload.filename or ""
        if not (fname.endswith(".nii.gz") or fname.endswith(".nii")):
            raise HTTPException(
                status_code=400,
                detail=f"Only NIfTI files (.nii.gz or .nii) are accepted for {label}.",
            )
        return fname

    flair_fname = _validate_and_save(flair, "FLAIR")

    flair_contents = await flair.read()

    if len(flair_contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="FLAIR file too large (max 500 MB)")

    def _write_tmp(contents: bytes, fname: str) -> str:
        suffix = ".nii.gz" if fname.endswith(".nii.gz") else ".nii"
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix=f"mri_{session_id}_",
        )
        tmp.write(contents)
        tmp.close()
        return tmp.name

    flair_path = _write_tmp(flair_contents, flair_fname)

    # Mark the job as processing BEFORE scheduling the background task, so a
    # status poll that lands the instant after this response is sent never
    # sees "no job found" (a brief false-negative race).
    create_mri_job(session_id)

    background_tasks.add_task(
        _run_mri_analysis_background,
        session_id,
        {"flair": flair_path, "t1": None},
    )

    logger.info(
        "[MRI Upload] session={} flair={} ({:.1f}MB) — analysis scheduled in background",
        session_id,
        flair_fname, len(flair_contents) / (1024 * 1024),
    )

    return {
        "status":       "processing",
        "message":      "MRI file received. Analysis has started — poll /mri/status for progress.",
        "flair_mb":     round(len(flair_contents) / (1024 * 1024), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Background MRI analysis runner
#
# Runs the exact same agent turn the old synchronous flow used
# (api.agent.chat.chat with nifti_paths set), just off the request/response
# cycle. FastAPI executes sync BackgroundTasks in a worker thread, so this
# does not block the event loop or any other request.
# ─────────────────────────────────────────────────────────────────────────────

def _run_mri_analysis_background(session_id: str, nifti_paths: dict) -> None:
    from api.agent.chat import chat as agent_chat
    from api.database import (
        get_session_state, save_message, update_session_state,
        touch_session, set_mri_job_status, get_messages,
    )

    _SYNTHETIC_USER_MSG = "I have uploaded my FLAIR MRI scan for analysis."

    try:
        db_state = get_session_state(session_id)
        history  = [{"role": m["role"], "content": m["content"]} for m in get_messages(session_id)]

        # Persist the synthetic trigger message so the transcript stays complete —
        # mirrors what the old frontend-sent /chat message did.
        save_message(session_id, "human", _SYNTHETIC_USER_MSG)
        human_turns = sum(1 for m in history if m["role"] == "human") + 1

        result = agent_chat(
            session_id=session_id,
            user_message=_SYNTHETIC_USER_MSG,
            history=history,
            phase=db_state["phase"],
            tier=db_state["tier"],
            features=db_state["features"],
            nifti_paths=nifti_paths,
            dis_regions=db_state.get("dis_regions", []),
            dit_episodes=db_state.get("dit_episodes", 0),
            symptom_timeline=db_state.get("symptom_timeline", []),
        )

        save_message(session_id, "assistant", result["prose"])
        update_session_state(
            session_id, result["next_phase"], result["tier"], result["features"],
            dis_regions=result["dis_regions"],
            dit_episodes=result["dit_episodes"],
            symptom_timeline=result["symptom_timeline"],
            human_turn=human_turns,
        )
        touch_session(session_id)
        set_mri_job_status(session_id, "done")

        logger.info("[MRI Background] session={} analysis complete, tier={}", session_id, result["tier"])

    except Exception as e:
        logger.error("[MRI Background] session={} failed: {}", session_id, e)
        try:
            set_mri_job_status(session_id, "error", error=str(e)[:300])
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# /mri/status/{session_id}  — cheap poll target for the frontend timer
# /mri/result/{session_id}  — fetched once, after status is "done"
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/mri/status/{session_id}")
@limiter.limit("120/minute")
async def mri_status(
    request: Request,
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Lightweight status check — meant to be polled every few seconds while an
    MRI analysis is running. Never blocks; always returns immediately.

    status: "processing" | "done" | "error" | "none" (no job found for this session)
    """
    from api.database import get_session_owner, get_mri_job

    owner = get_session_owner(session_id)
    if owner is not None and owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    job = get_mri_job(session_id)
    if not job:
        return {"status": "none"}

    return {"status": job["status"], "error": job.get("error")}


@router.get("/mri/result/{session_id}")
async def mri_result(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Fetch the assistant's response once /mri/status reports "done".
    Returns 409 if the job isn't finished yet — the frontend should keep
    polling /mri/status instead of calling this early.
    """
    from api.database import get_session_owner, get_mri_job, get_messages

    owner = get_session_owner(session_id)
    if owner is not None and owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    job = get_mri_job(session_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=409, detail="MRI analysis is not finished yet")

    messages = get_messages(session_id)
    last_assistant = next((m for m in reversed(messages) if m["role"] == "assistant"), None)

    return {
        "session_id": session_id,
        "response":   last_assistant["content"] if last_assistant else "",
        "turn":       len(messages),
    }


# ── NIfTI queue — DB-backed (multi-worker safe) ───────────────────────────────
# Delegates entirely to database.py; this layer has no state of its own.
# NOTE: no longer used by /mri/upload (v12 triggers the background task
# directly instead of queueing for the next /chat message), but kept for any
# other/legacy callers so nothing else breaks.

def _store_nifti_paths(session_id: str, flair_path: str, t1_path: str = None) -> None:
    from api.database import store_nifti_paths
    store_nifti_paths(session_id, flair_path, t1_path)


def _pop_nifti_paths(session_id: str) -> dict | None:
    from api.database import pop_nifti_paths
    return pop_nifti_paths(session_id)


# Keep the old single-path wrappers for any legacy callers
def _store_nifti_path(session_id: str, path: str) -> None:
    from api.database import store_nifti_path
    store_nifti_path(session_id, path)


def _pop_nifti_path(session_id: str) -> str | None:
    from api.database import pop_nifti_path
    return pop_nifti_path(session_id)


# ─────────────────────────────────────────────────────────────────────────────
# /session/{session_id}/export  — clinical summary for neurologist referral
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/session/{session_id}/export")
async def export_session_summary(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Generate a structured clinical pre-assessment summary.
    The patient can print or share this with their GP / neurologist.

    Returns a JSON document with:
      - chief_complaint, symptom_summary
      - ms_consistent_features
      - mcdonald_assessment (DIS/DIT status)
      - recommended_workup
      - confidence, urgency
      - neurologist_note
      - tier_log (risk trajectory across the conversation)
    """
    from api.database import get_session_owner, get_session_export_data, get_tier_log
    from api.agent.chat import generate_summary

    owner = get_session_owner(session_id)
    if owner is not None and owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    export_data = get_session_export_data(session_id)
    if not export_data:
        raise HTTPException(status_code=404, detail="Session not found")

    # Only generate summary when there's something to summarise
    tier     = export_data.get("tier", "LOW")
    features = export_data.get("features", [])

    if not features and tier == "LOW":
        return {
            "session_id":   session_id,
            "status":       "insufficient_data",
            "message":      "No MS-consistent features found in this session yet. Continue the conversation to gather more information.",
            "tier":         tier,
            "tier_log":     get_tier_log(session_id),
        }

    summary = generate_summary(
        tier=tier,
        features=features,
        symptom_timeline=export_data.get("symptom_timeline", []),
        dis_regions=export_data.get("dis_regions", []),
        dit_episodes=export_data.get("dit_episodes", 0),
    )

    return {
        "session_id":   session_id,
        "generated_at": export_data.get("updated_at"),
        "summary":      summary,
        "tier_log":     get_tier_log(session_id),
        "disclaimer":   "This assessment is for clinical decision support only and is not a diagnosis.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# /session/{session_id}/tier-log  — risk trajectory (for clinician dashboard)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/session/{session_id}/tier-log")
async def get_tier_log_endpoint(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return the risk tier progression across the conversation."""
    from api.database import get_session_owner, get_tier_log

    owner = get_session_owner(session_id)
    if owner is not None and owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    return {"session_id": session_id, "tier_log": get_tier_log(session_id)}


# ─────────────────────────────────────────────────────────────────────────────
# Session management routes
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/session/new")
async def new_session(current_user: dict = Depends(get_current_user)):
    from api.database import create_db_session
    sid = str(_uuid.uuid4())
    create_db_session(sid, current_user["id"])
    return {"session_id": sid}


@router.get("/user/sessions")
async def user_sessions(current_user: dict = Depends(get_current_user)):
    from api.database import get_user_sessions
    return {"sessions": get_user_sessions(current_user["id"])}


@router.get("/session/{session_id}/history")
async def session_history(session_id: str, current_user: dict = Depends(get_current_user)):
    from api.database import get_messages, get_session_owner
    owner = get_session_owner(session_id)
    if owner is not None and owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return {"session_id": session_id, "messages": get_messages(session_id)}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """
    Soft-delete a session.
    Idempotent: deleting an already-deleted or non-existent session returns 200.

    NOTE: we intentionally do NOT pre-create a replacement session here.
    Pre-creating caused a phantom "New Chat" entry to appear in the sidebar
    immediately after deletion, making it look like the chat wasn't deleted.
    The frontend creates a new session naturally when the user sends their
    first message after deletion.
    """
    from api.database import get_session_owner, delete_db_session

    owner = get_session_owner(session_id)

    if owner is None:
        # Already deleted or never existed — treat as success (idempotent)
        return {"deleted": session_id}
    if owner != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    delete_db_session(session_id)
    logger.info("[chat] Deleted session={}", session_id)
    return {"deleted": session_id}
