"""
AxonIQ — Admin Routes
All endpoints require is_admin=True and write to the append-only admin_audit table
(admin_id, action, target, ip, time) BEFORE returning data — reading patient chats
is itself an audited action.

No `from __future__ import annotations` (FastAPI).
"""
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from api.auth import get_current_user, client_ip, create_reset_token
from api.core.config import IS_DEV, PUBLIC_BASE_URL, ADMIN_RESET_TOKEN_MINUTES
from api.core.passwords import hash_password

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(current_user: dict = Depends(get_current_user)):
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def _audit(admin: dict, request: Request, action: str, target: str = None) -> None:
    from api.database import audit
    try:
        audit(admin["id"], admin["username"], action, target, client_ip(request))
    except Exception as e:                        # an un-auditable admin action must not proceed
        logger.error("[admin] audit write failed: {}", type(e).__name__)
        raise HTTPException(status_code=500, detail="Audit trail unavailable; action refused.")


@router.get("/users")
async def get_users(request: Request, admin=Depends(require_admin)):
    from api.database import get_all_users_admin, get_all_sessions_admin
    _audit(admin, request, "ADMIN_LIST_USERS")
    users = get_all_users_admin()
    total_sessions, total_messages = get_all_sessions_admin()
    return {"users": users, "total_sessions": total_sessions, "total_messages": total_messages}


@router.get("/user/{user_id}/sessions")
async def get_user_sessions(user_id: int, request: Request, admin=Depends(require_admin)):
    from api.database import get_user_sessions_with_messages
    _audit(admin, request, "ADMIN_READ_PATIENT_CHATS", f"user:{user_id}")
    return {"user_id": user_id, "sessions": get_user_sessions_with_messages(user_id)}


@router.post("/user/{user_id}/reset-password")
async def reset_user_password(user_id: int, request: Request, admin=Depends(require_admin)):
    """Never returns a password. Mints a single-use reset link, e-mails it to the user, kills the
    old password + all sessions, and flags must_change_password. If the e-mail can't be sent
    (and we're not in dev) nothing is changed and 503 is returned."""
    from api.database import get_user_by_id, admin_force_password_reset
    from api.email_utils import send_admin_reset_email
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not PUBLIC_BASE_URL and not IS_DEV:
        raise HTTPException(status_code=503, detail="PUBLIC_BASE_URL is not configured; cannot build a reset link.")
    _audit(admin, request, "ADMIN_RESET_PASSWORD", f"user:{user_id}")

    base = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    next_ver = {**user, "token_version": int(user.get("token_version", 0)) + 1}
    token = create_reset_token(next_ver, minutes=ADMIN_RESET_TOKEN_MINUTES)
    link = f"{base}/chat-ui#reset={token}"        # fragment: never sent to servers/logs/referrers

    sent = send_admin_reset_email(user["email"], user["username"], link, ADMIN_RESET_TOKEN_MINUTES)
    if not sent and not IS_DEV:
        raise HTTPException(status_code=503, detail="Could not e-mail the reset link; no changes were made.")
    admin_force_password_reset(user_id, hash_password(secrets.token_urlsafe(48)))
    out = {"user_id": user_id, "username": user["username"], "emailed": sent,
           "message": "Reset link sent to the user. Their old password and sessions are now invalid."}
    if not sent and IS_DEV:
        out["dev_link"] = link
    return out


@router.get("/logs")
async def get_all_logs(request: Request, admin=Depends(require_admin)):
    from api.database import get_all_logs
    _audit(admin, request, "ADMIN_VIEW_ALL_LOGS")
    return {"logs": get_all_logs(limit=500)}


@router.get("/user/{user_id}/logs")
async def get_user_logs(user_id: int, request: Request, admin=Depends(require_admin)):
    from api.database import get_user_logs
    _audit(admin, request, "ADMIN_VIEW_USER_LOGS", f"user:{user_id}")
    return {"user_id": user_id, "logs": get_user_logs(user_id, limit=200)}


@router.get("/audit")
async def get_audit(request: Request, admin=Depends(require_admin)):
    from api.database import get_audit_entries
    _audit(admin, request, "ADMIN_VIEW_AUDIT")
    return {"entries": get_audit_entries(500)}


@router.get("/health")
async def admin_health(request: Request, admin=Depends(require_admin)):
    """Detailed LLM / MRI engine status (moved here from the public /health)."""
    from api.agent.llm import llm
    from api.mri_engine import jobs as mri_jobs
    _audit(admin, request, "ADMIN_VIEW_HEALTH")
    return {"llm_provider": type(llm).__name__, "llm_model": getattr(llm, "_model", "ollama"),
            "mri_engine": mri_jobs.health()}


# ── Doctor approval ───────────────────────────────────────────────────────────

@router.get("/doctors")
async def list_doctor_accounts(request: Request, status: str = None, admin=Depends(require_admin)):
    from api.database import list_doctors
    if status not in (None, "pending", "approved", "removed"):
        raise HTTPException(status_code=400, detail="status must be pending, approved or removed")
    _audit(admin, request, "ADMIN_LIST_DOCTORS", status)
    return {"doctors": list_doctors(status)}


def _review(user_id: int, new_status: str, request: Request, admin: dict, action: str):
    from api.database import get_user_by_id, set_doctor_status
    u = get_user_by_id(user_id)
    if not u or u.get("role") != "doctor":
        raise HTTPException(status_code=404, detail="Doctor not found")
    _audit(admin, request, action, f"user:{user_id}")
    from api.database import release_doctor_patients, assign_unassigned
    set_doctor_status(user_id, new_status, admin["username"])
    if new_status == "removed":
        release_doctor_patients(user_id)            # their patients go back into the pool …
    moved = assign_unassigned()                      # … and waiting patients get the least-loaded doctor
    return {"user_id": user_id, "username": u["username"], "doctor_status": new_status,
            "patients_assigned": moved}


@router.post("/doctors/{user_id}/approve")
async def approve_doctor(user_id: int, request: Request, admin=Depends(require_admin)):
    """Permit a doctor (also re-permits a previously removed one)."""
    return _review(user_id, "approved", request, admin, "ADMIN_APPROVE_DOCTOR")


@router.post("/doctors/{user_id}/remove")
async def remove_doctor(user_id: int, request: Request, admin=Depends(require_admin)):
    """Remove a doctor's access (also rejects a pending request). Live sessions die immediately."""
    return _review(user_id, "removed", request, admin, "ADMIN_REMOVE_DOCTOR")
