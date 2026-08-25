"""
AxonIQ — Admin Routes
All endpoints require is_admin=True on the JWT user.
"""
from __future__ import annotations
import secrets

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from api.auth import get_current_user

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(current_user: dict = Depends(get_current_user)):
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.get("/users")
async def get_users(admin=Depends(require_admin)):
    from api.database import get_all_users_admin, get_all_sessions_admin, migrate_add_admin_column
    migrate_add_admin_column()
    users = get_all_users_admin()
    total_sessions, total_messages = get_all_sessions_admin()
    return {"users": users, "total_sessions": total_sessions, "total_messages": total_messages}


@router.get("/user/{user_id}/sessions")
async def get_user_sessions(user_id: int, admin=Depends(require_admin)):
    from api.database import get_user_sessions_with_messages
    return {"user_id": user_id, "sessions": get_user_sessions_with_messages(user_id)}


@router.post("/user/{user_id}/reset-password")
async def reset_user_password(user_id: int, admin=Depends(require_admin)):
    import bcrypt as _bcrypt
    from api.database import get_user_by_id, get_conn
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    new_password = secrets.token_urlsafe(10)
    new_hash = _bcrypt.hashpw(new_password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?", (new_hash, user_id))
    return {
        "user_id":   user_id,
        "username":  user["username"],
        "new_password": new_password,
        "message":   "Password reset. Share with the user and ask them to change it after login.",
    }


@router.get("/logs")
async def get_all_logs(admin=Depends(require_admin)):
    from api.database import get_all_logs
    return {"logs": get_all_logs(limit=500)}


@router.get("/user/{user_id}/logs")
async def get_user_logs(user_id: int, admin=Depends(require_admin)):
    from api.database import get_user_logs
    return {"user_id": user_id, "logs": get_user_logs(user_id, limit=200)}
