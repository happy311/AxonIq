"""
AxonIQ — Password Management Routes
Change password, forgot password (OTP flow), reset password.
"""
from __future__ import annotations
import hashlib
import os
import random
import string
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from api.auth import get_current_user, create_access_token, decode_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/change-password")
async def change_password(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    import bcrypt as _bcrypt
    from api.database import get_conn
    body   = await request.json()
    old_pw = body.get("old_password", "")
    new_pw = body.get("new_password", "")
    if not old_pw or not new_pw:
        raise HTTPException(status_code=400, detail="Both old and new password are required")
    if len(new_pw) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
    if not _bcrypt.checkpw(old_pw.encode("utf-8"), current_user["password_hash"].encode("utf-8")):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    new_hash = _bcrypt.hashpw(new_pw.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?", (new_hash, current_user["id"]))
    return {"message": "Password changed successfully"}


@router.post("/forgot-password")
async def forgot_password(request: Request):
    from api.database import get_user_by_email, save_otp, migrate_add_otp_table
    migrate_add_otp_table()
    body  = await request.json()
    email = body.get("email", "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    user = get_user_by_email(email)
    if not user:
        # Don't confirm/deny account existence to an unauthenticated caller.
        return {"message": "If an account exists for this email, a code has been sent."}
    otp      = "".join(random.choices(string.digits, k=6))
    otp_hash = hashlib.sha256(otp.encode()).hexdigest()
    expires  = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    save_otp(user["id"], email, otp_hash, expires)

    smtp_ok = bool(os.environ.get("SMTP_EMAIL") and os.environ.get("SMTP_PASSWORD"))
    if smtp_ok:
        try:
            from api.email_utils import send_otp_email
            if send_otp_email(email, user["username"], otp):
                return {"message": "Code sent to your email"}
        except Exception as e:
            logger.error("[Password Reset] send_otp_email failed for {}: {}", email, e)

    # NEVER return the OTP itself. If we get here, delivery genuinely failed
    # or SMTP isn't configured — the caller needs a way to reset without an
    # attacker being able to self-serve the code for any known email address.
    logger.warning(
        "[Password Reset] SMTP not configured or delivery failed for {} — "
        "OTP was generated but NOT returned to the client.", email
    )
    raise HTTPException(
        status_code=503,
        detail="Password reset email could not be sent right now. Please contact support or try again later.",
    )


@router.post("/verify-otp")
async def verify_otp(request: Request):
    from api.database import get_valid_otp, mark_otp_used, migrate_add_otp_table
    migrate_add_otp_table()
    body  = await request.json()
    email = body.get("email", "").strip().lower()
    otp   = body.get("otp", "").strip()
    if not email or not otp:
        raise HTTPException(status_code=400, detail="Email and code are required")
    record = get_valid_otp(email)
    if not record:
        raise HTTPException(status_code=400, detail="Code expired or not found. Request a new one.")
    if hashlib.sha256(otp.encode()).hexdigest() != record["otp_hash"]:
        raise HTTPException(status_code=400, detail="Incorrect code. Please try again.")
    mark_otp_used(record["id"])
    reset_token = create_access_token(record["user_id"], "reset:" + email)
    return {"reset_token": reset_token, "message": "Code verified"}


@router.post("/reset-password")
async def reset_password(request: Request):
    import bcrypt as _bcrypt
    from api.database import update_user_password
    body   = await request.json()
    token  = body.get("reset_token", "")
    new_pw = body.get("new_password", "")
    if not token or not new_pw:
        raise HTTPException(status_code=400, detail="Token and password are required")
    if len(new_pw) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub", 0))
        if not str(payload.get("username", "")).startswith("reset:"):
            raise ValueError("Not a reset token")
    except Exception:
        raise HTTPException(status_code=400, detail="Reset link expired. Please request a new code.")
    new_hash = _bcrypt.hashpw(new_pw.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")
    update_user_password(user_id, new_hash)
    return {"message": "Password updated successfully"}
