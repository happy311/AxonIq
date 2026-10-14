"""
AxonIQ — Password Management Routes
Change password, forgot password (OTP flow), reset password.

No `from __future__ import annotations` (FastAPI + slowapi).
"""
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from loguru import logger
from pydantic import BaseModel, EmailStr, Field, field_validator

from api.auth import (
    get_current_user_allow_pw_change, create_access_token, decode_token,
    set_auth_cookies, client_ip,
)
from api.core import ratelimit
from api.core.config import (
    CHANGE_PW_RATE, FORGOT_RATE_EMAIL, FORGOT_RATE_IP, VERIFY_RATE_EMAIL, VERIFY_RATE_IP,
    OTP_MAX_ATTEMPTS,
)
from api.core.limiter import limiter
from api.core.passwords import hash_password, verify_password, validate_password_policy
from api.database import (
    get_user_by_email, get_user_by_id, save_otp, get_active_otp, burn_otp_attempt,
    mark_otp_used, update_user_password, consume_jti,
)
from api.email_utils import generate_otp, new_salt, hash_otp, verify_otp, otp_expiry, send_otp_email

router = APIRouter(prefix="/auth", tags=["auth"])

_GENERIC_FORGOT = {"message": "If an account exists for this email, a code has been sent."}
_BAD_CODE = "Invalid or expired code. Request a new one if needed."
_BAD_RESET = "Reset link expired or already used. Please request a new code."
_DUMMY_SALT = "0" * 32


def _norm_email(v: str) -> str:
    v = str(v).strip().lower()
    if set("'\"<>") & set(v):
        raise ValueError("Invalid email")
    return v


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=1, max_length=128)


class _EmailModel(BaseModel):
    email: EmailStr

    @field_validator("email")
    @classmethod
    def _chars(cls, v):
        return _norm_email(v)


class ForgotRequest(_EmailModel):
    pass


class VerifyRequest(_EmailModel):
    otp: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


class ResetRequest(BaseModel):
    reset_token: str = Field(..., min_length=10, max_length=2000)
    new_password: str = Field(..., min_length=1, max_length=128)


@router.post("/change-password")
@limiter.limit(CHANGE_PW_RATE)
async def change_password(
    request: Request, response: Response, body: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user_allow_pw_change),
):
    ratelimit.enforce(f"chpw:user:{current_user['id']}", 5, 60)
    if not verify_password(body.old_password, current_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    try:
        validate_password_policy(body.new_password, username=current_user["username"], email=current_user["email"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if body.old_password == body.new_password:
        raise HTTPException(status_code=400, detail="New password must differ from the current one")
    update_user_password(current_user["id"], hash_password(body.new_password))   # bumps token_version
    # every old token (incl. this session's) is now dead — issue a fresh one for this browser
    fresh = get_user_by_id(current_user["id"])
    set_auth_cookies(response, create_access_token(fresh))
    return {"message": "Password changed successfully"}


@router.post("/forgot-password")
async def forgot_password(request: Request, body: ForgotRequest, background: BackgroundTasks):
    """Always answers identically (200 + generic text) — whether the account exists, whether
    e-mail delivery works, and regardless of timing (sending happens in a background task)."""
    email = body.email
    ip = client_ip(request)
    ratelimit.enforce(f"forgot:ip:{ip}", *FORGOT_RATE_IP)
    ratelimit.enforce(f"forgot:email:{email}", *FORGOT_RATE_EMAIL)

    user = get_user_by_email(email)
    if user:
        otp = generate_otp()
        salt = new_salt()
        save_otp(user["id"], email, hash_otp(otp, salt, email), otp_expiry(), salt)
        background.add_task(_deliver_otp, email, user["username"], otp)
    return _GENERIC_FORGOT


def _deliver_otp(email: str, username: str, otp: str) -> None:
    try:
        if not send_otp_email(email, username, otp):
            logger.warning("[Password Reset] OTP generated but e-mail delivery failed/unconfigured")
    except Exception as e:
        logger.error("[Password Reset] delivery error: {}", type(e).__name__)


@router.post("/verify-otp")
async def verify_otp_route(request: Request, body: VerifyRequest):
    from api.auth import create_reset_token
    email, otp = body.email, body.otp
    ip = client_ip(request)
    ratelimit.enforce(f"verify:ip:{ip}", *VERIFY_RATE_IP)
    ratelimit.enforce(f"verify:email:{email}", *VERIFY_RATE_EMAIL)

    record = get_active_otp(email)
    if not record:
        verify_otp(otp, _DUMMY_SALT, email, "0" * 64)          # equalise timing
        raise HTTPException(status_code=400, detail=_BAD_CODE)
    # spend the attempt first (atomic) — locks the OTP on the 5th wrong try
    if not burn_otp_attempt(record["id"], OTP_MAX_ATTEMPTS):
        raise HTTPException(status_code=400, detail=_BAD_CODE)
    if not verify_otp(otp, record["otp_salt"], email, record["otp_hash"]):
        raise HTTPException(status_code=400, detail=_BAD_CODE)
    if not mark_otp_used(record["id"]):                          # single use, race-safe
        raise HTTPException(status_code=400, detail=_BAD_CODE)
    user = get_user_by_id(record["user_id"])
    if not user:
        raise HTTPException(status_code=400, detail=_BAD_CODE)
    return {"reset_token": create_reset_token(user), "message": "Code verified"}


@router.post("/reset-password")
async def reset_password(request: Request, body: ResetRequest):
    """The ONLY endpoint that accepts a typ="reset" token. Single use, bound to token_version."""
    ratelimit.enforce(f"reset:ip:{client_ip(request)}", 20, 3600)
    try:
        payload = decode_token(body.reset_token, "reset")
        user = get_user_by_id(int(payload.get("sub", 0)))
    except (HTTPException, ValueError):
        raise HTTPException(status_code=400, detail=_BAD_RESET)
    if not user or int(payload.get("tv", -1)) != int(user.get("token_version", 0)):
        raise HTTPException(status_code=400, detail=_BAD_RESET)
    try:
        validate_password_policy(body.new_password, username=user["username"], email=user["email"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))     # token NOT consumed — user can retry
    exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc).isoformat()
    if not consume_jti(payload["jti"], exp):
        raise HTTPException(status_code=400, detail=_BAD_RESET)
    update_user_password(user["id"], hash_password(body.new_password))   # bumps token_version, clears must_change
    return {"message": "Password updated successfully"}
