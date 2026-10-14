"""
NeuroCheck — Authentication Layer

  POST /auth/register   — create account (generic response; never reveals duplicates)
  POST /auth/login      — sets HttpOnly session cookie (+ returns the token for API clients)
  POST /auth/logout     — clears cookies
  GET  /auth/me         — current user info

Token model
  typ="access"  8h default (JWT_EXPIRE_MINUTES ≤ 1440), carries jti + token_version (tv).
  typ="reset"   10 min, single-use (jti stored as consumed), accepted ONLY by /auth/reset-password.
  A password change/reset/admin-reset bumps users.token_version, which kills every
  previously-issued access token.

Browser sessions use an HttpOnly + Secure + SameSite=Strict cookie with a CSRF
double-submit token (cookie nc_csrf ↔ header X-CSRF-Token) on state-changing requests.
Bearer-header clients (scripts, tests) are not CSRF-able and skip that check.

NOTE: no `from __future__ import annotations` here — FastAPI + slowapi decorators
need real (non-string) annotations.
"""
import hmac
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt as _bcrypt  # noqa: F401  (kept importable for legacy callers)
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from loguru import logger
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from api.core import ratelimit
from api.core.config import (
    JWT_SECRET as SECRET_KEY, JWT_ALGORITHM as ALGORITHM,
    JWT_EXPIRE_MINUTES as ACCESS_TOKEN_EXPIRE_MINUTES,
    RESET_TOKEN_MINUTES, AUTH_COOKIE_NAME, CSRF_COOKIE_NAME, COOKIE_SECURE,
    LOGIN_RATE_IP, LOGIN_RATE_USER, REGISTER_RATE_IP, LOCKOUT_THRESHOLD, LOCKOUT_MAX_MINUTES,
)
from api.core.limiter import limiter
from api.core.passwords import (
    hash_password, verify_password, needs_rehash, validate_password_policy, DUMMY_HASH,
)
from api.database import (
    create_user, get_user_by_username, get_user_by_email, get_user_by_id,
    get_lockout_until, record_login_failure, clear_login_failures, set_password_hash_only,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
router = APIRouter(prefix="/auth", tags=["auth"])

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_FORBIDDEN_EMAIL_CHARS = set("'\"<>")
PRIVACY_NOTICE_VERSION = "2026-10b"   # b: adds disclosure that admin-approved doctors can view assessments/MRI


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=40, pattern=r"^[a-zA-Z0-9_\-]+$")
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)
    accept_privacy: bool = False
    role: str = Field("patient", pattern=r"^(patient|doctor)$")
    license_number: Optional[str] = Field(None, max_length=60)
    institution: Optional[str] = Field(None, max_length=120)

    @field_validator("license_number", "institution")
    @classmethod
    def _clean_free_text(cls, v):
        if v is None:
            return None
        v = " ".join(v.split())
        if _FORBIDDEN_EMAIL_CHARS & set(v):
            raise ValueError("Invalid characters in text field")
        return v or None

    @field_validator("email")
    @classmethod
    def _email_chars(cls, v):
        v = str(v).strip().lower()
        if _FORBIDDEN_EMAIL_CHARS & set(v):
            raise ValueError("Email contains invalid characters")
        if len(v) > 254:
            raise ValueError("Email too long")
        return v

    @model_validator(mode="after")
    def _policy(self):
        if not self.accept_privacy:
            raise ValueError("You must accept the privacy notice to register")
        if self.role == "doctor" and not (self.license_number and self.institution):
            raise ValueError("Doctors must provide a medical licence number and institution")
        validate_password_policy(self.password, username=self.username, email=self.email)
        return self


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    must_change_password: bool = False
    role: str = "patient"                 # patient | doctor | admin


class UserInfo(BaseModel):
    user_id: int
    username: str
    email: str
    created_at: str
    is_admin: bool = False
    must_change_password: bool = False
    role: str = "patient"
    doctor_status: Optional[str] = None


# ── Tokens ────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user: dict) -> str:
    now = _now()
    payload = {
        "sub": str(user["id"]), "username": user["username"],
        "typ": "access", "jti": uuid.uuid4().hex,
        "tv": int(user.get("token_version", 0)),
        "iat": now, "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_reset_token(user: dict, minutes: int = RESET_TOKEN_MINUTES) -> str:
    now = _now()
    payload = {
        "sub": str(user["id"]), "typ": "reset", "jti": uuid.uuid4().hex,
        "tv": int(user.get("token_version", 0)),
        "iat": now, "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str, expected_typ: str) -> dict:
    """Decode + verify signature/expiry AND enforce the token type."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
    if payload.get("typ") != expected_typ or not payload.get("jti"):
        raise HTTPException(status_code=401, detail="Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
    return payload


# ── Cookies / CSRF ────────────────────────────────────────────────────────────

def set_auth_cookies(response: Response, token: str) -> None:
    max_age = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    response.set_cookie(AUTH_COOKIE_NAME, token, max_age=max_age, httponly=True,
                        secure=COOKIE_SECURE, samesite="strict", path="/")
    # readable by our JS so it can echo it in X-CSRF-Token (double-submit)
    response.set_cookie(CSRF_COOKIE_NAME, secrets.token_urlsafe(32), max_age=max_age,
                        httponly=False, secure=COOKIE_SECURE, samesite="strict", path="/")


def clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")


def _check_csrf(request: Request) -> None:
    cookie = request.cookies.get(CSRF_COOKIE_NAME, "")
    header = request.headers.get("x-csrf-token", "")
    if not cookie or not header or not hmac.compare_digest(cookie, header):
        raise HTTPException(status_code=403, detail="CSRF validation failed")


# ── Dependencies ──────────────────────────────────────────────────────────────

def _resolve_user(request: Request, bearer: Optional[str], *, allow_pw_change: bool) -> dict:
    token, via_cookie = bearer, False
    if not token:
        token, via_cookie = request.cookies.get(AUTH_COOKIE_NAME), True
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})
    if via_cookie and request.method not in _SAFE_METHODS:
        _check_csrf(request)

    payload = decode_token(token, "access")
    try:
        user_id = int(payload.get("sub", 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if int(payload.get("tv", -1)) != int(user.get("token_version", 0)):
        raise HTTPException(status_code=401, detail="Session expired — please sign in again",
                            headers={"WWW-Authenticate": "Bearer"})
    if user.get("role") == "doctor" and user.get("doctor_status") != "approved":
        raise HTTPException(status_code=403, detail="DOCTOR_NOT_APPROVED")
    if user.get("must_change_password") and not allow_pw_change:
        raise HTTPException(status_code=403, detail="PASSWORD_CHANGE_REQUIRED")
    return user


def get_current_user(request: Request, token: Optional[str] = Depends(oauth2_scheme)) -> dict:
    """Protected-route dependency: access tokens only, current token_version only."""
    return _resolve_user(request, token, allow_pw_change=False)


def get_current_user_allow_pw_change(request: Request, token: Optional[str] = Depends(oauth2_scheme)) -> dict:
    return _resolve_user(request, token, allow_pw_change=True)


def get_optional_user(request: Request, token: Optional[str] = Depends(oauth2_scheme)) -> Optional[dict]:
    try:
        return _resolve_user(request, token, allow_pw_change=False)
    except HTTPException:
        return None


def client_ip(request: Request) -> str:
    # request.client is already the real client IP when uvicorn runs with
    # --proxy-headers and the peer is in --forwarded-allow-ips (see start.py / Dockerfile).
    return request.client.host if request.client else "unknown"


# ── Routes ────────────────────────────────────────────────────────────────────

_REGISTER_MSG = {"message": "Registration received. You can now sign in. "
                            "If you already have an account, use Sign In or Reset password."}
_REGISTER_MSG_DOCTOR = {"message": "Doctor registration received. An administrator must verify and approve "
                                   "your account before you can sign in."}


@router.post("/register", status_code=201)
@limiter.limit(REGISTER_RATE_IP)
async def register(request: Request, req: RegisterRequest):
    """Create an account. The response is identical whether or not the username/email
    already exists (no enumeration); the bcrypt hash is always computed so timing matches."""
    hashed = hash_password(req.password)
    exists = get_user_by_username(req.username) or get_user_by_email(req.email)
    if not exists:
        try:
            uid = create_user(req.username, req.email, hashed, consent_version=PRIVACY_NOTICE_VERSION,
                              role=req.role, doctor_license=req.license_number,
                              doctor_institution=req.institution)
            logger.info("[auth] registered user id={}", uid)
        except sqlite3.IntegrityError:
            pass   # lost a race with another registration — same generic response
    else:
        logger.info("[auth] registration attempt for existing account (suppressed)")
    return _REGISTER_MSG_DOCTOR if req.role == "doctor" else _REGISTER_MSG


@router.post("/login", response_model=LoginResponse)
@limiter.limit(LOGIN_RATE_IP)
async def login(request: Request, response: Response, form: OAuth2PasswordRequestForm = Depends()):
    """Accepts username OR email. Unknown users still pay a bcrypt check; lockout and
    rate-limit responses are identical for existing and non-existing accounts."""
    typed = (form.username or "").strip()[:200]
    user = get_user_by_username(typed) or get_user_by_email(typed)
    ident = f"u:{user['id']}" if user else f"x:{typed.lower()}"

    ratelimit.enforce(f"login:user:{typed.lower()}", *LOGIN_RATE_USER,
                      detail="Too many sign-in attempts — please wait a minute.")
    if get_lockout_until(ident):
        # still burn CPU equivalently, then refuse
        verify_password(form.password, DUMMY_HASH)
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.")

    ok = verify_password(form.password, user["password_hash"] if user else DUMMY_HASH)
    if not user or not ok:
        record_login_failure(ident, LOCKOUT_THRESHOLD, LOCKOUT_MAX_MINUTES)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username/email or password",
                            headers={"WWW-Authenticate": "Bearer"})

    clear_login_failures(ident)
    if user.get("role") == "doctor" and user.get("doctor_status") != "approved":
        # Only reached with the CORRECT password, so this reveals nothing to a guesser.
        st = user.get("doctor_status")
        msg = ("Your doctor account is awaiting administrator approval." if st == "pending"
               else "Your doctor access has been removed. Contact the administrator.")
        raise HTTPException(status_code=403, detail=msg)
    if needs_rehash(user["password_hash"]):
        set_password_hash_only(user["id"], hash_password(form.password))

    token = create_access_token(user)
    set_auth_cookies(response, token)
    return LoginResponse(access_token=token, user_id=user["id"], username=user["username"],
                         must_change_password=bool(user.get("must_change_password")),
                         role=("admin" if user.get("is_admin") else user.get("role") or "patient"))


@router.post("/logout")
async def logout(response: Response):
    clear_auth_cookies(response)
    return {"message": "Signed out"}


@router.get("/me", response_model=UserInfo)
async def me(current_user: dict = Depends(get_current_user_allow_pw_change)):
    return UserInfo(
        user_id=current_user["id"], username=current_user["username"],
        email=current_user["email"], created_at=current_user["created_at"],
        is_admin=bool(current_user.get("is_admin", 0)),
        must_change_password=bool(current_user.get("must_change_password", 0)),
        role=("admin" if current_user.get("is_admin") else current_user.get("role") or "patient"),
        doctor_status=current_user.get("doctor_status"),
    )
