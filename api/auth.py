"""
NeuroCheck — Authentication Layer
JWT-based stateless auth with bcrypt password hashing.

Endpoints added to main.py via include_router:
  POST /auth/register   — create account
  POST /auth/login      — returns access_token
  GET  /auth/me         — returns current user info  (requires Bearer token)
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt as _bcrypt
from pydantic import BaseModel, Field

from api.database import (
    create_user,
    get_user_by_username,
    get_user_by_email,
    get_user_by_id,
    init_db,
)

# ── Config ────────────────────────────────────────────────────────────────────
from api.core.config import JWT_SECRET as SECRET_KEY, JWT_ALGORITHM as ALGORITHM, JWT_EXPIRE_MINUTES as ACCESS_TOKEN_EXPIRE_MINUTES

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

router = APIRouter(prefix="/auth", tags=["auth"])

# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=40, pattern=r"^[a-zA-Z0-9_\-]+$")
    email: str = Field(..., pattern=r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
    password: str = Field(..., min_length=6, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str


class UserInfo(BaseModel):
    user_id: int
    username: str
    email: str
    created_at: str
    is_admin: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: int, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Dependency: get current user from Bearer token ────────────────────────────

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency — inject into any protected route."""
    payload = decode_token(token)
    user_id = int(payload.get("sub", 0))
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[dict]:
    """Like get_current_user but returns None instead of raising (for semi-public routes)."""
    if not token:
        return None
    try:
        return get_current_user(token)
    except HTTPException:
        return None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
async def register(req: RegisterRequest):
    """Create a new user account."""
    init_db()

    if get_user_by_username(req.username):
        raise HTTPException(status_code=409, detail="Username already taken")
    if get_user_by_email(req.email):
        raise HTTPException(status_code=409, detail="Email already registered")

    hashed = hash_password(req.password)
    uid = create_user(req.username, req.email, hashed)
    token = create_access_token(uid, req.username)

    return LoginResponse(
        access_token=token,
        user_id=uid,
        username=req.username,
    )


@router.post("/login", response_model=LoginResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Authenticate and return JWT. Accepts username OR email in the username field."""
    user = get_user_by_username(form.username) or get_user_by_email(form.username)
    if not user or not verify_password(form.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user["id"], user["username"])
    return LoginResponse(
        access_token=token,
        user_id=user["id"],
        username=user["username"],
    )


@router.get("/me", response_model=UserInfo)
async def me(current_user: dict = Depends(get_current_user)):
    """Return current user's profile info."""
    return UserInfo(
        user_id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        created_at=current_user["created_at"],
        is_admin=bool(current_user.get("is_admin", 0)),
    )
