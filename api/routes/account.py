"""
AxonIQ — account / data-subject routes.

  GET    /user/me/export   — everything we hold about you (JSON download)
  DELETE /user/me          — permanent erasure of the account and ALL data (needs password)

No `from __future__ import annotations` (FastAPI).
"""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from loguru import logger
from pydantic import BaseModel, Field

from api.auth import get_current_user_allow_pw_change, clear_auth_cookies, client_ip
from api.core import ratelimit
from api.core.limiter import limiter
from api.core.passwords import verify_password

router = APIRouter(tags=["account"])


class DeleteAccountRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=128)


@router.get("/user/me/export")
@limiter.limit("5/hour")
async def export_my_data(request: Request, current_user: dict = Depends(get_current_user_allow_pw_change)):
    from api.database import export_user_data, log_action
    log_action("DATA_EXPORT", user_id=current_user["id"], username=current_user["username"],
               ip_address=client_ip(request))
    data = export_user_data(current_user["id"])
    from fastapi.responses import JSONResponse
    return JSONResponse(data, headers={"Content-Disposition": 'attachment; filename="axoniq_my_data.json"'})


@router.delete("/user/me")
@limiter.limit("3/hour")
async def delete_my_account(request: Request, response: Response, body: DeleteAccountRequest,
                            current_user: dict = Depends(get_current_user_allow_pw_change)):
    """Hard-deletes messages, sessions, MRI results, queued files, logs and the user row.
    Re-authenticates with the password first (a stolen session alone can't wipe an account)."""
    from api.database import delete_user_everything, audit
    ratelimit.enforce(f"delacct:{current_user['id']}", 3, 3600)
    if not verify_password(body.password, current_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Password is incorrect")
    uid = current_user["id"]
    for f in delete_user_everything(uid):
        try:
            Path(f).unlink(missing_ok=True)
        except OSError:
            pass
    audit(None, None, "USER_SELF_DELETE", f"user:{uid}", client_ip(request))   # no PHI, no name
    clear_auth_cookies(response)
    logger.info("[account] user id={} erased", uid)
    return {"deleted": True}
