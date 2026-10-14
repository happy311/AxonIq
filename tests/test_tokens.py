"""Items 2 & 3 — token typing/jti/single-use reset tokens; token_version invalidation; lifetimes."""
import time

import pytest
from jose import jwt

from conftest import register, login, GOOD_PW

NEW_PW = "Another-Str0ng-Pass!9"


def _reset_token(client, email="alice@example.com"):
    import api.routes.password as pw
    pw.SENT.clear()
    client.post("/auth/forgot-password", json={"email": email})
    otp = pw.SENT[-1][1]
    r = client.post("/auth/verify-otp", json={"email": email, "otp": otp})
    return r.json()["reset_token"]


def _claims(token):
    from api.core.config import JWT_SECRET
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])


def test_access_token_claims(make_user):
    u = make_user()
    c = _claims(u["token"])
    assert c["typ"] == "access" and c["jti"] and "tv" in c
    from api.core.config import JWT_EXPIRE_MINUTES
    assert JWT_EXPIRE_MINUTES <= 1440
    assert c["exp"] - c["iat"] <= 24 * 3600


def test_reset_token_claims_and_ttl(client, make_user):
    make_user()
    c = _claims(_reset_token(client))
    assert c["typ"] == "reset" and c["jti"]
    assert c["exp"] - c["iat"] == 10 * 60


def test_reset_token_rejected_by_get_current_user(client, make_user):
    make_user()
    rt = _reset_token(client)
    for path in ("/auth/me", "/user/sessions"):
        r = client.get(path, headers={"Authorization": f"Bearer {rt}"})
        assert r.status_code == 401


def test_access_token_rejected_by_reset_endpoint(client, make_user):
    u = make_user()
    r = client.post("/auth/reset-password", json={"reset_token": u["token"], "new_password": NEW_PW})
    assert r.status_code == 400


def test_token_without_typ_is_rejected(client, make_user):
    u = make_user()
    from api.core.config import JWT_SECRET
    legacy = jwt.encode({"sub": str(u["id"]), "username": "alice", "exp": time.time() + 600}, JWT_SECRET, algorithm="HS256")
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {legacy}"}).status_code == 401


def test_reset_token_is_single_use(client, make_user):
    make_user()
    rt = _reset_token(client)
    assert client.post("/auth/reset-password", json={"reset_token": rt, "new_password": NEW_PW}).status_code == 200
    again = client.post("/auth/reset-password", json={"reset_token": rt, "new_password": "Third-Str0ng-Pass#7"})
    assert again.status_code == 400
    assert login(client, "alice", NEW_PW).status_code == 200            # first reset stuck


def test_weak_password_does_not_burn_reset_token(client, make_user):
    make_user()
    rt = _reset_token(client)
    assert client.post("/auth/reset-password", json={"reset_token": rt, "new_password": "short"}).status_code == 400
    assert client.post("/auth/reset-password", json={"reset_token": rt, "new_password": NEW_PW}).status_code == 200


def test_expired_reset_token_rejected(client, make_user):
    u = make_user()
    from api.core.config import JWT_SECRET
    expired = jwt.encode({"sub": str(u["id"]), "typ": "reset", "jti": "abc123abc123", "tv": 0,
                          "exp": time.time() - 5}, JWT_SECRET, algorithm="HS256")
    assert client.post("/auth/reset-password", json={"reset_token": expired, "new_password": NEW_PW}).status_code == 400


def test_password_reset_invalidates_old_access_tokens(client, make_user):
    u = make_user()
    assert client.get("/auth/me", headers=u["h"]).status_code == 200
    rt = _reset_token(client)
    client.post("/auth/reset-password", json={"reset_token": rt, "new_password": NEW_PW})
    assert client.get("/auth/me", headers=u["h"]).status_code == 401    # issued before the reset


def test_change_password_invalidates_old_tokens_but_not_the_fresh_session(client, make_user):
    u = make_user()
    r = client.post("/auth/change-password", headers=u["h"],
                    json={"old_password": GOOD_PW, "new_password": NEW_PW})
    assert r.status_code == 200
    assert client.get("/auth/me", headers=u["h"]).status_code == 401
    assert client.get("/auth/me").status_code == 200                    # cookie was re-issued with new tv


def test_admin_reset_invalidates_tokens_and_returns_no_password(client, make_user):
    admin = make_user("boss")
    target = make_user("carol")
    from api.database import set_admin
    set_admin("boss", True)
    r = client.post(f"/admin/user/{target['id']}/reset-password", headers=admin["h"])
    assert r.status_code == 200
    body = r.json()
    assert "new_password" not in body and "password" not in str(body).lower().replace("old password", "")
    assert client.get("/auth/me", headers=target["h"]).status_code == 401
    assert login(client, "carol", GOOD_PW).status_code == 401           # old password dead
    # dev_link carries a single-use reset token the user must use
    token = body["dev_link"].split("#reset=")[1]
    assert client.post("/auth/reset-password", json={"reset_token": token, "new_password": NEW_PW}).status_code == 200
    assert login(client, "carol", NEW_PW).status_code == 200


def test_token_version_bumped_everywhere(client, make_user):
    from api.database import get_user_by_id
    u = make_user()
    v0 = get_user_by_id(u["id"])["token_version"]
    client.post("/auth/change-password", headers=u["h"], json={"old_password": GOOD_PW, "new_password": NEW_PW})
    assert get_user_by_id(u["id"])["token_version"] == v0 + 1
