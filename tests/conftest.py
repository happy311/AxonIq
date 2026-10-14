"""Test harness: isolated temp DB, dev ENV, no network, no lifespan (no LLM/RAG/MRI startup)."""
import os
import secrets
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="axoniq_test_")
os.environ.update({
    "JWT_SECRET": secrets.token_urlsafe(48),
    "DATA_DIR": _TMP,
    "NEUROCHECK_DB": str(Path(_TMP) / "test.db"),
    "ENV": "dev",
    "COOKIE_SECURE": "0",
    "MRI_WORK_DIR": str(Path(_TMP) / "mri"),
    "MRI_DEVICE": "cpu",
    "DATA_ENCRYPTION_KEY": "",
    "SMTP_EMAIL": "", "SMTP_PASSWORD": "",
})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

GOOD_PW = "Tr0ub4dor&3-horse"


@pytest.fixture(scope="session")
def app():
    from api.main import app as _app
    return _app


@pytest.fixture(autouse=True)
def fresh_state(app):
    """Fresh DB + cleared rate-limit state for every test."""
    from api import database as db
    from api.core import ratelimit
    from api.core.limiter import limiter
    p = Path(os.environ["NEUROCHECK_DB"])
    for suffix in ("", "-wal", "-shm"):
        Path(str(p) + suffix).unlink(missing_ok=True)
    db.init_all()
    ratelimit.reset()
    try:
        limiter.reset()
    except Exception:
        pass
    # OTP mail must never leave the box in tests; capture instead
    import api.routes.password as pw
    pw.SENT = []
    pw.send_otp_email = lambda to, user, otp: (pw.SENT.append((to, otp)) or True)
    yield


@pytest.fixture
def client(app):
    return TestClient(app, base_url="http://testserver")


def register(client, username="alice", email="alice@example.com", password=GOOD_PW):
    return client.post("/auth/register", json={"username": username, "email": email,
                                               "password": password, "accept_privacy": True})


def login(client, username="alice", password=GOOD_PW):
    return client.post("/auth/login", data={"username": username, "password": password})


@pytest.fixture
def make_user(client):
    def _make(username="alice", email=None, password=GOOD_PW):
        email = email or f"{username}@example.com"
        assert register(client, username, email, password).status_code == 201
        r = login(client, username, password)
        assert r.status_code == 200, r.text
        return {"token": r.json()["access_token"], "id": r.json()["user_id"],
                "h": {"Authorization": f"Bearer {r.json()['access_token']}"}, "email": email,
                "username": username}
    return _make
