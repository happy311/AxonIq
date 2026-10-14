"""Item 1 — OTP: secure generation, HMAC storage, attempt lockout, single use, constant-time, rate limits."""
import hashlib
import re
import sqlite3

import pytest

from conftest import register, GOOD_PW


def _request_code(client, email):
    import api.routes.password as pw
    pw.SENT.clear()
    r = client.post("/auth/forgot-password", json={"email": email})
    return r, (pw.SENT[-1][1] if pw.SENT else None)


def test_otp_uses_secrets_not_random():
    import inspect
    from api import email_utils
    src = inspect.getsource(email_utils.generate_otp)
    assert "secrets.randbelow" in src and "random." not in src
    codes = {email_utils.generate_otp() for _ in range(200)}
    assert all(re.fullmatch(r"\d{6}", c) for c in codes) and len(codes) > 150


def test_otp_stored_as_salted_hmac_not_bare_sha256(client):
    register(client)
    _, otp = _request_code(client, "alice@example.com")
    from api.core.config import DB_PATH
    row = sqlite3.connect(str(DB_PATH)).execute(
        "SELECT otp_hash, otp_salt, attempts, locked FROM password_reset_otps").fetchone()
    stored, salt, attempts, locked = row
    assert salt and len(salt) >= 32
    assert stored != hashlib.sha256(otp.encode()).hexdigest()           # not a bare SHA-256
    from api.email_utils import hash_otp
    assert stored == hash_otp(otp, salt, "alice@example.com")           # HMAC(JWT_SECRET, salt|email|otp)
    assert (attempts, locked) == (0, 0)


def test_verify_uses_constant_time_compare():
    import inspect
    from api import email_utils
    assert "hmac.compare_digest" in inspect.getsource(email_utils.verify_otp)


def test_otp_locks_after_5_wrong_tries_even_with_correct_code_afterwards(client):
    register(client)
    _, otp = _request_code(client, "alice@example.com")
    wrong = "000000" if otp != "000000" else "111111"
    for _ in range(5):
        r = client.post("/auth/verify-otp", json={"email": "alice@example.com", "otp": wrong})
        assert r.status_code == 400
    # 6th try with the CORRECT code must fail — OTP is locked
    r = client.post("/auth/verify-otp", json={"email": "alice@example.com", "otp": otp})
    assert r.status_code == 400
    from api.core.config import DB_PATH
    attempts, locked = sqlite3.connect(str(DB_PATH)).execute(
        "SELECT attempts, locked FROM password_reset_otps").fetchone()
    assert locked == 1 and attempts == 5


def test_otp_single_use(client):
    register(client)
    _, otp = _request_code(client, "alice@example.com")
    r = client.post("/auth/verify-otp", json={"email": "alice@example.com", "otp": otp})
    assert r.status_code == 200 and r.json()["reset_token"]
    r2 = client.post("/auth/verify-otp", json={"email": "alice@example.com", "otp": otp})
    assert r2.status_code == 400


def test_forgot_password_response_identical_for_unknown_email(client):
    register(client)
    known, _ = _request_code(client, "alice@example.com")
    unknown, code = _request_code(client, "nobody@example.com")
    assert known.status_code == unknown.status_code == 200
    assert known.json() == unknown.json() and code is None


def test_forgot_password_rate_limited_per_email(client):
    register(client)
    codes = [client.post("/auth/forgot-password", json={"email": "alice@example.com"}).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and codes[5] == 429


def test_forgot_password_rate_limited_per_ip(client):
    codes = [client.post("/auth/forgot-password", json={"email": f"u{i}@example.com"}).status_code for i in range(12)]
    assert codes[:10] == [200] * 10 and 429 in codes[10:]


def test_verify_otp_rate_limited_per_email(client):
    register(client)
    _request_code(client, "alice@example.com")
    statuses = [client.post("/auth/verify-otp", json={"email": "alice@example.com", "otp": "123456"}).status_code
                for _ in range(12)]
    assert 429 in statuses


def test_otp_never_returned_to_client(client):
    register(client)
    r, otp = _request_code(client, "alice@example.com")
    assert otp not in r.text and "otp" not in r.json()
