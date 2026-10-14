"""Item 11 — require_session_owner: 404 missing/deleted, 403 someone else's; applied everywhere."""
import pytest


@pytest.fixture
def two_users(make_user):
    return make_user("alice"), make_user("bob")


@pytest.fixture
def alice_session(two_users):
    from api.database import create_db_session
    alice, _ = two_users
    sid = "11111111-2222-3333-4444-555555555555"
    create_db_session(sid, alice["id"])
    from api.database import save_message
    save_message(sid, "human", "hello")
    return sid


READ_ROUTES = [
    "/mri/status/{sid}", "/mri/result/{sid}", "/session/{sid}/export",
    "/session/{sid}/history", "/session/{sid}/tier-log",
]


@pytest.mark.parametrize("route", READ_ROUTES)
def test_other_users_get_403(client, two_users, alice_session, route):
    _, bob = two_users
    assert client.get(route.format(sid=alice_session), headers=bob["h"]).status_code == 403


@pytest.mark.parametrize("route", READ_ROUTES)
def test_missing_session_gets_404(client, two_users, route):
    alice, _ = two_users
    assert client.get(route.format(sid="99999999-0000-0000-0000-000000000000"), headers=alice["h"]).status_code == 404


@pytest.mark.parametrize("route", READ_ROUTES)
def test_malformed_session_id_gets_404(client, two_users, route):
    alice, _ = two_users
    assert client.get(route.format(sid="..%2Fetc"), headers=alice["h"]).status_code in (404, 422)


def test_owner_can_read_history(client, two_users, alice_session):
    alice, _ = two_users
    r = client.get(f"/session/{alice_session}/history", headers=alice["h"])
    assert r.status_code == 200 and r.json()["messages"][0]["content"] == "hello"


def test_chat_403_on_foreign_session_and_404_on_missing(client, two_users, alice_session):
    alice, bob = two_users
    assert client.post("/chat", headers=bob["h"], json={"message": "hi", "session_id": alice_session}).status_code == 403
    assert client.post("/chat", headers=alice["h"],
                       json={"message": "hi", "session_id": "99999999-0000-0000-0000-000000000000"}).status_code == 404


def test_deleted_session_is_404_not_reanimated(client, two_users, alice_session):
    alice, _ = two_users
    assert client.delete(f"/session/{alice_session}", headers=alice["h"]).status_code == 200
    assert client.get(f"/session/{alice_session}/history", headers=alice["h"]).status_code == 404
    assert client.post("/chat", headers=alice["h"], json={"message": "hi", "session_id": alice_session}).status_code == 404


def test_no_inline_owner_checks_remain():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "api" / "routes" / "chat.py").read_text()
    assert "owner is not None and owner !=" not in src
    assert src.count("require_session_owner(") >= 6


def test_unauthenticated_requests_rejected(client, alice_session):
    client.cookies.clear()      # make_user logged in via cookie too
    assert client.get(f"/session/{alice_session}/history").status_code == 401
