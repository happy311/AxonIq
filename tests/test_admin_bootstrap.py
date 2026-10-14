"""Item 6 — no name-based admin bootstrap; CLI-only grant; append-only audit trail."""
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import register, login

ROOT = Path(__file__).resolve().parents[1]


def test_startup_has_no_name_based_admin_grant():
    """Admin comes only from the env-seeded, env_managed account (api/core/bootstrap.py);
    main.py itself never promotes a registered username."""
    src = (ROOT / "api" / "main.py").read_text()
    assert "set_admin" not in src.replace("make_admin.py", "")
    assert "ensure_env_admin" in src


def test_registering_a_name_never_confers_admin(client, monkeypatch):
    register(client, "root", "root@example.com")
    r = login(client, "root")
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"})
    assert me.json()["is_admin"] is False
    assert client.get("/admin/users", headers={"Authorization": f"Bearer {r.json()['access_token']}"}).status_code == 403


def test_cli_grant_works_and_is_audited(client):
    register(client, "dana", "dana@example.com")
    env = {"PATH": "/usr/bin:/bin", "JWT_SECRET": __import__("os").environ["JWT_SECRET"],
           "NEUROCHECK_DB": __import__("os").environ["NEUROCHECK_DB"], "DATA_DIR": __import__("os").environ["DATA_DIR"],
           "ENV": "dev", "PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, "make_admin.py", "dana", "--yes"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    r = login(client, "dana")
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert client.get("/auth/me", headers=h).json()["is_admin"] is True
    from api.database import get_audit_entries
    assert any(e["action"] == "CLI_GRANT_ADMIN" for e in get_audit_entries())


def test_cli_refuses_unknown_user_and_bad_confirmation(client):
    env = {"PATH": "/usr/bin:/bin", "JWT_SECRET": __import__("os").environ["JWT_SECRET"],
           "NEUROCHECK_DB": __import__("os").environ["NEUROCHECK_DB"], "DATA_DIR": __import__("os").environ["DATA_DIR"],
           "ENV": "dev", "PYTHONPATH": str(ROOT)}
    assert subprocess.run([sys.executable, "make_admin.py", "ghost", "--yes"], cwd=ROOT, env=env,
                          capture_output=True, text=True).returncode == 1
    register(client, "erin", "erin@example.com")
    p = subprocess.run([sys.executable, "make_admin.py", "erin"], cwd=ROOT, env=env, input="wrong\n",
                       capture_output=True, text=True)
    assert p.returncode == 1


def test_every_admin_action_is_audited_with_ip(client, make_user):
    admin = make_user("boss")
    victim = make_user("frank")
    from api.database import set_admin, get_audit_entries
    set_admin("boss", True)
    client.get("/admin/users", headers=admin["h"])
    client.get(f"/admin/user/{victim['id']}/sessions", headers=admin["h"])
    client.get("/admin/logs", headers=admin["h"])
    client.get(f"/admin/user/{victim['id']}/logs", headers=admin["h"])
    client.post(f"/admin/user/{victim['id']}/reset-password", headers=admin["h"])
    entries = get_audit_entries()
    actions = {e["action"] for e in entries}
    assert {"ADMIN_LIST_USERS", "ADMIN_READ_PATIENT_CHATS", "ADMIN_VIEW_ALL_LOGS",
            "ADMIN_VIEW_USER_LOGS", "ADMIN_RESET_PASSWORD"} <= actions
    e = next(e for e in entries if e["action"] == "ADMIN_READ_PATIENT_CHATS")
    assert e["admin_id"] == admin["id"] and e["target"] == f"user:{victim['id']}" and e["ip"] and e["created_at"]


def test_audit_table_is_append_only():
    from api.database import audit
    from api.core.config import DB_PATH
    audit(1, "x", "TEST", "t", "1.2.3.4")
    con = sqlite3.connect(str(DB_PATH))
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("UPDATE admin_audit SET action='HACKED'")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("DELETE FROM admin_audit")


def test_non_admin_cannot_use_admin_routes(client, make_user):
    u = make_user()
    for path in ("/admin/users", "/admin/logs", "/admin/audit", "/admin/health"):
        assert client.get(path, headers=u["h"]).status_code == 403
