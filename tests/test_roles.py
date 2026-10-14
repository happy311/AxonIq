"""Doctor role, admin-from-env, admin approval, doctor patient/MRI access, DICOM zip."""
import io
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import pytest

from conftest import GOOD_PW, register, login

ADMIN_PW = "Adm1n-Str0ng-Passphrase!"


def _env_admin(monkeypatch, name="sysadmin", pw=ADMIN_PW, email="sysadmin@example.org"):
    from api.core import config
    monkeypatch.setattr(config, "ADMIN_USERNAME", name)
    monkeypatch.setattr(config, "ADMIN_PASSWORD", pw)
    monkeypatch.setattr(config, "ADMIN_EMAIL", email)
    from api.core.bootstrap import ensure_env_admin
    ensure_env_admin()


def _h(r):
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _reg_doctor(client, name="drsmith", email="drsmith@hospital.org", **kw):
    body = {"username": name, "email": email, "password": GOOD_PW, "accept_privacy": True,
            "role": "doctor", "license_number": "MCI-12345", "institution": "City Hospital"}
    body.update(kw)
    return client.post("/auth/register", json=body)


@pytest.fixture
def admin(client, monkeypatch):
    _env_admin(monkeypatch)
    r = login(client, "sysadmin", ADMIN_PW)
    assert r.status_code == 200 and r.json()["role"] == "admin"
    return _h(r)


@pytest.fixture
def approved_doctor(client, admin):
    assert _reg_doctor(client).status_code == 201
    did = [d for d in client.get("/admin/doctors", headers=admin).json()["doctors"]][0]["id"]
    assert client.post(f"/admin/doctors/{did}/approve", headers=admin).status_code == 200
    r = login(client, "drsmith")
    assert r.status_code == 200 and r.json()["role"] == "doctor"
    return {"h": _h(r), "id": did}


# ── admin from env ────────────────────────────────────────────────────────────
def test_env_admin_created_and_can_use_admin_api(client, admin):
    assert client.get("/admin/users", headers=admin).status_code == 200
    me = client.get("/auth/me", headers=admin).json()
    assert me["is_admin"] and me["role"] == "admin"


def test_env_admin_weak_password_aborts(monkeypatch):
    with pytest.raises(RuntimeError):
        _env_admin(monkeypatch, pw="short")


def test_env_admin_refuses_to_promote_existing_squatter(client, monkeypatch):
    register(client, "sysadmin", "x@example.com")
    with pytest.raises(RuntimeError):
        _env_admin(monkeypatch)
    assert login(client, "sysadmin").status_code == 200       # still just a patient
    t = login(client, "sysadmin").json()["access_token"]
    assert client.get("/admin/users", headers={"Authorization": f"Bearer {t}"}).status_code == 403


def test_env_admin_password_resync_and_rotation(client, monkeypatch):
    _env_admin(monkeypatch)
    _env_admin(monkeypatch, pw="Another-Str0ng-Passphrase#2")
    assert login(client, "sysadmin", ADMIN_PW).status_code == 401
    assert login(client, "sysadmin", "Another-Str0ng-Passphrase#2").status_code == 200
    _env_admin(monkeypatch, name="otheradmin", email="o@example.org")       # rename ⇒ old one demoted
    t = login(client, "sysadmin", "Another-Str0ng-Passphrase#2").json()["access_token"]
    assert client.get("/admin/users", headers={"Authorization": f"Bearer {t}"}).status_code == 403


# ── doctor registration / approval ───────────────────────────────────────────
def test_doctor_cannot_login_until_approved(client, admin):
    r = _reg_doctor(client)
    assert r.status_code == 201 and "administrator" in r.json()["message"]
    assert login(client, "drsmith").status_code == 403
    assert login(client, "drsmith", "wrong-password-xx").status_code == 401   # no status leak on bad password
    did = client.get("/admin/doctors?status=pending", headers=admin).json()["doctors"][0]["id"]
    assert client.post(f"/admin/doctors/{did}/approve", headers=admin).status_code == 200
    assert login(client, "drsmith").status_code == 200


def test_doctor_registration_requires_licence_and_institution(client):
    assert _reg_doctor(client, license_number=None).status_code == 422
    assert _reg_doctor(client, institution="").status_code == 422


def test_cannot_self_register_as_admin(client):
    r = client.post("/auth/register", json={"username": "evil", "email": "e@example.com", "password": GOOD_PW,
                                            "accept_privacy": True, "role": "admin"})
    assert r.status_code == 422


def test_non_admin_cannot_manage_doctors(client, admin, make_user):
    _reg_doctor(client)
    did = client.get("/admin/doctors", headers=admin).json()["doctors"][0]["id"]
    u = make_user("pat1")
    assert client.post(f"/admin/doctors/{did}/approve", headers=u["h"]).status_code == 403
    assert client.get("/admin/doctors", headers=u["h"]).status_code == 403


def test_remove_doctor_kills_live_session_immediately(client, admin, approved_doctor):
    assert client.get("/doctor/patients?email=abc", headers=approved_doctor["h"]).status_code == 200
    assert client.post(f"/admin/doctors/{approved_doctor['id']}/remove", headers=admin).status_code == 200
    assert client.get("/doctor/patients?email=abc", headers=approved_doctor["h"]).status_code == 401
    assert login(client, "drsmith").status_code == 403
    assert client.post(f"/admin/doctors/{approved_doctor['id']}/approve", headers=admin).status_code == 200   # re-permit
    assert login(client, "drsmith").status_code == 200


def test_approve_endpoint_only_touches_doctors(client, admin, make_user):
    u = make_user("pat2")
    assert client.post(f"/admin/doctors/{u['id']}/approve", headers=admin).status_code == 404


# ── doctor access ─────────────────────────────────────────────────────────────
def test_patient_and_anonymous_blocked_from_doctor_routes(client, make_user, admin):
    u = make_user("pat3")
    assert client.get("/doctor/patients?email=pat", headers=u["h"]).status_code == 403
    from fastapi.testclient import TestClient
    anon = TestClient(client.app, base_url="http://testserver")          # no cookies
    assert anon.get("/doctor/patients?email=pat").status_code == 401
    assert client.get("/doctor/mri/1/download", headers=u["h"]).status_code == 403
    assert client.get("/doctor/patients?email=pat", headers=admin).status_code == 403   # admin ≠ doctor


def test_doctor_cannot_use_admin_routes(client, approved_doctor):
    assert client.get("/admin/users", headers=approved_doctor["h"]).status_code == 403


def test_doctor_search_by_email_patients_only(client, approved_doctor, make_user):
    make_user("bob", "bob.jones@example.com")
    make_user("carol", "carol@example.com")
    h = approved_doctor["h"]
    r = client.get("/doctor/patients?email=bob.jo", headers=h).json()["patients"]
    assert [p["email"] for p in r] == ["bob.jones@example.com"]
    assert client.get("/doctor/patients?email=hospital", headers=h).json()["patients"] == []   # doctors not listed
    assert client.get("/doctor/patients?email=ab", headers=h).status_code == 400               # too short
    assert client.get("/doctor/patients?email=%25%25%25", headers=h).json()["patients"] == []   # wildcards are literal


def test_doctor_sees_symptoms_not_transcript(client, approved_doctor, make_user):
    from api import database as db
    u = make_user("dave", "dave@example.com")
    db.create_db_session("sess-1234abcd", u["id"], "title")
    db.save_message("sess-1234abcd", "human", "SECRET-CHAT-TEXT my left eye hurts")
    db.update_session_state("sess-1234abcd", "gathering", "HIGH", ["optic_neuritis"], dis_regions=["optic"],
                            dit_episodes=1, symptom_timeline=[], human_turn=1)
    with db.get_conn() as c:
        c.execute("UPDATE chat_sessions SET ms_tier='HIGH', ms_tier_label='High risk', ms_symptoms=? WHERE session_uuid=?",
                  ('["Optic neuritis", "Numbness"]', "sess-1234abcd"))
    r = client.get(f"/doctor/patients/{u['id']}/symptoms", headers=approved_doctor["h"])
    assert r.status_code == 200
    s = r.json()["sessions"][0]
    assert s["symptoms"] == ["Optic neuritis", "Numbness"] and s["concluded_tier"] == "HIGH"
    assert "SECRET-CHAT-TEXT" not in r.text
    assert client.get("/doctor/patients/999999/symptoms", headers=approved_doctor["h"]).status_code == 404


def _make_nifti(path: Path):
    arr = (np.random.rand(16, 20, 5) * 1000).astype(np.int16)
    aff = np.diag([1.2, 0.9, 3.0, 1.0])
    nib.save(nib.Nifti1Image(arr, aff), str(path))
    return arr


def test_mri_list_and_dicom_zip_download_and_audit(client, approved_doctor, make_user, admin, tmp_path):
    from api import database as db
    from api.mri_engine.scan_store import store_scan
    u = make_user("erin", "erin@example.com")
    db.create_db_session("sess-erin0001", u["id"], "t")
    src = tmp_path / "scan.nii.gz"
    arr = _make_nifti(src)
    sid = store_scan(str(src), u["id"], "sess-erin0001", src.stat().st_size, ".nii.gz")
    h = approved_doctor["h"]
    scans = client.get(f"/doctor/patients/{u['id']}/mri", headers=h).json()["scans"]
    assert [s["id"] for s in scans] == [sid]
    assert "file_path" not in scans[0]
    r = client.get(f"/doctor/mri/{sid}/download", headers=h)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "erin" not in r.headers.get("content-disposition", "")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    dcms = sorted(n for n in z.namelist() if n.endswith(".dcm"))
    assert len(dcms) == arr.shape[2]
    ds = pydicom.dcmread(io.BytesIO(z.read(dcms[0])))
    assert ds.Modality == "MR" and ds.Rows * ds.Columns == arr.shape[0] * arr.shape[1]
    assert "erin" not in str(ds.PatientName).lower() and "erin@" not in str(ds)
    assert ds.pixel_array.shape == (ds.Rows, ds.Columns)
    audit = [e["action"] for e in client.get("/admin/audit", headers=admin).json()["entries"]]
    assert "DOCTOR_DOWNLOAD_MRI" in audit
    # stored bytes are not the raw NIfTI when a key is configured; at minimum the file is private
    stored = Path(db.get_scan(sid)["file_path"])
    assert oct(stored.stat().st_mode & 0o777) == "0o600"


def test_scan_files_removed_when_session_deleted(make_user, tmp_path):
    from api import database as db
    from api.mri_engine.scan_store import store_scan
    u = make_user("frank", "frank@example.com")
    db.create_db_session("sess-frank001", u["id"], "t")
    src = tmp_path / "s.nii.gz"
    _make_nifti(src)
    sid = store_scan(str(src), u["id"], "sess-frank001", 1, ".nii.gz")
    path = db.get_scan(sid)["file_path"]
    files = db.delete_db_session("sess-frank001")
    assert path in files and db.get_scan(sid) is None


def test_file_encryption_roundtrip_with_key(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    from api.core import crypto
    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    crypto.reload_key()
    try:
        plain = tmp_path / "p.bin"
        data = b"A" * (9 * 1024 * 1024 + 123)
        plain.write_bytes(data)
        enc, dec = tmp_path / "e.bin", tmp_path / "d.bin"
        crypto.encrypt_file(str(plain), str(enc))
        assert enc.read_bytes()[:4] == b"AXQ1" and data[:100] not in enc.read_bytes()
        crypto.decrypt_file(str(enc), str(dec))
        assert dec.read_bytes() == data
    finally:
        monkeypatch.delenv("DATA_ENCRYPTION_KEY")
        crypto.reload_key()


# ── least-loaded assignment ───────────────────────────────────────────────────
def _scan(db, uid, sess, tmp_path):
    from api.mri_engine.scan_store import store_scan
    db.create_db_session(sess, uid, "t")
    src = tmp_path / f"{sess}.nii.gz"
    _make_nifti(src)
    return store_scan(str(src), uid, sess, 1, ".nii.gz")


def _patients(prefix, n):
    """Create patients directly (the register endpoint is rate-limited to 5/hour/IP)."""
    from api import database as db
    from api.core.passwords import hash_password
    return [{"id": db.create_user(f"{prefix}{i}", f"{prefix}{i}@example.com", hash_password(GOOD_PW))}
            for i in range(n)]


def _approved(client, admin, name):
    assert _reg_doctor(client, name, f"{name}@hospital.org").status_code == 201
    did = [d for d in client.get("/admin/doctors", headers=admin).json()["doctors"] if d["username"] == name][0]["id"]
    r = client.post(f"/admin/doctors/{did}/approve", headers=admin)
    assert r.status_code == 200
    return did


def test_assignment_balances_to_least_loaded_and_is_sticky(client, admin, make_user, tmp_path):
    from api import database as db
    d1, d2 = _approved(client, admin, "doca"), _approved(client, admin, "docb")
    pats = _patients("pat", 5)
    got = [db.assign_patient(p["id"]) for p in pats]
    counts = {d1: got.count(d1), d2: got.count(d2)}
    assert sorted(counts.values()) == [2, 3]                 # never differs by more than 1
    assert got[0] == d1 and got[1] == d2                     # tie → never-assigned/lowest id first
    assert [db.assign_patient(p["id"]) for p in pats] == got  # sticky: same doctor again, no double count
    assert db.assign_patient(pats[0]["id"]) == got[0]


def test_assignment_picks_doctor_with_fewest_after_uneven_load(client, admin, make_user):
    from api import database as db
    d1, d2 = _approved(client, admin, "docc"), _approved(client, admin, "docd")
    ps = _patients("qpt", 4)
    with db.get_conn() as c:     # force d1 to carry 3 patients, d2 to carry 0
        for p in ps[:3]:
            c.execute("INSERT INTO assignments VALUES (?,?,?)", (p["id"], d1, "2026-01-01T00:00:00"))
    assert db.assign_patient(ps[3]["id"]) == d2


def test_upload_flow_assigns_and_doctor_sees_in_my_patients(client, admin, make_user, tmp_path):
    from api import database as db
    did = _approved(client, admin, "doce")
    dr = _h(login(client, "doce"))
    u = make_user("gina", "gina@example.com")
    _scan(db, u["id"], "sess-gina0001", tmp_path)
    db.assign_patient(u["id"])
    mine = client.get("/doctor/my-patients", headers=dr).json()["patients"]
    assert [p["email"] for p in mine] == ["gina@example.com"] and mine[0]["scan_count"] == 1


def test_waiting_patients_assigned_when_first_doctor_approved(client, admin, make_user, tmp_path):
    from api import database as db
    u = make_user("hank", "hank@example.com")
    _scan(db, u["id"], "sess-hank0001", tmp_path)
    assert db.assign_patient(u["id"]) is None               # no approved doctor yet → waits
    did = _approved(client, admin, "docf")
    assert [p["id"] for p in db.list_assigned_patients(did)] == [u["id"]]


def test_removed_doctors_patients_move_to_remaining_doctor(client, admin, make_user, tmp_path):
    from api import database as db
    d1, d2 = _approved(client, admin, "docg"), _approved(client, admin, "doch")
    ps = _patients("rpt", 4)
    for i, p in enumerate(ps):
        _scan(db, p["id"], f"sess-rrrr000{i}", tmp_path)
        db.assign_patient(p["id"])
    assert client.post(f"/admin/doctors/{d1}/remove", headers=admin).json()["patients_assigned"] >= 1
    assert len(db.list_assigned_patients(d1)) == 0 and len(db.list_assigned_patients(d2)) == 4
    doctors = {d["id"]: d for d in client.get("/admin/doctors", headers=admin).json()["doctors"]}
    assert doctors[d2]["assigned_count"] == 4


def test_my_patients_requires_doctor(client, make_user):
    u = make_user("ivy")
    assert client.get("/doctor/my-patients", headers=u["h"]).status_code == 403
