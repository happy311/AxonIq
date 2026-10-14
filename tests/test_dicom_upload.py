"""Upload support: .nii, .nii.gz, and a .zip of DICOM slices (converted to NIfTI in a sandbox)."""
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from conftest import GOOD_PW, register, login

ROOT = Path(__file__).resolve().parents[1]


def _vol(tmp_path, shape=(20, 24, 12), aff=None, name="v.nii.gz"):
    rng = np.random.default_rng(1)
    arr = (rng.random(shape) * 1000).astype(np.int16)
    aff = np.array(aff) if aff is not None else np.array([[-1.2, 0, 0, 90], [0, -0.9, 0, 100], [0, 0, 3.0, -40], [0, 0, 0, 1]])
    p = tmp_path / name
    nib.save(nib.Nifti1Image(arr, aff), str(p))
    return p, arr, aff


def _to_dicom_zip(nii, tmp_path, name="d.zip"):
    from api.mri_engine.dicom_export import nifti_to_dicom_zip
    z = tmp_path / name
    nifti_to_dicom_zip(str(nii), str(z), patient_id="AXQ1")
    return z


def _convert(zip_path, out, **kw):
    args = [sys.executable, "-m", "api.mri_engine.dicom_import", str(zip_path), str(out),
            str(kw.get("unzipped", 500_000_000)), "1024", str(512 * 512 * 400), "4096", "120"]
    r = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)})
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


# ── converter correctness ─────────────────────────────────────────────────────
def test_dicom_zip_roundtrip_preserves_data_and_geometry(tmp_path):
    nii, arr, aff = _vol(tmp_path)
    z = _to_dicom_zip(nii, tmp_path)
    out = tmp_path / "out.nii.gz"
    v = _convert(z, out)
    assert v["ok"] and v["slices"] == 12 and v["shape"] == [20, 24, 12]
    orig = nib.as_closest_canonical(nib.load(str(nii)))
    back = nib.as_closest_canonical(nib.load(str(out)))
    assert np.array_equal(np.asarray(back.dataobj), np.asarray(orig.dataobj))
    assert np.allclose(back.affine, orig.affine, atol=1e-3)


def test_dicom_zip_with_rescale_and_subfolders_and_junk(tmp_path):
    nii, arr, _ = _vol(tmp_path)
    z = _to_dicom_zip(nii, tmp_path)
    # float source → export uses RescaleSlope/Intercept; add folders + junk files
    fl = tmp_path / "f.nii.gz"
    nib.save(nib.Nifti1Image((arr / 7.0).astype(np.float32), np.eye(4)), str(fl))
    z2 = _to_dicom_zip(fl, tmp_path, "f.zip")
    with zipfile.ZipFile(z2) as src, zipfile.ZipFile(tmp_path / "nested.zip", "w") as dst:
        for n in src.namelist():
            dst.writestr("study/series1/" + n if n.endswith(".dcm") else n, src.read(n))
        dst.writestr("__MACOSX/._junk", b"junk")
        dst.writestr("DICOMDIR", b"not really")
    out = tmp_path / "o2.nii.gz"
    v = _convert(tmp_path / "nested.zip", out)
    assert v["ok"]
    back = np.asarray(nib.load(str(out)).dataobj)
    assert np.allclose(back, arr / 7.0, atol=(arr / 7.0).max() / 65535 * 1.01)


def test_largest_series_is_chosen(tmp_path):
    nii, _, _ = _vol(tmp_path, shape=(16, 16, 12))
    nii2, _, _ = _vol(tmp_path, shape=(16, 16, 9), name="b.nii.gz")
    za, zb = _to_dicom_zip(nii, tmp_path, "a.zip"), _to_dicom_zip(nii2, tmp_path, "b.zip")
    with zipfile.ZipFile(tmp_path / "both.zip", "w") as dst:
        for tag, zz in (("A", za), ("B", zb)):
            with zipfile.ZipFile(zz) as src:
                for n in src.namelist():
                    if n.endswith(".dcm"):
                        dst.writestr(f"{tag}/{n.split('/')[-1]}", src.read(n))
    v = _convert(tmp_path / "both.zip", tmp_path / "o.nii.gz")
    assert v["ok"] and v["slices"] == 12


@pytest.mark.parametrize("case", ["notzip", "empty", "nodicom", "toofew", "bomb", "toomany", "dupes", "missing"])
def test_bad_zips_are_rejected_cleanly(tmp_path, case):
    nii, _, _ = _vol(tmp_path)
    good = _to_dicom_zip(nii, tmp_path)
    z = tmp_path / "bad.zip"
    if case == "notzip":
        z.write_bytes(b"this is not a zip at all")
    elif case == "empty":
        zipfile.ZipFile(z, "w").close()
    elif case == "nodicom":
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("a.txt", "hello")
    elif case == "toofew":
        with zipfile.ZipFile(good) as src, zipfile.ZipFile(z, "w") as dst:
            for n in [x for x in src.namelist() if x.endswith(".dcm")][:3]:
                dst.writestr(n, src.read(n))
    elif case == "bomb":
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
            f.writestr("big.bin", b"\0" * 60_000_000)       # ratio ≫ 250
    elif case == "toomany":
        with zipfile.ZipFile(z, "w") as f:
            for i in range(5100):
                f.writestr(f"{i}.x", b"")
    elif case == "dupes":
        with zipfile.ZipFile(good) as src, zipfile.ZipFile(z, "w") as dst:
            names = [x for x in src.namelist() if x.endswith(".dcm")]
            for i, n in enumerate(names):
                dst.writestr(n, src.read(n))
            dst.writestr("copy.dcm", src.read(names[0]))     # duplicate slice position
    elif case == "missing":
        with zipfile.ZipFile(good) as src, zipfile.ZipFile(z, "w") as dst:
            names = [x for x in src.namelist() if x.endswith(".dcm")]
            for i, n in enumerate(names):
                if i not in (4, 5, 6):
                    dst.writestr(n, src.read(n))
    out = tmp_path / "out.nii.gz"
    v = _convert(z, out)
    assert v["ok"] is False and v["error"]
    assert not out.exists()


def test_total_unzipped_limit(tmp_path):
    nii, _, _ = _vol(tmp_path)
    z = _to_dicom_zip(nii, tmp_path)
    v = _convert(z, tmp_path / "o.nii.gz", unzipped=10_000)
    assert v["ok"] is False and "too large" in v["error"]


def test_path_traversal_member_names_are_harmless(tmp_path):
    nii, _, _ = _vol(tmp_path)
    good = _to_dicom_zip(nii, tmp_path)
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(z, "w") as dst:
        for i, n in enumerate(x for x in src.namelist() if x.endswith(".dcm")):
            dst.writestr(f"../../etc/evil{i}.dcm", src.read(n))
    victim = tmp_path.parent / "etc"
    v = _convert(z, tmp_path / "o.nii.gz")
    assert v["ok"] and not victim.exists()                   # names are never used as paths


# ── endpoint ──────────────────────────────────────────────────────────────────
@pytest.fixture
def uploader(client, monkeypatch):
    import api.routes.chat as chat
    monkeypatch.setattr(chat, "_run_mri_analysis_background", lambda *a, **k: None)   # skip the heavy agent
    assert register(client, "upl", "upl@example.com").status_code == 201
    t = login(client, "upl").json()
    from api import database as db
    db.create_db_session("sess-upload01", t["user_id"], "t")
    return {"h": {"Authorization": f"Bearer {t['access_token']}"}, "uid": t["user_id"]}


def _post(client, up, path, filename):
    return client.post("/mri/upload?session_id=sess-upload01", headers=up["h"],
                       files={"flair": (filename, open(path, "rb"), "application/octet-stream")})


@pytest.mark.parametrize("ext", [".nii", ".nii.gz"])
def test_endpoint_accepts_nifti_variants(client, uploader, tmp_path, ext):
    nii, _, _ = _vol(tmp_path, name="x" + ext)
    r = _post(client, uploader, nii, "scan" + ext)
    assert r.status_code == 200, r.text
    from api import database as db
    assert db.count_scans(uploader["uid"]) == 1


def test_endpoint_accepts_dicom_zip_and_stores_converted_nifti(client, uploader, tmp_path):
    nii, arr, _ = _vol(tmp_path)
    z = _to_dicom_zip(nii, tmp_path)
    r = _post(client, uploader, z, "MY_SCAN.ZIP")
    assert r.status_code == 200, r.text
    assert r.json()["converted_from_dicom_slices"] == 12
    from api import database as db
    from api.core import crypto
    scan = db.list_scans_for_user(uploader["uid"])[0]
    stored = db.get_scan(scan["id"])
    assert stored["suffix"] == ".nii.gz"
    plain = tmp_path / "plain.nii.gz"
    crypto.decrypt_file(stored["file_path"], str(plain))
    assert nib.load(str(plain)).shape == arr.shape
    from api.mri_engine.config import UPLOAD_DIR
    assert not list(Path(UPLOAD_DIR).glob("*.zip"))            # the original zip (may hold patient identity) is gone


def test_endpoint_rejects_bad_zip_and_wrong_extension(client, uploader, tmp_path):
    from api.mri_engine.config import UPLOAD_DIR
    before = set(Path(UPLOAD_DIR).glob("*"))
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"PK\x03\x04garbage-not-really-a-zip")
    r = _post(client, uploader, bad, "bad.zip")
    assert r.status_code == 400 and "rejected" in r.json()["detail"].lower()
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    assert _post(client, uploader, txt, "a.txt").status_code == 400
    fake = tmp_path / "fake.zip"
    fake.write_bytes(b"not a zip")
    assert _post(client, uploader, fake, "fake.zip").status_code == 400
    assert set(Path(UPLOAD_DIR).glob("*")) == before            # nothing left behind
    from api import database as db
    assert db.count_scans(uploader["uid"]) == 0
