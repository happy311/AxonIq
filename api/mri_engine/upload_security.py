"""
Upload hardening: chunked streaming to a private file with a hard byte cap, then
real validation in a sandboxed subprocess. No user-controlled names ever touch the filesystem.
"""
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from loguru import logger

from api.core.config import (
    MRI_MAX_UPLOAD_BYTES, MRI_MAX_DECOMPRESSED, MRI_MAX_DIM, MRI_MAX_VOXELS,
    MRI_VALIDATE_TIMEOUT_S, MRI_VALIDATE_MEM_MB, BASE_DIR,
)
from api.mri_engine.config import UPLOAD_DIR

CHUNK = 1024 * 1024


def allowed_suffix(filename: str) -> str:
    """Return '.nii.gz', '.nii' or '.zip' (a zip of DICOM slices) — decided by the *claimed* name
    only to pick an extension; the real format is verified from the bytes. Raises 400 otherwise."""
    name = (filename or "").lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    if name.endswith(".zip"):
        return ".zip"
    raise HTTPException(status_code=400,
                        detail="Accepted files: NIfTI (.nii, .nii.gz) or a .zip of DICOM slices.")


def _new_path(suffix: str) -> str:
    return os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{suffix}")


async def stream_to_disk(upload: UploadFile, suffix: str, max_bytes: int = MRI_MAX_UPLOAD_BYTES) -> tuple[str, int]:
    """Copy the upload in 1 MB chunks into a fresh 0600 file; abort + delete as soon as
    max_bytes is exceeded. Returns (path, size)."""
    path = _new_path(suffix)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await upload.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413,
                                        detail=f"File too large (max {max_bytes // (1024 * 1024)} MB).")
                out.write(chunk)
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    if size == 0:
        Path(path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty file.")
    return path, size


def quick_magic_check(path: str, suffix: str) -> None:
    """Cheap pre-check before spawning the sandbox: gzip magic for .nii.gz."""
    with open(path, "rb") as f:
        head = f.read(4)
    is_gz = head[:2] == b"\x1f\x8b"
    if suffix == ".zip":
        if head[:4] not in (b"PK\x03\x04", b"PK\x05\x06"):
            raise HTTPException(status_code=400, detail="File is not a valid .zip archive.")
        return
    if suffix == ".nii.gz" and not is_gz:
        raise HTTPException(status_code=400, detail="File is not a valid gzip-compressed NIfTI (.nii.gz).")
    if suffix == ".nii" and is_gz:
        raise HTTPException(status_code=400, detail="Compressed file must be named .nii.gz.")


async def validate_in_sandbox(path: str) -> dict:
    """Run api.mri_engine.validate_nifti in a child process with RLIMIT_AS/CPU, scrubbed env,
    and a wall-clock timeout. Any non-clean outcome ⇒ 400."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(BASE_DIR),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, "-m", "api.mri_engine.validate_nifti", path,
           str(MRI_MAX_DECOMPRESSED), str(MRI_MAX_DIM), str(MRI_MAX_VOXELS),
           str(MRI_VALIDATE_MEM_MB), str(MRI_VALIDATE_TIMEOUT_S)]
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(BASE_DIR), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=MRI_VALIDATE_TIMEOUT_S + 5)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise HTTPException(status_code=400, detail="File validation timed out; the file was rejected.")
    try:
        verdict = json.loads(out.decode().strip().splitlines()[-1])
    except Exception:
        logger.warning("[upload] validator produced no verdict (rc={}) — rejecting", proc.returncode)
        raise HTTPException(status_code=400, detail="File could not be validated as a NIfTI image.")
    if not verdict.get("ok"):
        logger.info("[upload] rejected: {}", verdict.get("error"))
        raise HTTPException(status_code=400, detail=f"Invalid NIfTI file: {verdict.get('error', 'rejected')}")
    return verdict


async def convert_dicom_zip_in_sandbox(zip_path: str) -> tuple[str, int]:
    """Convert an uploaded zip of DICOM slices to a private .nii.gz (sandboxed subprocess).
    On success the zip is deleted and (nifti_path, slice_count) returned. On any failure the partial
    output is removed and a 400 raised (the caller removes the zip)."""
    from api.core.config import MRI_CONVERT_MEM_MB, MRI_CONVERT_TIMEOUT_S
    out = _new_path(".nii.gz")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(BASE_DIR),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, "-m", "api.mri_engine.dicom_import", zip_path, out,
           str(MRI_MAX_DECOMPRESSED), str(MRI_MAX_DIM), str(MRI_MAX_VOXELS),
           str(MRI_CONVERT_MEM_MB), str(MRI_CONVERT_TIMEOUT_S)]
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(BASE_DIR), env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        try:
            raw, _ = await asyncio.wait_for(proc.communicate(), timeout=MRI_CONVERT_TIMEOUT_S + 10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise HTTPException(status_code=400, detail="Converting the DICOM zip timed out; it was rejected.")
        try:
            verdict = json.loads(raw.decode().strip().splitlines()[-1])
        except Exception:
            logger.warning("[upload] dicom converter produced no verdict (rc={})", proc.returncode)
            raise HTTPException(status_code=400, detail="The zip could not be converted (it may be too large or corrupt).")
        if not verdict.get("ok"):
            logger.info("[upload] dicom zip rejected: {}", verdict.get("error"))
            raise HTTPException(status_code=400, detail=f"DICOM zip rejected: {verdict.get('error', 'invalid')}")
        Path(zip_path).unlink(missing_ok=True)          # original (may carry patient identity) is not kept
        return out, int(verdict.get("slices", 0))
    except BaseException:
        Path(out).unlink(missing_ok=True)
        raise
