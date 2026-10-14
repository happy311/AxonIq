"""
AxonIQ — Doctor routes. Every endpoint requires an ADMIN-APPROVED doctor account.

  GET /doctor/patients?email=…          search patients by e-mail (min 3 chars, contains-match)
  GET /doctor/patients/{id}/symptoms    per-session symptom picture (tier, symptoms, McDonald, MRI summary)
  GET /doctor/patients/{id}/mri         list of stored scans
  GET /doctor/mri/{scan_id}/download    DICOM zip of the scan

Every access is written to the append-only admin_audit table BEFORE data is returned;
if the audit write fails the request is refused. Chat transcripts are NOT exposed to doctors.
No `from __future__ import annotations` (FastAPI).
"""
import asyncio
import os
import tempfile
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from loguru import logger
from starlette.background import BackgroundTask

from api.auth import get_current_user, client_ip
from api.core import ratelimit
from api.core.config import (
    DOCTOR_SEARCH_MIN_CHARS, DOCTOR_SEARCH_MAX_RESULTS, DOCTOR_DOWNLOAD_RATE, MRI_STORE_DIR,
)
from api.core.limiter import limiter

router = APIRouter(prefix="/doctor", tags=["doctor"])


def require_doctor(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "doctor" or current_user.get("doctor_status") != "approved":
        raise HTTPException(status_code=403, detail="Approved doctor access required")
    return current_user


def _audit(doc: dict, request: Request, action: str, target: str = None) -> None:
    from api.database import audit
    try:
        audit(doc["id"], doc["username"], action, target, client_ip(request))
    except Exception as e:
        logger.error("[doctor] audit write failed: {}", type(e).__name__)
        raise HTTPException(status_code=500, detail="Audit trail unavailable; request refused.")


@router.get("/my-patients")
@limiter.limit("60/minute")
async def my_patients(request: Request, doc=Depends(require_doctor)):
    """Patients automatically assigned to this doctor (least-loaded rule, on MRI submission)."""
    from api.database import list_assigned_patients
    rows = list_assigned_patients(doc["id"])
    _audit(doc, request, "DOCTOR_LIST_ASSIGNED", f"results:{len(rows)}")
    return {"patients": rows}


@router.get("/patients")
@limiter.limit("60/minute")
async def search_patients(request: Request, email: str = "", doc=Depends(require_doctor)):
    from api.database import search_patients_by_email
    q = (email or "").strip().lower()
    if len(q) < DOCTOR_SEARCH_MIN_CHARS or len(q) > 254:
        raise HTTPException(status_code=400,
                            detail=f"Enter at least {DOCTOR_SEARCH_MIN_CHARS} characters of the patient's e-mail.")
    ratelimit.enforce(f"docsearch:{doc['id']}", 60, 60, detail="Too many searches — please wait.")
    rows = search_patients_by_email(q, DOCTOR_SEARCH_MAX_RESULTS)
    _audit(doc, request, "DOCTOR_SEARCH_PATIENTS", f"results:{len(rows)}")      # query text not stored (PHI-ish)
    return {"patients": rows, "limit": DOCTOR_SEARCH_MAX_RESULTS}


def _patient_or_404(pid: int) -> dict:
    from api.database import get_patient
    p = get_patient(pid)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return p


@router.get("/patients/{patient_id}/symptoms")
@limiter.limit("60/minute")
async def patient_symptoms(patient_id: int, request: Request, doc=Depends(require_doctor)):
    from api.database import get_patient_clinical_overview
    p = _patient_or_404(patient_id)
    _audit(doc, request, "DOCTOR_VIEW_SYMPTOMS", f"patient:{patient_id}")
    return {"patient": p, "sessions": get_patient_clinical_overview(patient_id)}


@router.get("/patients/{patient_id}/mri")
@limiter.limit("60/minute")
async def patient_mri(patient_id: int, request: Request, doc=Depends(require_doctor)):
    from api.database import list_scans_for_user
    p = _patient_or_404(patient_id)
    _audit(doc, request, "DOCTOR_LIST_MRI", f"patient:{patient_id}")
    return {"patient": p, "scans": list_scans_for_user(patient_id)}


def _rm(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


@router.get("/mri/{scan_id}/download")
@limiter.limit("30/minute")
async def download_mri(scan_id: int, request: Request, doc=Depends(require_doctor)):
    """Always a zip of DICOM slices (converted from the stored NIfTI)."""
    from api.database import get_scan, get_patient
    from api.mri_engine.dicom_export import make_zip_from_stored
    scan = get_scan(scan_id)
    if not scan or not get_patient(scan["user_id"]) or not os.path.isfile(scan["file_path"]):
        raise HTTPException(status_code=404, detail="Scan not found")
    ratelimit.enforce(f"docdl:{doc['id']}", 10, 3600, detail="Download limit reached — try again later.")
    _audit(doc, request, "DOCTOR_DOWNLOAD_MRI", f"scan:{scan_id} patient:{scan['user_id']}")
    try:
        when = datetime.fromisoformat(scan["created_at"])
    except Exception:
        when = datetime.now(timezone.utc)
    try:
        zpath = await asyncio.to_thread(
            make_zip_from_stored, scan["file_path"], scan.get("suffix") or ".nii.gz",
            f"AXQ{scan['user_id']}", when, str(MRI_STORE_DIR.parent / "dicom_tmp"))
    except Exception as e:
        logger.opt(exception=e).error("[doctor] DICOM export failed for scan {}", scan_id)
        raise HTTPException(status_code=500, detail="Could not prepare the DICOM export.")
    return FileResponse(zpath, media_type="application/zip",
                        filename=f"axoniq_patient{scan['user_id']}_scan{scan_id}_dicom.zip",
                        headers={"Cache-Control": "no-store"}, background=BackgroundTask(_rm, zpath))
