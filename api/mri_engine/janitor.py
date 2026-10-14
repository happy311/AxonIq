"""
AxonIQ — janitor: keeps temp files, stale job rows and expired data from accumulating.
run_startup(): after a crash/restart nothing is still processing (state is in-memory) →
               fail orphaned jobs and wipe the private upload dir.
run_once():    every 10 min — time out stale jobs, delete orphaned upload files/dirs,
               prune security tables, enforce the retention policy.
"""
import glob
import os
import shutil
import tempfile
import time
from pathlib import Path

from loguru import logger

from api.core.config import MRI_JOB_STALE_MINUTES, RETENTION_DAYS, DELETED_SESSION_PURGE_DAYS
from api.mri_engine.config import UPLOAD_DIR


def _unlink_all(paths) -> None:
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


def _sweep(directory: str, older_than_s: float, pattern: str = "*") -> int:
    n, cutoff = 0, time.time() - older_than_s
    for p in glob.glob(os.path.join(directory, pattern)):
        try:
            if older_than_s <= 0 or os.path.getmtime(p) < cutoff:
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.unlink(p)
                n += 1
        except OSError:
            pass
    return n


def run_startup() -> None:
    from api import database as db
    failed = db.fail_all_processing_mri_jobs()
    wiped = _sweep(UPLOAD_DIR, 0)
    _sweep(tempfile.gettempdir(), 0, "mricase_*")
    run_once()
    logger.info("[janitor] startup: {} orphaned job(s) failed, {} leftover upload file(s) removed", failed, wiped)


def run_once() -> None:
    from api import database as db
    stale = db.fail_stale_mri_jobs(MRI_JOB_STALE_MINUTES)
    old_jobs = db.delete_old_mri_jobs(7)
    db.housekeeping_db()
    purged, files = db.purge_old_data(RETENTION_DAYS, DELETED_SESSION_PURGE_DAYS)
    _unlink_all(files)
    # upload staging: anything older than the stale-job window is orphaned by definition
    swept = _sweep(UPLOAD_DIR, MRI_JOB_STALE_MINUTES * 60 + 300)
    swept += _sweep(tempfile.gettempdir(), 2 * 3600, "mricase_*")
    if stale or old_jobs or purged or swept:
        logger.info("[janitor] stale_jobs={} old_job_rows={} purged_sessions={} swept_files={}",
                    stale, old_jobs, purged, swept)
