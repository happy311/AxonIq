"""
In-process replacement for the old Flask /predict + /result endpoints.

Previously node_mri.py POSTed a file over HTTP to a separate Kaggle server
and polled a /result/<case_id> URL. Now everything runs in one process, so
submission is just "hand the file to a background thread" and polling is
just "read a dict" — no network, no tunnel, no MRI_SERVICE_URL.

IMPORTANT (single-worker requirement): job state and loaded models live in
this process's memory. If you run Uvicorn/Gunicorn with more than one
worker process, a request handled by worker B won't see a job submitted on
worker A. Run this app with a single worker (see the deployment README) —
CPU inference is the bottleneck anyway, so extra worker processes wouldn't
add throughput, only duplicate the ~3 loaded models in RAM.
"""
from __future__ import annotations
import datetime
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from api.mri_engine.config import MAX_CONCURRENT_JOBS, DEVICE
from api.mri_engine import pipeline

_job_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="mri-case-worker")
_jobs: dict = {}   # case_id -> {"status": "processing"|"done"|"error", "report": ..., "error": ...}
_jobs_lock = threading.Lock()


def _run_job(flair_path: str, case_id: str, tmp_dir: str):
    try:
        report = pipeline.run_pipeline(flair_path, case_id)
        report.pop("_output_files", None)   # local server paths -- not useful to the caller
        with _jobs_lock:
            _jobs[case_id] = {"status": "done", "report": report}
    except Exception as e:
        logger.opt(exception=e).error("[MRI Job {}] Pipeline failed", case_id)
        with _jobs_lock:
            _jobs[case_id] = {"status": "error", "error": "Pipeline failed (see server log)"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def submit(flair_path: str, tmp_dir: str, case_id: str | None = None) -> dict:
    """Submit a FLAIR file (already saved to disk) for background processing.
    Returns {"case_id": ..., "status": "processing"} immediately, mirroring
    the old server's POST /predict response — or {"error": ...} if the engine
    isn't ready."""
    if not pipeline.is_ready():
        err = pipeline.init_error() or "MRI engine is still starting up"
        return {"error": err}

    case_id = case_id or datetime.datetime.utcnow().strftime("case_%Y%m%dT%H%M%S%f")
    with _jobs_lock:
        if len(_jobs) > 200:      # keep the in-memory table bounded (results are persisted in the DB)
            for k in [k for k, j in _jobs.items() if j["status"] != "processing"][: len(_jobs) - 100]:
                _jobs.pop(k, None)
        _jobs[case_id] = {"status": "processing"}
    _job_pool.submit(_run_job, flair_path, case_id, tmp_dir)
    return {"case_id": case_id, "status": "processing"}


def get_status(case_id: str) -> dict | None:
    """Mirrors the old server's GET /result/<case_id> response shape:
    {"status": "processing"} | {"status": "done", ...report fields} |
    {"status": "error", "error": "..."} | None if unknown case_id."""
    with _jobs_lock:
        job = _jobs.get(case_id)

    if job is None:
        return None
    if job["status"] == "processing":
        return {"case_id": case_id, "status": "processing"}
    if job["status"] == "error":
        return {"case_id": case_id, "status": "error", "error": job["error"]}
    return {"status": "done", **job["report"]}


def health() -> dict:
    with _jobs_lock:
        n_processing = sum(1 for j in _jobs.values() if j["status"] == "processing")
    return {
        "ready": pipeline.is_ready(),
        "init_error": pipeline.init_error(),
        "device": DEVICE,
        "n_models_in_ensemble": pipeline.n_models(),
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "jobs_processing": n_processing,
    }
