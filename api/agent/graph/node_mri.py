"""
AxonIQ — MRI Analysis Node (v15 FLAIR-only)

Handles two cases:
  A. User uploaded a FLAIR NIfTI file → POST it to the segmentation-ensemble service
  B. User typed/pasted their radiologist report as text → parse with LLM

v15 changes (FLAIR-only):
  - The backend MRI service is now the trained fold×orientation segmentation
    ensemble (server-ensemble.ipynb), not LST-AI, and it only needs FLAIR —
    T1 is no longer read, sent, or required anywhere in this node.
  - nifti_paths still arrives as a dict ({"flair": path[, "t1": path]}) for
    backward compatibility with anything still queuing a "t1" key, but only
    "flair" is ever used.
  - _call_nifti_service() now sends a single file with field name "flair".

v14 changes (kept for history):
  - nifti_path (single str) → nifti_paths (dict {"flair": ..., "t1": ...})
  - Timeout extended to 600 s (10 min) to match GPU inference time
  - Parses the JSON response format:
      {status, model, report: {overall: {total_lesions, total_voxels, total_volume_mm3},
                                regions: [{region, num_lesions, num_voxels, lesion_volume_mm3}],
                                report_text: "..."}}
  - Passes report_text through analyse_mri_text() so the LLM gets full findings
"""
from __future__ import annotations
from loguru import logger
from api.agent.state import AgentState
from api.agent.tools.mri_analyzer import analyse_mri_text

# ── Messages ──────────────────────────────────────────────────────────────────

_UPLOAD_RETRY_MSG = (
    "I received your MRI scan file, but the analysis server is not reachable right now.\n\n"
    "The MRI analysis server may still be starting up — please wait a minute and try again.\n\n"
    "If you have a written report from your radiologist in the meantime, "
    "you can paste the key findings here and I will work with that instead."
)

# Shown when goal_setter routed to mri_analysis but neither NIfTI files
# nor an actual text report were present in the message (safety net).
_NO_MRI_RECEIVED_MSG = (
    "I don't see any MRI files or a radiology report yet.\n\n"
    "To continue please either:\n"
    "• Upload your **FLAIR** NIfTI file (.nii / .nii.gz) using the MRI button, or\n"
    "• Paste the key findings from your radiologist's written report directly into the chat."
)

_VALID_DIS_REGIONS = {"cerebral", "brainstem", "cerebellar", "spinal_cord", "optic_nerve"}

# Shown by the real "check the mri again" recheck when the ensemble server
# confirms the job is still processing — grounded in an actual fetch, never
# invented (see: hallucinated case-ID retrieval bug).
_RECHECK_STILL_PROCESSING_MSG = (
    "Just checked with the analysis server — your scan is still processing. "
    "This can take up to 20 minutes depending on server load. Feel free to "
    "ask again in a bit, or paste your radiologist's written report in the "
    "meantime if you have one."
)

_RECHECK_NO_CASE_MSG = (
    "I don't have a scan on file to check — I don't see a case ID for this "
    "session. Please upload your FLAIR NIfTI file (.nii or .nii.gz) using the "
    "MRI button, or paste your radiologist's written report."
)

_RECHECK_FAILED_MSG = (
    "I checked with the analysis server, but it reports the earlier scan "
    "processing failed (or the case is no longer known to it — it may have "
    "restarted). You'll need to upload your FLAIR file again."
)


def node_mri_analysis(state: AgentState) -> dict:
    from api.agent.llm import llm
    from api.agent.graph.node_goal_setter import _is_mri_report  # reuse keyword check

    user_msg    = state.get("user_message", "")
    nifti_paths = state.get("nifti_paths")   # {"flair": path[, "t1": path]} or None

    # ── Path C: "check the mri again" — a REAL single-shot re-poll of the
    # ensemble server by case_id, not a hallucinated one. See node_goal_setter
    # ._asks_to_recheck_mri for the trigger phrases.
    if state.get("goal") == "mri_recheck":
        return _handle_recheck(state, llm)

    # ── Path A: NIfTI file uploaded ──────────────────────────────────────────
    if nifti_paths and nifti_paths.get("flair"):
        flair_path = nifti_paths["flair"]

        raw_response = _call_nifti_service(flair_path, session_id=state.get("session_id"))

        # Clean up the temp NIfTI file immediately after the service call —
        # it can be 100-500 MB and is not needed once sent.
        import os as _os
        try:
            _os.unlink(flair_path)
        except Exception:
            pass

        if raw_response is None:
            return {
                "mri_report":         None,
                "mri_service_failed": True,
                "response":           _UPLOAD_RETRY_MSG,
                "goal":               "mri_received",
                "next_phase":         "mri_requested",
            }

        # Convert ensemble-server response → internal findings dict
        findings = _parse_ensemble_response(raw_response, llm)
        logger.info("[MRI Node] Ensemble findings parsed: {}", findings)
        return _merge_mri_findings(state, findings, from_nifti=True)

    # ── Path B: Text/pasted report ────────────────────────────────────────────
    # SAFETY NET: only analyse as a text report if the message actually looks like
    # a radiology report. Without this guard, any casual message in mri_requested
    # phase (e.g. "ok", "I uploaded it") would fall here and the LLM would return
    # a spurious LOW-risk assessment from 23 words of non-clinical text.
    if not _is_mri_report(user_msg):
        logger.warning(
            "[MRI Node] Path B skipped — message does not look like an MRI report "
            "(len={}, nifti_paths={}). Returning guidance bubble.",
            len(user_msg), nifti_paths,
        )
        return {
            "mri_report":         None,
            "mri_service_failed": False,
            "response":           _NO_MRI_RECEIVED_MSG,
            "goal":               "request_mri",
            "next_phase":         "mri_requested",
        }

    findings = analyse_mri_text(user_msg, llm)
    logger.info("[MRI Node] Text report analysed: {}", findings)
    return _merge_mri_findings(state, findings, from_nifti=False)


# ── Ensemble-server response parser ───────────────────────────────────────────

def _parse_ensemble_response(raw: dict, llm) -> dict:
    """
    Convert the segmentation-ensemble server's JSON response into the internal
    findings dict that _merge_mri_findings() expects.

    server-ensemble.ipynb /predict response shape (FLAT — no "status"/"report"
    wrapper, unlike the old LST-AI service):
    {
      "case_id": "...",
      "generated_at": "...",
      "model": {"architecture": ..., "cell_name": ..., "n_models_in_ensemble": ..., "modalities": [...]},
      "preprocessing": {...},
      "overall": {"total_lesions": 12, "total_voxels": 1523, "total_volume_mm3": 3.89},
      "regions": [
        {"region": "Periventricular", "num_lesions": 3, "num_voxels": 410, "lesion_volume_mm3": 1.20},
        {"region": "Deep white matter", "num_lesions": 9, "num_voxels": 1113, "lesion_volume_mm3": 2.69}
      ],
      "report_text": "AI ENSEMBLE MS LESION SEGMENTATION REPORT\n..."
    }
    On failure the server instead returns {"error": "...", "case_id": "..."} with HTTP 500,
    which _call_nifti_service() already turns into None before this function is ever called.
    """
    overall = raw.get("overall", {})
    regions = raw.get("regions", [])
    report_text = raw.get("report_text", "")

    total_lesions = overall.get("total_lesions", 0)
    total_vol     = overall.get("total_volume_mm3", 0.0)

    # Build a human-readable summary for the LLM text analyser
    region_lines = []
    for r in regions:
        region_lines.append(
            f"  - {r.get('region','?')}: {r.get('num_lesions',0)} lesions, "
            f"{r.get('lesion_volume_mm3',0):.2f} mm³"
        )
    region_summary = "\n".join(region_lines) if region_lines else "  - No region breakdown available"

    # Compose a structured text report from the numeric data + report_text so
    # that analyse_mri_text() can extract McDonald criteria etc.
    combined_text = (
        f"MRI Lesion Segmentation Report\n"
        f"Total lesions: {total_lesions}\n"
        f"Total lesion volume: {total_vol:.2f} mm³\n"
        f"By region:\n{region_summary}\n\n"
        f"{report_text}"
    ).strip()

    logger.info(
        "[MRI Node] Ensemble total_lesions={} total_vol={:.2f}mm³ regions={}",
        total_lesions, total_vol, [r.get("region") for r in regions],
    )

    # Use the text analyser to extract structured MS findings from the combined report
    findings = analyse_mri_text(combined_text, llm)

    # Augment findings with the original numeric data for reference.
    # Key names kept as "lst_ai_*" for backward compatibility with anything else
    # reading these fields (e.g. _merge_mri_findings below) — purely a naming
    # holdover from the previous LST-AI backend, not tied to LST-AI itself.
    findings["lst_ai_total_lesions"] = total_lesions
    findings["lst_ai_total_volume_mm3"] = total_vol
    findings["lst_ai_regions"] = regions
    findings["lst_ai_report_text"] = report_text

    return findings


# ── NIfTI service call ────────────────────────────────────────────────────────

def _call_nifti_service(flair_path: str, session_id: str | None = None) -> dict | None:
    """
    Submit a FLAIR NIfTI file to the external segmentation-ensemble service and
    poll for the result.

    v17 change: as soon as the server acknowledges the submission with a
    case_id, that case_id is persisted to the mri_jobs table (keyed on
    session_id) BEFORE polling even starts. This means that if our own
    polling later times out client-side (deadline reached, worker restarted,
    etc.) the case_id survives and a later "check the mri again" from the
    patient can do a real, grounded re-fetch of /result/<case_id> — instead
    of the LLM having nothing to work with and inventing a result.

    v16 change (async submit+poll): the Kaggle server used to run the whole
    multi-minute pipeline (template downloads + registration + skull-strip +
    15-model ensemble) inside a single blocking POST /predict. Held open that
    long over a free ngrok tunnel + Flask's dev server, the connection would
    stall and ngrok would return ERR_NGROK_3004 ("invalid or incomplete HTTP
    response") even though the server eventually finished fine. The server now
    returns immediately from POST /predict with {case_id, status: "processing"}
    and runs the pipeline in a background thread; this function submits, then
    polls GET /result/<case_id> every POLL_INTERVAL_S seconds until status is
    "done" or "error", for up to MRI_SERVICE_TIMEOUT seconds total.

    Field name:   flair=<file>   (the service is FLAIR-only — no "t1" field is sent)
    Timeout:      MRI_SERVICE_TIMEOUT env var (default 1500 s = 25 min) is now the
                  total submit+poll budget, not a single request's read timeout.
    Returns:      parsed JSON report dict, or None on any failure.
    """
    import os
    import time
    import httpx
    from api.core.config import MRI_SERVICE_URL, MRI_SERVICE_TIMEOUT

    POLL_INTERVAL_S = 5.0

    if not MRI_SERVICE_URL:
        logger.warning("[MRI] MRI_SERVICE_URL not set — skipping POST")
        return None

    # MRI_SERVICE_URL historically pointed straight at /predict — derive the
    # service's base URL from it so we can also hit /result/<case_id>.
    base_url = MRI_SERVICE_URL.rsplit("/predict", 1)[0] if MRI_SERVICE_URL.endswith("/predict") else MRI_SERVICE_URL.rstrip("/")

    logger.info("[MRI] Submitting FLAIR scan to {} …", MRI_SERVICE_URL)

    try:
        files: dict = {}
        with open(flair_path, "rb") as f_flair:
            files["flair"] = (os.path.basename(flair_path), f_flair.read(), "application/gzip")

        # Submit: short timeout is fine now, this call returns almost instantly.
        submit_timeout = httpx.Timeout(60.0, connect=30.0)
        resp = httpx.post(MRI_SERVICE_URL, files=files, timeout=submit_timeout)
        resp.raise_for_status()
        submitted = resp.json()

        if submitted.get("error"):
            logger.warning("[MRI] Service rejected submission: {}", submitted.get("error", "?"))
            return None

        case_id = submitted.get("case_id")
        if not case_id:
            logger.warning("[MRI] Service response missing case_id: {}", submitted)
            return None

        logger.info("[MRI] Submitted as case_id={} — polling for result …", case_id)

        if session_id:
            try:
                from api.db import set_mri_job_case_id
                set_mri_job_case_id(session_id, case_id)
            except Exception as e:
                # Never let a DB hiccup here abort a submission that already
                # succeeded server-side — worst case, "check again" later
                # just won't have a case_id to work with.
                logger.warning("[MRI] Could not persist case_id for session {}: {}", session_id, e)

        # Poll.
        result_url = f"{base_url}/result/{case_id}"
        poll_timeout = httpx.Timeout(30.0, connect=15.0)
        deadline = time.monotonic() + float(MRI_SERVICE_TIMEOUT)

        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_S)

            # Transient hiccups (Kaggle CPU busy running the ensemble, tunnel
            # blip, etc.) must NOT abort the whole wait — only a hard 404,
            # explicit "error" status, or the outer deadline should stop us.
            try:
                poll_resp = httpx.get(result_url, timeout=poll_timeout)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                logger.warning(
                    "[MRI] Poll hiccup ({}: {}) — Kaggle CPU likely busy, retrying …",
                    type(e).__name__, e,
                )
                continue

            if poll_resp.status_code == 404:
                logger.warning("[MRI] case_id {} unknown to service (server restarted mid-job?)", case_id)
                return None

            poll_resp.raise_for_status()
            document = poll_resp.json()
            status = document.get("status")

            if status == "processing":
                continue
            if status == "error":
                logger.warning("[MRI] Service reported pipeline error: {}", document.get("error", "?"))
                return None
            if status == "done":
                logger.info("[MRI] Service returned keys: {}", list(document.keys()))
                return document

            logger.warning("[MRI] Unexpected status in poll response: {}", document)
            return None

        logger.warning(
            "[MRI] Polling exceeded {} s without a result — asking patient to retry",
            MRI_SERVICE_TIMEOUT,
        )
        return None

    except httpx.ConnectError as e:
        # Log the real underlying error (e.g. "Name or service not known" = DNS
        # resolution failure/blocked vs "Connection refused" = reached the host
        # but nothing listening) instead of a generic message, since these have
        # very different fixes.
        logger.warning("[MRI] Service unreachable — {}: {}", type(e.__cause__ or e).__name__, e)
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(
            "[MRI] Service returned HTTP {} — {}",
            e.response.status_code, e.response.text[:200],
        )
        return None
    except Exception as e:
        logger.error("[MRI] Unexpected error: {}", e)
        return None


# ── Real recheck-by-case-id (Path C) ──────────────────────────────────────────

def _derive_base_url(mri_service_url: str) -> str:
    return (
        mri_service_url.rsplit("/predict", 1)[0]
        if mri_service_url.endswith("/predict")
        else mri_service_url.rstrip("/")
    )


def _fetch_result_once(base_url: str, case_id: str) -> dict:
    """
    Single, non-looping GET /result/<case_id>. Used for an explicit patient
    "check the mri again" — grounded truth from the real server, not a poll
    loop and not an LLM guess.

    Returns {"outcome": "done", "document": {...}}
          | {"outcome": "processing"}
          | {"outcome": "error", "detail": str}
          | {"outcome": "not_found"}
    """
    import httpx

    result_url = f"{base_url}/result/{case_id}"
    try:
        resp = httpx.get(result_url, timeout=httpx.Timeout(30.0, connect=15.0))
    except (httpx.TimeoutException, httpx.TransportError) as e:
        logger.warning("[MRI Recheck] Transient error fetching {}: {}", result_url, e)
        return {"outcome": "error", "detail": str(e)}

    if resp.status_code == 404:
        return {"outcome": "not_found"}

    try:
        resp.raise_for_status()
        document = resp.json()
    except Exception as e:
        logger.warning("[MRI Recheck] Bad response from {}: {}", result_url, e)
        return {"outcome": "error", "detail": str(e)}

    status = document.get("status")
    if status == "done":
        return {"outcome": "done", "document": document}
    if status == "error":
        return {"outcome": "error", "detail": document.get("error", "?")}
    return {"outcome": "processing"}


def _handle_recheck(state: AgentState, llm) -> dict:
    """
    Goal == "mri_recheck": the patient explicitly asked us to check the
    server again. Do a REAL single-shot fetch by case_id and answer with
    whatever it actually says — done, still processing, or failed. Never
    fabricated, and never a silent no-op either.
    """
    from api.core.config import MRI_SERVICE_URL

    session_id = state.get("session_id")
    case_id    = state.get("mri_case_id")

    if not case_id and session_id:
        try:
            from api.db import get_mri_job
            job = get_mri_job(session_id)
            case_id = job.get("case_id") if job else None
        except Exception as e:
            logger.warning("[MRI Recheck] Could not look up case_id for session {}: {}", session_id, e)

    if not case_id or not MRI_SERVICE_URL:
        return {
            "mri_report":         None,
            "mri_service_failed": True,
            "response":           _RECHECK_NO_CASE_MSG,
            "next_phase":         "mri_requested",
        }

    base_url = _derive_base_url(MRI_SERVICE_URL)
    result   = _fetch_result_once(base_url, case_id)

    if result["outcome"] == "done":
        findings = _parse_ensemble_response(result["document"], llm)
        logger.info("[MRI Recheck] case_id={} came back done: {}", case_id, findings)
        return _merge_mri_findings(state, findings, from_nifti=True)

    if result["outcome"] == "processing":
        logger.info("[MRI Recheck] case_id={} still processing", case_id)
        return {
            "mri_report":         None,
            "mri_service_failed": True,
            "response":           _RECHECK_STILL_PROCESSING_MSG,
            "next_phase":         "mri_requested",
        }

    logger.warning("[MRI Recheck] case_id={} outcome={}", case_id, result["outcome"])
    return {
        "mri_report":         None,
        "mri_service_failed": True,
        "response":           _RECHECK_FAILED_MSG,
        "next_phase":         "mri_requested",
    }


# ── State merger ──────────────────────────────────────────────────────────────

def _merge_mri_findings(state: AgentState, findings: dict, *, from_nifti: bool) -> dict:
    """
    Merge MRI findings into agent state, updating:
      - tier      (ratchet up only)
      - features  (union)
      - dis_regions (union with validated region strings)
      - dit_episodes (ratchet up: dit_met → at least 2 episodes confirmed)
    """
    _ORDER = ["LOW", "WATCH", "MODERATE", "HIGH", "CRITICAL_EMERGENCY"]

    # ── Tier ratchet ──────────────────────────────────────────────────────────
    current  = state.get("tier", "LOW")
    mri_tier = findings.get("tier", "LOW")
    cur_idx  = _ORDER.index(current)  if current  in _ORDER else 0
    mri_idx  = _ORDER.index(mri_tier) if mri_tier in _ORDER else 0
    final    = _ORDER[max(cur_idx, mri_idx)]

    # ── Features union ────────────────────────────────────────────────────────
    locations = findings.get("lesion_locations", [])
    mri_feats = [f"MRI: {loc}" for loc in locations]

    # For LST-AI results, also add quantitative feature tags
    if from_nifti and findings.get("lst_ai_total_lesions") is not None:
        n = findings["lst_ai_total_lesions"]
        v = findings.get("lst_ai_total_volume_mm3", 0)
        mri_feats.append(f"MRI: {n} lesions ({v:.1f} mm³ total)")

    all_feats = list(dict.fromkeys(state.get("features", []) + mri_feats))

    # ── McDonald DIS: merge dis_regions from MRI into state ──────────────────
    raw_mri_dis   = findings.get("dis_regions", [])
    valid_mri_dis = [r for r in raw_mri_dis if r in _VALID_DIS_REGIONS]
    existing_dis  = state.get("dis_regions", [])
    merged_dis    = list(dict.fromkeys(existing_dis + valid_mri_dis))

    # ── McDonald DIT: if MRI confirms DIT (enhancing + non-enhancing) ─────────
    existing_dit = state.get("dit_episodes", 0)
    new_dit = max(existing_dit, 2) if findings.get("dit_met") else existing_dit

    logger.info(
        "[MRI Node] merged: tier={} dis_regions={} dit_episodes={} features_added={}",
        final, merged_dis, new_dit, mri_feats,
    )

    return {
        "mri_report":         findings,
        "mri_service_failed": False,
        "tier":               final,
        "features":           all_feats,
        "dis_regions":        merged_dis,
        "dit_episodes":       new_dit,
    }
