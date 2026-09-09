"""
AxonIQ — MRI Analysis Node (v21 — merged single-server, local CPU engine)

Handles two cases:
  A. User uploaded a FLAIR NIfTI file → hand it to the local mri_engine job queue
  B. User typed/pasted their radiologist report as text → parse with LLM

v21 changes (merged single-server):
  - The segmentation ensemble used to run on a separate Kaggle notebook behind
    a Cloudflare/ngrok tunnel, reached over HTTP via MRI_SERVICE_URL. It now
    runs in-process inside this same FastAPI app (api/mri_engine/), CPU-only,
    with a fixed 3-model orientation ensemble (one checkpoint each for axial,
    coronal, sagittal) instead of a fold×orientation ensemble of up to 15.
  - _call_nifti_service() and the recheck path now call api.mri_engine.jobs
    directly (in-process function calls) instead of httpx over the network.
    No MRI_SERVICE_URL, no tunnel, no separate server to keep alive.
  - The response SHAPE from the engine is unchanged from the old service, so
    _parse_ensemble_response() below is untouched:
      {status, model, report: {overall: {total_lesions, total_voxels, total_volume_mm3},
                                regions: [{region, num_lesions, num_voxels, lesion_volume_mm3}],
                                report_text: "..."}}

v15 changes (kept for history, FLAIR-only):
  - nifti_paths still arrives as a dict ({"flair": path[, "t1": path]}) for
    backward compatibility with anything still queuing a "t1" key, but only
    "flair" is ever used.
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

# v19: shown when the extraction LLM fails to produce valid structured data
# from a pasted text report — see mri_analyzer.analyse_mri_text's
# "parse_failed" flag. Without this, a parse failure silently fell through
# to a default tier="LOW" dict and _merge_mri_findings would confidently
# tell the patient their scan "doesn't look like MS" — a false reassurance
# caused by an extraction bug, not an actual normal scan.
_PARSE_FAILED_TEXT_MSG = (
    "I had trouble reading the structure of that report — the wording or "
    "formatting may have thrown off the extraction. Could you paste the "
    "full radiologist report text again, or try uploading the FLAIR file "
    "directly with the MRI button instead?"
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
    Submit a FLAIR NIfTI file to the local, in-process segmentation engine and
    poll for the result.

    v21 change (merged single-server): this used to POST over HTTP to a
    separate Kaggle notebook behind a Cloudflare tunnel. It now calls
    api.mri_engine.jobs directly — same submit-then-poll shape (so the rest
    of this file, and the "check the mri again" recovery path, needed no
    changes), just no network hop. The engine itself still runs the job on a
    background thread (CPU inference legitimately takes minutes), so this
    function still polls rather than blocking synchronously.

    v17 change: as soon as the job is accepted with a case_id, that case_id
    is persisted to the mri_jobs table (keyed on session_id) BEFORE polling
    even starts. This means that if our own polling later times out
    client-side (deadline reached, process restarted, etc.) the case_id
    survives and a later "check the mri again" from the patient can do a
    real, grounded re-fetch — instead of the LLM having nothing to work with
    and inventing a result.

    Field name:   n/a — local call, no multipart form.
    Timeout:      MRI_INFERENCE_TIMEOUT env var (default 1500 s = 25 min) is
                  the total submit+poll budget.
    Returns:      parsed report dict, or None on any failure.
    """
    import os
    import shutil
    import tempfile
    import time
    from api.mri_engine import jobs as mri_jobs
    from api.mri_engine.config import MRI_INFERENCE_TIMEOUT

    POLL_INTERVAL_S = 2.0

    tmp_dir = tempfile.mkdtemp(prefix="mricase_")
    case_flair_path = os.path.join(tmp_dir, "FLAIR.nii.gz")
    try:
        shutil.copy(flair_path, case_flair_path)
    except Exception as e:
        logger.error("[MRI] Could not stage upload for the local engine: {}", e)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    logger.info("[MRI] Submitting FLAIR scan to the local segmentation engine …")

    submitted = mri_jobs.submit(case_flair_path, tmp_dir)

    if submitted.get("error"):
        logger.warning("[MRI] Engine rejected submission: {}", submitted.get("error", "?"))
        return None

    case_id = submitted.get("case_id")
    if not case_id:
        logger.warning("[MRI] Engine response missing case_id: {}", submitted)
        return None

    logger.info("[MRI] Submitted as case_id={} — polling for result …", case_id)

    if session_id:
        try:
            from api.db import set_mri_job_case_id
            set_mri_job_case_id(session_id, case_id)
        except Exception as e:
            # Never let a DB hiccup here abort a submission that already
            # succeeded — worst case, "check again" later just won't have a
            # case_id to work with.
            logger.warning("[MRI] Could not persist case_id for session {}: {}", session_id, e)

    deadline = time.monotonic() + float(MRI_INFERENCE_TIMEOUT)

    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_S)

        document = mri_jobs.get_status(case_id)
        if document is None:
            logger.warning("[MRI] case_id {} unknown to engine (unexpected)", case_id)
            return None

        status = document.get("status")
        if status == "processing":
            continue
        if status == "error":
            logger.warning("[MRI] Engine reported pipeline error: {}", document.get("error", "?"))
            return None
        if status == "done":
            logger.info("[MRI] Engine returned keys: {}", list(document.keys()))
            return document

        logger.warning("[MRI] Unexpected status in poll response: {}", document)
        return None

    logger.warning(
        "[MRI] Polling exceeded {} s without a result — asking patient to retry",
        MRI_INFERENCE_TIMEOUT,
    )
    return None


# ── Real recheck-by-case-id (Path C) ──────────────────────────────────────────

def _fetch_result_once(case_id: str) -> dict:
    """
    Single, non-looping status check. Used for an explicit patient
    "check the mri again" — grounded truth from the real engine, not a poll
    loop and not an LLM guess.

    Returns {"outcome": "done", "document": {...}}
          | {"outcome": "processing"}
          | {"outcome": "error", "detail": str}
          | {"outcome": "not_found"}
    """
    from api.mri_engine import jobs as mri_jobs

    document = mri_jobs.get_status(case_id)
    if document is None:
        return {"outcome": "not_found"}

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
    session_id = state.get("session_id")
    case_id    = state.get("mri_case_id")

    if not case_id and session_id:
        try:
            from api.db import get_mri_job
            job = get_mri_job(session_id)
            case_id = job.get("case_id") if job else None
        except Exception as e:
            logger.warning("[MRI Recheck] Could not look up case_id for session {}: {}", session_id, e)

    if not case_id:
        return {
            "mri_report":         None,
            "mri_service_failed": True,
            "response":           _RECHECK_NO_CASE_MSG,
            "next_phase":         "mri_requested",
        }

    result = _fetch_result_once(case_id)

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

def _handle_parse_failure(state: AgentState, findings: dict, *, from_nifti: bool) -> dict:
    """
    findings["parse_failed"] is True — the extraction LLM (analyse_mri_text)
    did not return usable structured data. Never merge this as if it were a
    real result (that would silently report a false "looks normal" tier).
    """
    if not from_nifti:
        # Path B (pasted text): no independent data to fall back on.
        logger.warning("[MRI Node] Text-report extraction failed to parse — not merging as a result.")
        return {
            "mri_report":         None,
            "mri_service_failed": True,
            "response":           _PARSE_FAILED_TEXT_MSG,
            "next_phase":         "mri_requested",
        }

    # Path A (NIfTI/ensemble): the segmentation ensemble itself succeeded —
    # we have REAL lesion counts/volumes/regions. Only the McDonald-criteria
    # extraction step (an LLM call over the composed report text) failed.
    # Report the real numbers honestly; do NOT claim a tier or risk statement
    # — that part is genuinely unknown this time, not "normal".
    total   = findings.get("lst_ai_total_lesions")
    vol     = findings.get("lst_ai_total_volume_mm3")
    regions = findings.get("lst_ai_regions", [])
    region_lines = "; ".join(
        f"{r.get('region', '?')}: {r.get('num_lesions', 0)} lesions" for r in regions
    ) or "no regional breakdown available"
    total_str = str(total) if total is not None else "an unknown number of"
    vol_str   = f"{vol:.0f} mm³" if vol is not None else "an unknown volume"

    message = (
        f"Your scan has been processed — the segmentation found {total_str} "
        f"lesion(s) (total volume {vol_str}). Regional breakdown: {region_lines}.\n\n"
        "I wasn't able to automatically classify what these findings mean "
        "against the McDonald criteria this time, so I can't give you a risk "
        "level right now. Please share the full report with a neurologist for "
        "direct review, or feel free to paste the report text here again and "
        "I'll try once more.\n\n"
        "This assessment is for clinical decision support only and is not a diagnosis."
    )
    logger.warning("[MRI Node] Ensemble succeeded but McDonald-criteria extraction failed to parse.")
    return {
        "mri_report":         findings,
        "mri_service_failed": True,
        "response":           message,
        "next_phase":         "mri_requested",
    }


def _merge_mri_findings(state: AgentState, findings: dict, *, from_nifti: bool) -> dict:
    """
    Merge MRI findings into agent state, updating:
      - tier      (ratchet up only)
      - features  (union)
      - dis_regions (union with validated region strings)
      - dit_episodes (ratchet up: dit_met → at least 2 episodes confirmed)
    """
    if findings.get("parse_failed"):
        return _handle_parse_failure(state, findings, from_nifti=from_nifti)

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

    # v18: build the patient-facing results message deterministically — see
    # api/agent/tools/results_summary.py for why this can't be left to the
    # (often weak/free-tier fallback) LLM alone. node_llm.py short-circuits
    # on "mri_results_ready" the same way it already does for
    # mri_service_failed, so this is guaranteed to reach the patient intact.
    from api.agent.tools.results_summary import build_mri_results_message
    results_message = build_mri_results_message(final, findings, merged_dis, new_dit)

    return {
        "mri_report":         findings,
        "mri_service_failed": False,
        "mri_results_ready":  True,
        "response":           results_message,
        "next_phase":         "mri_received",
        "tier":               final,
        "features":           all_feats,
        "dis_regions":        merged_dis,
        "dit_episodes":       new_dit,
    }
