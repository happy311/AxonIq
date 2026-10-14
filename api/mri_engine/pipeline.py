"""
End-to-end pipeline: preprocess -> 3-model orientation-ensemble inference ->
zone report. Also owns process-wide engine state (loaded models, GPU/CPU
inference lock) and the one-time startup warm-up.

This replaces server-my-logical-model.ipynb's Flask server: same logic, but
loaded once at FastAPI startup and called directly as a Python function from
a background thread (see jobs.py) instead of over HTTP through a Cloudflare
tunnel.
"""
from __future__ import annotations
import os
import threading

import numpy as np

from api.mri_engine.config import OUTPUT_DIR, MIN_COMPONENT_VOXELS
from api.mri_engine.checkpoints import load_ensemble, CheckpointError
from api.mri_engine.model import predict_volume_probs_multimod
from api.mri_engine.preprocessing import (
    decide_preprocess_steps, preprocess_flair_case, load_icbm152_template, run_skull_strip,
)
from api.mri_engine.atlas import build_zone_atlas
from api.mri_engine.postprocess import suppress_small_components, build_zone_report, build_report

# ── Process-wide engine state ────────────────────────────────────────────────
_state_lock = threading.Lock()
_loaded_models: list | None = None      # [(model, axis, orientation, fold), ...] — exactly 3
_inference_lock = threading.Semaphore(1)      # serializes the actual forward-pass step (also protects a single GPU's VRAM if MRI_DEVICE=cuda)
_ready = False
_init_error: str | None = None


def is_ready() -> bool:
    return _ready


def init_error() -> str | None:
    return _init_error


def n_models() -> int:
    return len(_loaded_models) if _loaded_models else 0


def initialize():
    """Load the 3-checkpoint ensemble and warm up template/atlas downloads.
    Call once at FastAPI startup. Safe to call from a worker thread — does
    not touch any event loop."""
    global _loaded_models, _ready, _init_error
    with _state_lock:
        if _ready or _init_error:
            return
        try:
            from api.mri_engine.config import TEMPLATE_TF_NAME, TEMPLATE_RES_MM, TEMPLATE_MODALITY
            print("[MRI Engine] Loading checkpoints...")
            _loaded_models = load_ensemble()

            print("[MRI Engine] Warming up external resources (template/atlas downloads)...")
            warm_template = None
            try:
                warm_template = load_icbm152_template(TEMPLATE_TF_NAME, TEMPLATE_RES_MM)
                print("  [warmup] MNI template ready.")
            except Exception as e:
                print(f"  [warmup] WARNING: could not fetch the MNI template ({e}). "
                      f"Registration will fail for every case until this is resolved.")

            if warm_template is not None:
                try:
                    _, method = run_skull_strip(warm_template, TEMPLATE_MODALITY)
                    print(f"  [warmup] skull-strip path ready ({method}).")
                except Exception as e:
                    print(f"  [warmup] WARNING: skull-strip warm-up failed unexpectedly ({e}).")
                try:
                    build_zone_atlas(warm_template)
                    print("  [warmup] Zone atlas ready.")
                except Exception as e:
                    print(f"  [warmup] WARNING: zone atlas build failed even with its own "
                          f"fallback ({e}). Reports will omit the by-zone breakdown.")
            else:
                print("  [warmup] Skipping skull-strip / zone-atlas warm-up -- no template to test against.")

            _ready = True
            from api.mri_engine.config import DEVICE
            print(f"[MRI Engine] Ready — {len(_loaded_models)} models loaded (device: {DEVICE}).")
        except CheckpointError as e:
            _init_error = str(e)
            print(f"[MRI Engine] STARTUP FAILED: {e}")
        except Exception as e:
            _init_error = f"Unexpected startup error: {e}"
            print(f"[MRI Engine] STARTUP FAILED (unexpected): {e}")


def run_pipeline(flair_path: str, case_id: str) -> dict:
    if not _ready or not _loaded_models:
        raise RuntimeError(
            _init_error or "MRI engine not initialized — call initialize() at startup."
        )

    print(f"\n{'='*60}\nCase {case_id}\n{'='*60}")

    steps, inspect_info = decide_preprocess_steps(flair_path)
    processed_arrs, brain_mask, template, spacing, norm_method_used = preprocess_flair_case(flair_path, steps)

    # Serialize the actual model forward passes — preprocessing (registration,
    # skull-strip) for other in-flight cases can still proceed while this waits.
    with _inference_lock:
        all_probs = [predict_volume_probs_multimod(model, processed_arrs, axis)
                     for model, axis, orientation, fold in _loaded_models]
    ensemble_probs = np.mean(all_probs, axis=0)
    pred_mask = (ensemble_probs > 0.5).astype(np.uint8)
    print(f"Ensembled {len(_loaded_models)} models. Predicted lesion voxels (pre-CC-filter): {pred_mask.sum()}")

    ensemble_probs, pred_mask = suppress_small_components(ensemble_probs, pred_mask)
    print(f"Predicted lesion voxels (post-CC-filter, min_size={MIN_COMPONENT_VOXELS}): {pred_mask.sum()}")

    try:
        zone_vol, zone_spacing = build_zone_atlas(template)
        regions = build_zone_report(pred_mask, zone_vol, zone_spacing)
    except Exception as e:
        print(f"[zone-atlas] WARNING: anatomical zone labeling failed ({e}); "
              f"returning lesion totals without a by-zone breakdown.")
        regions = []

    report = build_report(case_id, pred_mask, regions, spacing, inspect_info,
                           norm_method_used, n_models=len(_loaded_models))

    case_out_dir = os.path.join(OUTPUT_DIR, case_id)
    os.makedirs(case_out_dir, exist_ok=True)
    import nibabel as nib
    affine = np.diag([-1, 1, 1, 1])  # template-space identity-ish affine
    mask_path = os.path.join(case_out_dir, f"{case_id}_prediction_mask.nii.gz")
    prob_path = os.path.join(case_out_dir, f"{case_id}_prediction_probability.nii.gz")
    nib.save(nib.Nifti1Image(pred_mask, affine), mask_path)
    nib.save(nib.Nifti1Image(ensemble_probs, affine), prob_path)

    report["_output_files"] = {"mask": mask_path, "probability": prob_path}
    print(report["report_text"])
    return report
