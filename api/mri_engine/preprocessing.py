"""
Preprocessing helpers — registration, skull-strip, normalization, and the
auto-detect heuristic that decides whether an uploaded FLAIR still needs
registration / skull-stripping.

Ported verbatim (logic-wise) from server-my-logical-model.ipynb cells 7 and 9.
"""
from __future__ import annotations
import os
import shutil
import time

import numpy as np
import nibabel as nib
import ants
import antspynet
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
import templateflow.api as tflow

from api.mri_engine.config import (
    TEMPLATE_CACHE, TARGET_SHAPE, CLIP_PCT, NORMALIZE_METHOD,
    SKULLSTRIP_PROB_THRESH, SKULLSTRIP_ATTEMPTS, SKULLSTRIP_RETRY_WAIT_S,
    MODALITY_TO_ANTSPYNET, REG_TYPE_OF_TRANSFORM,
)
from api.mri_engine.utils import crop_or_pad_to_shape

_tpl_cache: dict = {}


def n4_bias_correct_ants(img):
    return ants.n4_bias_field_correction(img)


# ── Real skull-stripping (antspynet.brain_extraction) ──────────────────────
def _try_antspynet_extraction(head, modality="t1", attempts=SKULLSTRIP_ATTEMPTS, wait_s=SKULLSTRIP_RETRY_WAIT_S):
    last_exc = None
    for i in range(attempts):
        try:
            prob = antspynet.brain_extraction(head, modality=modality, verbose=False)
            return ndimage.binary_fill_holes(prob.numpy() > SKULLSTRIP_PROB_THRESH).astype(np.uint8)
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(wait_s)
    raise last_exc


def run_skull_strip(img, modality_key):
    """Try the real antspynet model first; if the weight download or the extraction
    itself fails, fall back to the network-free morphological method instead of
    raising — a case should degrade to a coarser brain mask, not fail outright
    after registration has already run."""
    modality_arg = MODALITY_TO_ANTSPYNET.get(modality_key.upper(), "t1")
    try:
        mask = _try_antspynet_extraction(img, modality=modality_arg)
        return mask, "antspynet.brain_extraction"
    except Exception as e:
        print(f"  [skull-strip] WARNING: antspynet.brain_extraction failed after "
              f"{SKULLSTRIP_ATTEMPTS} attempt(s) ({e}); falling back to the morphological "
              f"whole-head threshold method (get_brain_mask_wholehead).")
        return get_brain_mask_wholehead(img.numpy()), "morphological fallback (get_brain_mask_wholehead)"


def load_icbm152_template(tf_name: str, res_mm: int):
    key = (tf_name, res_mm)
    if key in _tpl_cache:
        return _tpl_cache[key]
    dst = os.path.join(TEMPLATE_CACHE, f"{tf_name}_res{res_mm:02d}_brain.nii.gz")
    if not os.path.exists(dst):
        def _tf_get(**kw):
            try:
                p = tflow.get(tf_name, resolution=res_mm, **kw)
            except Exception:
                return None
            items = p if isinstance(p, list) else [p]
            return str(items[0]) if items else None
        brain_p = _tf_get(suffix="T1w", desc="brain", extension="nii.gz")
        if brain_p and os.path.exists(brain_p):
            shutil.copy(brain_p, dst)
        else:
            t1w_p = _tf_get(suffix="T1w", extension="nii.gz")
            t1w = nib.load(t1w_p)
            mask_p = _tf_get(suffix="mask", desc="brain", extension="nii.gz")
            if mask_p and os.path.exists(mask_p):
                bmask = nib.load(mask_p).get_fdata() > 0
            else:
                head = ants.image_read(t1w_p)
                bmask = _try_antspynet_extraction(head, modality="t1")
            brain = t1w.get_fdata(dtype=np.float32) * bmask
            nib.save(nib.Nifti1Image(brain, t1w.affine, t1w.header), dst)
    img = ants.image_read(dst)
    _tpl_cache[key] = img
    return img


def register_case_to_template(loaded_imgs: dict, template, primary_modality: str):
    reg = ants.registration(fixed=template, moving=loaded_imgs[primary_modality],
                             type_of_transform=REG_TYPE_OF_TRANSFORM)
    fwd = reg["fwdtransforms"]
    warped = {primary_modality: reg["warpedmovout"]}
    for m, img in loaded_imgs.items():
        if m == primary_modality:
            continue
        warped[m] = ants.apply_transforms(fixed=template, moving=img,
                                           transformlist=fwd, interpolator="linear")
    return warped, fwd


# ── Brain-mask helpers for the "skip" branch ────────────────────────────────
def get_brain_mask_from_stripped(arr: np.ndarray):
    mask = (arr > 0).astype(np.uint8)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    labeled, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    return mask


def get_brain_mask_wholehead(arr: np.ndarray):
    thresh = np.percentile(arr[arr > 0], 5) if np.any(arr > 0) else 0
    mask = (arr > thresh).astype(np.uint8)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    labeled, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    return mask


def percentile_clip_arr(arr, brain_mask, pct=CLIP_PCT):
    vals = arr[brain_mask > 0]
    if vals.size == 0:
        return arr
    lo, hi = np.percentile(vals, pct)
    return np.clip(arr, lo, hi)


def whitestripe_normalize(arr, brain_mask, smooth_sigma=2.0, n_bins=256):
    vals = arr[brain_mask > 0]
    if vals.size < 100:
        return arr, False
    hist, edges = np.histogram(vals, bins=n_bins)
    hist_s = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peaks = argrelextrema(hist_s, np.greater)[0]
    if len(peaks) == 0:
        return arr, False
    mode_val = centers[peaks[np.argmax(hist_s[peaks])]]
    band = 0.15 * (vals.max() - vals.min())
    nawm_vals = vals[(vals > mode_val - band) & (vals < mode_val + band)]
    if nawm_vals.size < 50:
        return arr, False
    mean, std = nawm_vals.mean(), nawm_vals.std()
    if std < 1e-6:
        return arr, False
    return (arr - mean) / std, True


def nyul_fallback_normalize(arr, brain_mask):
    vals = arr[brain_mask > 0]
    p1, p99 = np.percentile(vals, [1, 99])
    return (arr - p1) / max(p99 - p1, 1e-6)


def normalize_intensity(arr, brain_mask, method=None):
    method = method or NORMALIZE_METHOD
    if method == "whitestripe":
        normed, ok = whitestripe_normalize(arr, brain_mask)
        if ok:
            return normed, "whitestripe"
        return nyul_fallback_normalize(arr, brain_mask), "nyul_fallback"
    elif method == "nyul":
        return nyul_fallback_normalize(arr, brain_mask), "nyul"
    elif method == "none":
        return arr, "none"
    else:
        raise ValueError(f"Unknown NORMALIZE_METHOD={method!r}")


# ── Auto-detect: does this upload need registration / skull-stripping? ──────
def inspect_flair(flair_path: str) -> dict:
    """Inspect an uploaded FLAIR volume and decide whether it still needs
    registration and/or skull-stripping before it can go through the model."""
    img = nib.load(flair_path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]

    hdr = img.header
    shape = img.shape[:3]
    zooms = tuple(round(float(z), 3) for z in hdr.get_zooms()[:3])
    axcodes = "".join(nib.aff2axcodes(img.affine))

    mni_shape_match = bool(shape == TARGET_SHAPE)
    mni_spacing_match = bool(all(abs(z - 1.0) < 0.02 for z in zooms))
    already_on_grid = bool(mni_shape_match and mni_spacing_match)

    vals = data[np.isfinite(data)]
    thresh = 0.01 * (np.percentile(vals, 99.5) - vals.min()) + vals.min()
    nz_mask = data > thresh
    # Cast to native Python float/bool immediately — numpy.bool_ is not JSON
    # serializable and Python's `and`/`or` don't coerce their operands.
    bg_frac = float(1 - nz_mask.sum() / data.size)
    if nz_mask.sum() > 0:
        c = np.argwhere(nz_mask); mins, maxs = c.min(0), c.max(0) + 1
        bbox_fill = tuple(round(float((maxs - mins)[i]) / data.shape[i], 3) for i in range(3))
    else:
        bbox_fill = (0.0, 0.0, 0.0)
    likely_stripped = bool(bg_frac > 0.55 and min(bbox_fill) < 0.85)

    registration_mode = "skip" if already_on_grid else "run"
    skull_strip_mode = "skip" if likely_stripped else "run"

    info = {
        "shape": tuple(int(s) for s in shape), "zooms": zooms, "orientation": axcodes,
        "already_on_mni_grid": already_on_grid,
        "background_voxel_fraction": round(bg_frac, 3),
        "bbox_fill": bbox_fill,
        "likely_already_skull_stripped": likely_stripped,
        "registration_mode": registration_mode,
        "skull_strip_mode": skull_strip_mode,
    }
    print(f"  [inspect] shape={shape} spacing={zooms} orientation={axcodes}")
    print(f"  [inspect] already_on_MNI_grid={already_on_grid} -> registration={registration_mode.upper()}")
    print(f"  [inspect] bg_frac={bg_frac:.3f} bbox_fill={bbox_fill} "
          f"likely_stripped={likely_stripped} -> skull_strip={skull_strip_mode.upper()}")
    return info


def decide_preprocess_steps(flair_path: str) -> dict:
    from api.mri_engine.config import PREPROCESS_STEPS_BASE
    info = inspect_flair(flair_path)
    steps = dict(PREPROCESS_STEPS_BASE)
    steps["registration"] = info["registration_mode"]
    steps["skull_strip"] = info["skull_strip_mode"]
    return steps, info


def preprocess_flair_case(flair_path: str, steps: dict):
    """Full preprocessing pipeline for one uploaded FLAIR. Returns the
    processed FLAIR array, the brain mask, the template ANTs image, the
    voxel spacing, and which normalization method actually ran."""
    from api.mri_engine.config import TEMPLATE_TF_NAME, TEMPLATE_RES_MM, TEMPLATE_MODALITY, NORMALIZE_METHOD

    loaded = {"FLAIR": ants.image_read(flair_path)}

    if steps["n4_bias_correction"]:
        loaded = {m: n4_bias_correct_ants(img) for m, img in loaded.items()}

    template = None
    if steps["registration"] == "run":
        template = load_icbm152_template(TEMPLATE_TF_NAME, TEMPLATE_RES_MM)
        loaded, _ = register_case_to_template(loaded, template, TEMPLATE_MODALITY)
        print("  registered to", TEMPLATE_TF_NAME, "via", REG_TYPE_OF_TRANSFORM)
    else:
        template = load_icbm152_template(TEMPLATE_TF_NAME, TEMPLATE_RES_MM)
        primary_arr = loaded[TEMPLATE_MODALITY].numpy().astype(np.float32)
        if not steps["crop_or_pad"] and primary_arr.shape != TARGET_SHAPE:
            raise ValueError(
                f"Upload shape {primary_arr.shape} != TARGET_SHAPE {TARGET_SHAPE} but "
                f"registration was auto-detected as 'skip' and crop_or_pad is False."
            )

    primary_arr = loaded[TEMPLATE_MODALITY].numpy().astype(np.float32)
    mode = steps["skull_strip"]
    if mode == "run":
        brain_mask, strip_method = run_skull_strip(loaded[TEMPLATE_MODALITY], TEMPLATE_MODALITY)
        print(f"  skull-stripped via {strip_method}")
    elif mode == "skip":
        brain_mask = get_brain_mask_from_stripped(primary_arr)
    else:
        raise ValueError(f"Unknown skull_strip mode {mode!r}")

    arr = primary_arr
    steps_applied = []

    if steps["brain_mask_multiply"]:
        arr = arr * brain_mask
        steps_applied.append("mask")

    if steps["percentile_clip"]:
        arr = percentile_clip_arr(arr, brain_mask)
        steps_applied.append("clip")

    norm_method_used = "none"
    if steps["intensity_normalize"]:
        arr, norm_method_used = normalize_intensity(arr, brain_mask, method=NORMALIZE_METHOD)
        steps_applied.append(f"norm({norm_method_used})")
        if steps["brain_mask_multiply"]:
            arr = arr * brain_mask

    if steps["crop_or_pad"]:
        arr = crop_or_pad_to_shape(arr)
        brain_mask = crop_or_pad_to_shape(brain_mask)
        steps_applied.append("crop/pad")

    print(f"  FLAIR: {' -> '.join(steps_applied) if steps_applied else '(no steps applied)'}")

    spacing = template.spacing if steps["registration"] == "run" else loaded[TEMPLATE_MODALITY].spacing
    return {
        "FLAIR": arr.astype(np.float32),
    }, brain_mask.astype(np.uint8), template, spacing, norm_method_used
