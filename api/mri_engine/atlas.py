from __future__ import annotations
import os
import shutil

import numpy as np
import nibabel as nib
import ants
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import templateflow.api as tflow
from nilearn import datasets as nl_datasets

from api.mri_engine.config import (
    TEMPLATE_CACHE, TEMPLATE_TF_NAME, TEMPLATE_RES_MM, TARGET_SHAPE,
    ATLAS_DATASET_DIR, GM_PROB_THRESH, WM_PROB_THRESH, CSF_PROB_THRESH,
    PERIVENTRICULAR_DIST_MM, JUXTACORTICAL_DIST_MM, INFRATENTORIAL_Z_FRACTION,
    ZONE_NAMES,
)
from api.mri_engine.utils import crop_or_pad_to_shape

_zone_atlas_cache = {}


def _get_tissue_probseg(tf_name, res_mm, label):
    dst = os.path.join(TEMPLATE_CACHE, f"{tf_name}_res{res_mm:02d}_label-{label}_probseg.nii.gz")
    if not os.path.exists(dst):
        try:
            p = tflow.get(tf_name, resolution=res_mm, label=label, suffix="probseg", extension="nii.gz")
            items = p if isinstance(p, list) else [p]
            src = str(items[0])
            shutil.copy(src, dst)
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch TemplateFlow tissue probseg label={label} for {tf_name} "
                f"res={res_mm} — {e}"
            )
    return nib.load(dst).get_fdata(dtype=np.float32)


def _resample_labels_to_template(label_nifti_path, template, label_indices):
    atlas_img = ants.image_read(label_nifti_path)
    resampled = ants.resample_image_to_target(atlas_img, template, interp_type="genericLabel")
    arr = resampled.numpy()
    mask = np.isin(np.round(arr).astype(np.int32), label_indices)
    return mask


def _atlas_data_dir():
    expected_fsl_dir = os.path.join(ATLAS_DATASET_DIR, "fsl")
    if os.path.isdir(ATLAS_DATASET_DIR) and os.path.isdir(expected_fsl_dir):
        print(f"[atlas] Using pre-downloaded atlases from {ATLAS_DATASET_DIR}")
        return ATLAS_DATASET_DIR
    print(f"[atlas] Pre-downloaded atlas dataset not found at {ATLAS_DATASET_DIR} -- "
          f"will attempt a live nilearn download instead.")
    return None


def build_zone_atlas_harvard_aal(template):
    shape = template.shape
    spacing = template.spacing
    data_dir = _atlas_data_dir()

    wm = _get_tissue_probseg(TEMPLATE_TF_NAME, TEMPLATE_RES_MM, "WM")
    if wm.shape != shape:
        src_img = ants.from_numpy(wm.astype(np.float32))
        wm = ants.resample_image_to_target(src_img, template, interp_type="linear").numpy()
    wm_mask = wm > WM_PROB_THRESH

    ho_sub = nl_datasets.fetch_atlas_harvard_oxford(
        "sub-maxprob-thr25-1mm", symmetric_split=False, data_dir=data_dir
    )
    ho_sub_labels = list(ho_sub.labels)
    ho_sub_path = ho_sub.maps if isinstance(ho_sub.maps, str) else ho_sub.filename

    def _indices_matching(names, keyword):
        return [i for i, n in enumerate(names) if keyword.lower() in n.lower()]

    ventricle_idx = _indices_matching(ho_sub_labels, "Lateral Ventricle")
    brainstem_idx = _indices_matching(ho_sub_labels, "Brain-Stem")
    assert ventricle_idx, "Harvard-Oxford subcortical atlas: no 'Lateral Ventricle' label found"
    assert brainstem_idx, "Harvard-Oxford subcortical atlas: no 'Brain-Stem' label found"

    ventricle_mask = _resample_labels_to_template(ho_sub_path, template, ventricle_idx)
    brainstem_mask = _resample_labels_to_template(ho_sub_path, template, brainstem_idx)

    ho_cort = nl_datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-1mm", symmetric_split=False, data_dir=data_dir
    )
    ho_cort_labels = list(ho_cort.labels)
    ho_cort_path = ho_cort.maps if isinstance(ho_cort.maps, str) else ho_cort.filename
    cortex_idx = [i for i in range(1, len(ho_cort_labels))]
    cortex_mask = _resample_labels_to_template(ho_cort_path, template, cortex_idx)

    aal = nl_datasets.fetch_atlas_aal(data_dir=data_dir)
    aal_labels = list(aal.labels)
    aal_indices_raw = [int(i) for i in aal.indices]
    aal_path = aal.maps
    cerebellum_names = [n for n in aal_labels if ("cerebel" in n.lower() or "vermis" in n.lower())]
    cerebellum_values = [aal_indices_raw[aal_labels.index(n)] for n in cerebellum_names]
    assert cerebellum_values, "AAL atlas: no cerebellum/vermis labels found"
    cerebellum_mask = _resample_labels_to_template(aal_path, template, cerebellum_values)

    infratentorial_mask = brainstem_mask | cerebellum_mask

    dist_from_ventricle = distance_transform_edt(~ventricle_mask, sampling=spacing)
    dist_from_cortex = distance_transform_edt(~cortex_mask, sampling=spacing)

    zone = np.zeros(shape, dtype=np.uint8)
    remaining = wm_mask.copy()

    periventricular = remaining & (dist_from_ventricle <= PERIVENTRICULAR_DIST_MM)
    zone[periventricular] = 1
    remaining &= ~periventricular

    juxtacortical = remaining & (dist_from_cortex <= JUXTACORTICAL_DIST_MM)
    zone[juxtacortical] = 2
    remaining &= ~juxtacortical

    infratentorial = remaining & infratentorial_mask
    zone[infratentorial] = 3
    remaining &= ~infratentorial

    zone[remaining] = 4

    zone = crop_or_pad_to_shape(zone, target_shape=TARGET_SHAPE)
    return zone, spacing


def build_zone_atlas_heuristic(template):
    shape = template.shape
    spacing = template.spacing

    gm = _get_tissue_probseg(TEMPLATE_TF_NAME, TEMPLATE_RES_MM, "GM")
    wm = _get_tissue_probseg(TEMPLATE_TF_NAME, TEMPLATE_RES_MM, "WM")
    csf = _get_tissue_probseg(TEMPLATE_TF_NAME, TEMPLATE_RES_MM, "CSF")

    if gm.shape != shape:
        def _resample_to_template(arr):
            src_img = ants.from_numpy(arr.astype(np.float32))
            return ants.resample_image_to_target(src_img, template, interp_type="linear").numpy()
        gm, wm, csf = _resample_to_template(gm), _resample_to_template(wm), _resample_to_template(csf)

    gm_mask = gm > GM_PROB_THRESH
    wm_mask = wm > WM_PROB_THRESH
    csf_mask = csf > CSF_PROB_THRESH

    brain_mask = (gm + wm + csf) > 0.1
    border_touch = np.zeros_like(csf_mask)
    border_touch[0, :, :] = border_touch[-1, :, :] = True
    border_touch[:, 0, :] = border_touch[:, -1, :] = True
    border_touch[:, :, 0] = border_touch[:, :, -1] = True
    eroded_brain = ndimage.binary_erosion(brain_mask, iterations=3)
    ventricle_mask = csf_mask & eroded_brain & (~border_touch)

    dist_from_ventricle = distance_transform_edt(~ventricle_mask, sampling=spacing)
    dist_from_cortex = distance_transform_edt(~gm_mask, sampling=spacing)

    z_cutoff = max(int(round(INFRATENTORIAL_Z_FRACTION * shape[2])), 1)
    infratentorial_mask = np.zeros(shape, dtype=bool)
    infratentorial_mask[:, :, :z_cutoff] = True

    zone = np.zeros(shape, dtype=np.uint8)
    remaining = wm_mask.copy()

    periventricular = remaining & (dist_from_ventricle <= PERIVENTRICULAR_DIST_MM)
    zone[periventricular] = 1
    remaining &= ~periventricular

    juxtacortical = remaining & (dist_from_cortex <= JUXTACORTICAL_DIST_MM)
    zone[juxtacortical] = 2
    remaining &= ~juxtacortical

    infratentorial = remaining & infratentorial_mask
    zone[infratentorial] = 3
    remaining &= ~infratentorial

    zone[remaining] = 4

    zone = crop_or_pad_to_shape(zone, target_shape=TARGET_SHAPE)
    return zone, spacing


def build_zone_atlas(template):
    """One-time construction of an integer zone-label volume, cached process-wide."""
    key = id(template)
    if key in _zone_atlas_cache:
        return _zone_atlas_cache[key]

    try:
        zone, spacing = build_zone_atlas_harvard_aal(template)
        method = "Harvard-Oxford + AAL"
    except Exception as e:
        print(f"[atlas] WARNING: Harvard-Oxford/AAL atlas build failed ({e}); "
              f"falling back to the tissue-probability heuristic.")
        zone, spacing = build_zone_atlas_heuristic(template)
        method = "tissue-probability heuristic"

    _zone_atlas_cache[key] = (zone, spacing)
    print(f"  zone atlas built ({method}): {zone.shape} | voxel counts -> "
          + ", ".join(f"{ZONE_NAMES[k]}={int((zone==k).sum())}" for k in ZONE_NAMES))
    return zone, spacing
