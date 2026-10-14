"""
AxonIQ — MRI Segmentation Engine Configuration

Ported from server-my-logical-model.ipynb (the Kaggle inference notebook) and
adapted to run in-process inside the FastAPI app instead of as a separate
Flask server behind a Cloudflare/ngrok tunnel.

Key changes from the notebook:
  - CHECKPOINT_DIR now points at a local `checkpoints/` folder shipped in the
    repo (see checkpoints/README.md) instead of a Kaggle "Add Data" mount.
  - Exactly ONE checkpoint per orientation (axial, coronal, sagittal) is
    expected — 3 total, not a fold×orientation ensemble of up to 15. See
    api/mri_engine/checkpoints.py for the loader that enforces this.
  - DEVICE auto-detects a GPU if one is visible to torch, otherwise falls
    back to CPU. Force one or the other with env var MRI_DEVICE=cuda|cpu.
  - No Flask app, no Swagger, no Cloudflare tunnel — this becomes a normal
    background job inside the same Uvicorn process. See jobs.py.
"""
from __future__ import annotations
import os
from pathlib import Path

from api.core.config import BASE_DIR

# ── Device ────────────────────────────────────────────────────────────────────
# Auto-detects: uses a GPU if torch can see one, otherwise CPU. Set
# MRI_DEVICE=cpu to force CPU even with a GPU present, or MRI_DEVICE=cuda to
# require a GPU (fails loudly at startup if none is actually available,
# rather than silently falling back).
_requested_device = os.environ.get("MRI_DEVICE", "auto").lower()

if _requested_device == "auto":
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        DEVICE = "cpu"
elif _requested_device == "cuda":
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "MRI_DEVICE=cuda was set but torch.cuda.is_available() is False. "
            "Either the instance has no GPU attached, the NVIDIA driver isn't "
            "installed/visible in this container, or a CPU-only torch build "
            "is installed (see requirements-gpu.txt). Set MRI_DEVICE=auto or "
            "MRI_DEVICE=cpu to run without a GPU."
        )
    DEVICE = "cuda"
else:
    DEVICE = _requested_device   # e.g. "cpu", or "cuda:1" for a specific GPU

# ── Paths ─────────────────────────────────────────────────────────────────────
# Checkpoints ship in the repo as an (empty, .gitkeep'd) folder — drop your
# 3 trained .pt files in here (see checkpoints/README.md).
CHECKPOINT_DIR = os.environ.get("MRI_CHECKPOINT_DIR", str(BASE_DIR / "checkpoints"))

# Scratch space for template/atlas caches + per-case working files.
# /tmp is fine on a single-instance AWS deployment (ephemeral is OK here —
# nothing in WORK_DIR needs to survive a restart; templates/atlases just get
# re-downloaded once on next startup).
WORK_DIR = os.environ.get("MRI_WORK_DIR", "/tmp/axoniq_mri")
# In the container image the templates/atlases/pretrained nets are BAKED at build time into a
# read-only path (MRI_TEMPLATE_CACHE / MRI_ATLAS_DATASET_DIR) — no runtime downloads.
TEMPLATE_CACHE = os.environ.get("MRI_TEMPLATE_CACHE", os.path.join(WORK_DIR, "templateflow"))
os.makedirs(TEMPLATE_CACHE, exist_ok=True)
os.environ.setdefault("TEMPLATEFLOW_HOME", TEMPLATE_CACHE)

OUTPUT_DIR = os.path.join(WORK_DIR, "case_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional pre-downloaded Harvard-Oxford/AAL atlas dataset (same idea as the
# Kaggle "Add Data" mount in the notebook). If this folder doesn't exist,
# build_zone_atlas() automatically falls back to a live nilearn download, and
# if that also fails, to the tissue-probability heuristic — see atlas.py.
ATLAS_DATASET_DIR = os.environ.get(
    "MRI_ATLAS_DATASET_DIR", str(BASE_DIR / "atlases" / "nilearn_atlases")
)

# ── Model / ensemble ──────────────────────────────────────────────────────────
CELL_NAME = os.environ.get("MRI_CELL_NAME", "stage1_flair__syn_n4on")   # metadata only, shown in reports
CELL_MODALITIES = ["FLAIR"]
ORIENTATIONS = ["axial", "coronal", "sagittal"]
AXIS_FOR_ORIENTATION = {"axial": 2, "coronal": 1, "sagittal": 0}
N_CHECKPOINTS_EXPECTED = 3   # exactly one per orientation — see checkpoints.py

STACK_HALF = 3
IN_CHANNELS = len(CELL_MODALITIES) * (2 * STACK_HALF + 1)
TARGET_SHAPE = (182, 218, 182)   # MNI152 1mm grid
CLIP_PCT = (0.5, 99.5)

TEMPLATE_MODALITY = "FLAIR"
REG_TYPE_OF_TRANSFORM = "SyN"
TEMPLATE_TF_NAME = "MNI152NLin2009cAsym"
TEMPLATE_RES_MM = 1

SKULLSTRIP_PROB_THRESH = 0.5
SKULLSTRIP_ATTEMPTS = 2
SKULLSTRIP_RETRY_WAIT_S = 5.0
MODALITY_TO_ANTSPYNET = {"FLAIR": "flair", "T1": "t1", "T1KS": "t1", "T2": "t2"}

MIN_COMPONENT_VOXELS = 10   # connected components smaller than this are treated as noise

# ══════════════════════════════════════════════════════════════════════════
# PREPROCESSING:
#   • intensity normalization is ALWAYS ON, ALWAYS whitestripe
#   • registration / skull-strip are auto-DETECTED per uploaded scan
# ══════════════════════════════════════════════════════════════════════════
NORMALIZE_METHOD = "whitestripe"

PREPROCESS_STEPS_BASE = {
    "n4_bias_correction":  True,
    "brain_mask_multiply": False,
    "percentile_clip":     False,
    "intensity_normalize": True,
    "crop_or_pad":         True,
    # "registration" and "skull_strip" are decided per-case at request time
}

# ── MS-style lesion-location zones ───────────────────────────────────────────
GM_PROB_THRESH   = 0.5
WM_PROB_THRESH   = 0.5
CSF_PROB_THRESH  = 0.5
PERIVENTRICULAR_DIST_MM = 3.0
JUXTACORTICAL_DIST_MM   = 4.0
INFRATENTORIAL_Z_FRACTION = 0.30

ZONE_NAMES = {
    1: "Periventricular",
    2: "Juxtacortical",
    3: "Infratentorial",
    4: "Deep white matter",
}

# ── Job concurrency ───────────────────────────────────────────────────────────
# Registration + skull-strip (ANTs, CPU-bound regardless of DEVICE) still
# dominate the wall clock per case even on a GPU — only the 3-model ensemble
# forward pass itself gets faster. Default to running one case at a time;
# raise MRI_MAX_CONCURRENT_JOBS if your instance (CPU cores, or GPU + enough
# host RAM for concurrent registrations) has headroom to spare.
MAX_CONCURRENT_JOBS = int(os.environ.get("MRI_MAX_CONCURRENT_JOBS", "1"))

# Total submit→done budget a caller should wait before giving up (used by
# node_mri.py's polling loop). CPU inference with only 3 models is faster
# than the old 15-model ensemble, but registration/skull-strip dominate the
# wall clock either way — keep a generous default.
MRI_INFERENCE_TIMEOUT = int(os.environ.get("MRI_INFERENCE_TIMEOUT", "1500"))

# ── Upload staging (dedicated, private dir; never user-controlled names) ──────
UPLOAD_DIR = os.path.join(WORK_DIR, "uploads")
os.makedirs(UPLOAD_DIR, mode=0o700, exist_ok=True)
try:
    os.chmod(UPLOAD_DIR, 0o700)
except OSError:
    pass
