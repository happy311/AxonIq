#!/usr/bin/env python3
"""
Build-time asset bake — run ONCE inside the Docker build (network available there), so the
running container needs no outbound downloads and can have a read-only filesystem.

Fetches: TemplateFlow MNI152NLin2009cAsym (template + GM/WM/CSF probseg), Harvard-Oxford +
AAL atlases (nilearn), and the ANTsXNet brain-extraction network.

Env (set by the Dockerfile): MRI_WORK_DIR, MRI_TEMPLATE_CACHE, MRI_ATLAS_DATASET_DIR, HOME
"""
import os
import sys

os.environ.setdefault("JWT_SECRET", "build-time-only-" + "x1Y2z3" * 8)   # api.core imports require one; unused here
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.mri_engine import config as C                      # noqa: E402
import templateflow.api as tflow                            # noqa: E402
from nilearn import datasets as nl                          # noqa: E402

print("[bake] TemplateFlow →", C.TEMPLATE_CACHE)
from api.mri_engine.atlas import _get_tissue_probseg        # noqa: E402
from api.mri_engine.preprocessing import load_icbm152_template  # noqa: E402

load_icbm152_template(C.TEMPLATE_TF_NAME, C.TEMPLATE_RES_MM)
for label in ("GM", "WM", "CSF"):
    _get_tissue_probseg(C.TEMPLATE_TF_NAME, C.TEMPLATE_RES_MM, label)

print("[bake] atlases →", C.ATLAS_DATASET_DIR)
os.makedirs(C.ATLAS_DATASET_DIR, exist_ok=True)
nl.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm", symmetric_split=False, data_dir=C.ATLAS_DATASET_DIR)
nl.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm", symmetric_split=False, data_dir=C.ATLAS_DATASET_DIR)
nl.fetch_atlas_aal(data_dir=C.ATLAS_DATASET_DIR)

print("[bake] ANTsXNet brain-extraction weights → ~/.keras")
import antspynet, ants, numpy as np                           # noqa: E402
tpl = ants.from_numpy(np.random.rand(64, 64, 64).astype("float32"))
try:
    for mod in ("flair", "t1"):
        antspynet.brain_extraction(tpl, modality=mod, verbose=False)   # forces the weight download
except Exception as e:  # inference on noise may fail AFTER the download succeeded — that's fine
    print("[bake] note:", type(e).__name__)
print("[bake] done")
