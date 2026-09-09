"""
Loads the fixed 3-checkpoint ensemble (one model per orientation) from
CHECKPOINT_DIR. This is deliberately stricter than the notebook's original
"load every fold*_*.pt found" loop: that server ran a fold×orientation
ensemble (up to 15 models); this deployment is scoped down to exactly one
checkpoint per orientation for a lighter, CPU-only footprint.

Each checkpoint file is expected to be a torch.save()'d dict with the same
keys the training pipeline writes:
    {
      "model_state":            state_dict for build_model(...),
      "orientation":             "axial" | "coronal" | "sagittal",
      "modalities":               [...]   (must equal CELL_MODALITIES, i.e. ["FLAIR"]),
      "in_channels":              int,
      "fold":                     int (optional, metadata only),
      "validation_fold_dice":     float (optional, metadata only),
    }
Filenames don't matter — orientation is read from inside the checkpoint, not
parsed from the filename — so any 3 files named however you like will work
as long as each one is the right shape and covers a distinct orientation.
"""
from __future__ import annotations
import glob
import os

import torch
from loguru import logger

from api.mri_engine.config import (
    CHECKPOINT_DIR, CELL_MODALITIES, ORIENTATIONS, AXIS_FOR_ORIENTATION,
    N_CHECKPOINTS_EXPECTED, DEVICE,
)
from api.mri_engine.model import build_model


class CheckpointError(RuntimeError):
    pass


def load_ensemble():
    """Returns a list of (model, axis, orientation, fold) tuples — exactly one
    per orientation. Raises CheckpointError with an actionable message if the
    checkpoints/ folder doesn't contain exactly the right set."""
    ckpt_paths = sorted(
        p for p in glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))
    )
    if not ckpt_paths:
        raise CheckpointError(
            f"No checkpoint files (*.pt) found under {CHECKPOINT_DIR}. "
            f"Drop your 3 trained checkpoints (one each for axial, coronal, "
            f"sagittal) into that folder — see checkpoints/README.md."
        )

    by_orientation: dict[str, tuple] = {}
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        orientation = ckpt.get("orientation")
        if orientation not in ORIENTATIONS:
            logger.warning(
                "[MRI Checkpoints] Skipping {} — unrecognised/missing 'orientation' "
                "key ({!r}), expected one of {}",
                os.path.basename(ckpt_path), orientation, ORIENTATIONS,
            )
            continue
        if ckpt.get("modalities") != CELL_MODALITIES:
            raise CheckpointError(
                f"{os.path.basename(ckpt_path)} was trained on modalities="
                f"{ckpt.get('modalities')} but this server expects {CELL_MODALITIES}. "
                f"These MUST match or predictions are meaningless."
            )
        if orientation in by_orientation:
            raise CheckpointError(
                f"Found more than one checkpoint for orientation={orientation!r} "
                f"({os.path.basename(ckpt_paths[0])} and others). This build expects "
                f"exactly {N_CHECKPOINTS_EXPECTED} checkpoints total — one per "
                f"orientation. Remove the extra file(s) from {CHECKPOINT_DIR}."
            )

        model = build_model(in_channels=ckpt["in_channels"]).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        by_orientation[orientation] = (
            model, AXIS_FOR_ORIENTATION[orientation], orientation, ckpt.get("fold"),
        )
        dice = ckpt.get("validation_fold_dice")
        dice_str = f"{dice:.4f}" if isinstance(dice, float) else "n/a"
        logger.info(
            "[MRI Checkpoints] loaded {} (orientation={}, fold={}, val_dice={})",
            os.path.basename(ckpt_path), orientation, ckpt.get("fold"), dice_str,
        )

    missing = [o for o in ORIENTATIONS if o not in by_orientation]
    if missing:
        raise CheckpointError(
            f"Missing checkpoint(s) for orientation(s): {missing}. This build "
            f"requires exactly {N_CHECKPOINTS_EXPECTED} checkpoints — one each "
            f"for {ORIENTATIONS} — in {CHECKPOINT_DIR}."
        )

    loaded = [by_orientation[o] for o in ORIENTATIONS]
    logger.info("[MRI Checkpoints] {} models loaded (one per orientation).", len(loaded))
    return loaded
