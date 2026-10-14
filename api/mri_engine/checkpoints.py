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
import hashlib
import io
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


def _pinned_hashes() -> dict:
    """{filename: sha256} from MRI_CHECKPOINT_SHA256 ("a.pt:<hex>,b.pt:<hex>") and/or the
    SHA256SUMS file ("<hex>  <filename>" per line)."""
    from api.core.config import MRI_CHECKPOINT_SHA256, MRI_CHECKPOINT_SHA256_FILE
    pins: dict = {}
    if os.path.isfile(MRI_CHECKPOINT_SHA256_FILE):
        for line in open(MRI_CHECKPOINT_SHA256_FILE, encoding="utf-8").read().splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                pins[os.path.basename(parts[1].lstrip("*"))] = parts[0].lower()
    for item in filter(None, (x.strip() for x in MRI_CHECKPOINT_SHA256.split(","))):
        name, _, digest = item.rpartition(":")
        if name and digest:
            pins[os.path.basename(name)] = digest.lower()
    return pins


def read_verified(path: str, pins: dict | None = None) -> bytes:
    """Read the file ONCE, check its SHA-256 against the pinned list, and return the exact
    bytes that were hashed (so there is no verify-then-reload TOCTOU window)."""
    pins = _pinned_hashes() if pins is None else pins
    name = os.path.basename(path)
    expected = pins.get(name)
    if not expected:
        raise CheckpointError(f"{name} has no pinned SHA-256 (set MRI_CHECKPOINT_SHA256 or checkpoints/SHA256SUMS). "
                              f"Refusing to load an unverified checkpoint.")
    data = open(path, "rb").read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise CheckpointError(f"{name} SHA-256 mismatch (expected {expected[:12]}…, got {actual[:12]}…). "
                              f"The file was modified or is not the pinned checkpoint.")
    return data


def verify_checkpoint_hashes() -> None:
    """Called synchronously at app startup: raise (→ app refuses to start) on any mismatch or
    unpinned *.pt present. No checkpoints at all is allowed (engine just reports not-ready)."""
    paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt")))
    pins = _pinned_hashes()
    for p in paths:
        read_verified(p, pins)


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

    pins = _pinned_hashes()
    by_orientation: dict[str, tuple] = {}
    for ckpt_path in ckpt_paths:
        # weights_only=True: the pickle is restricted to tensors/primitive containers, so a
        # tampered file cannot execute code. Bytes are hash-verified first (see read_verified).
        ckpt = torch.load(io.BytesIO(read_verified(ckpt_path, pins)), map_location=DEVICE, weights_only=True)
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
