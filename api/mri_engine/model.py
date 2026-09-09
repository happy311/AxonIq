from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from api.mri_engine.config import STACK_HALF, CELL_MODALITIES, DEVICE

MODEL_ARCH_NAME = "smp_unet_resnet34_scse_imagenet"


class PaddedUnetWrapper(nn.Module):
    def __init__(self, base_model, multiple=32):
        super().__init__()
        self.base_model = base_model
        self.multiple = multiple

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (self.multiple - h % self.multiple) % self.multiple
        pad_w = (self.multiple - w % self.multiple) % self.multiple
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h])
        out = self.base_model(x)
        if pad_h or pad_w:
            out = out[..., :h, :w]
        return out


def build_model(in_channels, out_channels=1, encoder_name="resnet34", pretrained=True):
    base = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        decoder_attention_type="scse",
        in_channels=in_channels,
        classes=out_channels,
        activation=None,
    )
    return PaddedUnetWrapper(base, multiple=32)


def reslice_stack(volume, axis, center, half=STACK_HALF):
    slicer = [slice(None)] * 3
    slabs = []
    for offset in range(-half, half + 1):
        slicer[axis] = center + offset
        slabs.append(volume[tuple(slicer)])
    return np.stack(slabs, axis=0)


def enumerate_slice_indices(volume_shape, axis, half=STACK_HALF):
    n = volume_shape[axis]
    return list(range(half, n - half))


@torch.no_grad()
def predict_volume_probs_multimod(model, mod_arrs: dict, axis, batch_size=16):
    any_vol = next(iter(mod_arrs.values()))
    prob_vol = np.zeros_like(any_vol, dtype=np.float32)
    idxs = enumerate_slice_indices(any_vol.shape, axis)
    for start in range(0, len(idxs), batch_size):
        batch_idx = idxs[start:start + batch_size]
        batch_stacks = []
        for s in batch_idx:
            stacks = [reslice_stack(mod_arrs[m], axis, s) for m in CELL_MODALITIES]
            batch_stacks.append(np.concatenate(stacks, axis=0))
        x = torch.from_numpy(np.stack(batch_stacks, axis=0).astype(np.float32)).to(DEVICE)
        probs = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
        for b, s in enumerate(batch_idx):
            slicer = [slice(None)] * 3; slicer[axis] = s
            prob_vol[tuple(slicer)] = probs[b]
    return prob_vol
