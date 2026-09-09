from __future__ import annotations
import numpy as np

from api.mri_engine.config import TARGET_SHAPE


def crop_or_pad_to_shape(arr, target_shape=TARGET_SHAPE):
    out = np.zeros(target_shape, dtype=arr.dtype)
    slabs_src, slabs_dst = [], []
    for d in range(3):
        src, tgt = arr.shape[d], target_shape[d]
        if src >= tgt:
            start = (src - tgt) // 2
            slabs_src.append(slice(start, start + tgt)); slabs_dst.append(slice(0, tgt))
        else:
            pad = (tgt - src) // 2
            slabs_src.append(slice(0, src)); slabs_dst.append(slice(pad, pad + src))
    out[tuple(slabs_dst)] = arr[tuple(slabs_src)]
    return out
