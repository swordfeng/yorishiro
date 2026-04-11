"""Shared audio resampling helpers."""

from __future__ import annotations

import numpy as np


def resample(
    x: np.ndarray,
    *,
    orig_sr: int | float,
    target_sr: int | float,
    quality: str = "HQ",
) -> np.ndarray:
    """Resample mono or multi-channel audio with soxr.

    Input:
    - mono: shape (samples,)
    - multi-channel: shape (samples, channels)
    """
    arr = np.asarray(x, dtype=np.float32)
    if int(orig_sr) == int(target_sr):
        return arr

    import soxr

    out = soxr.resample(arr, float(orig_sr), float(target_sr), quality=quality)
    return np.asarray(out, dtype=np.float32)

