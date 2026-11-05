"""Pixel combination (mean / median / sigma-clip / max) and sky-ground blending."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .calibration import fast_nanmedian

SKY_METHODS = {
    "sigma": "Sigma-clipped mean  (best - removes planes & satellites)",
    "median": "Median  (robust, slightly noisier)",
    "mean": "Mean  (lowest noise, keeps planes)",
    "winsor": "Winsorized sigma mean  (great for 15+ frames)",
    "max": "Maximum  (lighten)",
}
GROUND_METHODS = {
    "median": "Median  (removes moving cars / people)",
    "sigma": "Sigma-clipped mean",
    "mean": "Mean  (lowest noise)",
}
BLEND_MODES = {
    "min_noise": "Minimum horizon noise  (recommended)",
    "standard": "Standard feathered edge",
}


def combine(data: np.ndarray, valid: Optional[np.ndarray], method: str, kappa: float = 2.5,
            iterations: int = 3):
    """data: (N, h, w, 3) float32, valid: (N, h, w) bool or None. Returns (image, count)."""
    n = data.shape[0]
    if valid is None:
        valid = np.ones(data.shape[:3], bool)
    v = valid[..., None]
    count = valid.sum(0).astype(np.float32)
    if method == "max":
        out = np.where(v, data, -np.inf).max(0)
    elif method == "mean" or (method in ("sigma", "winsor") and n < 4):
        out = (data * v).sum(0) / np.maximum(count, 1)[..., None]
    elif method == "median":
        out = fast_nanmedian(np.where(v, data, np.nan), 0)
    elif method == "sigma":
        out = _sigma_clip_sorted(np.where(v, data, np.nan), kappa, iterations)
    elif method == "winsor":
        d = np.where(v, data, np.nan)
        for _ in range(iterations):
            med = fast_nanmedian(d, 0)
            std = _nanstd(d)
            d = np.where(np.isnan(d), np.nan, np.clip(d, med - kappa * std, med + kappa * std))
        cnt = np.sum(~np.isnan(d), 0)
        out = np.nansum(d, 0) / np.maximum(cnt, 1)
    else:
        raise ValueError(method)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out[count == 0] = 0
    return out, count


def _gather(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """arr[idx[p], p] for every pixel p (fast 1-D take instead of take_along_axis)."""
    n = arr.shape[0]
    p = idx.size
    flat = idx.ravel().astype(np.int64) * p + np.arange(p, dtype=np.int64)
    return np.take(arr.reshape(n * p), flat).reshape(idx.shape)


def _sigma_clip_sorted(d: np.ndarray, kappa: float, iterations: int) -> np.ndarray:
    """Kappa-sigma clipped mean with a single sort.

    Clipping only ever removes values from the two ends of the sorted samples, so each
    pixel keeps a [lo, hi) window into its sorted array; median and std of the window
    come from index lookups and cumulative sums - no re-sorting per iteration.
    """
    srt = np.sort(d, axis=0)                      # NaN (invalid) go to the end
    n_valid = np.sum(~np.isnan(srt), 0)
    z = np.nan_to_num(srt)
    zero = np.zeros((1,) + z.shape[1:], np.float32)
    cs = np.concatenate([zero, np.cumsum(z, 0, dtype=np.float32)], 0)
    cs2 = np.concatenate([zero, np.cumsum(z * z, 0, dtype=np.float32)], 0)
    lo = np.zeros(n_valid.shape, np.int64)
    hi = n_valid.astype(np.int64)
    filled = np.where(np.isnan(srt), np.inf, srt)
    for _ in range(iterations):
        cnt = np.maximum(hi - lo, 1)
        m1 = (_gather(cs, hi) - _gather(cs, lo)) / cnt
        m2 = (_gather(cs2, hi) - _gather(cs2, lo)) / cnt
        std = np.sqrt(np.clip(m2 - m1 ** 2, 0, None) * cnt / np.maximum(cnt - 1, 1))
        a = np.clip((lo + hi - 1) // 2, 0, srt.shape[0] - 1)
        b = np.clip((lo + hi) // 2, 0, srt.shape[0] - 1)
        med = 0.5 * (_gather(z, a) + _gather(z, b))
        new_lo = np.sum(filled < (med - kappa * std)[None], 0)
        new_hi = np.sum(filled <= (med + kappa * std)[None], 0)
        new_lo = np.maximum(new_lo, lo)
        new_hi = np.minimum(new_hi, hi)
        ok = new_hi - new_lo >= np.minimum(3, n_valid)  # never clip away everything
        lo = np.where(ok, new_lo, lo)
        hi = np.where(ok, new_hi, hi)
    cnt = np.maximum(hi - lo, 1)
    out = (_gather(cs, hi) - _gather(cs, lo)) / cnt
    return out.astype(np.float32)


def _nanstd(d: np.ndarray) -> np.ndarray:
    cnt = np.sum(~np.isnan(d), 0)
    mean = np.nansum(d, 0) / np.maximum(cnt, 1)
    var = np.nansum((d - mean) ** 2, 0) / np.maximum(cnt - 1, 1)
    return np.sqrt(var)


def soft_mask(mask_u8: np.ndarray, feather: float) -> np.ndarray:
    m = (mask_u8 > 127).astype(np.float32)
    if feather <= 0:
        return m
    return cv2.GaussianBlur(m, (0, 0), max(0.5, feather / 2.5))


def blend(sky: np.ndarray, ground: np.ndarray, sky_mask: np.ndarray, count: np.ndarray, n_sky: int,
          mode: str = "min_noise", feather: float = 4.0, strength: float = 1.0):
    """Combine the aligned-sky stack and the fixed-ground stack. Returns (image, alpha)."""
    m = (sky_mask > 127).astype(np.float32)
    if mode == "standard":
        alpha = soft_mask(sky_mask, feather) * (count > 0)
    else:
        # Keep the edge crisp on the ground side (no sky halo over trees), feather into the sky.
        alpha = np.minimum(m, soft_mask(sky_mask, feather))
        # Near the horizon only some aligned frames contribute sky data, so the aligned stack
        # is noisier there.  Lean on the all-frames (fixed) stack in proportion to that loss.
        f = count / max(1, n_sky)
        f0 = 0.35
        w = np.clip((f - f0) / (1 - f0), 0, 1)
        w = 1 - strength * (1 - w)
        w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 3.0)
        w = np.clip(np.minimum(w, (count > 0).astype(np.float32)), 0, 1)
        alpha = alpha * w
    out = sky - ground            # ground + alpha * (sky - ground), computed in place
    out *= alpha[..., None]
    out += ground
    return out, alpha.astype(np.float32)
