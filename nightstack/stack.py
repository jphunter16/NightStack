"""Pixel combination with Siril-style rejection, normalization weights, and sky/ground blending."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .calibration import fast_nanmedian

SKY_METHODS = {
    "sigma": "Sigma-clipped mean  (best all-round, removes planes)",
    "winsor": "Winsorized sigma  (great for 15+ frames)",
    "linear": "Linear-fit clipping  (best when sky brightness drifts)",
    "mad": "MAD clipping  (very robust, few frames)",
    "percentile": "Percentile clipping  (for 3-6 frames)",
    "minmax": "Min/max rejection  (best for 4-9 frames)",
    "median": "Median  (robust, slightly noisier)",
    "mean": "Mean  (lowest noise, keeps planes)",
    "max": "Maximum  (lighten)",
}
GROUND_METHODS = {
    "median": "Median  (removes moving cars / people)",
    "sigma": "Sigma-clipped mean",
    "winsor": "Winsorized sigma",
    "minmax": "Min/max rejection",
    "mean": "Mean  (lowest noise)",
    "best": "Best single frame  (crisp trees / water)",
}
BLEND_MODES = {
    "min_noise": "Minimum horizon noise  (recommended)",
    "standard": "Standard feathered edge",
}
CLIP_METHODS = ("sigma", "winsor", "linear", "mad", "percentile")
# These estimate a spread from the samples themselves, which is unreliable with few frames:
# one airplane streak inflates sigma so much that the streak survives the clipping.
STATISTICAL_METHODS = CLIP_METHODS
SMALL_STACK = 10   # below this many frames, statistical clipping is swapped for min/max or median


def small_stack_method(method: str, n_frames: int) -> str:
    """Pick a rejection that works for short sequences (applied automatically)."""
    if method not in STATISTICAL_METHODS or n_frames >= SMALL_STACK:
        return method
    return "median" if n_frames <= 3 else "minmax"


def combine(data: np.ndarray, valid: Optional[np.ndarray], method: str, kappa: float = 2.5,
            iterations: int = 3, weights: Optional[np.ndarray] = None):
    """data (N, h, w, 3) float32; valid (N, h, w) bool or None; weights (N,) per-frame or None.

    Returns (image, count of contributing frames per pixel).
    """
    n = data.shape[0]
    if valid is None:
        valid = np.ones(data.shape[:3], bool)
    v = valid[..., None]
    count = valid.sum(0).astype(np.float32)
    w = np.ones(n, np.float32) if weights is None else np.asarray(weights, np.float32)
    if method == "max":
        out = np.where(v, data, -np.inf).max(0)
    elif method == "median":
        out = fast_nanmedian(np.where(v, data, np.nan), 0)
    elif method == "minmax":
        out = minmax_reject(data, v, w)
    elif method == "mean" or (method in CLIP_METHODS and n < 3):
        wv = v * w[:, None, None, None]
        out = (data * wv).sum(0) / np.maximum(wv.sum(0), 1e-12)
    elif method in CLIP_METHODS:
        out = clip_sorted(data, v, w, method, kappa, iterations)
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


def minmax_reject(data, v, w, k: int = 1) -> np.ndarray:
    """Drop the k highest and k lowest samples of each pixel, weighted-average the rest.

    An airplane / satellite crosses a pixel in one frame, so it is the highest sample there
    and is removed no matter how few frames there are.  Pixels with fewer than 2k+1 valid
    samples (near the horizon) keep what they have.
    """
    d = np.where(v, data, np.nan)
    order = np.argsort(d, axis=0)
    srt = np.take_along_axis(d, order, 0)
    n_valid = np.sum(~np.isnan(srt), 0)
    z = np.nan_to_num(srt)
    N = z.shape[0]
    ws = np.take_along_axis(np.broadcast_to(w[:, None, None, None], d.shape), order, 0) * (~np.isnan(srt))
    kk = np.where(n_valid >= 2 * k + 1, k, 0)
    lo, hi = kk, n_valid - kk
    zero = np.zeros((1,) + z.shape[1:], np.float32)
    cw = np.concatenate([zero, np.cumsum(ws, 0, dtype=np.float32)], 0)
    cwz = np.concatenate([zero, np.cumsum(ws * z, 0, dtype=np.float32)], 0)
    out = (_gather(cwz, hi) - _gather(cwz, lo)) / np.maximum(_gather(cw, hi) - _gather(cw, lo), 1e-12)
    return out.astype(np.float32)


def _window(keep: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Shrink [lo, hi) to the first..last kept sample (rejection happens at the sorted ends)."""
    n = keep.shape[0]
    j = np.arange(n).reshape((n,) + (1,) * (keep.ndim - 1))
    inside = keep & (j >= lo[None]) & (j < hi[None])
    any_ = inside.any(0)
    first = np.argmax(inside, 0)
    last = n - np.argmax(inside[::-1], 0)
    return np.where(any_, first, lo), np.where(any_, last, hi)


def clip_sorted(data, v, w, method: str, kappa: float, iterations: int) -> np.ndarray:
    """Rejection stacking with a single (arg)sort.

    Every algorithm rejects only from the low or high end of a pixel's sorted samples, so
    each pixel keeps a window [lo, hi) into its sorted values.  Means / variances of the
    window come from cumulative sums; the weighted mean of the survivors is the result.
    Methods follow Siril: sigma, winsorized sigma, linear-fit, MAD and percentile clipping.
    """
    d = np.where(v, data, np.nan)
    order = np.argsort(d, axis=0)                 # NaN (invalid samples) sort to the end
    srt = np.take_along_axis(d, order, 0)
    n_valid = np.sum(~np.isnan(srt), 0)
    z = np.nan_to_num(srt)
    N = z.shape[0]
    ws = np.broadcast_to(w[:, None, None, None], d.shape)
    ws = np.take_along_axis(ws, order, 0) * (~np.isnan(srt))
    zero = np.zeros((1,) + z.shape[1:], np.float32)
    cs = np.concatenate([zero, np.cumsum(z, 0, dtype=np.float32)], 0)
    cs2 = np.concatenate([zero, np.cumsum(z * z, 0, dtype=np.float32)], 0)
    lo = np.zeros(n_valid.shape, np.int64)
    hi = n_valid.astype(np.int64)
    filled = np.where(np.isnan(srt), np.inf, srt)
    jidx = np.arange(N, dtype=np.float32).reshape((N,) + (1,) * (z.ndim - 1))
    if method == "linear":
        csj = np.concatenate([zero, np.cumsum(z * jidx, 0, dtype=np.float32)], 0)
    if method == "winsor":
        zw = z.copy()
    for _ in range(iterations):
        cnt = np.maximum(hi - lo, 1).astype(np.float32)
        a = np.clip((lo + hi - 1) // 2, 0, N - 1)
        b = np.clip((lo + hi) // 2, 0, N - 1)
        med = 0.5 * (_gather(z, a) + _gather(z, b))
        if method in ("sigma", "winsor"):
            src_cs, src_cs2 = cs, cs2
            if method == "winsor":
                zc = np.concatenate([zero, np.cumsum(zw, 0, dtype=np.float32)], 0)
                zc2 = np.concatenate([zero, np.cumsum(zw * zw, 0, dtype=np.float32)], 0)
                src_cs, src_cs2 = zc, zc2
            m1 = (_gather(src_cs, hi) - _gather(src_cs, lo)) / cnt
            m2 = (_gather(src_cs2, hi) - _gather(src_cs2, lo)) / cnt
            std = np.sqrt(np.clip(m2 - m1 ** 2, 0, None) * cnt / np.maximum(cnt - 1, 1))
            if method == "winsor":
                std = std * 1.134  # Huber correction used by Siril / PixInsight
                zw = np.clip(z, (med - 1.5 * std)[None], (med + 1.5 * std)[None])
            keep = (filled >= (med - kappa * std)[None]) & (filled <= (med + kappa * std)[None])
        elif method == "mad":
            j = np.arange(N).reshape((N,) + (1,) * (z.ndim - 1))
            inwin = (j >= lo[None]) & (j < hi[None])
            dev = np.where(inwin, np.abs(z - med[None]), np.nan)
            mad = 1.4826 * fast_nanmedian(dev, 0)
            keep = np.abs(filled - med[None]) <= kappa * mad[None]
        elif method == "percentile":
            p = 0.25 * kappa / 2.5   # +/-25% around the median at the default strength
            keep = (filled >= (med * (1 - p))[None]) & (filled <= (med * (1 + p))[None])
        elif method == "linear":
            # Fit sorted values against their rank; reject samples far from the line.
            S0 = cnt
            Sj = ((hi - 1) * hi / 2 - (lo - 1) * lo / 2).astype(np.float32)
            Sjj = (((hi - 1) * hi * (2 * hi - 1) - (lo - 1) * lo * (2 * lo - 1)) / 6).astype(np.float32)
            Sy = _gather(cs, hi) - _gather(cs, lo)
            Sjy = _gather(csj, hi) - _gather(csj, lo)
            den = S0 * Sjj - Sj * Sj
            slope = np.where(np.abs(den) > 1e-9, (S0 * Sjy - Sj * Sy) / np.where(den == 0, 1, den), 0)
            icpt = (Sy - slope * Sj) / S0
            res = filled - (icpt[None] + slope[None] * jidx)
            j = np.arange(N).reshape((N,) + (1,) * (z.ndim - 1))
            inwin = (j >= lo[None]) & (j < hi[None]) & np.isfinite(filled)
            # Siril uses the mean absolute deviation from the fit (robust to the outliers being fitted)
            sig = np.where(inwin, np.abs(res), 0).sum(0) / np.maximum(cnt, 1) * 1.2533
            keep = np.abs(res) <= kappa * sig[None]
        else:
            raise ValueError(method)
        keep &= np.isfinite(filled)
        new_lo, new_hi = _window(keep, lo, hi)
        ok = (new_hi - new_lo) >= np.minimum(3, n_valid)  # never reject everything
        lo = np.where(ok, new_lo, lo)
        hi = np.where(ok, new_hi, hi)
    wz = ws * (zw if method == "winsor" else z)
    cw = np.concatenate([zero, np.cumsum(ws, 0, dtype=np.float32)], 0)
    cwz = np.concatenate([zero, np.cumsum(wz, 0, dtype=np.float32)], 0)
    out = (_gather(cwz, hi) - _gather(cwz, lo)) / np.maximum(_gather(cw, hi) - _gather(cw, lo), 1e-12)
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
          mode: str = "min_noise", feather: float = 4.0, strength: float = 1.0,
          fill: Optional[np.ndarray] = None):
    """Combine the aligned-sky stack and the fixed-ground stack. Returns (image, alpha).

    fill: what to use under the sky where aligned data is thin (just above the horizon).
    Defaults to `ground`; when the foreground is a single frame, pass the multi-frame stack
    here so that horizon band stays low-noise.
    """
    m = (sky_mask > 127).astype(np.float32)
    if fill is not None:
        ground = fill + (1 - m)[..., None] * (ground - fill)  # single frame below, stack above
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
