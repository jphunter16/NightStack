"""Calibration frames (bias / dark / flat) and hot-pixel removal - the Siril-style pre-processing."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .imageio import FrameCache, load_image

Progress = Callable[[float, str], None]


def _noop(p, m):
    pass


def fast_nanmedian(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median along axis ignoring NaN. Much faster than np.nanmedian for stacks."""
    s = np.sort(data, axis=axis)  # NaN sorts to the end
    n = np.sum(~np.isnan(data), axis=axis)
    lo = np.clip((n - 1) // 2, 0, None)
    hi = np.clip(n // 2, 0, None)
    a = np.take_along_axis(s, np.expand_dims(lo, axis), axis=axis).squeeze(axis)
    b = np.take_along_axis(s, np.expand_dims(hi, axis), axis=axis).squeeze(axis)
    out = 0.5 * (a + b)
    out[n == 0] = np.nan
    return out


def build_master(paths: Sequence[str], cache_dir: str, name: str, half_size: bool = False,
                 subtract: Optional[np.ndarray] = None, progress: Progress = _noop,
                 p0: float = 0.0, p1: float = 1.0) -> Optional[np.ndarray]:
    """Median-combine calibration frames into a master frame."""
    paths = list(paths)
    if not paths:
        return None
    first = load_image(paths[0], half_size)
    h, w, _ = first.shape
    if len(paths) == 1:
        return first - subtract if subtract is not None else first
    cache = FrameCache(cache_dir, f"master-{name}", len(paths), h, w, dtype=np.float32)
    try:
        for i, p in enumerate(paths):
            progress(p0 + (p1 - p0) * 0.8 * i / len(paths), f"Reading {name} frame {i + 1}/{len(paths)}")
            img = first if i == 0 else load_image(p, half_size)
            if img.shape != first.shape:
                raise ValueError(f"{name} frame {p} has a different size than the others")
            if subtract is not None:
                img = img - subtract
            cache.write(i, img)
        progress(p0 + (p1 - p0) * 0.85, f"Combining master {name}")
        out = np.empty((h, w, 3), np.float32)
        step = 128
        for r in range(0, h, step):
            out[r:r + step] = np.median(cache.rows(r, min(h, r + step)), axis=0)
        return out
    finally:
        cache.close()


def normalise_flat(flat: np.ndarray) -> np.ndarray:
    f = cv2.GaussianBlur(flat, (0, 0), 1.0)
    means = np.array([np.median(f[..., c]) for c in range(3)], np.float32)
    f = f / np.maximum(means, 1e-6)
    return np.clip(f, 0.05, None).astype(np.float32)


def calibrate(img: np.ndarray, bias=None, dark=None, flat_norm=None) -> np.ndarray:
    out = img
    if dark is not None:  # master dark still contains bias, so subtracting it removes both
        out = out - dark
    elif bias is not None:
        out = out - bias
    if flat_norm is not None:
        out = out / flat_norm
    return np.clip(out, 0.0, None).astype(np.float32)


# ----------------------------------------------------------------------------- hot pixels
class HotPixelDetector:
    """Hot / cold pixels stay fixed on the sensor while stars drift, so we vote across frames."""

    def __init__(self, shape, k: float = 5.0):
        self.count = np.zeros(shape, np.uint8)
        self.k = k
        self.n = 0

    def add(self, img: np.ndarray) -> None:
        med = cv2.medianBlur(img, 3)
        diff = img - med
        sample = diff[::7, ::7].reshape(-1, 3)
        sig = 1.4826 * np.median(np.abs(sample - np.median(sample, 0)), 0) + 1e-6
        bad = np.abs(diff) > self.k * sig + 0.002
        self.count += bad.astype(np.uint8)
        self.n += 1

    def hot_map(self) -> np.ndarray:
        if self.n < 3:
            return np.zeros(self.count.shape, bool)
        return self.count >= max(3, int(np.ceil(0.6 * self.n)))


def fix_pixels(img: np.ndarray, bad: np.ndarray) -> np.ndarray:
    if not bad.any():
        return img
    med = cv2.medianBlur(img, 5) if img.dtype == np.float32 else cv2.medianBlur(img.astype(np.float32), 5)
    out = img.copy()
    out[bad] = med[bad]
    return out
