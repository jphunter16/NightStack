"""Star detection with sub-pixel centroids."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def highpass(gray: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    return gray - cv2.GaussianBlur(gray, (0, 0), sigma)


def point_highpass(gray: np.ndarray) -> np.ndarray:
    """Keeps stars, removes smooth glow AND sharp edges (horizon, tree lines).

    A Gaussian high-pass leaves a bright static ridge along the horizon that would act as a
    fake 'star' which never moves; a median background does not.
    """
    g = gray.astype(np.float32)
    bg = cv2.medianBlur(g, 5)
    bg = cv2.medianBlur(bg, 5)
    return g - bg


def detect_stars(gray: np.ndarray, mask: Optional[np.ndarray] = None, max_stars: int = 1500,
                 k: float = 5.0, border: int = 12, grid: int = 10):
    """Return (points[N,2] as x,y float32, flux[N]) sorted by brightness, spread across the frame."""
    h, w = gray.shape
    sub = point_highpass(gray)
    sm = cv2.GaussianBlur(sub, (0, 0), 1.0)
    sel = sm[::3, ::3] if mask is None else sm[::3, ::3][mask[::3, ::3] > 0]
    if sel.size < 100:
        sel = sm[::3, ::3].ravel()
    med = float(np.median(sel))
    mad = float(1.4826 * np.median(np.abs(sel - med))) + 1e-9
    thr = med + k * mad
    dil = cv2.dilate(sm, np.ones((7, 7), np.uint8))
    peaks = (sm >= dil) & (sm > thr)
    peaks[:border] = False
    peaks[-border:] = False
    peaks[:, :border] = False
    peaks[:, -border:] = False
    if mask is not None:
        peaks &= mask > 0
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
    flux = sm[ys, xs]
    # Reject single hot pixels: a real star spreads light to its neighbours.
    nb = (sub[ys - 1, xs] + sub[ys + 1, xs] + sub[ys, xs - 1] + sub[ys, xs + 1]) / 4.0
    ok = nb > 0.18 * sub[ys, xs]
    ys, xs, flux = ys[ok], xs[ok], flux[ok]
    # Even spatial spread: keep the brightest few per grid cell, then overall top.
    order = np.argsort(-flux)
    ys, xs, flux = ys[order], xs[order], flux[order]
    cell = (ys * grid // h) * grid + (xs * grid // w)
    per_cell = max(4, int(2.5 * max_stars / (grid * grid)))
    rank = np.zeros_like(cell)
    seen = {}
    for j, c in enumerate(cell):
        seen[c] = seen.get(c, 0) + 1
        rank[j] = seen[c]
    keep = rank <= per_cell
    ys, xs, flux = ys[keep][:max_stars], xs[keep][:max_stars], flux[keep][:max_stars]
    # Intensity-weighted centroid in a 5x5 window.
    off = np.arange(-2, 3)
    yy = ys[:, None, None] + off[None, :, None]
    xx = xs[:, None, None] + off[None, None, :]
    win = np.clip(sub[yy, xx] - med, 0, None)
    tot = win.sum(axis=(1, 2)) + 1e-12
    cy = ys + (win * off[None, :, None]).sum(axis=(1, 2)) / tot
    cx = xs + (win * off[None, None, :]).sum(axis=(1, 2)) / tot
    pts = np.stack([cx, cy], 1).astype(np.float32)
    return pts, flux.astype(np.float32)


def snap_to_star(gray: np.ndarray, x: float, y: float, radius: int = 8):
    """Move a user click to the brightest nearby peak (sub-pixel)."""
    h, w = gray.shape
    x0, x1 = int(max(2, x - radius)), int(min(w - 3, x + radius + 1))
    y0, y1 = int(max(2, y - radius)), int(min(h - 3, y + radius + 1))
    if x1 <= x0 or y1 <= y0:
        return float(x), float(y)
    patch = cv2.GaussianBlur(gray[y0 - 2:y1 + 2, x0 - 2:x1 + 2], (0, 0), 1.0)[2:-2, 2:-2]
    iy, ix = np.unravel_index(int(np.argmax(patch)), patch.shape)
    py, px = y0 + iy, x0 + ix
    win = gray[py - 2:py + 3, px - 2:px + 3]
    win = np.clip(win - np.median(patch), 0, None)
    tot = win.sum()
    if tot <= 0:
        return float(px), float(py)
    off = np.arange(-2, 3)
    return float(px + (win.sum(0) * off).sum() / tot), float(py + (win.sum(1) * off).sum() / tot)
