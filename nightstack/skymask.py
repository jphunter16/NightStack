"""Automatic sky / foreground separation and the user-editable mask model.

Detection combines three cues computed on small versions of every frame:
  * brightness   - the night sky is usually brighter than a silhouetted foreground
  * star motion  - stars drift between exposures, the ground does not
  * star density - point sources are much denser in the sky
These are fused into a "skyness" score and a horizon line is found with dynamic
programming (best boundary per column with a smoothness penalty).  The line is then
refined with GrabCut in a narrow band so trees and rocks get detailed edges.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

from .stars import highpass


def _norm(x: np.ndarray, lo=2, hi=98) -> np.ndarray:
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-12), 0, 1).astype(np.float32)


def _mhp(img: np.ndarray) -> np.ndarray:
    """Median high-pass: keeps point sources, ignores brightness steps such as the horizon edge."""
    return img - cv2.medianBlur(img.astype(np.float32), 5)


def skyness_map(smalls: Sequence[np.ndarray], ref_idx: int):
    """Return (score in [0,1], vertical-edge map) at the small working resolution."""
    stack = np.stack(smalls).astype(np.float32)
    n = len(smalls)
    ref = stack[ref_idx]
    med = np.median(stack, 0) if n > 1 else ref
    lb = np.log(np.clip(cv2.GaussianBlur(med, (0, 0), 2.0), 1e-5, None))
    bright = _norm(lb)
    hp_ref = _mhp(ref)
    # Local noise level (sky glow near the horizon is noisier than the zenith or the ground).
    noise = cv2.GaussianBlur(np.abs(hp_ref), (0, 0), 10.0) * 1.25 + 1e-9
    dil = cv2.dilate(hp_ref, np.ones((3, 3), np.uint8))
    peaks = ((hp_ref >= dil) & (hp_ref > 3.5 * noise)).astype(np.float32)
    density = _norm(cv2.GaussianBlur(peaks, (0, 0), 8.0), 0, 90)
    if n >= 3:
        hp = np.stack([_mhp(f) for f in stack])
        hp_med = np.median(hp, 0)
        moving = np.mean(np.clip(np.abs(hp - hp_med) / noise - 3.0, 0, None), 0)
        motion = _norm(cv2.GaussianBlur(np.sqrt(moving), (0, 0), 8.0), 0, 90)
        score = 0.10 * bright + 0.45 * density + 0.45 * motion
    else:
        score = 0.15 * bright + 0.85 * density
    score = cv2.GaussianBlur(score, (0, 0), 1.5)
    # Horizon edges: brightness drops going down from sky to silhouetted ground.
    lb_s = cv2.GaussianBlur(lb, (0, 0), 1.5)
    edge = np.zeros_like(lb_s)
    edge[2:-2] = np.clip(lb_s[:-4] - lb_s[4:], 0, None)
    edge = _norm(edge, 0, 99.5)
    return score, edge


def dp_horizon(score: np.ndarray, edge: Optional[np.ndarray] = None, threshold: Optional[float] = None,
               smooth: float = 0.02, max_jump_frac: float = 0.35, edge_weight: float = 3.0,
               dark_sky_weight: float = 0.08, ground_reward: float = 0.5) -> np.ndarray:
    """Return, per column, the first ground row (rows above it are sky)."""
    h, w = score.shape
    score = cv2.GaussianBlur(score, (0, 0), max(2.0, 0.012 * w))  # fill gaps between stars
    if threshold is None:
        u8 = (np.clip(score, 0, 1) * 255).astype(np.uint8)
        otsu = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
        low = float(np.mean(score[score <= otsu])) if (score <= otsu).any() else 0.0
        threshold = low + 0.55 * (otsu - low)  # ground is uniform; sky is patchy, so lean low
    s = score - threshold
    # Sky-labelled pixels that look dark/empty (e.g. the zenith) cost less than starry ground.
    above = np.where(s < 0, -dark_sky_weight * s, -s)
    ca = np.vstack([np.zeros((1, w), np.float32), np.cumsum(above, 0)])
    below = np.where(s < 0, ground_reward * s, s)  # empty-looking areas are only weak ground evidence
    cb = np.vstack([np.zeros((1, w), np.float32), np.cumsum(below, 0)])
    cost = ca + (cb[-1][None, :] - cb)  # (h+1, w): boundary at row y
    scale = float(np.mean(np.abs(s))) + 1e-6
    if edge is not None:
        e = np.vstack([edge, np.zeros((1, w), np.float32)])
        cost = cost - edge_weight * scale * h * 0.1 * e
    lam = smooth * scale * 10
    K = max(2, int(h * max_jump_frac))
    D = cost[:, 0].astype(np.float32).copy()
    back = np.zeros((w, h + 1), np.int32)
    idx = np.arange(h + 1)
    for c in range(1, w):
        best = np.full(h + 1, np.inf, np.float32)
        arg = np.zeros(h + 1, np.int32)
        for k in range(-K, K + 1):
            src = idx - k
            valid = (src >= 0) & (src <= h)
            cand = np.full(h + 1, np.inf, np.float32)
            cand[valid] = D[src[valid]] + lam * abs(k)
            better = cand < best
            best[better] = cand[better]
            arg[better] = src[better]
        D = best + cost[:, c]
        back[c] = arg
    y = np.zeros(w, np.int32)
    y[-1] = int(np.argmin(D))
    for c in range(w - 1, 0, -1):
        y[c - 1] = back[c, y[c]]
    return y


def horizon_to_mask(yc: np.ndarray, h: int) -> np.ndarray:
    return ((np.arange(h)[:, None] < yc[None, :]) * 255).astype(np.uint8)


def grabcut_refine(color_u8: np.ndarray, mask: np.ndarray, band: int) -> np.ndarray:
    """Refine a sky mask (255 = sky) inside a band around its edge."""
    m = (mask > 127).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    sure_sky = cv2.erode(m, k)
    sure_gnd = cv2.erode(1 - m, k)
    if sure_sky.sum() < 50 or sure_gnd.sum() < 50:
        return mask
    gc = np.full(m.shape, cv2.GC_PR_BGD, np.uint8)
    gc[m == 1] = cv2.GC_PR_FGD
    gc[sure_sky == 1] = cv2.GC_FGD
    gc[sure_gnd == 1] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(color_u8, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Keep only sky regions connected to the image top / large blobs (removes speckles).
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(out, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        top = stats[i, cv2.CC_STAT_TOP]
        if top == 0 or stats[i, cv2.CC_STAT_AREA] > 0.002 * out.size:
            keep[i] = True
    out = np.where(keep[lab], 255, 0).astype(np.uint8)
    inv = 255 - out  # fill tiny ground holes in the sky
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    small = np.zeros(n, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.0005 * out.size:
            small[i] = True
    out[small[lab]] = 255
    return out


def simplify_horizon(yc: np.ndarray, max_points: int = 120) -> List[tuple]:
    w = len(yc)
    curve = np.stack([np.arange(w), yc], 1).astype(np.int32).reshape(-1, 1, 2)
    eps = 1.0
    while True:
        approx = cv2.approxPolyDP(curve, eps, False).reshape(-1, 2)
        if len(approx) <= max_points:
            break
        eps *= 1.5
    return [(float(x), float(y)) for x, y in approx]


def mask_to_horizon(mask: np.ndarray) -> np.ndarray:
    """Per column: number of sky rows counted from the top until the first ground pixel."""
    h, w = mask.shape
    g = mask <= 127
    first = np.where(g.any(0), g.argmax(0), h)
    return first.astype(np.int32)


def auto_detect(smalls: Sequence[np.ndarray], ref_idx: int, ref_color_small: np.ndarray,
                out_size: tuple, refine: bool = True):
    """Returns (mask at out_size (w,h) 255=sky, horizon_points in out coords, no_foreground flag)."""
    score, edge = skyness_map(smalls, ref_idx)
    sh, sw = score.shape
    yc = dp_horizon(score, edge)
    no_fg = np.mean(yc >= sh - 2) > 0.9
    W, H = out_size
    sx, sy = W / sw, H / sh
    if no_fg:
        return np.full((H, W), 255, np.uint8), [], True
    from scipy.ndimage import median_filter
    yc = median_filter(yc, size=3, mode="nearest")
    mask_small = horizon_to_mask(yc, sh)
    if refine:
        # Work at ~1200px for detailed edges.
        rs = min(1.0, 1200.0 / max(W, H))
        rw, rh = max(8, int(W * rs)), max(8, int(H * rs))
        col = cv2.resize(ref_color_small, (rw, rh), interpolation=cv2.INTER_AREA)
        col = np.clip(stretch_for_display(col) * 255, 0, 255).astype(np.uint8)
        m0 = cv2.resize(mask_small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        m1 = grabcut_refine(col, m0, band=max(4, int(0.06 * rh)))
        mask = cv2.resize(m1, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    return mask, horizon_points_from_mask(mask), False


def horizon_points_from_mask(mask: np.ndarray, max_points: int = 90) -> List[tuple]:
    """Green control points that trace the top edge of the foreground."""
    yc = mask_to_horizon(mask)
    if np.all(yc >= mask.shape[0]):
        return []
    return simplify_horizon(np.minimum(yc, mask.shape[0] - 1), max_points)


def stretch_for_display(img: np.ndarray, target_bg: float = 0.22, linked: bool = True) -> np.ndarray:
    """Siril / PixInsight-style auto stretch (midtones transfer) for previewing linear data."""
    x = img.astype(np.float32)
    sample = x[::4, ::4].reshape(-1, x.shape[-1]) if x.ndim == 3 else x[::4, ::4].reshape(-1, 1)
    med = np.median(sample, 0)
    mad = 1.4826 * np.median(np.abs(sample - med), 0) + 1e-6
    if linked:
        med = np.full_like(med, med.mean())
        mad = np.full_like(mad, mad.mean())
    shadows = np.clip(med - 2.8 * mad, 0, None)
    y = np.clip((x - shadows) / np.maximum(1 - shadows, 1e-6), 0, 1)
    m = np.clip(med - shadows, 1e-6, 1)  # where the background now sits
    # midtones balance so that background -> target_bg
    mb = (m * (target_bg - 1)) / ((2 * target_bg - 1) * m - target_bg)
    mb = np.clip(mb, 1e-4, 0.9999)
    return ((mb - 1) * y / ((2 * mb - 1) * y - mb)).astype(np.float32)


# ----------------------------------------------------------------------------- editable model
class SkyMaskModel:
    """The mask the user sees and edits, held at display resolution.

    base      - automatic (or horizon-point derived) mask, 255 = sky
    edits     - user brush layer: +1 force sky, -1 force ground, 0 untouched
    horizon   - green control points along the horizon (display coords)
    """

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.base = np.full((h, w), 255, np.uint8)
        self.edits = np.zeros((h, w), np.int8)
        self.horizon: List[tuple] = []
        self.no_foreground = False

    def mask(self) -> np.ndarray:
        if self.no_foreground:
            return np.full((self.h, self.w), 255, np.uint8)
        m = self.base.copy()
        m[self.edits > 0] = 255
        m[self.edits < 0] = 0
        return m

    def rebuild_from_horizon(self) -> None:
        if len(self.horizon) == 0:
            self.base[:] = 255
            return
        pts = sorted(self.horizon)
        xs = np.array([p[0] for p in pts], np.float32)
        ys = np.array([p[1] for p in pts], np.float32)
        yc = np.interp(np.arange(self.w), xs, ys)
        self.base = horizon_to_mask(np.round(yc).astype(np.int32), self.h)

    def paint(self, x: float, y: float, radius: float, value: int, last=None) -> None:
        val = int(np.sign(value))
        layer = self.edits.view(np.uint8)  # draw onto int8 via uint8 view
        colour = int(np.uint8(np.int8(val)))
        if last is not None:
            cv2.line(layer, (int(last[0]), int(last[1])), (int(x), int(y)), colour,
                     max(1, int(2 * radius)), cv2.LINE_8)
        cv2.circle(layer, (int(x), int(y)), max(1, int(radius)), colour, -1, cv2.LINE_8)

    def clear_edits(self) -> None:
        self.edits[:] = 0

    def full_res(self, W: int, H: int) -> np.ndarray:
        m = cv2.resize(self.mask(), (W, H), interpolation=cv2.INTER_LINEAR)
        return np.where(m > 127, 255, 0).astype(np.uint8)
