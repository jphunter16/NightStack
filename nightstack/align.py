"""Sky registration.

For a tripod-mounted rectilinear lens the apparent sky rotation is exactly a
homography in image coordinates (H = K R K^-1), so a homography fitted to tracked
stars aligns the sky precisely.  Wide-angle lens distortion leaves small residuals,
which the optional "local" model removes with a smooth displacement field.

Stars are tracked from the reference frame with pyramidal Lucas-Kanade optical flow
(forward-backward checked), seeded by the previous frame's solution plus a masked
phase-correlation shift, and the transform is fitted with RANSAC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np

from .stars import point_highpass


@dataclass
class AlignResult:
    index: int
    ok: bool
    H: np.ndarray
    n_tracked: int = 0
    n_inliers: int = 0
    rms: float = 0.0
    shift_px: float = 0.0
    field: Optional[np.ndarray] = None  # (gh, gw, 2) residual displacement grid
    message: str = ""
    enabled: bool = True


def tracking_image(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    sub = point_highpass(gray)
    sel = sub[::4, ::4] if mask is None else sub[::4, ::4][mask[::4, ::4] > 0]
    if sel.size < 100:
        sel = sub[::4, ::4].ravel()
    noise = 1.4826 * float(np.median(np.abs(sel - np.median(sel)))) + 1e-9
    # asinh stretch scaled by noise: robust to satellites, planes and bright planets
    u = np.arcsinh(np.clip(sub, 0, None) / (2 * noise)) / np.arcsinh(200.0)
    return (np.clip(u, 0, 1) * 255).astype(np.uint8)


def refine_centroids(hp: np.ndarray, pts: np.ndarray, r: int = 3) -> np.ndarray:
    """Sub-pixel intensity-weighted centroids on the full-precision high-passed image."""
    h, w = hp.shape
    p = pts.reshape(-1, 2).astype(np.float64)
    out = p.copy()
    for _ in range(2):
        xi = np.clip(np.round(out[:, 0]).astype(int), r, w - r - 1)
        yi = np.clip(np.round(out[:, 1]).astype(int), r, h - r - 1)
        off = np.arange(-r, r + 1)
        win = hp[yi[:, None, None] + off[None, :, None], xi[:, None, None] + off[None, None, :]]
        win = np.clip(win - np.median(win.reshape(len(p), -1), 1)[:, None, None], 0, None)
        tot = win.sum((1, 2)) + 1e-12
        cx = xi + (win * off[None, None, :]).sum((1, 2)) / tot
        cy = yi + (win * off[None, :, None]).sum((1, 2)) / tot
        ok = tot > 1e-9
        out[ok, 0] = cx[ok]
        out[ok, 1] = cy[ok]
    return out.astype(np.float32)


def _coarse(u8: np.ndarray, mask: Optional[np.ndarray], f: int = 4) -> np.ndarray:
    small = cv2.resize(u8, (u8.shape[1] // f, u8.shape[0] // f), interpolation=cv2.INTER_AREA).astype(np.float32)
    if mask is not None:
        ms = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        k = max(3, int(0.02 * max(ms.shape)))
        ms = cv2.erode(ms, np.ones((k, k), np.uint8))  # stay clear of the static horizon edge
        small *= (ms > 200)
    return small


def _phase_shift(a: np.ndarray, b: np.ndarray, f: int = 4):
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return dx * f, dy * f, resp


def _local_field(p_ref, p_obs, H, shape, grid: int = 40):
    h, w = shape
    pred = cv2.perspectiveTransform(p_ref.reshape(-1, 1, 2), H).reshape(-1, 2)
    res = p_obs - pred
    if len(res) < 30 or np.median(np.linalg.norm(res, axis=1)) < 0.12:
        return None
    gw = grid
    gh = max(4, int(round(grid * h / w)))
    gx = np.linspace(0, w - 1, gw)
    gy = np.linspace(0, h - 1, gh)
    GX, GY = np.meshgrid(gx, gy)
    G = np.stack([GX.ravel(), GY.ravel()], 1)
    sigma = 0.08 * max(h, w)
    d2 = ((G[:, None, :] - p_ref[None, :, :]) ** 2).sum(-1)
    wgt = np.exp(-d2 / (2 * sigma * sigma))
    lam = 2.0  # pulls the field to zero where there are few stars
    field = (wgt @ res) / (wgt.sum(1, keepdims=True) + lam)
    return field.reshape(gh, gw, 2).astype(np.float32)


def _apply(H, field, pts, shape):
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    if field is not None:
        h, w = shape
        gh, gw, _ = field.shape
        cx = (pts[:, 0] * (gw - 1) / (w - 1)).astype(np.float32).reshape(-1, 1)
        cy = (pts[:, 1] * (gh - 1) / (h - 1)).astype(np.float32).reshape(-1, 1)
        d = cv2.remap(field, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, 2)
        out = out + d
    return out


def align_sequence(get_gray: Callable[[int], np.ndarray], n: int, ref_idx: int, ref_pts: np.ndarray,
                   sky_mask: Optional[np.ndarray], local: bool = False,
                   progress: Callable[[float, str], None] = lambda p, m: None) -> List[AlignResult]:
    results: List[Optional[AlignResult]] = [None] * n
    ref_gray = get_gray(ref_idx)
    shape = ref_gray.shape
    ref_u8 = tracking_image(ref_gray, sky_mask)
    ref_coarse = _coarse(ref_u8, sky_mask)
    p0 = np.ascontiguousarray(ref_pts, np.float32).reshape(-1, 1, 2)
    ref_cent = refine_centroids(point_highpass(ref_gray), p0.reshape(-1, 2))
    results[ref_idx] = AlignResult(ref_idx, True, np.eye(3), len(p0), len(p0), 0.0, 0.0, None, "Reference")
    lk = dict(winSize=(25, 25), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    order = [list(range(ref_idx + 1, n)), list(range(ref_idx - 1, -1, -1))]
    done = 0
    for seq in order:
        H_prev = np.eye(3)
        prev_coarse = ref_coarse
        for i in seq:
            done += 1
            progress(done / max(1, n - 1), f"Tracking stars in frame {i + 1}/{n}")
            g = get_gray(i)
            u8 = tracking_image(g, sky_mask)
            cur_coarse = _coarse(u8, sky_mask)
            dx, dy, resp = _phase_shift(prev_coarse, cur_coarse)
            base = cv2.perspectiveTransform(p0.astype(np.float64), H_prev).astype(np.float32)
            guesses = [base]
            if resp > 0.03:
                guesses.insert(0, base + np.array([dx, dy], np.float32))
            best = None
            for guess in guesses:  # phase guess first, plain previous solution as fallback
                p1, st, _ = cv2.calcOpticalFlowPyrLK(ref_u8, u8, p0, guess.copy(),
                                                     flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(u8, ref_u8, p1, p0.copy(),
                                                      flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
                fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
                good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.75)
                if best is None or good.sum() > best[1].sum():
                    best = (p1, good)
                if good.sum() > 0.5 * len(p0):
                    break
            p1, good = best
            a = ref_cent[good]
            b = refine_centroids(point_highpass(g), p1.reshape(-1, 2)[good])
            res = AlignResult(i, False, H_prev.copy(), int(good.sum()))
            if len(a) < 8:
                res.message = "Too few stars tracked (clouds or a big jump?)"
                results[i] = res
                continue
            H, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.0, maxIters=4000, confidence=0.999)
            if H is None:
                res.message = "Could not fit a transform"
                results[i] = res
                continue
            inl = inl.ravel().astype(bool)
            if inl.sum() >= 8:
                H2, _ = cv2.findHomography(a[inl], b[inl], 0)
                if H2 is not None:
                    H = H2
            field = _local_field(a[inl], b[inl], H, shape) if local else None
            pred = _apply(H, field, a[inl], shape)
            rms = float(np.sqrt(np.mean(np.sum((pred - b[inl]) ** 2, 1)))) if inl.any() else 99.0
            shift = float(np.median(np.linalg.norm(b - a, axis=1)))
            res.H, res.field, res.n_inliers, res.rms, res.shift_px = H, field, int(inl.sum()), rms, shift
            if inl.sum() < max(8, 0.15 * len(a)) or rms > 3.0:
                res.message = "Alignment unreliable - frame will be skipped for the sky"
            else:
                res.ok = True
                res.message = "Aligned"
                H_prev = H
                prev_coarse = cur_coarse
            results[i] = res
    return results  # type: ignore[return-value]


def warp(img: np.ndarray, H: np.ndarray, field: Optional[np.ndarray], mask: Optional[np.ndarray] = None,
         interp: str = "lanczos", strip: int = 256):
    """Resample a frame into the reference sky geometry. Returns (warped, warped_mask)."""
    h, w = img.shape[:2]
    flag = {"lanczos": cv2.INTER_LANCZOS4, "cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR}[interp]
    out = np.empty_like(img)
    mout = np.zeros((h, w), np.uint8) if mask is not None else None
    xs = np.arange(w, dtype=np.float64)
    if field is not None:
        gh, gw, _ = field.shape
        cxs = (xs * (gw - 1) / (w - 1)).astype(np.float32)
    for r0 in range(0, h, strip):
        r1 = min(h, r0 + strip)
        ys = np.arange(r0, r1, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        den = H[2, 0] * X + H[2, 1] * Y + H[2, 2]
        mx = ((H[0, 0] * X + H[0, 1] * Y + H[0, 2]) / den).astype(np.float32)
        my = ((H[1, 0] * X + H[1, 1] * Y + H[1, 2]) / den).astype(np.float32)
        if field is not None:
            cx = np.broadcast_to(cxs, mx.shape).astype(np.float32)
            cy = np.broadcast_to((ys * (gh - 1) / (h - 1)).astype(np.float32)[:, None], mx.shape).copy()
            d = cv2.remap(field, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mx += d[..., 0]
            my += d[..., 1]
        out[r0:r1] = cv2.remap(img, mx, my, flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if mask is not None:
            mout[r0:r1] = cv2.remap(mask, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
    np.clip(out, 0, None, out=out)
    return out, mout
