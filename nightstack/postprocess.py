"""Finishing: gradient removal, colour neutralisation, noise reduction and tone mapping."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .imageio import linear_to_srgb, to_gray


@dataclass
class FinishParams:
    exposure: float = 0.0          # EV
    black_point: float = 0.0       # fraction of sky background removed
    saturation: float = 1.0
    contrast: float = 0.0          # -1..1
    remove_gradient: bool = False
    gradient_degree: int = 2
    neutralize: bool = False
    sky_denoise: float = 0.0       # 0..1
    ground_denoise: float = 0.0    # 0..1
    protect_highlights: bool = False
    linear_output: bool = False
    enhance_stars: float = 0.0     # 0..1  (Sequator "Enhance star light")
    tone: str = "natural"          # natural | asinh (colour-preserving astro stretch, Siril-style)
    asinh_strength: float = 30.0

    def to_dict(self):
        return asdict(self)


def _bg_stats(img: np.ndarray, mask: np.ndarray):
    small = cv2.resize(img, (max(8, img.shape[1] // 4), max(8, img.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    if ms.sum() < 100:
        ms = np.ones(ms.shape, bool)
    px = small[ms]
    med = np.median(px, 0)
    return med


def gradient_model(img: np.ndarray, mask: np.ndarray, degree: int = 2, grid: int = 24) -> np.ndarray:
    h, w = img.shape[:2]
    sw = 512
    sh = max(8, int(h * sw / w))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST) > 127
    gy = max(4, int(grid * sh / sw))
    xs, ys, vals = [], [], []
    for j in range(gy):
        for i in range(grid):
            y0, y1 = j * sh // gy, (j + 1) * sh // gy
            x0, x1 = i * sw // grid, (i + 1) * sw // grid
            mm = ms[y0:y1, x0:x1]
            if mm.mean() < 0.7:
                continue
            px = small[y0:y1, x0:x1][mm]
            xs.append((x0 + x1) / 2 / sw)
            ys.append((y0 + y1) / 2 / sh)
            vals.append(np.median(px, 0))
    if len(vals) < (degree + 1) * (degree + 2) // 2 + 3:
        return np.zeros_like(img)
    xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)

    def terms(x, y):
        return np.stack([x ** a * y ** b for a in range(degree + 1) for b in range(degree + 1 - a)], -1)

    A = terms(xs, ys)
    keep = np.ones(len(vals), bool)
    lum = vals.mean(1)
    for _ in range(4):  # reject Milky Way / bright nebula boxes
        coef, *_ = np.linalg.lstsq(A[keep], lum[keep], rcond=None)
        r = lum - A @ coef
        s = 1.4826 * np.median(np.abs(r[keep])) + 1e-9
        keep = r < 1.5 * s
    X, Y = np.meshgrid(np.linspace(0, 1, sw), np.linspace(0, 1, sh))
    T = terms(X.ravel(), Y.ravel())
    model = np.zeros((sh, sw, 3), np.float32)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], vals[keep, c], rcond=None)
        model[..., c] = (T @ coef).reshape(sh, sw)
    return cv2.resize(model, (w, h), interpolation=cv2.INTER_CUBIC)


def apply_finish(img: np.ndarray, mask: np.ndarray, p: FinishParams, display: bool = True) -> np.ndarray:
    """img: linear stack; mask: 255 = sky. Returns linear (if p.linear_output and not display) or sRGB."""
    out = np.array(img, dtype=np.float32)  # always a fresh float32 copy
    m = cv2.GaussianBlur((mask > 127).astype(np.float32), (0, 0), 2.0)[..., None]
    if p.remove_gradient and (mask > 127).mean() > 0.05:
        model = gradient_model(out, mask, p.gradient_degree)
        level = np.median(model[mask > 127], 0) if (mask > 127).any() else 0
        out = out - m * (model - level)
    if p.neutralize:
        bg = _bg_stats(out, mask)
        target = bg.mean()
        # additive light-pollution cast in the sky, gentle white balance on the ground
        out = out - m * (bg - target)
        out = np.clip(out, 0, None)
    if p.sky_denoise > 0 or p.ground_denoise > 0:
        noise = float(np.median(np.abs(out[::4, ::4] - cv2.medianBlur(out, 3)[::4, ::4]))) * 1.4826 + 1e-6
        if p.sky_denoise > 0:
            dn = cv2.bilateralFilter(out, 5, noise * 6 * p.sky_denoise, 1.5 + p.sky_denoise)
            out = out * (1 - m * p.sky_denoise) + dn * (m * p.sky_denoise)
        if p.ground_denoise > 0:
            dn = cv2.bilateralFilter(out, 7, noise * 8 * p.ground_denoise, 2 + 2 * p.ground_denoise)
            g = (1 - m) * p.ground_denoise
            out = out * (1 - g) + dn * g
    if p.enhance_stars > 0:
        # Boost point sources only (median background keeps nebulae / Milky Way / ground untouched).
        g = out.astype(np.float32)
        bgp = cv2.medianBlur(cv2.medianBlur(g, 5), 5)
        stars = np.clip(g - bgp, 0, None)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        stars = cv2.GaussianBlur(cv2.dilate(stars, k), (0, 0), 0.6) * 0.5 + stars * 0.5
        out = out + m * stars * (2.0 * p.enhance_stars)
    if p.linear_output and not display:
        return np.clip(out, 0, 1)
    bg = _bg_stats(out, mask).mean()
    out = np.clip(out - p.black_point * bg, 0, None)
    out *= 2.0 ** p.exposure
    if p.tone == "asinh":
        # Colour-preserving arcsinh stretch: lifts faint sky while keeping star colour (Siril / Lupton).
        lum = to_gray(out)[..., None]
        b = max(p.asinh_strength, 1e-3)
        scale = np.arcsinh(b * lum) / (np.arcsinh(b) * np.maximum(lum, 1e-7))
        s = np.clip(out * scale, 0, None)
        s = s / np.maximum(1.0, s.max(-1, keepdims=True))  # keep hue when a channel saturates
        if p.contrast != 0:
            s = np.clip((s - 0.5) * (1 + p.contrast) + 0.5, 0, 1)
        if p.saturation != 1:
            lm = to_gray(s)[..., None]
            s = np.clip(lm + (s - lm) * p.saturation, 0, 1)
        return s.astype(np.float32)
    if p.protect_highlights:
        knee = 0.6  # soft shoulder: bright stars & lights roll off instead of clipping hard
        over = np.clip(out - knee, 0, None)
        out = np.where(out > knee, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee))), out)
    s = linear_to_srgb(out)
    if p.contrast != 0:
        k = 1 + p.contrast
        s = np.clip((s - 0.5) * k + 0.5, 0, 1)
    if p.saturation != 1:
        lum = to_gray(s)[..., None]
        s = np.clip(lum + (s - lum) * p.saturation, 0, 1)
    return s.astype(np.float32)


def auto_exposure(img: np.ndarray, mask: np.ndarray, black_point: float = 0.5, target_srgb: float = 0.16) -> float:
    bg = float(_bg_stats(img, mask).mean())
    if bg <= 0:
        return 0.0
    remaining = (1 - black_point) * bg
    # find linear value whose sRGB is target
    t = ((target_srgb + 0.055) / 1.055) ** 2.4
    return float(np.clip(np.log2(t / max(remaining, 1e-7)), -4, 12))
