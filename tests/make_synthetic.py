"""Generate a synthetic starry-landscape sequence with known ground truth."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nightstack.imageio import linear_to_srgb, save_image  # noqa: E402


def rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def make(outdir, n=10, W=1800, H=1200, f=1100.0, deg_per_frame=0.12, seed=1, noise=0.004):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    Kc = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    pole = np.array([0.35, -0.75, 0.55])  # celestial pole direction in camera coords (y down)
    # stars: random directions in front of camera
    m = 9000
    d = rng.normal(size=(m, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[d[:, 2] > 0.25]
    flux = (rng.pareto(1.6, len(d)) + 0.15) * 0.03
    flux = np.clip(flux, 0, 3)
    # foreground silhouette
    xs = np.arange(W)
    hor = (0.70 * H + 0.06 * H * np.sin(xs / W * 5.1) + 0.03 * H * np.sin(xs / W * 17 + 1)
           - 0.18 * H * np.exp(-((xs - 0.68 * W) / (0.09 * W)) ** 2))
    for tx in (0.2 * W, 0.25 * W, 0.31 * W):  # pine trees
        hw = 0.02 * W
        th = 0.22 * H
        reg = np.abs(xs - tx) < hw
        hor[reg] = np.minimum(hor[reg], 0.72 * H - th * (1 - np.abs(xs[reg] - tx) / hw))
    sky_true = (np.arange(H)[:, None] < hor[None, :])
    gtex = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 3) * 0.02 + 0.006
    ground = np.stack([gtex * 1.1, gtex * 1.0, gtex * 0.8], -1)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    hot = rng.integers(0, [H, W], size=(150, 2))
    Hs = []
    ref = n // 2
    frames = []
    for t in range(n):
        R = rot(pole, np.deg2rad(deg_per_frame * (t - ref)))
        Hs.append(Kc @ R @ np.linalg.inv(Kc))
        p = (Kc @ (R @ d.T)).T
        uv = p[:, :2] / p[:, 2:3]
        sky = np.zeros((H, W), np.float32)
        ok = (uv[:, 0] > -5) & (uv[:, 0] < W + 5) & (uv[:, 1] > -5) & (uv[:, 1] < H + 5)
        for (u, v), fl in zip(uv[ok], flux[ok]):
            x0, y0 = int(u), int(v)
            xa, xb, ya, yb = max(0, x0 - 4), min(W, x0 + 5), max(0, y0 - 4), min(H, y0 + 5)
            if xa >= xb or ya >= yb:
                continue
            gx = X[ya:yb, xa:xb] - u
            gy = Y[ya:yb, xa:xb] - v
            sky[ya:yb, xa:xb] += fl * np.exp(-(gx ** 2 + gy ** 2) / (2 * 1.15 ** 2))
        # Milky way band + skyglow (also rotates, approximated by warping a static band)
        band = np.exp(-((X - 0.5 * W - 0.4 * (Y - 0.5 * H)) / (0.12 * W)) ** 2) * 0.02
        band = cv2.warpPerspective(band, Hs[-1], (W, H))
        glow = 0.012 + 0.03 * (Y / H) ** 2
        skyc = np.stack([sky + band * 0.9 + glow * 1.2, sky + band + glow, sky + band * 1.1 + glow * 0.7], -1)
        img = np.where(sky_true[..., None], skyc, ground)
        img = img + rng.normal(0, noise, img.shape).astype(np.float32) * np.sqrt(np.clip(img / 0.02, 0.3, 30))
        img[hot[:, 0], hot[:, 1]] += 0.25
        if t == 2:  # satellite trail
            cv2.line(img, (100, 80), (1500, 500), (0.3, 0.3, 0.3), 2)
        img = np.clip(img, 0, 1).astype(np.float32)
        frames.append(img)
        save_image(os.path.join(outdir, f"frame_{t:02d}.tif"), linear_to_srgb(img), bits=16)
    np.save(os.path.join(outdir, "truth_H.npy"), np.array(Hs))
    np.save(os.path.join(outdir, "truth_sky.npy"), sky_true)
    return frames, Hs, sky_true


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
