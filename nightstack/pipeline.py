"""The Project object ties the whole workflow together (used by both the GUI and the CLI)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Callable, List, Optional

import cv2
import numpy as np

from . import align as _align
from .calibration import HotPixelDetector, build_master, calibrate, fix_pixels, normalise_flat
from .imageio import (FrameCache, load_image, new_cache_dir, remove_cache_dir, resize_max,
                      save_image, to_gray)
from .postprocess import FinishParams, apply_finish, auto_exposure
from .skymask import SkyMaskModel, auto_detect, stretch_for_display
from .stack import blend, combine
from .stars import detect_stars, snap_to_star


class Cancelled(Exception):
    pass


Progress = Callable[[float, str], None]


def _noop(p, m):
    pass


@dataclass
class Settings:
    half_size: bool = False
    hot_pixels: bool = True
    ref_index: Optional[int] = None
    align_local: bool = True
    interp: str = "lanczos"
    mode: str = "landscape"          # landscape | trails
    sky_method: str = "sigma"
    ground_method: str = "median"
    blend_mode: str = "min_noise"
    feather: float = 4.0
    horizon_margin: int = 3
    horizon_strength: float = 1.0
    kappa: float = 2.5
    max_align_stars: int = 800


@dataclass
class StarSet:
    pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    flux: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    removed: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


class Project:
    DISPLAY_MAX = 2400

    def __init__(self):
        self.lights: List[str] = []
        self.darks: List[str] = []
        self.flats: List[str] = []
        self.biases: List[str] = []
        self.settings = Settings()
        self.cache_dir: Optional[str] = None
        self.orig: Optional[FrameCache] = None
        self.aligned: Optional[FrameCache] = None
        self.valid: Optional[FrameCache] = None
        self.shape = None
        self.ref_idx = 0
        self.ref_gray: Optional[np.ndarray] = None
        self.display_linear: Optional[np.ndarray] = None
        self.display_scale = 1.0
        self.smalls: List[np.ndarray] = []
        self.stars = StarSet()
        self.mask_model: Optional[SkyMaskModel] = None
        self.align_results: List[_align.AlignResult] = []
        self.sky_stack = self.ground_stack = self.count = self.result = self.alpha = None
        self.final_mask: Optional[np.ndarray] = None
        self.n_sky = 0
        self.hot_count = 0
        self.prepared = self.aligned_ok = self.stacked = False

    # ------------------------------------------------------------------ housekeeping
    def cleanup(self):
        for c in (self.orig, self.aligned, self.valid):
            if c is not None:
                c.close()
        self.orig = self.aligned = self.valid = None
        if self.cache_dir:
            remove_cache_dir(self.cache_dir)
        self.cache_dir = None

    def invalidate(self, from_step: str):
        if from_step in ("prepare",):
            self.prepared = False
        if from_step in ("prepare", "mask", "align"):
            self.aligned_ok = False
        self.stacked = False

    # ------------------------------------------------------------------ step 1/2: load & calibrate
    def prepare(self, progress: Progress = _noop):
        if len(self.lights) == 0:
            raise ValueError("Add some photos first.")
        self.cleanup()
        self.cache_dir = new_cache_dir()
        s = self.settings
        n = len(self.lights)
        has_cal = bool(self.darks or self.flats or self.biases)
        span = 0.25 if has_cal else 0.0
        bias = build_master(self.biases, self.cache_dir, "bias", s.half_size, progress=progress,
                            p0=0, p1=span / 3) if self.biases else None
        dark = build_master(self.darks, self.cache_dir, "dark", s.half_size, progress=progress,
                            p0=span / 3, p1=2 * span / 3) if self.darks else None
        flat = build_master(self.flats, self.cache_dir, "flat", s.half_size, subtract=bias,
                            progress=progress, p0=2 * span / 3, p1=span) if self.flats else None
        flat_norm = normalise_flat(flat) if flat is not None else None

        first = load_image(self.lights[0], s.half_size)
        H, W, _ = first.shape
        for name, m in (("dark", dark), ("bias", bias), ("flat", flat)):
            if m is not None and m.shape != first.shape:
                raise ValueError(f"The master {name} ({m.shape[1]}x{m.shape[0]}) does not match the "
                                 f"photos ({W}x{H}).")
        self.shape = (H, W)
        self.orig = FrameCache(self.cache_dir, "orig", n, H, W)
        hot = HotPixelDetector((H, W, 3)) if (s.hot_pixels and n >= 3) else None
        self.smalls = []
        sw = 640
        sh = max(8, int(round(H * sw / W)))
        for i, p in enumerate(self.lights):
            progress(span + (0.75 - span) * i / n, f"Loading photo {i + 1} of {n}: {os.path.basename(p)}")
            img = first if i == 0 else load_image(p, s.half_size)
            if img.shape != first.shape:
                raise ValueError(f"{os.path.basename(p)} is {img.shape[1]}x{img.shape[0]} but the first photo "
                                 f"is {W}x{H}. All photos must be the same size/orientation.")
            img = calibrate(img, bias, dark, flat_norm)
            if hot is not None:
                hot.add(img)
            self.orig.write(i, img)
            self.smalls.append(cv2.resize(to_gray(img), (sw, sh), interpolation=cv2.INTER_AREA))
        del first
        if hot is not None:
            bad = hot.hot_map()
            self.hot_count = int(bad.any(-1).sum())
            if self.hot_count:
                for i in range(n):
                    progress(0.75 + 0.15 * i / n, f"Removing {self.hot_count:,} hot pixels ({i + 1}/{n})")
                    self.orig.write(i, fix_pixels(self.orig.read(i), bad))
                self.smalls = [cv2.resize(to_gray(self.orig.read(i)), (sw, sh), interpolation=cv2.INTER_AREA)
                               for i in range(n)] if n <= 60 else self.smalls
        self.orig.flush()
        self.set_reference(s.ref_index if s.ref_index is not None else n // 2, progress)
        self.prepared = True
        self.invalidate("mask")
        progress(1.0, "Photos ready")

    def set_reference(self, idx: int, progress: Progress = _noop):
        self.ref_idx = int(np.clip(idx, 0, len(self.lights) - 1))
        ref = self.orig.read(self.ref_idx)
        self.ref_gray = to_gray(ref)
        H, W = self.shape
        self.display_scale = min(1.0, self.DISPLAY_MAX / max(H, W))
        self.display_linear = resize_max(ref, self.DISPLAY_MAX)
        progress(0.95, "Finding stars")
        pts, flux = detect_stars(self.ref_gray, None, max_stars=3000)
        self.stars = StarSet(pts, flux, np.zeros(len(pts), bool))
        dh, dw = self.display_linear.shape[:2]
        self.mask_model = SkyMaskModel(dw, dh)

    def display_image(self) -> np.ndarray:
        return stretch_for_display(self.display_linear)

    # ------------------------------------------------------------------ step 3: sky mask
    def auto_mask(self, refine: bool = True, progress: Progress = _noop):
        progress(0.1, "Looking for the horizon")
        dh, dw = self.display_linear.shape[:2]
        mask, pts, no_fg = auto_detect(self.smalls, self.ref_idx, self.display_linear, (dw, dh), refine=refine)
        mm = self.mask_model
        mm.base = mask
        mm.horizon = pts
        mm.no_foreground = no_fg
        mm.clear_edits()
        self.invalidate("mask")
        progress(1.0, "Sky detected" if not no_fg else "No foreground found - treating the whole image as sky")

    def refine_mask(self, progress: Progress = _noop):
        from .skymask import grabcut_refine
        progress(0.2, "Refining edges")
        mm = self.mask_model
        m = mm.mask()
        col = np.clip(stretch_for_display(self.display_linear) * 255, 0, 255).astype(np.uint8)
        rs = min(1.0, 1200.0 / max(mm.w, mm.h))
        rw, rh = int(mm.w * rs), int(mm.h * rs)
        out = grabcut_refine(cv2.resize(col, (rw, rh), interpolation=cv2.INTER_AREA),
                             cv2.resize(m, (rw, rh), interpolation=cv2.INTER_NEAREST), max(3, int(0.015 * rh)))
        mm.base = np.where(cv2.resize(out, (mm.w, mm.h), interpolation=cv2.INTER_LINEAR) > 127, 255, 0).astype(np.uint8)
        mm.clear_edits()
        from .skymask import horizon_points_from_mask
        mm.horizon = horizon_points_from_mask(mm.base)
        self.invalidate("mask")
        progress(1.0, "Edges refined")

    def star_display_points(self):
        """Star positions in display coordinates and a boolean 'shown' (in sky & not erased)."""
        p = self.stars.pts * self.display_scale
        if len(p) == 0:
            return p, np.zeros(0, bool)
        m = self.mask_model.mask()
        xi = np.clip(p[:, 0].astype(int), 0, m.shape[1] - 1)
        yi = np.clip(p[:, 1].astype(int), 0, m.shape[0] - 1)
        shown = (m[yi, xi] > 127) & ~self.stars.removed
        return p, shown

    def erase_stars(self, x: float, y: float, r: float):
        p = self.stars.pts * self.display_scale
        if len(p):
            hit = (p[:, 0] - x) ** 2 + (p[:, 1] - y) ** 2 <= r * r
            self.stars.removed |= hit

    def restore_stars(self, x: float, y: float, r: float):
        p = self.stars.pts * self.display_scale
        if len(p):
            hit = (p[:, 0] - x) ** 2 + (p[:, 1] - y) ** 2 <= r * r
            self.stars.removed &= ~hit

    def add_star(self, x: float, y: float):
        fx, fy = snap_to_star(self.ref_gray, x / self.display_scale, y / self.display_scale,
                              radius=max(4, int(6 / self.display_scale)))
        self.stars.pts = np.vstack([self.stars.pts, [[fx, fy]]]).astype(np.float32)
        self.stars.flux = np.append(self.stars.flux, np.float32(1e9))  # user stars always used
        self.stars.removed = np.append(self.stars.removed, False)
        return fx * self.display_scale, fy * self.display_scale

    def alignment_points(self) -> np.ndarray:
        p, shown = self.star_display_points()
        pts = self.stars.pts[shown]
        flux = self.stars.flux[shown]
        order = np.argsort(-flux)[: self.settings.max_align_stars]
        return pts[order]

    def full_mask(self) -> np.ndarray:
        H, W = self.shape
        return self.mask_model.full_res(W, H)

    # ------------------------------------------------------------------ step 4: align
    def align(self, progress: Progress = _noop):
        s = self.settings
        n = len(self.lights)
        H, W = self.shape
        sky = self.full_mask()
        pts = self.alignment_points()
        if len(pts) < 12:
            raise ValueError(f"Only {len(pts)} alignment stars are marked in the sky. Add stars or "
                             "paint more sky before aligning.")
        res = _align.align_sequence(lambda i: to_gray(self.orig.read(i)), n, self.ref_idx, pts,
                                    sky, local=s.align_local,
                                    progress=lambda f, m: progress(0.45 * f, m))
        for c in (self.aligned, self.valid):
            if c is not None:
                c.close()
        self.aligned = FrameCache(self.cache_dir, "aligned", n, H, W)
        self.valid = FrameCache(self.cache_dir, "valid", n, H, W, channels=1, dtype=np.uint8)
        for k, r in enumerate(res):
            progress(0.45 + 0.55 * k / n, f"Warping frame {k + 1} of {n}")
            if not r.ok:
                continue
            img = self.orig.read(k)
            if k == self.ref_idx:
                w_img, w_mask = img, sky
            else:
                w_img, w_mask = _align.warp(img, r.H, r.field, sky, s.interp)
            self.aligned.write(k, w_img)
            self.valid.write(k, w_mask)
        self.aligned.flush()
        self.valid.flush()
        self.align_results = res
        self.aligned_ok = True
        self.stacked = False
        progress(1.0, f"Aligned {sum(r.ok for r in res)} of {n} frames")
        return res

    # ------------------------------------------------------------------ step 5: stack
    def stack(self, progress: Progress = _noop):
        s = self.settings
        H, W = self.shape
        n = len(self.lights)
        enabled = [i for i in range(n) if (not self.align_results or self.align_results[i].enabled)]
        sky_idx = [i for i in enabled if self.align_results and self.align_results[i].ok] \
            if s.mode == "landscape" else enabled
        if s.mode == "landscape" and not sky_idx:
            raise ValueError("No aligned frames to stack. Run alignment first.")
        if not enabled:
            raise ValueError("All frames are excluded.")
        sky_mask = self.full_mask()
        self.final_mask = sky_mask
        vtmp = None
        if s.mode == "landscape":
            vtmp = FrameCache(self.cache_dir, "valid-eff", len(sky_idx), H, W, channels=1, dtype=np.uint8)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s.horizon_margin + 1,) * 2) \
                if s.horizon_margin > 0 else None
            for j, i in enumerate(sky_idx):
                v = (np.asarray(self.valid.mm[i]) > 200).astype(np.uint8)
                if k is not None:
                    v = cv2.erode(v, k)
                vtmp.write(j, v)
        sky = np.zeros((H, W, 3), np.float32)
        gnd = np.zeros((H, W, 3), np.float32)
        cnt = np.zeros((H, W), np.float32)
        workers = max(1, min(4, (os.cpu_count() or 2)))
        nbytes = max(len(sky_idx), len(enabled)) * W * 3 * 4 * 8
        step = int(np.clip(160e6 // max(nbytes, 1), 8, 256))
        strips = [(r0, min(H, r0 + step)) for r0 in range(0, H, step)]
        vmm = vtmp.mm if vtmp is not None else None

        def do_strip(r):
            r0, r1 = r
            g = self.orig.rows(r0, r1, enabled)
            gnd[r0:r1], _ = combine(g, None, s.ground_method, s.kappa)
            if s.mode == "trails":
                sky[r0:r1], _ = combine(g, None, "max")
                cnt[r0:r1] = len(enabled)
            else:
                del g
                d = self.aligned.rows(r0, r1, sky_idx)
                v = np.asarray(vmm[:, r0:r1]) > 0
                sky[r0:r1], cnt[r0:r1] = combine(d, v, s.sky_method, s.kappa)

        from concurrent.futures import ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(workers) as ex:  # numpy releases the GIL: real multi-core speed-up
                futures = [ex.submit(do_strip, r) for r in strips]
                for k, f in enumerate(futures):
                    f.result()
                    progress(k / len(futures), f"Stacking ({100 * k // len(futures)}%)")
        except BaseException:
            for f in futures:
                f.cancel()
            raise
        finally:
            if vtmp is not None:
                vtmp.close()
        progress(0.97, "Blending sky and foreground")
        self.n_sky = len(sky_idx)
        mode = "standard" if s.mode == "trails" else s.blend_mode
        self.result, self.alpha = blend(sky, gnd, sky_mask, cnt, self.n_sky, mode, s.feather, s.horizon_strength)
        # keep the separate layers compactly (for previews / layer export)
        self.sky_stack, self.ground_stack, self.count = sky.astype(np.float16), gnd.astype(np.float16), cnt
        del sky, gnd
        self.stacked = True
        progress(1.0, "Stack complete")
        return self.result

    # ------------------------------------------------------------------ step 6: finish & export
    def preview_inputs(self, max_side: int = 1600):
        img = resize_max(self.result, max_side)
        m = cv2.resize(self.final_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return img, m

    def suggest_exposure(self, black_point: float = 0.5) -> float:
        img, m = self.preview_inputs(800)
        return auto_exposure(img, m, black_point)

    def export(self, path: str, params: FinishParams, bits: int = 16, layers: bool = False,
               progress: Progress = _noop):
        progress(0.1, "Applying finishing touches")
        out = apply_finish(self.result, self.final_mask, params, display=False)
        progress(0.7, "Writing file")
        save_image(path, out, bits=bits)
        written = [path]
        if layers:
            base, ext = os.path.splitext(path)
            ext = ext if ext.lower() in (".tif", ".tiff", ".png") else ".tif"
            for name, img in (("sky", self.sky_stack), ("foreground", self.ground_stack)):
                fp = f"{base}_{name}{ext}"
                save_image(fp, apply_finish(img, self.final_mask, params, display=False), bits=bits)
                written.append(fp)
            fp = f"{base}_mask{ext}"
            save_image(fp, self.final_mask.astype(np.float32) / 255.0, bits=8 if ext == ".png" else 16)
            written.append(fp)
        progress(1.0, "Saved")
        return written

    def summary(self) -> dict:
        d = {"lights": self.lights, "darks": self.darks, "flats": self.flats, "biases": self.biases,
             "settings": asdict(self.settings)}
        return d

    def save_settings(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
