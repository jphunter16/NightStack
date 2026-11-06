"""The Project object ties the whole workflow together (used by both the GUI and the CLI)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Callable, List, Optional

import cv2
import numpy as np

from . import align as _align
from .calibration import (CFACalibrator, HotPixelDetector, build_master, build_master_cfa, calibrate, fix_pixels,
                          normalise_flat, normalise_flat_cfa)
from .imageio import (RAW_EXTS, FrameCache, load_cfa, load_image, new_cache_dir, remove_cache_dir, resize_max,
                      save_image, to_gray)
from .postprocess import FinishParams, apply_finish, auto_exposure
from .skymask import SkyMaskModel, auto_detect, stretch_for_display
from .stack import blend, combine, small_stack_method
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
    demosaic: str = "AHD"
    normalize: bool = True           # Siril-style per-frame background/scale normalisation
    weighting: str = "quality"       # none | quality | noise | fwhm
    auto_reject: bool = True         # drop blurry / cloudy frames automatically
    match_foreground: bool = True    # scale separate foreground photos to the night frames
    ground_frame: Optional[int] = None  # "best single frame": chosen frame (None = automatic)
    auto_small_stack: bool = True    # <10 frames: use min/max (or median) instead of sigma-type clipping


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
        self.foregrounds: List[str] = []   # optional separate foreground exposures (blue hour, long exposure)
        self.settings = Settings()
        self.cache_dir: Optional[str] = None
        self.orig: Optional[FrameCache] = None
        self.aligned: Optional[FrameCache] = None
        self.valid: Optional[FrameCache] = None
        self.fg: Optional[FrameCache] = None
        self.cfa_calibrated = False
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
        self.ground_choice = None        # (label, scores) of the frame used by "best single frame"
        self.method_notes: List[str] = []  # automatic method changes, shown to the user
        self.prepared = self.aligned_ok = self.stacked = False

    # ------------------------------------------------------------------ housekeeping
    def cleanup(self):
        for c in (self.orig, self.aligned, self.valid, self.fg):
            if c is not None:
                c.close()
        self.orig = self.aligned = self.valid = self.fg = None
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
    def _all_raw(self, paths) -> bool:
        return bool(paths) and all(os.path.splitext(p)[1].lower() in RAW_EXTS for p in paths)

    def prepare(self, progress: Progress = _noop):
        if len(self.lights) == 0:
            raise ValueError("Add some photos first.")
        self.cleanup()
        self.cache_dir = new_cache_dir()
        s = self.settings
        n = len(self.lights)
        has_cal = bool(self.darks or self.flats or self.biases)
        span = 0.25 if has_cal else 0.0
        cal_paths = self.darks + self.flats + self.biases
        # Siril-style: calibrate on the raw Bayer data before demosaicing when everything is RAW.
        use_cfa = has_cal and not s.half_size and self._all_raw(self.lights) and self._all_raw(cal_paths)
        self.cfa_calibrated = use_cfa
        bias = dark = flat_norm = None
        cfa_cal = None
        if use_cfa:
            bias = build_master_cfa(self.biases, self.cache_dir, "bias", progress=progress, p0=0, p1=span / 3)
            dark = build_master_cfa(self.darks, self.cache_dir, "dark", progress=progress,
                                    p0=span / 3, p1=2 * span / 3)
            flat = build_master_cfa(self.flats, self.cache_dir, "flat", subtract=bias, progress=progress,
                                    p0=2 * span / 3, p1=span)
            if flat is not None:
                flat_norm = normalise_flat_cfa(flat, load_cfa(self.flats[0])[1])
            ref_shape = load_cfa(self.lights[0])[0].shape
            for name, m in (("dark", dark), ("bias", bias), ("flat", flat_norm)):
                if m is not None and m.shape != ref_shape:
                    raise ValueError(f"The {name} frames ({m.shape[1]}x{m.shape[0]}) come from a different "
                                     f"camera/mode than the photos.")
            cfa_cal = CFACalibrator(bias, dark, flat_norm)
            bias = dark = flat_norm = None   # already applied on the raw data
        else:
            bias = build_master(self.biases, self.cache_dir, "bias", s.half_size, progress=progress,
                                p0=0, p1=span / 3) if self.biases else None
            dark = build_master(self.darks, self.cache_dir, "dark", s.half_size, progress=progress,
                                p0=span / 3, p1=2 * span / 3) if self.darks else None
            flat = build_master(self.flats, self.cache_dir, "flat", s.half_size, subtract=bias,
                                progress=progress, p0=2 * span / 3, p1=span) if self.flats else None
            flat_norm = normalise_flat(flat) if flat is not None else None

        def load(p):
            return load_image(p, s.half_size, s.demosaic, cfa_cal)

        first = load(self.lights[0])
        H, W, _ = first.shape
        for name, m in (("dark", dark), ("bias", bias), ("flat", flat_norm)):
            if m is not None and m.shape != first.shape:
                raise ValueError(f"The master {name} ({m.shape[1]}x{m.shape[0]}) does not match the "
                                 f"photos ({W}x{H}).")
        self.shape = (H, W)
        self.orig = FrameCache(self.cache_dir, "orig", n, H, W)
        hot = HotPixelDetector((H, W, 3)) if (s.hot_pixels and n >= 3) else None
        self.smalls = []
        sw = 640
        sh = max(8, int(round(H * sw / W)))
        end = 0.68 if self.foregrounds else 0.75
        for i, p in enumerate(self.lights):
            progress(span + (end - span) * i / n, f"Loading photo {i + 1} of {n}: {os.path.basename(p)}")
            img = first if i == 0 else load(p)
            if img.shape != first.shape:
                raise ValueError(f"{os.path.basename(p)} is {img.shape[1]}x{img.shape[0]} but the first photo "
                                 f"is {W}x{H}. All photos must be the same size/orientation.")
            img = calibrate(img, bias, dark, flat_norm)
            if hot is not None:
                hot.add(img)
            self.orig.write(i, img)
            self.smalls.append(cv2.resize(to_gray(img), (sw, sh), interpolation=cv2.INTER_AREA))
        del first
        bad = hot.hot_map() if hot is not None else None
        self.hot_count = int(bad.any(-1).sum()) if bad is not None else 0
        if self.hot_count:
            for i in range(n):
                progress(end + 0.12 * i / n, f"Removing {self.hot_count:,} hot pixels ({i + 1}/{n})")
                self.orig.write(i, fix_pixels(self.orig.read(i), bad))
            if n <= 60:
                self.smalls = [cv2.resize(to_gray(self.orig.read(i)), (sw, sh), interpolation=cv2.INTER_AREA)
                               for i in range(n)]
        self.orig.flush()
        if self.foregrounds:
            nf = len(self.foregrounds)
            self.fg = FrameCache(self.cache_dir, "foreground", nf, H, W)
            for i, p in enumerate(self.foregrounds):
                progress(0.82 + 0.1 * i / nf, f"Loading foreground photo {i + 1} of {nf}")
                img = load_image(p, s.half_size, s.demosaic)
                if img.shape != (H, W, 3):
                    raise ValueError(f"Foreground photo {os.path.basename(p)} is a different size than the "
                                     "night photos. Shoot it from the same tripod position and camera mode.")
                if self.hot_count:
                    img = fix_pixels(img, bad)
                self.fg.write(i, img)
            self.fg.flush()
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
        self._grade_frames()
        self.aligned_ok = True
        self.stacked = False
        progress(1.0, f"Aligned {sum(r.ok for r in res)} of {n} frames")
        return res

    def score_ground_frames(self, cache, idx, scale=None, progress: Progress = _noop) -> np.ndarray:
        """Score each candidate foreground frame (lower = better).

        A good frame looks like the consensus (median) of all frames - no headlights, people,
        torch light or a bumped tripod - yet keeps the consensus's fine detail (not blurred).
        """
        if len(idx) == 1:
            return np.zeros(1)
        mask = self.final_mask if self.final_mask is not None else self.full_mask()
        # Full-resolution tiles from the foreground (downsampling would hide slight blur).
        H, W = mask.shape
        t = 160
        ground = mask <= 127
        cand = []
        for y in range(0, H - t, t):
            for x in range(0, W - t, t):
                if ground[y:y + t, x:x + t].mean() > 0.95:
                    cand.append((y, x))
        if len(cand) < 3:   # little or no foreground: use the whole frame
            cand = [(y, x) for y in range(0, H - t, t) for x in range(0, W - t, t)]
        rng = np.random.default_rng(0)
        if len(cand) > 30:
            cand = [cand[i] for i in rng.choice(len(cand), 30, replace=False)]
        gm = np.ones((len(cand), t, t), bool)
        lum = []
        for k, i in enumerate(idx):
            progress(0.02, f"Checking foreground frame {k + 1} of {len(idx)}")
            f = np.stack([np.asarray(cache.mm[i, y:y + t, x:x + t], np.float32) for y, x in cand])
            if scale is not None:
                f = f * scale[k]
            lum.append(to_gray(f.reshape(-1, t, 3)).reshape(len(cand), t, t))
        lum = np.stack([x.reshape(-1, t) for x in lum])   # tiles stacked vertically
        gm = gm.reshape(-1, t)
        med = np.median(lum, 0)
        hp_med = med - cv2.GaussianBlur(med, (0, 0), 1.5)
        b = hp_med[gm]
        devs, noises, gains = [], [], []
        for f in lum:
            r = f - med
            # Large-scale differences from the consensus: headlights, people, torch light, a shifted tripod.
            devs.append(float(np.mean(np.abs(cv2.GaussianBlur(r, (0, 0), 2.0))[gm])))
            # Per-frame noise.
            rr = (r - cv2.GaussianBlur(r, (0, 0), 1.5))[gm]
            noises.append(float(1.4826 * np.median(np.abs(rr - np.median(rr)))))
            # Detail kept: regression of the frame's fine detail on the consensus detail.
            # ~1 when sharp, < 1 when blurred; unlike a correlation it is not fooled by noise.
            a = (f - cv2.GaussianBlur(f, (0, 0), 1.5))[gm]
            gains.append(float(np.sum(a * b) / (np.sum(b * b) + 1e-12)))
        n0 = float(np.median(noises)) + 1e-9
        scores = [d / n0 + nz / n0 + 8.0 * abs(1.0 - g) for d, nz, g in zip(devs, noises, gains)]
        return np.array(scores)

    def _grade_frames_weights_only(self):
        """Re-compute weights after the user changes the weighting mode (keeps their Use ticks)."""
        enabled = [r.enabled for r in self.align_results]
        msgs = [r.message for r in self.align_results]
        auto = self.settings.auto_reject
        self.settings.auto_reject = False
        self._grade_frames()
        self.settings.auto_reject = auto
        for r, e, m in zip(self.align_results, enabled, msgs):
            r.enabled, r.message = e, m

    def _grade_frames(self):
        """Weights + automatic rejection of blurry / trailed / cloudy frames (Siril-style)."""
        res = self.align_results
        ok = [r for r in res if r.ok]
        if not ok:
            return
        fw = np.array([r.fwhm for r in ok])
        bg = np.array([r.background for r in ok])
        nz = np.array([max(r.noise, 1e-9) for r in ok])
        ns = np.array([r.n_inliers for r in ok])
        mf, mb, mn = np.median(fw), np.median(bg), np.median(ns)
        sb = 1.4826 * np.median(np.abs(bg - mb)) + 1e-9
        for r in res:
            if not r.ok:
                continue
            if self.settings.weighting == "quality":
                # Sky-limited photon noise grows with background brightness; blur spreads star light.
                # (Measured pixel noise would reward blurry frames, so it isn't used here.)
                r.weight = float((mb / max(r.background, 1e-9)) * (mf / max(r.fwhm, 1e-3)) ** 2)
            elif self.settings.weighting == "noise":
                r.weight = float((np.median(nz) / max(r.noise, 1e-9)) ** 2)
            elif self.settings.weighting == "fwhm":
                r.weight = float((mf / max(r.fwhm, 1e-3)) ** 2)
            else:
                r.weight = 1.0
            r.weight = float(np.clip(r.weight, 0.2, 5.0))
            if not self.settings.auto_reject or len(ok) < 5 or r.index == self.ref_idx:
                continue
            reason = None
            if r.fwhm > 1.45 * mf:
                reason = "blurry stars (wind, focus or dew?)"
            elif r.roundness < 0.55:
                reason = "stretched stars (camera bumped?)"
            elif r.n_inliers < 0.4 * mn:
                reason = "few stars visible (clouds?)"
            elif r.background > mb + max(6 * sb, 0.6 * mb):
                reason = "bright sky (clouds, moon or car lights?)"
            if reason:
                r.enabled = False
                r.message = "Auto-skipped: " + reason

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
                progress(0.0, f"Preparing frame {j + 1} of {len(sky_idx)}")   # also lets Cancel stop here
                v = (np.asarray(self.valid.mm[i]) > 200).astype(np.uint8)
                if k is not None:
                    v = cv2.erode(v, k)
                vtmp.write(j, v)
        sky = np.zeros((H, W, 3), np.float32)
        gnd = np.zeros((H, W, 3), np.float32)
        cnt = np.zeros((H, W), np.float32)
        progress(0.0, "Measuring frames")
        sub = (slice(None, None, 8), slice(None, None, 8))
        gmask = self.final_mask[sub] <= 127
        if gmask.sum() < 50:
            gmask = np.ones_like(gmask)

        def chan_stats(arr, m):
            progress(0.01, "Measuring frames")
            px = arr[m] if m.any() else arr.reshape(-1, 3)
            loc = np.median(px, 0)
            return loc, 1.4826 * np.median(np.abs(px - loc), 0) + 1e-7

        # Sky normalisation (Siril "additive with scaling"): match every frame's background level and
        # spread to the reference so changing skyglow / transparency doesn't fight the rejection.
        sky_norm = None
        sky_w = None
        if s.mode == "landscape":
            if s.normalize:
                stats = [chan_stats(np.asarray(self.aligned.mm[i][sub], np.float32),
                                    np.asarray(vtmp.mm[j][sub]) > 0) for j, i in enumerate(sky_idx)]
                rj = sky_idx.index(self.ref_idx) if self.ref_idx in sky_idx else 0
                L0, S0 = stats[rj]
                sky_norm = [(L, np.clip(S0 / S, 0.5, 2.0), L0) for L, S in stats]
            sky_w = np.array([self.align_results[i].weight for i in sky_idx], np.float32) \
                if s.weighting != "none" else None
        # Foreground source: separate foreground photos if given, else the night frames themselves.
        g_cache = self.fg if self.fg is not None else self.orig
        g_idx = list(range(self.fg.n)) if self.fg is not None else enabled
        g_scale = None
        if s.normalize or (self.fg is not None and s.match_foreground):
            night = [chan_stats(np.asarray(self.orig.mm[i][sub], np.float32), gmask)[0] for i in enabled]
            target = np.median(np.array(night), 0) if self.fg is not None else \
                night[enabled.index(self.ref_idx)] if self.ref_idx in enabled else night[0]
            src = night if self.fg is None else \
                [chan_stats(np.asarray(self.fg.mm[i][sub], np.float32), gmask)[0] for i in g_idx]
            if self.fg is None or s.match_foreground:
                lim = (0.5, 2.0) if self.fg is None else (1e-3, 1e3)
                g_scale = [np.clip(target / np.maximum(m, 1e-7), *lim).astype(np.float32) for m in src]
        best_pos = None
        fill = None
        self.ground_choice = None
        if s.ground_method == "best":
            progress(0.02, "Choosing the best foreground frame")
            scores = self.score_ground_frames(g_cache, g_idx, g_scale, progress)
            if s.ground_frame is not None and s.ground_frame in g_idx:
                best_pos = g_idx.index(s.ground_frame)
                how = "your choice"
            else:
                best_pos = int(np.argmin(scores))
                how = "auto"
            names = self.foregrounds if self.fg is not None else self.lights
            self.ground_choice = (f"{os.path.basename(names[g_idx[best_pos]])} ({how})", scores, g_idx[best_pos])
            if len(g_idx) > 1:
                fill = np.zeros((H, W, 3), np.float32)
        # Short sequences: sigma-type clipping can't estimate a spread from 4-9 samples (one airplane
        # streak inflates it), so switch to min/max rejection, or median for 3 frames or fewer.
        sky_method, ground_method = s.sky_method, s.ground_method
        self.method_notes = []
        if s.auto_small_stack:
            names = {"minmax": "min/max rejection", "median": "median"}
            if s.mode == "landscape":
                sky_method = small_stack_method(s.sky_method, len(sky_idx))
                if sky_method != s.sky_method:
                    self.method_notes.append(f"Sky: {len(sky_idx)} frames, so {names[sky_method]} was used "
                                             "instead of sigma-type clipping (removes airplane streaks).")
            if s.ground_method != "best":
                ground_method = small_stack_method(s.ground_method, len(g_idx))
                if ground_method != s.ground_method:
                    self.method_notes.append(f"Foreground: {len(g_idx)} frames, so {names[ground_method]} was "
                                             "used.")
        workers = max(1, min(4, (os.cpu_count() or 2)))
        nbytes = max(len(sky_idx), len(g_idx)) * W * 3 * 4 * 8
        step = int(np.clip(160e6 // max(nbytes, 1), 8, 256))
        strips = [(r0, min(H, r0 + step)) for r0 in range(0, H, step)]
        vmm = vtmp.mm if vtmp is not None else None

        def do_strip(r):
            r0, r1 = r
            g = g_cache.rows(r0, r1, g_idx)
            if g_scale is not None:
                g *= np.array(g_scale)[:, None, None, :]
            if best_pos is not None:
                gnd[r0:r1] = g[best_pos]
                if fill is not None:
                    fill[r0:r1], _ = combine(g, None, "median")
            else:
                gnd[r0:r1], _ = combine(g, None, ground_method, s.kappa)
            if s.mode == "trails":
                t = self.orig.rows(r0, r1, enabled) if self.fg is not None else g
                sky[r0:r1], _ = combine(t, None, "max")
                cnt[r0:r1] = len(enabled)
            else:
                del g
                d = self.aligned.rows(r0, r1, sky_idx)
                if sky_norm is not None:
                    for k, (L, kk, L0) in enumerate(sky_norm):
                        d[k] = (d[k] - L) * kk + L0
                v = np.asarray(vmm[:, r0:r1]) > 0
                sky[r0:r1], cnt[r0:r1] = combine(d, v, sky_method, s.kappa, weights=sky_w)

        from concurrent.futures import ThreadPoolExecutor
        # No `with` block: its exit waits for *every* queued strip, which made Cancel slow.
        ex = ThreadPoolExecutor(workers)   # numpy releases the GIL: real multi-core speed-up
        futures = []
        try:
            futures = [ex.submit(do_strip, r) for r in strips]
            for k, f in enumerate(futures):
                f.result()
                progress(k / len(futures), f"Stacking ({100 * k // len(futures)}%)")
        except BaseException:
            for f in futures:
                f.cancel()
            raise
        finally:
            ex.shutdown(wait=True, cancel_futures=True)   # only waits for the few strips already running
            if vtmp is not None:
                vtmp.close()
        progress(0.97, "Blending sky and foreground")
        self.n_sky = len(sky_idx)
        mode = "standard" if s.mode == "trails" else s.blend_mode
        self.result, self.alpha = blend(sky, gnd, sky_mask, cnt, self.n_sky, mode, s.feather, s.horizon_strength,
                                        fill=fill)
        del fill
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

    def export_frames(self, folder: str, params: FinishParams, ext: str = ".jpg", aligned: bool = True,
                      progress: Progress = _noop):
        """Sequator-style: save every frame (sky-aligned, foreground untouched) for time-lapses."""
        os.makedirs(folder, exist_ok=True)
        mask = self.final_mask if self.final_mask is not None else self.full_mask()
        alpha = cv2.GaussianBlur((mask > 127).astype(np.float32), (0, 0), 1.5)[..., None]
        written = []
        n = len(self.lights)
        for i in range(n):
            progress(i / n, f"Saving frame {i + 1} of {n}")
            img = self.orig.read(i)
            r = self.align_results[i] if self.align_results else None
            if aligned and r is not None and r.ok and self.aligned is not None:
                img = alpha * self.aligned.read(i) + (1 - alpha) * img
            fp = os.path.join(folder, f"frame_{i + 1:04d}{ext}")
            save_image(fp, apply_finish(img, mask, params, display=False), bits=8 if ext == ".jpg" else 16)
            written.append(fp)
        progress(1.0, "Frames saved")
        return written

    def summary(self) -> dict:
        d = {"lights": self.lights, "darks": self.darks, "flats": self.flats, "biases": self.biases,
             "settings": asdict(self.settings)}
        return d

    def save_settings(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
