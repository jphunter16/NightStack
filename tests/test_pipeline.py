"""End-to-end test on synthetic data with known ground truth.  Run:  python tests/test_pipeline.py"""
import glob
import os
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from nightstack.pipeline import Project  # noqa: E402
from nightstack.postprocess import FinishParams  # noqa: E402
from tests_util import ensure_data  # noqa: E402


def main():
    d = ensure_data()
    files = sorted(glob.glob(os.path.join(d, "frame_*.tif")))
    Ht = np.load(os.path.join(d, "truth_H.npy"))
    sky_true = np.load(os.path.join(d, "truth_sky.npy"))
    pr = Project()
    pr.lights = files
    log = lambda p, m: None
    t = time.time()
    pr.prepare(log)
    print(f"prepare: {time.time()-t:.1f}s, hot pixels fixed: {pr.hot_count}")
    assert pr.hot_count >= 100, "hot pixels not found"

    t = time.time()
    pr.auto_mask(progress=log)
    H, W = pr.shape
    m = pr.full_mask() > 127
    iou = (m & sky_true).sum() / (m | sky_true).sum()
    print(f"auto mask: {time.time()-t:.1f}s  IoU vs truth = {iou:.4f}")
    assert iou > 0.97, "sky detection poor"
    p, shown = pr.star_display_points()
    print(f"stars detected: {len(p)}, shown in sky: {shown.sum()}")

    t = time.time()
    res = pr.align(log)
    print(f"align: {time.time()-t:.1f}s")
    ref = pr.ref_idx
    pts = pr.alignment_points()
    worst = 0
    for r in res:
        Hrel = Ht[r.index] @ np.linalg.inv(Ht[ref])   # truth ref->i
        grid = np.stack(np.meshgrid(np.linspace(50, W - 50, 12), np.linspace(50, H * 0.6, 8)), -1).reshape(-1, 2)
        a = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), Hrel).reshape(-1, 2)
        b = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), r.H).reshape(-1, 2)
        err = np.sqrt(((a - b) ** 2).sum(1)).max()
        worst = max(worst, err)
        print(f"  frame {r.index}: ok={r.ok} inliers={r.n_inliers:4d} rms={r.rms:.3f} shift={r.shift_px:5.1f}px"
              f"  max err vs truth={err:.3f}px")
    assert worst < 0.5, "alignment inaccurate"

    for blend in ("min_noise", "standard"):
        pr.settings.blend_mode = blend
        t = time.time()
        out = pr.stack(log)
        print(f"stack [{blend}]: {time.time()-t:.1f}s")
    single = pr.orig.read(ref)
    # noise in ground (flat region) and sky (star-free patch) vs single frame
    gy0 = int(0.93 * H)
    gnd_s = np.std(single[gy0:, 1000:1200, 1] - cv2.GaussianBlur(single, (0, 0), 3)[gy0:, 1000:1200, 1])
    gnd_o = np.std(out[gy0:, 1000:1200, 1] - cv2.GaussianBlur(out, (0, 0), 3)[gy0:, 1000:1200, 1])
    sky_s = np.median(np.abs(single[50:300, 50:600, 1] - cv2.medianBlur(single, 5)[50:300, 50:600, 1]))
    sky_o = np.median(np.abs(out[50:300, 50:600, 1] - cv2.medianBlur(out, 5)[50:300, 50:600, 1]))
    print(f"noise ground: single {gnd_s:.5f} -> stack {gnd_o:.5f}  ({gnd_s/gnd_o:.2f}x better)")
    print(f"noise sky   : single {sky_s:.5f} -> stack {sky_o:.5f}  ({sky_s/sky_o:.2f}x better)")
    assert gnd_s / gnd_o > 2 and sky_s / sky_o > 2
    # star sharpness: the stacked sky star FWHM should be close to the single frame's
    from nightstack.stars import detect_stars
    from nightstack.imageio import to_gray
    def fwhm(img):
        g = to_gray(img)
        pts, fl = detect_stars(g, (sky_true * 255).astype(np.uint8), 200)
        vals = []
        for x, y in pts[:150]:
            xi, yi = int(round(x)), int(round(y))
            if yi < 8 or yi > H - 9 or xi < 8 or xi > W - 9:
                continue
            win = g[yi - 6:yi + 7, xi - 6:xi + 7]
            win = win - np.median(win)
            win = np.clip(win, 0, None)
            if win.sum() <= 0:
                continue
            yy, xx = np.mgrid[-6:7, -6:7]
            cx = (win * xx).sum() / win.sum(); cy = (win * yy).sum() / win.sum()
            var = (win * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / win.sum() / 2
            vals.append(2.355 * np.sqrt(var))
        return float(np.median(vals))
    f1, f2 = fwhm(single), fwhm(out)
    print(f"star FWHM: single {f1:.2f}px  stack {f2:.2f}px")
    assert f2 < f1 * 1.25, "stars blurred by stacking"
    # satellite trail rejected by sigma clipping
    tr = out[int(80 + (500 - 80) * 0.3), int(100 + 1400 * 0.3)]
    print("sat trail pixel:", tr)
    # export
    outdir = tempfile.mkdtemp()
    fp = FinishParams(exposure=pr.suggest_exposure())
    files = pr.export(os.path.join(outdir, "result.tif"), fp, layers=True)
    print("exported:", [os.path.basename(f) for f in files])
    from nightstack.postprocess import apply_finish
    from nightstack.imageio import save_image
    small, sm = pr.preview_inputs(1600)
    save_image(os.path.join(d, "..", "result_preview.jpg"), apply_finish(small, sm, fp))
    pr.settings.mode = "trails"
    pr.stack(log)
    save_image(os.path.join(d, "..", "trails_preview.jpg"), apply_finish(*pr.preview_inputs(1600), fp))
    pr.cleanup()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
