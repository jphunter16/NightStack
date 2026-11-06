"""Command-line batch mode:  python -m nightstack photos/*.CR2 -o result.tif"""
import argparse
import glob
import sys
import time

from .pipeline import Project
from .postprocess import FinishParams


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nightstack", description="Stack starry-landscape exposures.")
    ap.add_argument("lights", nargs="+", help="light frames (globs ok)")
    ap.add_argument("-o", "--output", default="nightstack.tif")
    ap.add_argument("--darks", nargs="*", default=[])
    ap.add_argument("--flats", nargs="*", default=[])
    ap.add_argument("--bias", nargs="*", default=[])
    ap.add_argument("--foreground", nargs="*", default=[], help="separate foreground exposures (same tripod spot)")
    ap.add_argument("--sky", default="sigma",
                    choices=["sigma", "winsor", "linear", "mad", "percentile", "median", "mean", "max"])
    ap.add_argument("--ground", default="median", choices=["median", "sigma", "winsor", "mean", "best"])
    ap.add_argument("--ground-frame", type=int, default=None,
                    help="with --ground best: use this frame number (1-based) instead of the automatic pick")
    ap.add_argument("--kappa", type=float, default=2.5, help="rejection strength (sigma)")
    ap.add_argument("--weighting", default="quality", choices=["quality", "noise", "fwhm", "none"])
    ap.add_argument("--no-normalize", action="store_true", help="disable frame normalisation")
    ap.add_argument("--keep-bad-frames", action="store_true", help="disable automatic frame rejection")
    ap.add_argument("--demosaic", default="AHD", choices=["AHD", "DCB", "DHT", "AAHD", "VNG", "PPG", "LINEAR"])
    ap.add_argument("--enhance-stars", type=float, default=0.0, help="0..1")
    ap.add_argument("--astro-stretch", type=float, default=None, help="use asinh tone curve with this strength")
    ap.add_argument("--remove-gradient", action="store_true")
    ap.add_argument("--timelapse", default=None, help="also save aligned frames into this folder")
    ap.add_argument("--blend", default="min_noise", choices=["min_noise", "standard"])
    ap.add_argument("--trails", action="store_true", help="star-trail (lighten) mode")
    ap.add_argument("--no-lens-correction", action="store_true")
    ap.add_argument("--half", action="store_true", help="half resolution (fast)")
    ap.add_argument("--no-foreground", action="store_true")
    ap.add_argument("--linear", action="store_true", help="save linear data")
    ap.add_argument("--layers", action="store_true", help="also save sky/foreground/mask")
    ap.add_argument("--exposure", default="0", help="EV, or 'auto' (default: 0 = as shot)")
    a = ap.parse_args(argv)

    def expand(items):
        out = []
        for it in items:
            out += sorted(glob.glob(it)) or [it]
        return out

    pr = Project()
    pr.lights, pr.darks, pr.flats, pr.biases = expand(a.lights), expand(a.darks), expand(a.flats), expand(a.bias)
    pr.foregrounds = expand(a.foreground)
    s = pr.settings
    s.kappa, s.weighting, s.demosaic = a.kappa, a.weighting, a.demosaic
    s.ground_frame = a.ground_frame - 1 if a.ground_frame else None
    s.normalize, s.auto_reject = not a.no_normalize, not a.keep_bad_frames
    s.half_size, s.sky_method, s.ground_method, s.blend_mode = a.half, a.sky, a.ground, a.blend
    s.align_local = not a.no_lens_correction
    s.mode = "trails" if a.trails else "landscape"
    t0 = time.time()

    def prog(p, m):
        sys.stdout.write(f"\r[{p * 100:5.1f}%] {m[:70]:<70}")
        sys.stdout.flush()
    try:
        pr.prepare(prog)
        pr.auto_mask(True, prog)
        if a.no_foreground:
            pr.mask_model.no_foreground = True
        if s.mode == "landscape":
            res = pr.align(prog)
            print(f"\nAligned {sum(r.ok for r in res)}/{len(res)} frames")
            for r in res:
                if r.ok and not r.enabled:
                    print(f"  frame {r.index + 1}: {r.message}")
        pr.stack(prog)
        if pr.ground_choice:
            print(f"\nForeground frame: {pr.ground_choice[0]}")
        fp = FinishParams(linear_output=a.linear, enhance_stars=a.enhance_stars,
                          remove_gradient=a.remove_gradient)
        if a.astro_stretch is not None:
            fp.tone, fp.asinh_strength = "asinh", a.astro_stretch
        fp.exposure = pr.suggest_exposure(fp.black_point) if a.exposure == "auto" else float(a.exposure)
        files = pr.export(a.output, fp, bits=32 if a.output.lower().endswith((".fit", ".fits")) else 16,
                          layers=a.layers, progress=prog)
        if a.timelapse:
            files += pr.export_frames(a.timelapse, fp, progress=prog)
        print(f"\nSaved {len(files)} file(s), e.g. {files[0]}, in {time.time() - t0:.0f}s")
    finally:
        pr.cleanup()


if __name__ == "__main__":
    main()
