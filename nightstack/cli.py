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
    ap.add_argument("--sky", default="sigma", choices=["sigma", "median", "mean", "winsor", "max"])
    ap.add_argument("--ground", default="median", choices=["median", "sigma", "mean"])
    ap.add_argument("--blend", default="min_noise", choices=["min_noise", "standard"])
    ap.add_argument("--trails", action="store_true", help="star-trail (lighten) mode")
    ap.add_argument("--no-lens-correction", action="store_true")
    ap.add_argument("--half", action="store_true", help="half resolution (fast)")
    ap.add_argument("--no-foreground", action="store_true")
    ap.add_argument("--linear", action="store_true", help="save linear data")
    ap.add_argument("--layers", action="store_true", help="also save sky/foreground/mask")
    ap.add_argument("--exposure", type=float, default=None, help="EV (default: auto)")
    a = ap.parse_args(argv)

    def expand(items):
        out = []
        for it in items:
            out += sorted(glob.glob(it)) or [it]
        return out

    pr = Project()
    pr.lights, pr.darks, pr.flats, pr.biases = expand(a.lights), expand(a.darks), expand(a.flats), expand(a.bias)
    s = pr.settings
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
        pr.stack(prog)
        fp = FinishParams(linear_output=a.linear)
        fp.exposure = a.exposure if a.exposure is not None else pr.suggest_exposure(fp.black_point)
        files = pr.export(a.output, fp, bits=32 if a.output.lower().endswith((".fit", ".fits")) else 16,
                          layers=a.layers, progress=prog)
        print(f"\nSaved {', '.join(files)} in {time.time() - t0:.0f}s")
    finally:
        pr.cleanup()


if __name__ == "__main__":
    main()
