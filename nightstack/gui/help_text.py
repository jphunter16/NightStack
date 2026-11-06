"""Hover descriptions for every setting: widget attribute -> (name, where, description).

Used for tooltips and for the "All settings" reference in Preferences.
"""

HELP = {
    # ---------------------------------------------------------------- 1 Photos
    "fast_mode": ("Fast mode", "1 Photos",
                  "Processes everything at half resolution. Roughly 4x faster and uses far less memory - "
                  "great for trying settings. Turn it off for the final image."),
    "photo_list": ("Photo list", "1 Photos",
                   "Your night exposures. Click one to preview it. The ★ marks the reference frame - the "
                   "one whose star positions the final image uses."),
    # ---------------------------------------------------------------- 2 Calibration
    "demosaic": ("Demosaic", "2 Calibration",
                 "How RAW sensor data is turned into colour. AHD is a safe default; DCB is a little sharper "
                 "with fewer colour artifacts on stars; Bilinear is fastest but softest."),
    "hot_cb": ("Remove hot & cold pixels", "2 Calibration",
               "Finds pixels that are stuck bright or dark in the same place on every frame (stars move, "
               "hot pixels don't) and repairs them. Works even without dark frames."),
    # ---------------------------------------------------------------- 3 Sky & Foreground
    "brush": ("Brush size", "3 Sky & Foreground",
              "Size of the Erase and Paint-sky brushes, in preview pixels. Shortcut: [ and ]."),
    "opacity": ("Overlay", "3 Sky & Foreground",
                "How strongly the orange foreground tint is drawn over the photo."),
    "show_mask": ("Show foreground tint", "3 Sky & Foreground",
                  "Shows the foreground in orange so you can check where the sky ends. Shortcut: M."),
    "erase_marks": ("Eraser also marks foreground", "3 Sky & Foreground",
                    "When on, erasing red dots also paints that area as foreground (Starry Landscape Stacker "
                    "style). Turn off to only remove alignment dots."),
    "no_fg": ("No foreground", "3 Sky & Foreground",
              "Treat the whole picture as sky (e.g. a shot with no landscape in it)."),
    # ---------------------------------------------------------------- 4 Align
    "align_model": ("Alignment model", "4 Align",
                    "Wide-angle lens correction adds a smooth local warp that fixes star misalignment in the "
                    "corners caused by lens distortion. Standard uses only the sky's rotation."),
    "interp": ("Resampling", "4 Align",
               "How pixels are re-sampled when the sky is rotated. Lanczos keeps stars sharpest; Bilinear is "
               "fastest but slightly softer."),
    "auto_reject": ("Skip bad frames automatically", "4 Align",
                    "Measures star sharpness, star shape, star count and sky brightness for every frame and "
                    "unticks outliers (wind blur, a bumped tripod, clouds, car lights). You can re-tick them."),
    "align_table": ("Frame list", "4 Align",
                    "Error: alignment accuracy in pixels (lower is better). FWHM: star size - lower is sharper. "
                    "Wt: how much the frame counts in the stack. Untick a frame to leave it out."),
    # ---------------------------------------------------------------- 5 Stack
    "mode_land": ("Starry landscape", "5 Stack",
                  "Aligns the stars so they are pinpoints, keeps the foreground still, and blends the two."),
    "mode_trails": ("Star trails", "5 Stack",
                    "Keeps the brightest value of every pixel across the frames, so stars draw arcs."),
    "sky_method": ("Sky stacking method", "5 Stack",
                   "How the aligned sky frames are combined. Sigma-clipped mean is the best all-round choice "
                   "for 10+ frames: low noise and removes planes and satellites. With fewer than 10 frames "
                   "NightStack switches to min/max rejection automatically (see Preferences)."),
    "ground_method": ("Foreground stacking method", "5 Stack",
                      "How the foreground is combined. Median removes cars, people and torch light. Best single "
                      "frame keeps moving trees or water crisp but has more noise."),
    "ground_pick": ("Foreground frame", "5 Stack",
                    "For Best single frame: Auto picks the frame closest to the consensus of all frames (no "
                    "headlights, people or bumps) that is also sharp and low-noise. Or choose one yourself."),
    "kappa": ("Clip strength", "5 Stack",
              "How far from typical a pixel must be before it is rejected (sigma-type methods). Lower rejects "
              "more (cleaner, slightly noisier); higher keeps more."),
    "normalize": ("Normalize frames", "5 Stack",
                  "Matches every frame's sky brightness and contrast to the reference before stacking, so "
                  "changing skyglow, moonlight or haze doesn't spoil the rejection. Leave on."),
    "weighting": ("Frame weights", "5 Stack",
                  "Lets better frames count more. Quality favours sharp stars and dark sky; Noise favours "
                  "cleaner frames; Equal treats all frames the same."),
    "match_fg": ("Match foreground photos", "5 Stack",
                 "Scales separate foreground photos (blue hour / long exposure) to the brightness of the night "
                 "frames so they blend naturally."),
    "small_stack_cb": ("Small-stack protection", "5 Stack",
                       "With fewer than 10 frames, sigma-type clipping can't judge what's normal - one airplane "
                       "streak survives. This switches to min/max rejection (or median for 3 frames or fewer)."),
    "blend_mode": ("Horizon blending", "5 Stack",
                   "Minimum horizon noise keeps the sky just above the horizon as clean as the rest by filling "
                   "it from all frames where few aligned frames saw sky. Standard uses a plain soft edge."),
    "horizon_strength": ("Noise fill", "5 Stack",
                         "How much Minimum Horizon Noise fills the thin band above the horizon. 100% = cleanest; "
                         "lower keeps more of the aligned (slightly noisier) data."),
    "feather": ("Edge softness", "5 Stack",
                "How soft the transition between sky and foreground is, in pixels."),
    "margin": ("Horizon guard", "5 Stack",
               "Ignores sky pixels this close to the horizon in each frame, preventing a glow or dark halo "
               "along the skyline."),
    "show_what": ("Show", "5 Stack",
                  "Switch the view: the final result, the sky or foreground stack alone, a heat map of how many "
                  "frames contributed sky at each pixel, or a single frame for comparison."),
    # ---------------------------------------------------------------- 6 Finish
    "exposure": ("Brightness", "6 Finish",
                 "Exposure in stops (EV). 0 = as shot. +1 doubles the brightness. Type an exact value in the box."),
    "black": ("Black point", "6 Finish",
              "Removes this share of the sky's background glow, making the sky darker and punchier. 0% = none "
              "(recommended if you edit in Lightroom - clipped shadows can't be recovered)."),
    "contrast": ("Contrast", "6 Finish", "Adds (+) or reduces (−) contrast. 0 = none."),
    "saturation": ("Saturation", "6 Finish", "Colour intensity. 1.00 = unchanged."),
    "tone": ("Tone curve", "6 Finish",
             "Natural is a normal photographic curve. Astro (asinh) lifts faint sky strongly while keeping star "
             "colours - the stretch used in Siril and PixInsight."),
    "asinh": ("Astro stretch", "6 Finish", "Strength of the Astro tone curve. Higher brings out fainter sky."),
    "gradient": ("Remove light-pollution gradient", "6 Finish",
                 "Fits the uneven glow from towns or the moon across the sky and removes it - in the sky only."),
    "grad_deg": ("Gradient shape", "6 Finish",
                 "How complex a glow to remove. Simple for a smooth one-sided glow; Complex for patchy light "
                 "pollution. Too complex can eat into the Milky Way."),
    "neutral": ("Neutral sky colour", "6 Finish",
                "Removes the colour cast of the sky background (orange skyglow, green airglow). Foreground "
                "colours are left alone."),
    "highlights": ("Protect bright stars & lights", "6 Finish",
                   "Rolls off the brightest areas gently instead of clipping them to white."),
    "stars_boost": ("Enhance stars", "6 Finish",
                    "Brightens stars only - the Milky Way glow and the foreground are untouched."),
    "sky_dn": ("Extra sky smoothing", "6 Finish", "Additional edge-preserving noise reduction in the sky."),
    "gnd_dn": ("Foreground smoothing", "6 Finish",
               "Noise reduction for the foreground (useful with Best single frame)."),
    "full_btn": ("Render 100% full preview", "6 Finish",
                 "Renders the full-resolution image so you can inspect stars at 1:1 pixels. Moving a slider "
                 "returns to the fast preview."),
    "before_btn": ("Compare with a single frame", "6 Finish",
                   "Hold to see one unstacked frame with the same adjustments - shows how much noise stacking removed."),
    "fmt": ("Save format", "6 Finish",
            "TIFF 16-bit keeps the most detail for Lightroom/Photoshop. JPEG for sharing. FITS for Siril/PixInsight."),
    "linear": ("Linear data", "6 Finish",
               "Saves the stack with no brightness curve, for Siril or PixInsight. Looks almost black in Lightroom."),
    "layers": ("Also save layers", "6 Finish",
               "Also saves the sky stack, the foreground stack and the mask as separate files for blending by hand."),
    "frames_btn": ("Export aligned frames", "6 Finish",
                   "Saves every frame with the sky aligned - for time-lapses."),
}
