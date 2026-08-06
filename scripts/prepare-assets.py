#!/usr/bin/env python3
"""Turn a raw nano-banana-pro render into a registry-ready asset.

Three steps, each of which exists because of something measured rather than
something assumed:

1. CROP THE BORDER. nano-banana-pro draws a newspaper-panel frame on every
   render, and it does so even when the prompt states NO border, NO frame, NO
   margin as CRITICAL. The cause is the house style block's own opening words —
   "A single panel from a 1933 newspaper comic strip". "Panel" summons a frame
   and it outranks the negative. Rewording the style block is not an option: it
   is shared and never varied. So the border comes off programmatically. It is a
   uniform low-variance margin, so walking in from each edge until the row/column
   median colour changes finds it exactly, and a few pixels of pad clears the ink
   rule inside it.

2. MATTE (optional). Renders destined for the pose library are generated on flat
   mid-grey #808080 — never green, because diffusion bleeds backdrop colour and
   there is no despill node in this stack. BiRefNet then cuts the subject out.
   LayerDiffuse, the obvious alternative, has been broken since ComfyUI 0.19.3
   with the fix PRs unmerged.

3. SQUARE AND RESIZE. Centre-crop to square, then to a fixed edge, so every asset
   in a set lands at identical dimensions. The detected border varies by a few
   pixels between renders; normalising here keeps figure scale consistent to
   about a percent, which is what makes a pose library composite cleanly.

4. QUANTISE to a 256-colour palette. The house style is sepia duotone shaded
   only with halftone dots, so the real colour count is tiny and truecolour PNG
   is paying for precision the art does not use. Measured on this set: 50-85%
   smaller at RMSE 1.4-2.1, which is invisible at this palette, and it brings
   the biggest style plates back under the ~2 MB per-asset guidance. The assets
   live in git forever and four characters at truecolour would have crowded the
   ~300 MB repo budget. Pass --no-quantize for anything not in the house style.

Steps 1 and 3 need only the repo's own dependencies. Step 2 needs rembg, which
is a dev-time tool and deliberately NOT a build dependency — the site build must
never need a 973 MB ONNX model. Install it in a venv:

    python3 -m venv .venv-matte && .venv-matte/bin/pip install rembg onnxruntime
    .venv-matte/bin/python scripts/prepare-assets.py --matte in.png out.png

    python3 scripts/prepare-assets.py raw.png cast/wilson/assets/training/still-01.png
    python3 scripts/prepare-assets.py --matte raw.png cast/wilson/assets/poses/walking.png
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

# How far the median colour of a line must move from the outermost line before
# that line counts as "past the border". The paper margin is much lighter than
# any staging background we render on, so this is a wide, forgiving threshold.
BORDER_TOLERANCE = 24
# Pixels to trim past the transition, to clear the ink rule drawn inside the
# margin. Measured at 6 on the first Wilson plate; 8 leaves headroom.
BORDER_PAD = 8


def _line_median(px, fixed, horizontal, width, height, samples=64):
    """Median colour of one row or column, from `samples` evenly spaced pixels."""
    span = width if horizontal else height
    step = max(1, span // samples)
    pts = [px[i, fixed] if horizontal else px[fixed, i] for i in range(0, span, step)]
    mid = len(pts) // 2
    return tuple(sorted(p[c] for p in pts)[mid] for c in range(3))


def detect_border(im, tolerance=BORDER_TOLERANCE, pad=BORDER_PAD):
    """Box of the artwork inside the drawn newspaper border."""
    im = im.convert("RGB")
    width, height = im.size
    px = im.load()

    def walk(count, horizontal, reverse):
        ref = _line_median(px, count - 1 if reverse else 0, horizontal, width, height)
        for i in range(count):
            line = count - 1 - i if reverse else i
            cur = _line_median(px, line, horizontal, width, height)
            if max(abs(cur[c] - ref[c]) for c in range(3)) > tolerance:
                return i
        return 0  # no border found — leave the edge alone

    top, left = walk(height, True, False), walk(width, False, False)
    bottom = height - walk(height, True, True)
    right = width - walk(width, False, True)

    box = (left + pad, top + pad, right - pad, bottom - pad)
    if box[2] - box[0] < width // 2 or box[3] - box[1] < height // 2:
        # A detector that eats half the image has misfired — almost certainly a
        # render with no border at all. Better to publish the whole frame than
        # to silently ship a crop of the character's midriff.
        return (0, 0, width, height)
    return box


def to_square(im):
    """Centre-crop to the shorter edge. Never scales one axis against the other."""
    width, height = im.size
    if width == height:
        return im
    edge = min(width, height)
    left, top = (width - edge) // 2, (height - edge) // 2
    return im.crop((left, top, left + edge, top + edge))


def matte(im):
    """Cut the subject out of the flat mid-grey plate with BiRefNet."""
    try:
        from rembg import new_session, remove
    except ImportError:
        sys.exit("error: --matte needs rembg. See this file's docstring for the venv "
                 "one-liner; it is intentionally not a build dependency.")
    return remove(im, session=new_session("birefnet-general"), post_process_mask=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("dest")
    ap.add_argument("--matte", action="store_true",
                    help="cut the subject out of the mid-grey plate (needs rembg)")
    ap.add_argument("--size", type=int, default=1024,
                    help="final square edge in pixels (default 1024)")
    ap.add_argument("--no-crop", action="store_true",
                    help="skip border detection, for a render that has none")
    ap.add_argument("--no-quantize", action="store_true",
                    help="keep truecolour; for art outside the sepia-duotone house style")
    ap.add_argument("--colors", type=int, default=256,
                    help="palette size when quantising (default 256)")
    ap.add_argument("--pad", type=int, default=BORDER_PAD)
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_file():
        sys.exit(f"error: no such file: {src}")

    im = Image.open(src)
    original = im.size

    if not args.no_crop:
        box = detect_border(im, pad=args.pad)
        im = im.crop(box)

    if args.matte:
        im = matte(im.convert("RGB"))

    im = to_square(im)
    if im.size[0] != args.size:
        im = im.resize((args.size, args.size), Image.LANCZOS)
    if not args.matte:
        im = im.convert("RGB")

    if not args.no_quantize:
        # FASTOCTREE rather than MEDIANCUT: it is the only Pillow method that
        # carries an alpha channel through into the palette, and the pose
        # cutouts need their transparency to survive this step.
        im = im.quantize(colors=args.colors, method=Image.FASTOCTREE)

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, optimize=True)
    print(f"{src.name}: {original[0]}x{original[1]} -> {args.size}x{args.size}"
          f"{' matted' if args.matte else ''}"
          f"{'' if args.no_quantize else f' {args.colors}col'} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
