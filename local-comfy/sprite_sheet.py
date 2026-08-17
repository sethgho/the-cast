#!/usr/bin/env python3
"""Turn a magenta-screen clip into a normalised sprite sheet the web can actually use.

    python3 sprite_sheet.py <clip.mp4> <name> [--frames 8] [--cell 256] [--outdir DIR]
                            [--anchor feet|centre] [--no-smooth]

Writes, into --outdir (default /home/wilson/artifacts/cast-local-comfy/sprites):

    <name>.png        the atlas — one row, uniform cells, transparent background
    <name>.json       frame table: cell size, count, fps, per-frame source bbox
    <name>.css        a ready-to-paste CSS steps() animation
    <name>-preview.webp   animated, for eyeballing the timing
    <name>-frames/    the individual cells, if you would rather pack them yourself

## Why a sheet and not a video

A PNG atlas plays in CSS `steps()`, Phaser, Pixi, Godot and Unity, and a state machine can jump
between moves on a frame boundary. Alpha video can do none of that, and alpha WebM is still a
coin-flip in Safari. The clip is the source; the sheet is the deliverable.

## The four jitter problems, and what fixes each

1. **Matte jitter** — a learned matte wobbles a pixel or two per frame. Fixed upstream by
   rendering on a flat magenta screen and keying it here by colour distance.
2. **Frame choice** — evenly spaced frames land on in-betweens and the animation reads mushy.
   Fixed by picking *pose extremes*: the frames where motion energy peaks.
3. **Drift** — the character's bounding box moves frame to frame, so a naive pack makes him
   swim in the cell. Fixed by aligning every frame on a stable anchor — by default the bottom
   centre of the silhouette, i.e. his feet, which is what a game engine treats as the origin.
4. **Ragged dimensions** — every frame has a different bbox. Fixed with one uniform cell sized
   to the largest frame plus padding, so every cell is interchangeable.

A 3-frame median smooths the anchor track, which kills the last of the sub-pixel shimmer without
flattening real movement. `--no-smooth` turns it off.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

KEY = np.array([255, 0, 255], dtype=np.float32)   # the magenta the sprite-clip app renders on
SOFT_IN, SOFT_OUT = 45.0, 110.0   # colour distance from the measured screen: background .. solid
EDGE_ERODE = 0                    # eroding eats the ink OUTLINE — the darkest pixels in the drawing


def extract_frames(clip, tmp):
    os.makedirs(tmp, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-vsync", "0", f"{tmp}/f_%04d.png"],
                   check=True)
    return sorted(os.path.join(tmp, f) for f in os.listdir(tmp) if f.startswith("f_"))


def despill(rgb, edge=None):
    """Magenta bleeds into every anti-aliased edge pixel and reads as a red fringe.

    Magenta is high red + high blue, low green. Where a pixel's red/blue average sits above its
    green, that excess is the screen showing through the edge — pull both channels down to the
    green level. This is the standard green-screen despill with the channels swapped, and it is
    why the fringe cannot be fixed by eroding the matte alone: the colour is IN the pixel.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    spill = np.maximum((r + b) * 0.5 - g, 0.0)
    if edge is not None:
        # Two places the screen shows up inside the drawing. The anti-aliased edge, and — once
        # RIFE is interpolating between poses — pink smears where a limb moved a long way between
        # keys. Both are magenta-ish: blue ABOVE green. The character's own warm colours are the
        # opposite (brown hair, tan skin, khaki trousers all sit blue below green), so that one
        # test separates them and lets the despill run at full strength on the smears without
        # touching the drawing.
        pinkish = (rgb[..., 2] > rgb[..., 1] + 6).astype(np.float32)
        spill = spill * np.maximum(edge, pinkish)
    out = rgb.copy()
    out[..., 0] = np.clip(r - spill, 0, 255)
    out[..., 2] = np.clip(b - spill, 0, 255)
    return out


def measure_key(im):
    """Take the key colour from the picture, not from a constant.

    The model does not paint pure #FF00FF — a punch still measured (238, 51, 163), which is 107
    units from the constant. Keyed against the constant that whole screen came back ~20% opaque:
    a translucent halo that also inflated every bounding box and shrank the packed character.
    The four corners are always screen, so their median IS the key.
    """
    h, w, _ = im.shape
    p = 12
    corners = np.concatenate([im[:p, :p].reshape(-1, 3), im[:p, -p:].reshape(-1, 3),
                              im[-p:, :p].reshape(-1, 3), im[-p:, -p:].reshape(-1, 3)])
    return np.median(corners, axis=0)


def key_out(path):
    """Screen -> soft alpha, plus despill. Colour distance, not an exact match: the codec moves it."""
    # float32, not int16: squaring a 230-unit channel difference overflows int16 and the
    # key silently eats most of the character.
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    key = measure_key(im)
    dist = np.sqrt(((im - key) ** 2).sum(axis=2))
    # A soft ramp, not a hard threshold: a binary matte stair-steps every outline.
    alpha = np.clip((dist - SOFT_IN) / (SOFT_OUT - SOFT_IN), 0.0, 1.0)
    if EDGE_ERODE:
        from PIL import ImageFilter
        a8 = Image.fromarray((alpha * 255).astype(np.uint8))
        alpha = np.asarray(a8.filter(ImageFilter.MinFilter(2 * EDGE_ERODE + 1))).astype(np.float32) / 255.0
    # edge weight: 1 in the transition band, 0 on solid pixels and on the screen itself
    edge = np.clip(1.0 - np.abs(alpha * 2.0 - 1.0), 0.0, 1.0)
    rgba = np.dstack([despill(im, edge).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def bbox(rgba):
    a = np.asarray(rgba)[:, :, 3]
    ys, xs = np.nonzero(a > 8)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def silhouettes(frames):
    """Small binary silhouettes — enough to compare poses, cheap enough to compare them all."""
    out = []
    for f in frames:
        a = np.asarray(f)[:, :, 3]
        im = Image.fromarray((a > 8).astype(np.uint8) * 255).resize((48, 48), Image.BILINEAR)
        out.append(np.asarray(im).astype(np.float32) / 255.0)
    return out


def find_cycle(sils, min_period=6):
    """Find the gait period by self-similarity, so the sheet spans exactly one loop.

    A walk repeats: frame t looks like frame t+P. Score every candidate period by how closely the
    clip matches itself shifted by P, and take the best. Sampling n frames across exactly one
    period is what makes the first and last cell adjacent poses — i.e. what makes the loop
    seamless. Without this the sheet spans some arbitrary 0.7 of a stride and visibly hitches.
    """
    n = len(sils)
    best, best_score = None, None
    for p in range(min_period, n // 2 + 1):
        pairs = n - p
        diff = sum(float(np.abs(sils[t] - sils[t + p]).mean()) for t in range(pairs)) / pairs
        if best_score is None or diff < best_score:
            best, best_score = p, diff
    return best, best_score


def pose_extremes(frames, count, cycle=True):
    """Pick the cells. Inside one gait period when the move loops, by motion energy when it does not."""
    sils = silhouettes(frames)
    if cycle and len(frames) >= 16:
        period, score = find_cycle(sils)
        # A loose match means the move genuinely is not periodic (a jump, a bow) — fall through.
        if period and score < 0.09:
            start = max(range(len(frames) - period),
                        key=lambda t: float(np.abs(sils[t] - sils[t + 1]).mean()))
            picks = [start + int(round(period * i / count)) for i in range(count)]
            return [min(p, len(frames) - 1) for p in picks], period
    energy = [0.0]
    prev = np.asarray(frames[0]).astype(np.float32)
    for f in frames[1:]:
        cur = np.asarray(f).astype(np.float32)
        energy.append(float(np.abs(cur - prev).mean()))
        prev = cur
    # cumulative motion, sampled evenly: equal *movement* between cells, not equal time
    cum = np.cumsum(energy)
    if cum[-1] <= 0:
        return list(np.linspace(0, len(frames) - 1, count).astype(int)), None
    targets = np.linspace(0, cum[-1], count, endpoint=False)
    return [int(np.searchsorted(cum, t)) for t in targets], None


def median3(values):
    out = list(values)
    for i in range(1, len(values) - 1):
        out[i] = sorted(values[i - 1:i + 2])[1]
    return out


def pack_set(jobs, cell, outdir, anchor, smooth, skip=6, cycle=True):
    """Pack several moves of ONE character with a SHARED scale.

    Packing each move on its own scales every sheet to its own content, so the character is a
    different size in every move and the state machine visibly resizes him mid-fight. One pass
    measures the tallest silhouette across the whole set; the second pass packs every move to
    that same pixels-per-character, which is what makes the sheets interchangeable.
    """
    measured = []
    for clip, name, n in jobs:
        prep = _prepare(clip, name, n, skip, cycle, anchor, smooth)
        measured.append((name, n, prep))
    tallest = max(p["natural"] for _, _, p in measured)
    scale = (cell * 0.92) / tallest if tallest else 1.0
    out = []
    for name, n, prep in measured:
        out.append(_emit(name, prep, cell, outdir, anchor, scale))
    return out


def pack(clip, name, n_frames, cell, outdir, anchor, smooth, skip=6, cycle=True):
    """Pack one move on its own. Use pack_set() for a character's whole move list."""
    prep = _prepare(clip, name, n_frames, skip, cycle, anchor, smooth)
    scale = (cell * 0.92) / prep["natural"] if prep["natural"] else 1.0
    return _emit(name, prep, cell, outdir, anchor, scale)


def _prepare(clip, name, n_frames, skip, cycle, anchor, smooth):
    """Key, pick the cells, and work out each cell's anchor and reach. No pixels written yet."""
    tmp = f"/tmp/sprite-{name}"
    paths = extract_frames(clip, tmp)[skip:]
    rgba = [key_out(p) for p in paths]
    boxes = [bbox(r) for r in rgba]
    keep = [i for i, b in enumerate(boxes) if b]
    if not keep:
        raise SystemExit(f"{name}: every frame keyed to nothing — is the clip actually on magenta?")

    picks, period = pose_extremes([rgba[i] for i in keep], n_frames, cycle)
    picks = [keep[p] for p in picks]
    print(f"  {name} cycle: {'period ' + str(period) + ' frames' if period else 'not periodic, spaced by motion'}")

    # anchor per frame: feet = bottom centre of the silhouette, the origin a game engine expects
    anchors = []
    for i in picks:
        x0, y0, x1, y1 = boxes[i]
        anchors.append(((x0 + x1) // 2, y1 if anchor == "feet" else (y0 + y1) // 2))
    if smooth and len(anchors) > 2:
        anchors = list(zip(median3([a[0] for a in anchors]), median3([a[1] for a in anchors])))

    spans = []
    for (i, (ax, ay)) in zip(picks, anchors):
        x0, y0, x1, y1 = boxes[i]
        spans.append((ax - x0, x1 - ax, ay - y0, y1 - ay))
    natural = max(max(s[0] for s in spans) + max(s[1] for s in spans),
                  max(s[2] for s in spans) + max(s[3] for s in spans))
    return {"rgba": rgba, "boxes": boxes, "picks": picks, "anchors": anchors,
            "natural": natural, "period": period, "clip": clip}


def _prepare_stills(paths, smooth=True):
    """Same measurement pass as _prepare, but every still IS a cell — no picking, no cycle hunt.

    Stills come from sprite_poses.py: one render per pose beat, so the frame order is the
    playback order and there is nothing to select.
    """
    rgba = [key_out(p) for p in paths]
    boxes = [bbox(r) for r in rgba]
    picks = [i for i, b in enumerate(boxes) if b]
    if not picks:
        raise SystemExit("every still keyed to nothing — are they on magenta?")
    anchors = []
    for i in picks:
        x0, y0, x1, y1 = boxes[i]
        anchors.append(((x0 + x1) // 2, y1))
    if smooth and len(anchors) > 2:
        anchors = list(zip(median3([a[0] for a in anchors]), median3([a[1] for a in anchors])))
    spans = []
    for (i, (ax, ay)) in zip(picks, anchors):
        x0, y0, x1, y1 = boxes[i]
        spans.append((ax - x0, x1 - ax, ay - y0, y1 - ay))
    # Scale off HEIGHT, with width only allowed to veto if it would overflow the cell. A wide
    # pose — a kick at full extension — would otherwise shrink every other move to match it.
    tall = max(s[2] for s in spans) + max(s[3] for s in spans)
    wide = max(s[0] for s in spans) + max(s[1] for s in spans)
    natural = max(tall, wide * 0.8)
    return {"rgba": rgba, "boxes": boxes, "picks": picks, "anchors": anchors,
            "natural": natural, "period": None, "clip": paths[0]}


def _emit(name, prep, cell, outdir, anchor, scale, unify_height=False):
    """Write the atlas, the cells, the JSON, the CSS and the preview at the given scale."""
    rgba, boxes, picks, anchors = prep["rgba"], prep["boxes"], prep["picks"], prep["anchors"]
    frames_dir = os.path.join(outdir, f"{name}-frames")
    os.makedirs(frames_dir, exist_ok=True)
    # Per-frame height normalisation. Every still is generated independently, so the character
    # comes out a few percent bigger or smaller each time and a looping sprite "breathes". For a
    # cycle that has to be seamless, each frame is scaled so its silhouette height matches the
    # set's median. Moves whose height genuinely changes — crouch, jump — must NOT use this.
    heights = [boxes[i][3] - boxes[i][1] for i in picks]
    median_h = sorted(heights)[len(heights) // 2]

    cells, table = [], []
    for (i, (ax, ay)) in zip(picks, anchors):
        src = rgba[i]
        canvas = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
        w, h = src.size
        f_scale = scale * (median_h / (boxes[i][3] - boxes[i][1])) if unify_height else scale
        sized = src.resize((max(1, int(w * f_scale)), max(1, int(h * f_scale))), Image.LANCZOS)
        sax, say = ax * f_scale, ay * f_scale
        target = (cell // 2, int(cell * 0.94) if anchor == "feet" else cell // 2)
        canvas.paste(sized, (int(target[0] - sax), int(target[1] - say)), sized)
        cells.append(canvas)
        canvas.save(os.path.join(frames_dir, f"{name}-{len(cells):02d}.png"))
        x0, y0, x1, y1 = boxes[i]
        table.append({"source_frame": i, "bbox": [x0, y0, x1, y1], "anchor": [ax, ay]})

    atlas = Image.new("RGBA", (cell * len(cells), cell), (0, 0, 0, 0))
    for n, c in enumerate(cells):
        atlas.paste(c, (n * cell, 0), c)
    atlas_path = os.path.join(outdir, f"{name}.png")
    atlas.save(atlas_path)

    cells[0].save(os.path.join(outdir, f"{name}-preview.webp"), save_all=True,
                  append_images=cells[1:], duration=1000 // 12, loop=0, lossless=True)

    meta = {"name": name, "cell": [cell, cell], "frames": len(cells), "fps": 12,
            "anchor": anchor, "layout": "single-row", "scale": round(scale, 4),
            "source_clip": os.path.basename(prep["clip"]),
            "cycle_period_frames": prep["period"], "loops": bool(prep["period"]),
            "frame_table": table}
    json.dump(meta, open(os.path.join(outdir, f"{name}.json"), "w"), indent=1)

    css = f""".{name} {{
  width: {cell}px; height: {cell}px;
  background: url("{name}.png") 0 0 / {cell * len(cells)}px {cell}px no-repeat;
  animation: {name}-play {len(cells) / 12:.2f}s steps({len(cells)}) infinite;
}}
@keyframes {name}-play {{ to {{ background-position: -{cell * len(cells)}px 0; }} }}
"""
    open(os.path.join(outdir, f"{name}.css"), "w").write(css)
    print(f"{name}: {len(cells)} frames, {cell}px cells, scale {scale:.3f} -> {atlas_path}")
    return atlas_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("name")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--outdir", default="/home/wilson/artifacts/cast-local-comfy/sprites")
    ap.add_argument("--anchor", choices=("feet", "centre"), default="feet")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--skip", type=int, default=6,
                    help="drop N settle-in frames off the head of the clip")
    ap.add_argument("--no-cycle", action="store_true",
                    help="do not look for a gait period; space cells by motion instead")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    pack(a.clip, a.name, a.frames, a.cell, a.outdir, a.anchor, not a.no_smooth, a.skip,
         not a.no_cycle)


if __name__ == "__main__":
    main()
