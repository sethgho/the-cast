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
# The gate on the cycle detector, as an absolute self-similarity score. A relative version was
# tried and is worse: a walk's own harmonics (2P, 3P) drag any within-clip baseline down, so the
# ratio stops discriminating. And the case that motivated it turned out not to need it -- Cadbury's
# walk scored 0.150 against a 0.159 random-pair baseline, i.e. his clip genuinely does not repeat.
# A non-periodic cycle clip is an upstream problem to re-render, not a threshold to loosen.
CYCLE_SCORE_MAX = 0.09
# The SECOND gate, and the one that separates a walk from an idle. The score above says how
# closely the clip matches itself at its BEST lag; it says nothing about whether that lag is a
# real gait period. Measured over all 42 character clips on gpu-worker, the score alone does not
# discriminate: Seth's shipped walk scores 0.0294 and his idle 0.0332 — a 12% margin, i.e. none.
#
# What DOES separate them is the shape of the self-similarity curve. A move that repeats carves a
# TROUGH: the curve falls to the period and climbs away on both sides. A move that only mills
# about — an idle, a breathe, a sway — never returns to a pose, so its curve is flat or climbs
# monotonically, and its minimum is pinned at the min_period edge. So score the trough:
#
#     trough = min(max(curve left of the minimum), max(curve right of it)) / minimum
#
# and require the minimum to be interior (a monotone curve has no trough by definition, and
# scores 1.0). This is NOT the relative gate the comment above warns off. That one divided by a
# within-clip RANDOM-PAIR baseline, which the walk's own harmonics at 2P/3P dragged down. This
# divides by the curve's SHOULDERS — maxima, not means — so harmonics deepen the trough instead
# of erasing it, which is exactly the behaviour we want.
#
# Measured trough ratios, clips that also pass CYCLE_SCORE_MAX (full table in the A/B report):
#     gnome walk        12.11      cadbury uppercut   1.77   <- highest non-cycle
#     cadbury walk 2     7.02      nowater walk       1.55   <- a walk take that does not close
#     sprite seth walk-c 5.63      cadbury punch      1.49
#     sprite seth walk   3.24      seth block 1       1.37
#     seth walk          2.89      seth/cadbury/nowater idles, blocks, kicks, crouches: 1.00
#     ake roll           2.23   <- lowest genuine cycle
# 2.0 is the geometric midpoint of 1.77 and 2.23, so both sides get ~11% headroom, and every
# idle in the set sits at the 1.00 floor — a 2.2x margin, not a 12% one.
CYCLE_TROUGH_MIN = 2.0
EDGE_ERODE = 0                  # eroding eats the ink OUTLINE — the darkest pixels in the drawing
CEL_CLEAN = True                  # off for repainted cells: they are drawn, not decoded, so there
                                  # is no codec noise to remove and the median only costs halftone


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
    cleaned = despill(im, edge)
    if CEL_CLEAN:
        cleaned = cel_clean(cleaned, alpha)
    rgba = np.dstack([cleaned.astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def cel_clean(rgb, alpha):
    """Make a video frame look drawn again.

    H3's output carries codec noise: chroma speckle in the flats and soft, slightly coloured
    edges. In a flat-ink cartoon both read as "ripped from video". A small median kills the
    speckle without touching line position, and an unsharp pass puts the bite back into the
    outlines. Applied inside the silhouette only, so the matte is untouched.
    """
    from PIL import ImageFilter
    im = Image.fromarray(rgb.astype(np.uint8))
    im = im.filter(ImageFilter.MedianFilter(3))
    im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=3))
    out = np.asarray(im).astype(np.float32)

    # Chroma fringe. The codec leaves blue and cyan speckle along the outlines, and it sits INSIDE
    # the silhouette, so an edge-band fix does not reach it. The palette is a sepia duotone: every
    # real colour in this drawing has red above green above blue. Anything that breaks that order
    # is codec noise by definition, so it gets pulled back to the palette.
    lum = out.mean(axis=2, keepdims=True)
    sepia = np.concatenate([lum * 1.06, lum, lum * 0.88], axis=2)
    r, g, b = out[..., 0], out[..., 1], out[..., 2]
    off_palette = np.clip(np.maximum(b - r, g - r) / 18.0, 0.0, 1.0)[..., None]
    out = out * (1 - off_palette) + sepia * off_palette

    keep = (alpha > 0.02)[..., None]
    return np.where(keep, out, rgb)


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


def hold_end(frames, warmup=0.25):
    """Where a go-out-hold-come-back move REACHES its held pose.

    A hold move freezes on its last cell while the key is down, so the cells must stop at the
    extreme -- keep the recovery and the player holds a shrug. The obvious test, "furthest pose
    from the first frame", is wrong twice over: it assumes frame one is neutral (Cadbury's crouch
    already starts half-crouched, so it chose the STANDING recovery), and an outlier artifact
    maximises distance-from-anything, so it actively selects a corrupted frame.

    Motion energy does not have those failure modes. The move goes out, rests, and comes back, so
    the hold is a plateau of near-zero movement after the opening. Take the quietest sustained run
    in the middle of the clip and cut at its end.
    """
    sils = silhouettes(frames)
    energy = [float(np.abs(sils[i + 1] - sils[i]).mean()) for i in range(len(sils) - 1)]
    if not energy:
        return len(frames) - 1
    start = int(len(energy) * warmup)
    win = max(3, len(energy) // 8)
    best, best_e = None, None
    for t in range(start, len(energy) - win):
        e = sum(energy[t:t + win]) / win
        if best_e is None or e < best_e:
            best, best_e = t, e
    # end of the plateau: the last frame before movement picks up again
    quiet = best_e * 2.0 + 1e-6
    end = best + win
    while end < len(energy) and energy[end] <= quiet:
        end += 1
    return min(end, len(frames) - 1)


def find_cycle(sils, min_period=6):
    """Find the gait period by self-similarity, so the sheet spans exactly one loop.

    A walk repeats: frame t looks like frame t+P. Score every candidate period by how closely the
    clip matches itself shifted by P, and take the best. Sampling n frames across exactly one
    period is what makes the first and last cell adjacent poses — i.e. what makes the loop
    seamless. Without this the sheet spans some arbitrary 0.7 of a stride and visibly hitches.

    Returns (period, score, trough). `trough` is how many times the curve's shoulders sit above
    its minimum — the flat-curve discriminator described at CYCLE_TROUGH_MIN. A curve whose
    minimum is at either end of the search range has no trough and scores 1.0, which is the right
    answer: a monotone curve is telling you the clip never comes back to a pose.
    """
    n = len(sils)
    periods = list(range(min_period, n // 2 + 1))
    if not periods:
        return None, None, 1.0
    curve = []
    for p in periods:
        pairs = n - p
        curve.append(sum(float(np.abs(sils[t] - sils[t + p]).mean()) for t in range(pairs)) / pairs)
    i = int(np.argmin(curve))
    best, best_score = periods[i], curve[i]
    if 0 < i < len(curve) - 1 and best_score > 0:
        trough = min(max(curve[:i]), max(curve[i + 1:])) / best_score
    else:
        trough = 1.0
    return best, float(best_score), float(trough)


def is_cycle(period, score, trough):
    """CYCLE or AMBIENT. Both gates are absolute; see CYCLE_SCORE_MAX and CYCLE_TROUGH_MIN.

    A move that passes is periodic and gets its cells cropped to one gait period. A move that
    fails is AMBIENT: it drifts, it never returns, and cropping it to a "period" that is really
    just the flattest part of a flat curve is what produced Seth's idle sheet with four duplicate
    cells across a six-frame window.
    """
    return bool(period and score is not None
                and score < CYCLE_SCORE_MAX and trough >= CYCLE_TROUGH_MIN)


# Which spacing rule the CYCLIC branch uses inside the gait period. "arclength" spaces by equal
# cumulative MOTION -- the same rule the non-cyclic branch below has always used -- and is now the
# default: A/B'd against equal TIME on Cadbury's walk it closed the loop 39% tighter (loop_closure
# 1.27 -> 0.77) at identical cell count, distinctness and every other QC number. "energy" is the
# old equal-time rule, kept as SELECT_MODE=energy for rollback. Span selection (where the period
# starts, how long it is) is identical either way, and so is the sharpest-frame nudge.
SELECT_MODE = os.environ.get("SELECT_MODE", "arclength")


def motion_curve(sils):
    """Per-frame motion: mean absolute difference between consecutive keyed silhouettes.

    Read off the 48x48 silhouettes rather than the RGBA, so it measures the CHARACTER moving and
    not the halftone or the codec breathing on a flat magenta field.
    """
    out = [0.0]
    for i in range(1, len(sils)):
        out.append(float(np.abs(sils[i] - sils[i - 1]).mean()))
    return np.asarray(out)


def arclength_raw(sils, start, period, count):
    """`count` frames at equal CUMULATIVE-MOTION intervals along one gait period.

    Equal-time spacing hands every beat of the stride the same cell budget, so a passing position
    that crosses in two frames gets as many cells as a contact that dwells for six. Walking the
    cumulative-motion curve instead places a cell every time the same AMOUNT of movement has
    happened, which is what an animator means by even spacing.

    It cannot manufacture frames that are not there: `count` picks across a `period`-frame span
    can only ever be as distinct as the span is long, whatever the spacing rule.
    """
    seg = motion_curve(sils[start:start + period + 1])
    cum = np.cumsum(seg)
    if cum[-1] <= 0:                          # a dead-still span: fall back to equal time
        return [start + int(round(period * i / count)) for i in range(count)]
    targets = np.linspace(0, cum[-1], count, endpoint=False)
    return [start + min(int(np.searchsorted(cum, t)), period - 1) for t in targets]


def farthest_poses(sils, count):
    """`count` maximally distinct poses across the WHOLE clip. Greedy farthest-point, deterministic.

    This is the AMBIENT rule. An idle has no period to crop to, so any window you pick is
    arbitrary, and a narrow one hands the repainter the same drawing several times over. Instead
    take the pose furthest from the clip's average, then repeatedly take the frame that is most
    different from EVERY pose already taken. That maximises the smallest gap between cells, which
    is the quantity `duplicate_cells` actually measures.

    Deterministic by construction: no randomness, and np.argmax breaks every tie on the lowest
    index, so the same clip always yields the same cells.
    """
    n = len(sils)
    if count >= n:
        return list(range(n))
    flat = np.stack([s.ravel() for s in sils])
    # Seed on the most extreme pose rather than frame 0: frame 0 is wherever H3 happened to start.
    first = int(np.argmax(np.abs(flat - flat.mean(axis=0)).mean(axis=1)))
    picks = [first]
    dist = np.abs(flat - flat[first]).mean(axis=1)
    while len(picks) < count:
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0:      # nothing distinct left — a frozen clip, not a selection failure
            break
        picks.append(nxt)
        dist = np.minimum(dist, np.abs(flat - flat[nxt]).mean(axis=1))
    return sorted(picks)


def pose_extremes(frames, count, cycle=True):
    """Pick the cells. Inside one gait period when the move loops, by motion energy when it does not."""
    sils = silhouettes(frames)
    if cycle and len(frames) >= 16:
        period, score, trough = find_cycle(sils)
        if is_cycle(period, score, trough):
            # Where the cycle STARTS decides whether it loops. Picking the busiest frame reads
            # well but says nothing about the seam, and the walk came back with a wrap step twice
            # the size of every other step. Start instead on the frame whose pose the cycle
            # returns to most exactly — that IS the seam, so minimising it closes the loop.
            start = min(range(len(frames) - period),
                        key=lambda t: float(np.abs(sils[t] - sils[t + period]).mean()))
            # A period-frame span holds at most `period` distinct drawings, whatever the spacing
            # rule. Asking for more only buys duplicate cells, which read as a stutter.
            if count > period:
                print(f"  warning: {count} cells asked for but the gait period is {period} frames"
                      f" — capping at {period}")
                count = period
            if SELECT_MODE == "energy":
                raw = [start + int(round(period * i / count)) for i in range(count)]
            else:
                raw = arclength_raw(sils, start, period, count)
            return _snap(frames, raw), period
        # AMBIENT: flat or featureless self-similarity curve, so there is no period to crop to.
        # Spread the cells over the whole clip, then put them back in clip order — an idle loops
        # as a subtle shuffle and still has to PLAY, so time order is the playback order.
        print(f"  ambient move (score {'n/a' if score is None else f'{score:.4f}'}, "
              f"trough {trough:.2f}x < {CYCLE_TROUGH_MIN}) — {count} distinct poses "
              f"across the whole clip")
        return sorted(_snap(frames, farthest_poses(sils, count))), None
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
    picks = [int(np.searchsorted(cum, t)) for t in targets]
    return _snap(frames, picks), None


def _snap(frames, raw):
    """Nudge every pick to a sharp frame WITHOUT letting two picks land on the same one.

    `sharpest_near` searches a couple of frames either side. When the cells are closely spaced —
    ten cells across a 23-frame gait period is 2.3 frames apart — that search window is wider than
    the gap, so neighbouring picks collapse onto the same frame. The walk sheet came back with four
    duplicate cells and a loop that jumped, because the cycle was really only six drawings.

    So the window is derived from the actual spacing, and any pick that still collides falls back
    to its unsnapped position. Distinct cells matter more than a marginally sharper one.
    """
    n = len(frames)
    gaps = [b - a for a, b in zip(raw, raw[1:])] or [3]
    window = max(0, min(2, (min(gaps) - 1) // 2))
    picks = []
    for p in raw:
        p = min(p, n - 1)
        q = sharpest_near(frames, p, window) if window else p
        if q in picks:
            q = p if p not in picks else q
        picks.append(q)
    return picks


def sharpness(rgba):
    """Mean gradient inside the silhouette — high means crisp ink, low means motion blur."""
    a = np.asarray(rgba).astype(np.float32)
    lum = a[:, :, :3].mean(axis=2)
    mask = a[:, :, 3] > 128
    if mask.sum() < 100:
        return 0.0
    gx = np.abs(np.diff(lum, axis=1))
    m = mask[:, :-1] & mask[:, 1:]
    return float(gx[m].mean()) if m.any() else 0.0


def sharpest_near(frames, index, window=2):
    """Nudge a pick to the crispest frame within a couple of frames.

    H3 motion-blurs a limb that is travelling fast, and the beat we want often lands on one of
    those. The neighbouring frames are the same pose a few milliseconds either side, so taking the
    sharpest of them costs nothing in timing and buys a clean drawing.
    """
    lo, hi = max(0, index - window), min(len(frames) - 1, index + window)
    return max(range(lo, hi + 1), key=lambda i: sharpness(frames[i]))


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
    highest = max(p.get("up", 0) for _, _, p in measured)
    # 0.94 of the cell is the feet line; leave 3% of headroom above the tallest reach
    scale = min((cell * 0.92) / tallest, (cell * 0.88) / highest) if tallest else 1.0
    out = []
    for name, n, prep in measured:
        out.append(_emit(name, prep, cell, outdir, anchor, scale))
    return out


def pack(clip, name, n_frames, cell, outdir, anchor, smooth, skip=6, cycle=True):
    """Pack one move on its own. Use pack_set() for a character's whole move list."""
    prep = _prepare(clip, name, n_frames, skip, cycle, anchor, smooth)
    scale = (cell * 0.92) / prep["natural"] if prep["natural"] else 1.0
    return _emit(name, prep, cell, outdir, anchor, scale)


def _prepare(clip, name, n_frames, skip, cycle, anchor, smooth, stop=False):
    """Key, pick the cells, and work out each cell's anchor and reach. No pixels written yet."""
    tmp = f"/tmp/sprite-{name}"
    paths = extract_frames(clip, tmp)[skip:]
    rgba = [key_out(p) for p in paths]
    boxes = [bbox(r) for r in rgba]
    keep = [i for i, b in enumerate(boxes) if b]
    if not keep:
        # RuntimeError, not SystemExit: this runs as a library call from the sprite agent's worker
        # thread. SystemExit is a BaseException and slips past a bare `except Exception`,
        # silently killing the worker thread and stalling every job queued behind it.
        raise RuntimeError(f"{name}: every frame keyed to nothing — is the clip actually on magenta?")

    # A HOLD move must stop at its held pose: the player freezes on the last cell, so the recovery
    # must never be picked. hold_end reads that off the clip's own motion.
    if stop:
        cut = hold_end([rgba[i] for i in keep])
        keep = keep[:cut + 1]
    picks, period = pose_extremes([rgba[i] for i in keep], n_frames, cycle)
    picks = [keep[p] for p in picks]
    print(f"  {name} cycle: {'period ' + str(period) + ' frames' if period else 'ambient / not periodic — cells spread over the whole clip'}")

    # anchor per frame: feet = bottom centre of the silhouette, the origin a game engine expects
    anchors = []
    for i in picks:
        x0, y0, x1, y1 = boxes[i]
        anchors.append((torso_cx(rgba[i], boxes[i]),
                        y1 if anchor == "feet" else (y0 + y1) // 2))
    if smooth and len(anchors) > 2:
        anchors = _smooth_anchors(anchors)

    spans = []
    for (i, (ax, ay)) in zip(picks, anchors):
        x0, y0, x1, y1 = boxes[i]
        spans.append((ax - x0, x1 - ax, ay - y0, y1 - ay))
    natural = max(max(s[0] for s in spans) + max(s[1] for s in spans),
                  max(s[2] for s in spans) + max(s[3] for s in spans))
    return {"rgba": rgba, "boxes": boxes, "picks": picks, "anchors": anchors, "up": max(s[2] for s in spans),
            "natural": natural, "period": period, "clip": clip}


def _prepare_stills(paths, smooth=True, unify="none"):
    """Same measurement pass as _prepare, but every still IS a cell — no picking, no cycle hunt.

    Stills come from sprite_poses.py: one render per pose beat, so the frame order is the
    playback order and there is nothing to select.
    """
    rgba = [key_out(p) for p in paths]
    boxes = [bbox(r) for r in rgba]
    picks = [i for i, b in enumerate(boxes) if b]
    if not picks:
        # RuntimeError, not SystemExit: see _prepare() above — this is a library function, and
        # a BaseException here would silently kill the sprite agent's worker thread.
        raise RuntimeError("every still keyed to nothing — are they on magenta?")
    if len(picks) != len(paths):
        # A dropped index must be loud: `picks` still carries each surviving cell's ORIGINAL
        # manifest position (consumers like _emit_sheet key metadata off that, not off position in
        # `picks`), but a silent drop shrinks the tag range by one frame with no trace in the log.
        dropped = [i for i in range(len(paths)) if i not in picks]
        print(f"  dropped {len(dropped)} fully-transparent frame(s) at index {dropped}: {[paths[i] for i in dropped]}")
    anchors = []
    for i in picks:
        x0, y0, x1, y1 = boxes[i]
        anchors.append((torso_cx(rgba[i], boxes[i]), y1))
    if smooth and len(anchors) > 2:
        anchors = _smooth_anchors(anchors)
    # Per-cell size correction, worked out BEFORE the set scale. A cell that gets scaled up by 6%
    # reaches 6% higher, so measuring the set's reach on uncorrected spans under-reads it and the
    # heads end up against the top of the cell.
    factors = _unify_factors(rgba, boxes, picks, unify)

    spans = []
    for (i, (ax, ay), f) in zip(picks, anchors, factors):
        x0, y0, x1, y1 = boxes[i]
        spans.append(((ax - x0) * f, (x1 - ax) * f, (ay - y0) * f, (y1 - ay) * f))
    # Scale off HEIGHT, with width only allowed to veto if it would overflow the cell. A wide
    # pose — a kick at full extension — would otherwise shrink every other move to match it.
    tall = max(s[2] for s in spans) + max(s[3] for s in spans)
    wide = max(s[0] for s in spans) + max(s[1] for s in spans)
    natural = max(tall, wide * 0.8)
    # A jump reaches higher above the anchor than a stand does, and the anchor sits at 94% of the
    # cell, so "fits the cell" is not the same as "fits above the feet". Keep the tallest reach.
    up = max(s[2] for s in spans)
    return {"rgba": rgba, "up": up, "boxes": boxes, "picks": picks, "anchors": anchors,
            "factors": factors, "natural": natural, "period": None, "clip": paths[0]}


# A raw anchor moves for two different reasons: sub-pixel matte shimmer, which should be smoothed
# away, and the character genuinely striding, which must not be. A plain median filter cannot tell
# them apart — it swaps in a neighbour's foot position wholesale, which floated the walk cells up
# to 16px off the ground line. So the smoothed value is only allowed to pull an anchor a few
# pixels: enough for shimmer, never enough to move a footfall.
ANCHOR_PULL = 3


def head_width(rgba, box):
    """A size measure that a stride cannot change.

    Unifying cells by total silhouette height is wrong for a walk: the body genuinely rises and
    falls, so height mixes real animation with generation drift and correcting it distorts the
    animation. The head does not change size when he takes a step, so the widest row of the top
    fifth of the silhouette is a clean scale reference.
    """
    a = np.asarray(rgba)[:, :, 3] > 32
    x0, y0, x1, y1 = box
    band = a[y0:y0 + max(2, (y1 - y0) // 5), :]
    rows = band.sum(axis=1)
    return float(np.median(rows[rows > 0])) if (rows > 0).any() else 0.0


def torso_cx(rgba, box):
    """Horizontal anchor that does not swing with the legs.

    The bottom-centre of the silhouette is the right ORIGIN but the wrong x: a stride throws one
    leg forward and one back, so the bounding box's centre swings with the legs and drags the
    whole drawing side to side — the walk cells slid 23px against each other while the feet were
    already planted. The head and shoulders travel with the body instead, so the x anchor is
    taken from the top third of the silhouette.
    """
    a = np.asarray(rgba)[:, :, 3] > 32
    x0, y0, x1, y1 = box
    band = a[y0:y0 + max(1, (y1 - y0) // 3), :]
    xs = np.nonzero(band)[1]
    return int(xs.mean()) if xs.size else (x0 + x1) // 2


def _smooth_anchors(anchors):
    xs = median3([a[0] for a in anchors])
    ys = median3([a[1] for a in anchors])
    out = []
    for (ax, ay), sx, sy in zip(anchors, xs, ys):
        out.append((ax + max(-ANCHOR_PULL, min(ANCHOR_PULL, sx - ax)),
                    ay + max(-ANCHOR_PULL, min(ANCHOR_PULL, sy - ay))))
    return out


def _unify_factors(rgba, boxes, picks, unify):
    """Per-cell scale correction, clamped. "none" -> all 1.0."""
    mode = "height" if unify is True else (unify or "none")
    if mode == "none":
        return [1.0] * len(picks)
    if mode == "head":
        sizes = [head_width(rgba[i], boxes[i]) for i in picks]
    else:
        sizes = [float(boxes[i][3] - boxes[i][1]) for i in picks]
    med = sorted(sizes)[len(sizes) // 2]
    # Past a few percent it is a mis-measurement, not drift, and acting on it distorts the drawing.
    return [max(0.94, min(1.06, med / z)) if z else 1.0 for z in sizes]


def cell_pivot(cell, anchor):
    """The point in the packed cell that a frame's anchor is placed on.

    0.94 of the cell height is the feet line the whole set is packed against; a consumer that
    draws a character standing on the ground draws this point on the ground.
    """
    return (cell // 2, int(cell * 0.94) if anchor == "feet" else cell // 2)


def _compose(src, anchor_xy, pivot, cell, f_scale):
    """Scale one keyed frame and place its anchor on `pivot` inside an empty cell.

    Both emitters go through here on purpose. The per-move atlas and the per-character sheet show
    the SAME cell, so the placement arithmetic — including where it truncates to whole pixels —
    has to exist once. Two copies drift by a pixel and nobody sees it until the two outputs are
    diffed against each other.
    """
    ax, ay = anchor_xy
    canvas = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
    w, h = src.size
    sized = src.resize((max(1, int(w * f_scale)), max(1, int(h * f_scale))), Image.LANCZOS)
    canvas.paste(sized, (int(pivot[0] - ax * f_scale), int(pivot[1] - ay * f_scale)), sized)
    return canvas


def _emit(name, prep, cell, outdir, anchor, scale, unify_height=False):
    """Write the atlas, the cells, the JSON, the CSS and the preview at the given scale."""
    rgba, boxes, picks, anchors = prep["rgba"], prep["boxes"], prep["picks"], prep["anchors"]
    frames_dir = os.path.join(outdir, f"{name}-frames")
    os.makedirs(frames_dir, exist_ok=True)
    # Per-frame height normalisation. Every still is generated independently, so the character
    # comes out a few percent bigger or smaller each time and a looping sprite "breathes". For a
    # cycle that has to be seamless, each frame is scaled so its silhouette height matches the
    # set's median. Moves whose height genuinely changes — crouch, jump — must NOT use this.
    factors = prep.get("factors") or [1.0] * len(picks)
    pivot = cell_pivot(cell, anchor)

    cells, table = [], []
    for (i, (ax, ay)) in zip(picks, anchors):
        canvas = _compose(rgba[i], (ax, ay), pivot, cell, scale * factors[len(cells)])
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


# One row is not an option for a whole character: 54 cells x 512px is 27,648px wide, and every
# GPU the browser runs on refuses a texture past 16,384px. A grid keeps the widest dimension at
# 10 x 512 = 5,120px, and a consumer reads a frame's cell straight off its index.
SHEET_COLUMNS = 10


def _emit_sheet(cid, parts, cell, outdir, scale, pivot):
    """Write ONE tagged atlas for a character: `<cid>.png` and `<cid>.json`.

    `parts` is [(tag, prep, frames)] in playback order — a tag, the measured pass over its cells,
    and its frame records from the character manifest. The tag ranges are recomputed here rather
    than copied, because the sheet IS this concatenation: computing them anywhere else lets the
    ranges and the pixels disagree.

    The pivot written per frame is the CONSTANT sheet origin (see `pivot`), the same for every
    frame regardless of that frame's `pivot_nudge`. The nudge shifts where the ARTWORK is pasted
    inside the cell, not where the reported origin sits — recording pivot+nudge instead would make
    the nudge invisible to a consumer that places the recorded pivot on the ground line, since the
    art and its reported origin would then move by the same vector every time.
    """
    cells, frames_meta, tags_meta = [], [], []
    for tag, prep, frames in parts:
        rgba, picks, anchors = prep["rgba"], prep["picks"], prep["anchors"]
        factors = prep.get("factors") or [1.0] * len(picks)
        start = len(cells)
        for n, (i, (ax, ay)) in enumerate(zip(picks, anchors)):
            # `i` is the frame's position in the manifest BEFORE any transparent-frame drop, and
            # `picks` already carries it — a dropped frame makes `i` skip ahead of the enumeration
            # counter `n`, so indexing `frames` by `n` would pull the previous frame's hold,
            # pivot_nudge and trim onto this cell. `factors` is exempt: it was built off `picks`
            # itself in _unify_factors, so it is already aligned by position, not by manifest index.
            nx, ny = frames[i].get("pivot_nudge", (0, 0))
            # The recorded sheet pivot is the CONSTANT origin every cell shares; the nudge moves
            # only the ARTWORK relative to it. Pasting at pivot+nudge but recording the bare pivot
            # keeps the two independent — recording pivot+nudge here looks more "correct" but makes
            # a consumer that draws the recorded pivot on the ground line see no effect from any
            # nudge, since the art and its reported origin would have moved by the same amount.
            fpivot = (pivot[0] + nx, pivot[1] + ny)
            canvas = _compose(rgba[i], (ax, ay), fpivot, cell, scale * factors[n])
            index = len(cells)
            cells.append(canvas)
            box = bbox(canvas)
            x0, y0, x1, y1 = box if box else (0, 0, 0, 0)
            frames_meta.append({
                "x": (index % SHEET_COLUMNS) * cell, "y": (index // SHEET_COLUMNS) * cell,
                "hold": int(frames[i].get("hold", 1)), "pivot": [pivot[0], pivot[1]],
                # Trim is the opaque rect INSIDE the cell. The cells stay full-size — trimming the
                # atlas itself would break the constant frame stride — so this is the margin a
                # consumer may skip, not a translation it must undo.
                "trim": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}})
        tags_meta.append({"name": tag["name"], "from": start, "to": len(cells) - 1,
                          "fps": tag["fps"], "direction": tag["direction"],
                          "loop": tag["loop"], "hold_key": tag["hold_key"]})

    rows = (len(cells) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    atlas = Image.new("RGBA", (cell * min(SHEET_COLUMNS, len(cells)), cell * rows), (0, 0, 0, 0))
    for f, c in zip(frames_meta, cells):
        # Copied, not composited. Pasting a cell through its own alpha as a mask squares that
        # alpha (a*a/255), so soft ink edges land in the atlas fainter than in the cell they came
        # from and the recorded trim no longer describes the pixels a consumer samples. The cells
        # never overlap, so there is nothing to composite against anyway.
        atlas.paste(c, (f["x"], f["y"]))
    atlas_path = os.path.join(outdir, f"{cid}.png")
    atlas.save(atlas_path)

    json.dump({"cell": cell, "columns": SHEET_COLUMNS, "frames": frames_meta, "tags": tags_meta},
              open(os.path.join(outdir, f"{cid}.json"), "w"), indent=1)
    print(f"{cid}: {len(cells)} frames in {len(tags_meta)} tags, "
          f"{atlas.width}x{atlas.height} -> {atlas_path}")
    return atlas_path


# ------------------------------------------------------------------ numeric QC
#
# Method borrowed from gary149/h3-game-sprites, which writes a `report.json` per move and says to
# read it before trusting anything. We measured the same things, but only by hand in conversation
# — which is exactly why every defect below was caught late and by eye. Now the pack measures
# them every run.
#
# The QC pass reads the atlas back off disk and measures the pixels that SHIPPED. It hands nothing
# back to the packer, so it cannot move a cell by a pixel.

QC_DECIDED = 8       # alpha within this of 0 or 255 is a pixel the key was sure about

# Every limit is set from what the two packed characters — Seth and Cadbury, both known good —
# actually measure, with room for normal variation. The measured range is written next to each
# one, so the next person can see whether a change moved the data or just moved the goalposts.
QC_LIMITS = {
    # measured 0.9477 (seth idle) .. 0.9989; a key against the wrong magenta reads far below 0.5,
    # because the whole screen comes back faintly opaque instead of decided.
    "bg_purity": 0.92,
    # wrap step / median step. Measured 0.32 .. 1.71 across the four cyclic tags. Past ~2.5 the
    # seam is jumping more than twice as far as a normal step, which is a visible reset.
    "loop_closure": 2.5,
    # Cells are anchored on their own feet, so this is residual only — smoothing pull (<=3px) plus
    # rounding. Measured 1 .. 4px. A double-digit spread means the anchor is reading something
    # other than the feet.
    "feet_drift_px": 8,
    # Head-width spread, judged only on tags the packer unifies (see below). Measured 1.1 .. 4.3%.
    "size_drift_pct": 8.0,
    # Measured 30 .. 60px of clear space above the head in a 512 cell. A clipped head reads 0, and
    # PAD exists precisely because eight of ten walk cells once came back flat-topped.
    "top_clearance_px": 12,
    # Two consecutive cells with the same silhouette are one drawing shown twice: an uneven hold,
    # which reads as a stutter. There is no benign amount of it.
    "duplicate_cells": 0,
}


def _qc_purity(rgba):
    """How much of a keyed frame the key was confident about (their `worst_bg_purity`).

    Ours is a SOFT key — `key_out` ramps alpha across a colour-distance band — so the analogue of
    their binary bg+fg fraction is the fraction of pixels that came out decided, either screen or
    character. The failure it catches is the one `measure_key` exists for: key against the wrong
    magenta and the whole screen comes back faintly opaque, which is a picture full of undecided
    pixels long before anyone can see the haze.
    """
    a = np.asarray(rgba)[:, :, 3]
    return float(((a <= QC_DECIDED) | (a >= 255 - QC_DECIDED)).mean())


def _qc_cells(atlas, meta, tmeta, cell):
    return [atlas.crop((meta["frames"][i]["x"], meta["frames"][i]["y"],
                        meta["frames"][i]["x"] + cell, meta["frames"][i]["y"] + cell))
            for i in range(tmeta["from"], tmeta["to"] + 1)]


def _qc_metric(value, limit, worse="high"):
    ok = value <= limit if worse == "high" else value >= limit
    return {"value": value, "limit": limit, "status": "ok" if ok else "warn"}


def sheet_qc(cid, parts, cell, outdir, pivot):
    """Measure the packed sheet and write `<outdir>/<cid>-qc.json`. Returns the report.

    Tag ranges and cell positions are read back out of the emitted JSON rather than recomputed,
    for the same reason `_emit_sheet` recomputes them from the concatenation: a second copy of
    the arithmetic is only somewhere for the report and the atlas to disagree.
    """
    atlas = Image.open(os.path.join(outdir, f"{cid}.png")).convert("RGBA")
    meta = json.load(open(os.path.join(outdir, f"{cid}.json")))
    tags, worst = {}, "ok"
    for (tag, prep, _frames), tmeta in zip(parts, meta["tags"]):
        cells = _qc_cells(atlas, meta, tmeta, cell)
        boxes = [bbox(c) for c in cells]
        if not all(boxes):
            # An empty cell has no silhouette to measure, so every measure below would be a lie.
            # It is also a defect on its own: a cell that keyed to nothing is a hole in the move.
            tags[tag["name"]] = {"cells": len(cells), "status": "warn", "warn": ["empty cell"],
                                 "empty_cells": [i for i, b in enumerate(boxes) if not b]}
            worst = "warn"
            continue

        purity = min(_qc_purity(prep["rgba"][i]) for i in prep["picks"])
        sils = silhouettes(cells)
        steps = [float(np.abs(sils[i] - sils[i - 1]).mean()) for i in range(1, len(sils))]
        bottoms = [b[3] - 1 - pivot[1] for b in boxes]
        heads = [head_width(c, b) for c, b in zip(cells, boxes)]
        med_head = sorted(heads)[len(heads) // 2]
        flat = [np.asarray(c)[:, :, 3] > 8 for c in cells]
        dupes = sum(1 for i in range(1, len(flat)) if np.array_equal(flat[i], flat[i - 1]))

        m = {
            "cells": len(cells),
            "bg_purity": _qc_metric(round(purity, 4), QC_LIMITS["bg_purity"], "low"),
            "feet_drift_px": _qc_metric(max(bottoms) - min(bottoms), QC_LIMITS["feet_drift_px"]),
            "feet_offset_px": [min(bottoms), max(bottoms)],
            "top_clearance_px": _qc_metric(min(b[1] for b in boxes),
                                           QC_LIMITS["top_clearance_px"], "low"),
            "duplicate_cells": _qc_metric(dupes, QC_LIMITS["duplicate_cells"]),
        }
        if tmeta["loop"] and len(steps) > 1:
            # The wrap step against the median step — the ratio we already compute by hand when a
            # cycle visibly resets. Absolute silhouette difference cannot be compared across
            # characters (a fence panel moves less than a lanky man), but "is the seam like every
            # other step?" is scale-free and is the question a loop actually asks.
            wrap = float(np.abs(sils[0] - sils[-1]).mean())
            med_step = sorted(steps)[len(steps) // 2]
            m["loop_closure"] = _qc_metric(round(wrap / med_step, 2) if med_step else 99.0,
                                           QC_LIMITS["loop_closure"])
        # `head_width` takes the widest row of the top fifth of the silhouette, which is the head
        # only while the head is the topmost thing. A punch, a kick and a jump all throw an arm or
        # a knee above it, so the number stops being a size measure there — which is also why the
        # packer unifies on it for walk and idle alone. Measured everywhere, judged where it means
        # something.
        size = round(100.0 * (max(heads) - min(heads)) / med_head, 1) if med_head else 100.0
        m["size_drift_pct"] = (_qc_metric(size, QC_LIMITS["size_drift_pct"]) if tag["unify"]
                               else {"value": size, "limit": None, "status": "n/a"})
        m["warn"] = [k for k, v in m.items() if isinstance(v, dict) and v["status"] == "warn"]
        m["status"] = "warn" if m["warn"] else "ok"
        worst = "warn" if m["status"] == "warn" else worst
        tags[tag["name"]] = m

    report = {"character": cid, "cell": cell, "pivot": list(pivot), "status": worst,
              "limits": QC_LIMITS, "tags": tags}
    path = os.path.join(outdir, f"{cid}-qc.json")
    json.dump(report, open(path, "w"), indent=1)
    return report


def print_qc(report):
    """One line per tag, so a bad tag is obvious without opening the file."""
    print(f"QC {report['character']}: {report['status'].upper()}  "
          f"(limits: purity>={QC_LIMITS['bg_purity']}, loop<={QC_LIMITS['loop_closure']}, "
          f"feet<={QC_LIMITS['feet_drift_px']}px, size<={QC_LIMITS['size_drift_pct']}%, "
          f"top>={QC_LIMITS['top_clearance_px']}px, dup<={QC_LIMITS['duplicate_cells']})")
    print(f"  {'tag':8} {'n':>2}  {'purity':>6} {'loop':>5} {'feet':>5} {'size':>6} "
          f"{'top':>5} {'dup':>3}  status        (bracketed = measured, not judged)")
    for name, t in report["tags"].items():
        if "bg_purity" not in t:
            print(f"  {name:8} {t['cells']:>2}  {'-- empty cells --':>34}  WARN")
            continue
        loop = f"{t['loop_closure']['value']:.2f}" if "loop_closure" in t else "-"
        flags = " ".join(t["warn"])
        size = (f"{t['size_drift_pct']['value']:.1f}%" if t["size_drift_pct"]["status"] != "n/a"
                else f"({t['size_drift_pct']['value']:.1f})")
        print(f"  {name:8} {t['cells']:>2}  {t['bg_purity']['value']:>6.4f} {loop:>5} "
              f"{t['feet_drift_px']['value']:>4}p {size:>6} "
              f"{t['top_clearance_px']['value']:>4}p {t['duplicate_cells']['value']:>3}  "
              f"{'WARN: ' + flags if flags else 'ok'}")



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
