#!/usr/bin/env python3
"""Cut one drawing into a rubber-hose puppet, then animate it with maths instead of samples.

    python3 puppet.py cut seth          # slice the splay render into parts + rig.json
    python3 puppet.py bake seth walk    # bake a move to a sprite sheet at any frame count

## Why a puppet

Every earlier attempt generated each animation frame independently — as a video frame, or as a
still from a written pose. That caps quality in a way no prompt fixes: two independent samples of
the same character are never the same drawing. Proportions breathe, the ink weight shifts, a shoe
grows. Interpolating between them (RIFE) then has to invent geometry, which is where the smears
came from.

A cut-out puppet removes the variable entirely. ONE crisp drawing is cut into parts; every frame
reuses those exact pixels under a rigid transform. So:

- **crisp** — the pixels are the 1024 drawing, never resampled from a video codec
- **fluid** — motion is joint angles interpolated with easing, so 8 or 60 frames is a parameter,
  not another 20 minutes of GPU
- **repeatable** — the same rig and the same keys give byte-identical output, every time
- **cheap** — one render per character (~50s), then every move thereafter is free

This is also the historically correct technique: 1933 rubber-hose characters were built from
tubes and circles precisely so they could be cut up and re-posed. The style is the rig.

## The one manual step

A character's parts have to be marked out once, on the splay render, in PARTS below: a box, a
pivot, and a parent. That is the whole per-character cost, and it is drawing coordinates, not
prompt engineering — it does not drift, and it never has to be done twice.
"""
import json
import math
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sprite_sheet import key_out  # noqa: E402

SPLAY = "/tmp/puppet/splay.png"
PAD = 260
RIG_DIR = "/home/wilson/artifacts/cast-fighter/puppet"

# box = (x0, y0, x1, y1) on the 1024 splay render · pivot = the joint it rotates about
# parent = what it hangs off · z = draw order, low first
PARTS = {
    "arm_far":       {"box": (632, 340, 1010, 470), "pivot": (655, 380), "parent": "torso",  "z": 0},
    "leg_far":       {"box": (545, 640, 930, 1010), "pivot": (585, 650), "parent": "torso",  "z": 1},
    "torso":         {"box": (398, 300, 672, 700),  "pivot": (530, 640), "parent": None,     "z": 2},
    "head":          {"box": (390, 40, 700, 380),   "pivot": (520, 350), "parent": "torso",  "z": 3},
    "leg_near":      {"box": (200, 640, 600, 990),  "pivot": (520, 650), "parent": "torso",  "z": 1.5},
    "arm_near":      {"box": (55, 340, 470, 470),   "pivot": (445, 385), "parent": "torso",  "z": 5},
}

# Joint caps. A cut has to end in a straight line somewhere, and that line shows as a seam when
# the limb swings. Rubber-hose characters are drawn as tubes with a round cap at every joint
# precisely so this is invisible — so the rig draws the cap, filled with a colour sampled from
# the drawing itself and rimmed in the same ink weight.
JOINTS = [
    {"at": (438, 390), "r": 24, "sample": (415, 372), "z": 4.9},   # near shoulder, in the sleeve
    {"at": (520, 652), "r": 30, "sample": (505, 634), "z": 1.6},   # near hip, in the trousers
]

# Rest offsets: the splay drawing holds every limb out straight so it can be cut cleanly, which
# is not a pose anyone stands in. These angles bring it back to a neutral stance, and every pose
# below is written relative to that.
REST = {"arm_near": 78, "arm_far": -78, "leg_near": 35, "leg_far": -35, "torso": 0, "head": 0}

# Joint angles per pose key, in degrees, measured from each part's rest angle in the splay
# drawing. The rig interpolates between these; the drawings never change.
POSES = {
    "walk": {
        "loop": True,
        "keys": [
            {"t": 0.00, "a": {"arm_near": 25, "arm_far": -25, "leg_near": 22, "leg_far": -22, "torso": 0, "head": 0, "y": 0}},
            {"t": 0.25, "a": {"arm_near": 0,  "arm_far": 0,   "leg_near": 0,  "leg_far": 0,   "torso": 0, "head": 0, "y": -6}},
            {"t": 0.50, "a": {"arm_near": -25, "arm_far": 25, "leg_near": -22, "leg_far": 22, "torso": 0, "head": 0, "y": 0}},
            {"t": 0.75, "a": {"arm_near": 0,  "arm_far": 0,   "leg_near": 0,  "leg_far": 0,   "torso": 0, "head": 0, "y": -6}},
        ],
    },
    "idle": {
        "loop": True,
        "keys": [
            {"t": 0.00, "a": {"arm_near": 4, "arm_far": -4, "leg_near": 2, "leg_far": -2, "torso": 0, "head": 0, "y": 0}},
            {"t": 0.50, "a": {"arm_near": 6, "arm_far": -6, "leg_near": 2, "leg_far": -2, "torso": -1.5, "head": 1.5, "y": 4}},
        ],
    },
    "punch": {
        "loop": False,
        "keys": [
            {"t": 0.00, "a": {"arm_near": 70, "arm_far": -40, "leg_near": 10, "leg_far": -10, "torso": 0, "head": 0, "y": 0}},
            {"t": 0.30, "a": {"arm_near": 95, "arm_far": -55, "leg_near": 12, "leg_far": -12, "torso": -6, "head": -4, "y": 0}},
            {"t": 0.55, "a": {"arm_near": -8, "arm_far": -35, "leg_near": 16, "leg_far": -16, "torso": 8, "head": 5, "y": -2}},
            {"t": 0.75, "a": {"arm_near": -8, "arm_far": -35, "leg_near": 16, "leg_far": -16, "torso": 8, "head": 5, "y": -2}},
            {"t": 1.00, "a": {"arm_near": 70, "arm_far": -40, "leg_near": 10, "leg_far": -10, "torso": 0, "head": 0, "y": 0}},
        ],
    },
    "kick": {
        "loop": False,
        "keys": [
            {"t": 0.00, "a": {"arm_near": 20, "arm_far": -20, "leg_near": 15, "leg_far": -15, "torso": 0, "head": 0, "y": 0}},
            {"t": 0.35, "a": {"arm_near": 45, "arm_far": -45, "leg_near": 55, "leg_far": -12, "torso": -10, "head": -6, "y": 0}},
            {"t": 0.60, "a": {"arm_near": 55, "arm_far": -55, "leg_near": 85, "leg_far": -10, "torso": -16, "head": -8, "y": 0}},
            {"t": 0.80, "a": {"arm_near": 55, "arm_far": -55, "leg_near": 85, "leg_far": -10, "torso": -16, "head": -8, "y": 0}},
            {"t": 1.00, "a": {"arm_near": 20, "arm_far": -20, "leg_near": 15, "leg_far": -15, "torso": 0, "head": 0, "y": 0}},
        ],
    },
    "jump": {
        "loop": False,
        "keys": [
            {"t": 0.00, "a": {"arm_near": 15, "arm_far": -15, "leg_near": 12, "leg_far": -12, "torso": 0, "head": 0, "y": 0}},
            {"t": 0.20, "a": {"arm_near": -30, "arm_far": 30, "leg_near": 30, "leg_far": -30, "torso": 10, "head": 6, "y": 18}},
            {"t": 0.45, "a": {"arm_near": 60, "arm_far": -60, "leg_near": 18, "leg_far": -18, "torso": -6, "head": -4, "y": -70}},
            {"t": 0.70, "a": {"arm_near": 40, "arm_far": -40, "leg_near": 24, "leg_far": -24, "torso": -3, "head": -2, "y": -40}},
            {"t": 1.00, "a": {"arm_near": 15, "arm_far": -15, "leg_near": 12, "leg_far": -12, "torso": 0, "head": 0, "y": 0}},
        ],
    },
    "block": {
        "loop": False,
        "keys": [
            {"t": 0.00, "a": {"arm_near": 20, "arm_far": -20, "leg_near": 10, "leg_far": -10, "torso": 0, "head": 0, "y": 0}},
            {"t": 1.00, "a": {"arm_near": 110, "arm_far": -105, "leg_near": 14, "leg_far": -14, "torso": 6, "head": 8, "y": 6}},
        ],
    },
    "crouch": {
        "loop": False,
        "keys": [
            {"t": 0.00, "a": {"arm_near": 15, "arm_far": -15, "leg_near": 10, "leg_far": -10, "torso": 0, "head": 0, "y": 0}},
            {"t": 1.00, "a": {"arm_near": 45, "arm_far": -45, "leg_near": 40, "leg_far": -40, "torso": 12, "head": 8, "y": 34}},
        ],
    },
}


def cut(cid):
    """Slice the splay drawing into parts and write the rig."""
    os.makedirs(RIG_DIR, exist_ok=True)
    full = key_out(SPLAY)
    rig = {"character": cid, "source": os.path.basename(SPLAY), "canvas": list(full.size),
           "parts": {}, "poses": POSES}
    for name, spec in sorted(PARTS.items(), key=lambda kv: kv[1]["z"]):
        x0, y0, x1, y1 = spec["box"]
        px, py = spec["pivot"]
        part = full.crop((x0, y0, x1, y1))
        # Pad the part into a square centred ON ITS PIVOT. An arm is a long thin strip; rotating
        # it inside its own bounding box slices the hand off. With the pivot at the centre of a
        # square whose half-side reaches the furthest corner, no rotation can ever clip, and the
        # placement maths collapses to "put the centre where the joint is".
        r = int(max(math.hypot(dx, dy) for dx in (x0 - px, x1 - px) for dy in (y0 - py, y1 - py))) + 2
        square = Image.new("RGBA", (2 * r, 2 * r), (0, 0, 0, 0))
        square.paste(part, (int(r - (px - x0)), int(r - (py - y0))))
        square.save(os.path.join(RIG_DIR, f"{cid}-{name}.png"))
        rig["parts"][name] = {
            "file": f"{cid}-{name}.png",
            "size": [2 * r, 2 * r],
            "pivot": [r, r],
            "canvas_pivot": [px, py],
            "parent": spec["parent"],
            "z": spec["z"],
        }
    rig["joints"] = [dict(j, fill=[int(v) for v in full.convert("RGB").getpixel(tuple(j["sample"]))])
                     for j in JOINTS]
    json.dump(rig, open(os.path.join(RIG_DIR, f"{cid}-rig.json"), "w"), indent=1)
    print(f"cut {len(PARTS)} parts -> {RIG_DIR}/{cid}-rig.json")
    return rig


def ease(t):
    """Smoothstep. Linear joint angles read mechanical; this is the difference between a rig that
    looks rigged and one that looks animated."""
    return t * t * (3 - 2 * t)


def sample(pose, u):
    """Angles at normalised time u, eased between the surrounding keys."""
    keys = pose["keys"] + ([dict(pose["keys"][0], t=1.0)] if pose["loop"] else [])
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if a["t"] <= u <= b["t"]:
            span = (b["t"] - a["t"]) or 1
            k = ease((u - a["t"]) / span)
            return {j: a["a"].get(j, 0) * (1 - k) + b["a"].get(j, 0) * k
                    for j in set(a["a"]) | set(b["a"])}
    return dict(keys[-1]["a"])


def compose(rig, angles):
    """Draw one frame at full resolution: every part rotated about its pivot, hung off its parent.

    Returns the raw RGBA canvas. Framing is NOT done here — the frames go through the same
    feet-anchored, uniform-cell packer the drawn stills use, so a puppet sheet and a drawn sheet
    are interchangeable.
    """
    cw, ch = rig["canvas"]
    # Room around the drawing: a leg swung forward reaches past the edge of the frame the splay
    # was drawn in, and a jump lifts the whole figure out of it.
    pad = PAD
    canvas = Image.new("RGBA", (cw + 2 * pad, ch + 2 * pad), (0, 0, 0, 0))
    root = rig["parts"]["torso"]["canvas_pivot"]
    lift = angles.get("y", 0)

    layers = [(p["z"], "part", n) for n, p in rig["parts"].items()]
    layers += [(j["z"], "joint", i) for i, j in enumerate(rig.get("joints", []))]

    for _, kind, ref in sorted(layers):
        if kind == "joint":
            j = rig["joints"][ref]
            jx, jy = j["at"]
            ta = math.radians(angles.get("torso", 0) + REST.get("torso", 0))
            rx, ry = root
            dx, dy = jx - rx, jy - ry
            jx = rx + dx * math.cos(ta) + dy * math.sin(ta)
            jy = ry - dx * math.sin(ta) + dy * math.cos(ta)
            cap = Image.new("RGBA", (2 * j["r"] + 8, 2 * j["r"] + 8), (0, 0, 0, 0))
            ImageDraw.Draw(cap).ellipse([4, 4, 2 * j["r"] + 4, 2 * j["r"] + 4],
                                        fill=tuple(j["fill"]) + (255,), outline=(20, 18, 15, 255),
                                        width=5)
            canvas.alpha_composite(cap, (int(jx - j["r"] - 4) + pad,
                                         int(jy - j["r"] - 4 + lift) + pad))
            continue
        name = ref
        part = rig["parts"][name]
        img = Image.open(os.path.join(RIG_DIR, part["file"])).convert("RGBA")
        angle = angles.get(name, 0) + REST.get(name, 0)
        rot = img.rotate(angle, resample=Image.BICUBIC, center=tuple(part["pivot"]), expand=False)
        px, py = part["canvas_pivot"]
        if part["parent"] == "torso" and name != "torso":
            # a child follows the torso's own rotation about the hips
            ta = math.radians(angles.get("torso", 0) + REST.get("torso", 0))
            rx, ry = root
            dx, dy = px - rx, py - ry
            px = rx + dx * math.cos(ta) + dy * math.sin(ta)
            py = ry - dx * math.sin(ta) + dy * math.cos(ta)
        ox = int(px - part["pivot"][0]) + pad
        oy = int(py - part["pivot"][1] + lift) + pad
        canvas.alpha_composite(rot, (ox, oy))
    return canvas


def bake(cid, move, frames=24, cell=256, outdir=None):
    """Bake a move to a normalised sprite sheet, through the same packer the drawings use."""
    from sprite_sheet import _emit, median3, bbox

    rig = json.load(open(os.path.join(RIG_DIR, f"{cid}-rig.json")))
    pose = rig["poses"][move]
    outdir = outdir or RIG_DIR
    rgba = [compose(rig, sample(pose, i / frames if pose["loop"] else i / max(frames - 1, 1)))
            for i in range(frames)]
    boxes = [bbox(r) for r in rgba]
    picks = [i for i, b in enumerate(boxes) if b]
    anchors = [((boxes[i][0] + boxes[i][2]) // 2, boxes[i][3]) for i in picks]
    anchors = list(zip(median3([a[0] for a in anchors]), median3([a[1] for a in anchors])))
    spans = [(a[0] - boxes[i][0], boxes[i][2] - a[0], a[1] - boxes[i][1], boxes[i][3] - a[1])
             for i, a in zip(picks, anchors)]
    natural = max(max(s[2] for s in spans) + max(s[3] for s in spans),
                  (max(s[0] for s in spans) + max(s[1] for s in spans)) * 0.8)
    prep = {"rgba": rgba, "boxes": boxes, "picks": picks, "anchors": anchors,
            "natural": natural, "period": frames if pose["loop"] else None,
            "clip": f"{cid}-rig.json"}
    _emit(f"{cid}-{move}", prep, cell, outdir, "feet", (cell * 0.92) / natural)
    return os.path.join(outdir, f"{cid}-{move}.png")


if __name__ == "__main__":
    cmd, cid = sys.argv[1], sys.argv[2]
    if cmd == "cut":
        cut(cid)
    elif cmd == "bake":
        for move in (sys.argv[3:] or POSES):
            bake(cid, move, frames=24 if POSES[move]["loop"] else 16)
