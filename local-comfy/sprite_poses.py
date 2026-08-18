#!/usr/bin/env python3
"""Sprite cells rendered as STILLS, one pose beat at a time.

    python3 sprite_poses.py seth walk            # one move
    python3 sprite_poses.py seth --all           # every move in POSE_BEATS
    python3 sprite_poses.py seth --pack          # pack what has already been rendered

Why not video. MiniMax H3 draws the character about 240px tall inside a soft 512 render; measured
edge sharpness 2.47 against 2.49 for a still at 1024, where the character fills the frame. Blown up
to sprite size the H3 frames read blurry and no packing fixes that — the detail was never there.
Two rescue attempts failed and are worth not repeating: img2img refine of an H3 frame at partial
denoise makes mush (1.42 sharpness) because the Lightning LoRA needs cfg 1.0 and a full denoise,
and a full Qwen re-render of an H3 frame comes back crisp but RE-POSED, because a reference image
is an identity reference, not a pose reference.

So the pose comes from words instead. Each move is a list of pose beats — the drawings an animator
would key — and every beat is one still render off the character's own plate. That makes the whole
thing repeatable: same beats, same plate, same seed, same sheet.

Rendering: ~48s a cell. A 6-cell move is 5 minutes, the seven-move fighter set about 26.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import smoke_test as S  # noqa: E402
from build_workflows import UNET, LORA, CLIP, VAE, STYLE_LOCK, STEPS  # noqa: E402
from build_transition import TRAITS  # noqa: E402

OUT_ROOT = "/tmp/poses"
# This is the older stills-based pipeline, kept for reference only. It must NOT point at
# /home/wilson/artifacts/cast-fighter/sprites: that's the live output directory for the
# grid-atlas pipeline (repaint_cells.py / sprite_sheet.py), and one --pack run here would
# resurrect the 70+ legacy per-move PNGs and <cid>-moves.json files the migration deleted.
SHEET_DIR = "/tmp/poses-sheets"

# The staging every cell shares. Said the same way every time, because any variation here is
# variation between cells, which reads as the character resizing or the camera moving.
STAGE_LOCK = (
    "Redraw the character in image 1 as one frame of a sprite animation: clean crisp 1933 "
    "rubber-hose cartoon line art, confident smooth ink outlines of even weight, sharp clear edges, "
    "soft halftone dot shading, muted sepia inks. He is seen from his left side in a locked-off "
    "wide side view, standing on an invisible ground line at the bottom of the frame, his whole "
    "body inside the picture at the same size and the same place in every frame, alone on a "
    "completely flat bright magenta screen of one solid colour that fills the whole picture behind "
    "him with nothing else drawn on it. Keep his identity exactly as image 1 draws it, and keep "
    "his colours exactly as deep as image 1 draws them — the dark near-black of his shirt, the dark "
    "brown of his hair, the mid tone of his trousers — with strong black ink lines and real "
    "contrast, never pale or washed out. He is "
)
POSE_LEAD = "The pose in this frame is:"

# ---------------------------------------------------------------- the pose beats
#
# One entry per drawing. These are the keys an animator would draw: extremes first, the useful
# in-betweens second. Order is playback order.

POSE_BEATS = {
    "idle": [
        "standing square and relaxed, weight even on both feet, arms hanging loose at his sides",
        "the same relaxed stand, but settled a little lower as he breathes out, shoulders dropped",
        "standing square again, chest lifted as he breathes in, shoulders up a touch",
        "the same relaxed stand, weight shifted a little onto his back foot, one shoulder easing back",
    ],
    "walk": [
        "mid-stride contact: his front leg stretched forward with the heel just touching down, his "
        "back leg stretched behind with the toe just leaving the ground, opposite arms swung forward "
        "and back",
        "the down position: his front knee bent taking the weight, his body at its lowest, the back "
        "foot lifting clear",
        "the passing position: his weight leg straight underneath him, the other knee lifted and "
        "passing close beside it, arms hanging near his sides",
        "the up position: pushed up onto the ball of the weight foot, body at its highest, the free "
        "leg swinging forward",
        "the opposite contact: the other leg stretched forward with the heel touching down, the "
        "first leg stretched behind, arms swapped",
        "the opposite down position: the forward knee bent taking the weight, body low",
        "the opposite passing position: weight on the other straight leg, the first knee lifted "
        "beside it",
        "the opposite up position: pushed up on the ball of the other foot, body high, free leg "
        "swinging through",
    ],
    "jump": [
        "crouched to spring: knees deeply bent, body compressed low, both arms swept back behind him",
        "the launch: legs snapping straight, body stretched tall, both arms thrown up overhead",
        "at the top of the jump: body stretched out in the air, legs tucked slightly, arms high",
        "coming down: legs reaching for the ground, arms dropping, body starting to compress",
        "the landing: knees bent deep absorbing the impact, arms out for balance, body low again",
    ],
    "punch": [
        "on guard: both fists up near his chin, elbows in, knees softly bent, body turned a little "
        "away",
        "the wind-up: his punching shoulder pulled back, that fist drawn back past his ribs, weight "
        "shifted onto the back foot",
        "the punch half out: the arm driving forward, fist at chest height, shoulders beginning to "
        "square",
        "full extension: the punching arm stretched straight out in front at shoulder height, fist "
        "clenched, body leaning into it over the front foot, the other fist tucked at his chin",
        "the recovery: the arm folding back in, fist returning toward his chin, weight settling back",
    ],
    "kick": [
        "on guard: fists up, knees softly bent, weight even",
        "the plant: one foot planted flat and firm, the other knee lifting, arms swinging out for "
        "balance",
        "the kick half out: the kicking leg unfolding forward, knee still bent, arms wide",
        "full extension: the kicking leg stretched straight out in front at waist height, toe "
        "pointed, body leaning back over the planted foot, arms out",
        "the recovery: the leg folding back down, foot reaching for the ground, arms coming back in",
    ],
    "block": [
        "on guard: fists up near his chin, elbows in, knees softly bent",
        "raising the guard: both forearms lifting up in front of his face, elbows coming together",
        "the full block: both forearms crossed up in front of his face and chest, shoulders hunched, "
        "head tucked down behind them, knees bent, braced",
    ],
    "crouch": [
        "beginning to duck: knees bending, body dropping, head starting to lower",
        "half crouched: knees well bent, body low, one hand dropping toward the ground",
        "the full crouch: knees folded right up, body low and compact over his feet, head tucked "
        "down, arms drawn in close",
    ],
}

# playback hints for whatever consumes the sheets
MOVE_META = {
    "idle":   {"fps": 6,  "loop": True},
    "walk":   {"fps": 12, "loop": True},
    "jump":   {"fps": 10, "loop": False},
    "punch":  {"fps": 14, "loop": False},
    "kick":   {"fps": 12, "loop": False},
    "block":  {"fps": 14, "loop": False, "hold": True},
    "crouch": {"fps": 14, "loop": False, "hold": True},
}


def graph(plate, prompt, seed):
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": LORA, "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "img": {"class_type": "LoadImage", "inputs": {"image": plate}},
        "pos": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["img", 0],
                           "prompt": prompt}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "prompt": ""}},
        "lat": {"class_type": "EmptySD3LatentImage",
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "k": {"class_type": "KSampler",
              "inputs": {"model": ["lora", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                         "latent_image": ["lat", 0], "seed": seed, "steps": STEPS, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["vae", 0]}},
        "RESULT": {"class_type": "SaveImage",
                   "inputs": {"images": ["dec", 0], "filename_prefix": "cast/pose"},
                   "_meta": {"title": "RESULT"}},
    }


def render_move(cid, move, seed=101, force=False):
    """One still per pose beat. Same plate, same seed, same staging — so it is repeatable."""
    beats = POSE_BEATS[move]
    outdir = os.path.join(OUT_ROOT, f"{cid}-{move}")
    os.makedirs(outdir, exist_ok=True)
    for n, beat in enumerate(beats, start=1):
        dest = os.path.join(outdir, f"{n:02d}.png")
        if os.path.exists(dest) and not force:
            print(f"  {cid}-{move} {n:02d}: have it")
            continue
        prompt = f"{STAGE_LOCK}{TRAITS[cid]}. {POSE_LEAD} {beat}. {STYLE_LOCK}"
        g = graph(f"cast-cutout-{cid}.png", prompt, seed)
        for node in g.values():
            node.setdefault("_meta", {"title": "node"})
        t0 = time.time()
        pid = S.api("/prompt", {"prompt": g, "client_id": "poses"})["prompt_id"]
        while True:
            hist = S.api(f"/history/{pid}")
            if pid in hist:
                break
            time.sleep(3)
        st = hist[pid]["status"]
        if st.get("status_str") != "success":
            print(f"  {cid}-{move} {n:02d}: FAILED {json.dumps(st)[:200]}")
            continue
        out = hist[pid]["outputs"]["RESULT"]["images"][0]
        subprocess.run(["curl", "-s",
                        f"{S.HOST}/view?filename={out['filename']}&subfolder={out.get('subfolder','')}&type=output",
                        "-o", dest], check=True)
        print(f"  {cid}-{move} {n:02d}: {time.time()-t0:.0f}s")


def pack(cid, moves=None, cell=256):
    """Pack every rendered move for one character at a SHARED scale."""
    from sprite_sheet import _prepare_stills, _emit
    moves = moves or [m for m in POSE_BEATS if os.path.isdir(os.path.join(OUT_ROOT, f"{cid}-{m}"))]
    os.makedirs(SHEET_DIR, exist_ok=True)
    preps = []
    for move in moves:
        # prefer the RIFE-smoothed sequence when interpolate_poses.py has produced one
        d = os.path.join(OUT_ROOT, f"{cid}-{move}-smooth")
        if not os.path.isdir(d):
            d = os.path.join(OUT_ROOT, f"{cid}-{move}")
        paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))
        if not paths:
            continue
        preps.append((f"{cid}-{move}", _prepare_stills(paths, smooth=True)))
    tallest = max(p["natural"] for _, p in preps)
    scale = (cell * 0.92) / tallest
    index = {}
    for name, prep in preps:
        move = name.split("-", 1)[1]
        # looping moves get per-frame height normalisation; crouch and jump must keep their real
        # height changes or the move stops reading as a crouch or a jump
        _emit(name, prep, cell, SHEET_DIR, "feet", scale,
              unify_height=MOVE_META.get(move, {}).get("loop", False))
        beats = len(POSE_BEATS[move])
        n = len(prep["picks"])
        meta = dict(MOVE_META.get(move, {"fps": 12, "loop": False}))
        # in-betweens only help if they are played at the matching rate
        # keep the move's duration, but cap the rate: a 24-frame walk at 36fps is a sprint, and
        # nothing on the web benefits from stepping a sprite faster than the display refresh.
        meta["fps"] = min(round(meta["fps"] * n / beats), 24)
        index[move] = dict(meta, file=name, frames=n, beats=beats)
    json.dump(index, open(os.path.join(SHEET_DIR, f"{cid}-moves.json"), "w"), indent=1)
    print(f"packed {len(preps)} moves at shared scale {scale:.3f} -> {SHEET_DIR}")


if __name__ == "__main__":
    cid = sys.argv[1]
    args = sys.argv[2:]
    if "--pack" in args:
        pack(cid)
    elif "--all" in args:
        for move in POSE_BEATS:
            render_move(cid, move)
    else:
        for move in args:
            render_move(cid, move)
