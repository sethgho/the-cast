#!/usr/bin/env python3
"""Repaint the chosen sprite cells as drawings, instead of shipping video frames.

    python3 repaint_cells.py seth                 # every move
    python3 repaint_cells.py seth punch walk      # just these

## Why

H3 gives motion; it does not give a clean drawing. Its frames carry codec speckle, soft coloured
edges and a background it repaints slightly differently every frame, and no amount of post
recovers ink that was never crisp. So the clip is demoted to a POSE SOURCE: the packer still picks
the cells off it, but each picked frame then goes back through Qwen-Image-Edit as an EDIT of
itself — same pose, same framing, redrawn.

The distinction that makes it work: the H3 frame is the *edit target*, not a reference image. A
reference image is an identity reference and Qwen will happily re-pose the character to suit the
prompt. Editing the frame in place keeps the pose and changes only the drawing. An earlier attempt
did it the other way round and came back crisp but re-posed, which is why this looked impossible.

Measured on the punch: edge sharpness 8.8 -> 14.9, and the magenta comes back flat, so the key
gets a hard boundary and cel_clean is not needed at all.

## The one thing to be careful about

Every cell is an independent generation, so the character comes back a few percent bigger or
smaller each time. One locked seed keeps the drawing consistent; `unify_height` in the packer
rescales each cell to the set median for moves whose height should not change (walk, idle). Moves
where the height IS the animation — jump, crouch — must not use it.

Cost: ~48s a cell. The seven-move set is 54 cells, about 45 minutes.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/wilson/scratch/local-cast")

import smoke_test as S  # noqa: E402
import sprite_sheet as SS  # noqa: E402
from build_workflows import UNET, LORA, CLIP, VAE  # noqa: E402

OUT = "/home/wilson/artifacts/cast-fighter/sprites"
REPAINT_DIR = "/tmp/repaint"
SEED = 77          # one seed for every cell, so the drawing itself does not wander
STEPS = 8
CELL = 512

# move -> (clip, cells, fps, loop, hold, unify_height)
MOVES = {
    "walk":   ("/tmp/seth-walk-cycle-832-20.mp4",   10, 14, True,  False, True),
    "idle":   ("/tmp/seth-idle-breathe-832-20.mp4",  8, 14, True,  False, True),
    "punch":  ("/tmp/seth-punch-832-20.mp4",         8, 16, False, False, False),
    "kick":   ("/tmp/seth-kick-832-20.mp4",          8, 16, False, False, False),
    "jump":   ("/tmp/seth-jump-832-20.mp4",          8, 16, False, False, False),
    "block":  ("/tmp/seth-block-832-20.mp4",         6, 16, False, True,  False),
    "crouch": ("/tmp/seth-crouch-832-20.mp4",        6, 16, False, True,  False),
}

# The repaint instruction. It says "repaint" three ways on purpose: the failure mode is Qwen
# treating the frame as inspiration and drawing a fresh pose, and every clause here exists to
# close that door.
REPAINT = (
    "Redraw the picture as clean crisp 1933 rubber-hose cartoon line art: confident smooth black "
    "ink outlines of even weight, sharp clear edges, soft halftone dot shading, muted sepia inks of "
    "aged newsprint, strong contrast, never pale or washed out. This is a repaint, not a new "
    "drawing: the man's pose, the angle of every limb, his size in the frame and his position in "
    "the frame are already exactly right and must not change at all. Keep the flat magenta "
    "background exactly as it is, one single solid magenta colour with nothing drawn on it. Only "
    "the drawing quality changes — remove all blur, smearing and video compression noise, and "
    "restate every edge as a crisp ink line."
)
NEG = ("blurry, smeared, soft focus, video artifacts, photographic, different pose, moved, "
       "resized, cropped")


def graph(image_name):
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": LORA, "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "pos": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": REPAINT,
                           "image1": ["img", 0]}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": NEG,
                           "image1": ["img", 0]}},
        "lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["img", 0], "vae": ["vae", 0]}},
        "k": {"class_type": "KSampler",
              "inputs": {"model": ["lora", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                         "latent_image": ["lat", 0], "seed": SEED, "steps": STEPS, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["vae", 0]}},
        "RESULT": {"class_type": "SaveImage",
                   "inputs": {"images": ["dec", 0], "filename_prefix": "cast/repaint"},
                   "_meta": {"title": "RESULT"}},
    }


def repaint(src, dst):
    import run as R
    g = graph(R.upload(src))
    for node in g.values():
        node.setdefault("_meta", {"title": "node"})
    pid = S.api("/prompt", {"prompt": g, "client_id": "repaint"})["prompt_id"]
    while True:
        hist = S.api(f"/history/{pid}")
        if pid in hist:
            break
        time.sleep(3)
    st = hist[pid]["status"]
    if st.get("status_str") != "success":
        raise SystemExit(f"{src}: FAILED {st}")
    im = hist[pid]["outputs"]["RESULT"]["images"][0]
    subprocess.run(["curl", "-s",
                    f"{S.HOST}/view?filename={im['filename']}&subfolder={im.get('subfolder','')}"
                    "&type=output", "-o", dst], check=True)
    return dst


def pick_frames(move, clip, n_cells):
    """Let the packer choose the cells off the CLIP, then hand back their file paths.

    Cell choice needs the motion, so it stays on the video: gait period, pose extremes and the
    sharpest-frame nudge all read the sequence. Only the chosen frames get repainted.
    """
    SS.CEL_CLEAN = True                        # picking reads video frames, so clean them
    name = f"seth-{move}-pick"
    prep = SS._prepare(clip, name, n_cells, skip=6, cycle=MOVES[move][3], anchor="feet",
                       smooth=True)
    paths = sorted(os.listdir(f"/tmp/sprite-{name}"))
    paths = [os.path.join(f"/tmp/sprite-{name}", p) for p in paths if p.endswith(".png")][6:]
    return [paths[i] for i in prep["picks"]]


def main():
    cid = sys.argv[1] if len(sys.argv) > 1 else "seth"
    wanted = sys.argv[2:] or list(MOVES)
    os.makedirs(REPAINT_DIR, exist_ok=True)

    prepared = []
    for move in wanted:
        clip, n, fps, loop, hold, unify = MOVES[move]
        t0 = time.time()
        picks = pick_frames(move, clip, n)
        out_paths = []
        for i, src in enumerate(picks, start=1):
            dst = os.path.join(REPAINT_DIR, f"{cid}-{move}-{i:02d}.png")
            if not os.path.exists(dst):
                repaint(src, dst)
            out_paths.append(dst)
            print(f"  {cid}-{move} cell {i}/{len(picks)}  {time.time()-t0:.0f}s", flush=True)
        SS.CEL_CLEAN = False                   # repaints are drawn, not decoded
        prep = SS._prepare_stills(out_paths, smooth=True)
        prepared.append((f"{cid}-{move}", prep, fps, loop, hold, unify))

    # One scale across the whole set, or he changes size when the state machine switches move.
    tallest = max(p["natural"] for _, p, *_ in prepared)
    highest = max(p["up"] for _, p, *_ in prepared)
    scale = min((CELL * 0.92) / tallest, (CELL * 0.91) / highest)

    SS.CEL_CLEAN = False
    meta = {}
    for name, prep, fps, loop, hold, unify in prepared:
        SS._emit(name, prep, CELL, OUT, "feet", scale, unify_height=unify)
        move = name.split("-", 1)[1]
        meta[move] = {"file": name, "frames": len(prep["picks"]), "fps": fps,
                      "loop": loop, "hold": hold}

    if len(wanted) == len(MOVES):
        import json
        path = os.path.join(OUT, f"{cid}-moves.json")
        json.dump(meta, open(path, "w"), indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
