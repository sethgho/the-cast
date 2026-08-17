#!/usr/bin/env python3
"""Fill in between the pose beats with RIFE, so a 4-cell move plays like a 16-cell one.

    python3 interpolate_poses.py seth walk --multiplier 3

Eight drawings at 12fps is two thirds of a second of walk, and the eye reads every cut. Rather
than pay 48s a drawing for twice as many drawings, the beats go through ComfyUI's
`FrameInterpolate` (RIFE v4.26 heavy) — 8 keys at ×3 gives 22 frames for about fifteen seconds of
GPU, and the in-betweens are generated from the keys either side so they cannot drift off-model.

Interpolation happens on the STILLS, while they are still on their magenta screen, for two
reasons: RIFE has no alpha channel to preserve, and the packer's keyer runs afterwards on every
frame identically — so an interpolated frame is matted exactly like a rendered one.

Cyclic moves (walk, idle) wrap: the last beat is interpolated back into the first, so the loop
point gets in-betweens too.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import smoke_test as S  # noqa: E402
from sprite_poses import OUT_ROOT, MOVE_META  # noqa: E402

RIFE = "rife_v4.26_heavy.safetensors"


def upload(path):
    sys.path.insert(0, "/home/wilson/scratch/local-cast")
    import run as R
    return R.upload(path)


def graph(names, multiplier):
    """LoadImage per still, chained through ImageBatch, then one FrameInterpolate."""
    g = {"loader": {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": RIFE}}}
    prev = None
    for i, n in enumerate(names):
        nid = f"img{i}"
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": n}}
        if prev is None:
            prev = (nid, 0)
        else:
            bid = f"batch{i}"
            g[bid] = {"class_type": "ImageBatch", "inputs": {"image1": [prev[0], prev[1]],
                                                            "image2": [nid, 0]}}
            prev = (bid, 0)
    g["interp"] = {"class_type": "FrameInterpolate",
                   "inputs": {"interp_model": ["loader", 0], "images": [prev[0], prev[1]],
                              "multiplier": multiplier}}
    g["RESULT"] = {"class_type": "SaveImage",
                   "inputs": {"images": ["interp", 0], "filename_prefix": "cast/interp"},
                   "_meta": {"title": "RESULT"}}
    for node in g.values():
        node.setdefault("_meta", {"title": "node"})
    return g


def run(cid, move, multiplier=3, wrap=None):
    src = os.path.join(OUT_ROOT, f"{cid}-{move}")
    stills = sorted(os.path.join(src, f) for f in os.listdir(src)
                    if f.endswith(".png") and not f.startswith("i_"))
    if wrap is None:
        wrap = MOVE_META.get(move, {}).get("loop", False)
    seq = stills + ([stills[0]] if wrap else [])
    names = [upload(p) for p in seq]

    t0 = time.time()
    pid = S.api("/prompt", {"prompt": graph(names, multiplier), "client_id": "interp"})["prompt_id"]
    while True:
        hist = S.api(f"/history/{pid}")
        if pid in hist:
            break
        time.sleep(3)
    st = hist[pid]["status"]
    if st.get("status_str") != "success":
        raise SystemExit(f"{cid}-{move}: FAILED {json.dumps(st)[:400]}")

    out_dir = os.path.join(OUT_ROOT, f"{cid}-{move}-smooth")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    images = hist[pid]["outputs"]["RESULT"]["images"]
    # A wrapped sequence ends on a copy of frame 1; drop it so the loop does not stutter.
    if wrap:
        images = images[:-1]
    for n, im in enumerate(images, start=1):
        subprocess.run(["curl", "-s",
                        f"{S.HOST}/view?filename={im['filename']}&subfolder={im.get('subfolder','')}&type=output",
                        "-o", os.path.join(out_dir, f"{n:03d}.png")], check=True)
    print(f"{cid}-{move}: {len(stills)} beats -> {len(images)} frames in {time.time()-t0:.0f}s")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cid")
    ap.add_argument("moves", nargs="+")
    ap.add_argument("--multiplier", type=int, default=3)
    a = ap.parse_args()
    for m in a.moves:
        run(a.cid, m, a.multiplier)
