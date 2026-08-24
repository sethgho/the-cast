#!/usr/bin/env python3
"""Drive the scene app headlessly: a face and a place in, a wide cartoon of Seth out.

    python3 scene_run.py <picture> "<scene description>" [--seed N] [--out DIR]

<picture> is a path on this machine and is uploaded to ComfyUI first, so a file gpu-worker has
never seen still works. Both stages run in the one graph. Three files land in --out:

  <name>-scene.png   1920x832, the deliverable
  <name>-head.png    the stage-one headshot, for when the face is wrong
  <name>-scene.json  where the head is in the wide image, so a caller can crop the square
                     avatar out of it without generating a second time

The upload and fetch helpers come from expression_run so there is one implementation of the
multipart quirk (ComfyUI renames a colliding upload, so the stored name is read back).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import smoke_test as S  # noqa: E402
from expression_run import fetch, upload  # noqa: E402
from build_scene import HEAD_BOX, OUT_H, OUT_W  # noqa: E402


def run(picture, scene, seed, outdir):
    graph = json.load(open(os.path.join(HERE, "api", "seth-scene.api.json")))
    graph[S.find(graph, "▶ 1 · SOURCE FACE")]["inputs"]["image"] = upload(picture)
    graph[S.find(graph, "▶ 2 · THE SCENE")]["inputs"]["value"] = scene
    graph[S.find(graph, "▶ 3 · SEED")]["inputs"]["value"] = seed

    t0 = time.time()
    pid = S.api("/prompt", {"prompt": graph, "client_id": "cast-scene"})["prompt_id"]
    while True:
        hist = S.api(f"/history/{pid}")
        if pid in hist:
            break
        time.sleep(3)
    status = hist[pid]["status"]
    if status.get("status_str") != "success":
        print(f"FAILED — {json.dumps(status)[:600]}")
        return 1
    elapsed = time.time() - t0

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(picture))[0]
    outputs = hist[pid]["outputs"]
    for title, suffix in (("RESULT", "scene"), ("HEADSHOT", "head")):
        dest = os.path.join(outdir, f"{stem}-{suffix}.png")
        fetch(outputs[S.find(graph, title)]["images"][0], dest)
        print(f"{elapsed:.0f}s  -> {dest}")

    meta = os.path.join(outdir, f"{stem}-scene.json")
    json.dump({"image": f"{stem}-scene.png", "width": OUT_W, "height": OUT_H,
               "head_box": HEAD_BOX, "scene": scene, "seed": seed,
               "seconds": round(elapsed, 1)}, open(meta, "w"), indent=1)
    print(f"       -> {meta}  head_box={HEAD_BOX}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("picture")
    p.add_argument("scene", help="where he is — a place, not an action")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="/tmp/cast-scene")
    a = p.parse_args()
    return run(a.picture, a.scene, a.seed, a.out)


if __name__ == "__main__":
    sys.exit(main())
