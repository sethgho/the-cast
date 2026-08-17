#!/usr/bin/env python3
"""Run the generated API graph through every switch combination.

    python3 smoke_test.py seth [outdir]

Proves the graph executes on gpu-worker and that TRANSPARENT really yields RGBA.
"""
import json
import os
import sys
import time
import urllib.request

HOST = os.environ.get("COMFY_HOST", "http://192.168.0.210:8188")
HERE = os.path.dirname(os.path.abspath(__file__))


def api(path, payload=None):
    req = urllib.request.Request(HOST + path)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def find(graph, title):
    for nid, node in graph.items():
        if node["_meta"]["title"].startswith(title):
            return nid
    raise KeyError(title)


def run(graph, label, outdir):
    t0 = time.time()
    pid = api("/prompt", {"prompt": graph, "client_id": "cast-smoke"})["prompt_id"]
    while True:
        hist = api(f"/history/{pid}")
        if pid in hist:
            break
        time.sleep(3)
    status = hist[pid]["status"]
    if status.get("status_str") != "success":
        print(f"{label}: FAILED — {json.dumps(status)[:400]}")
        return
    out = hist[pid]["outputs"][find(graph, "SAVE")]["images"][0]
    url = f"/view?filename={out['filename']}&subfolder={out.get('subfolder','')}&type=output"
    with urllib.request.urlopen(HOST + url, timeout=300) as r:
        data = r.read()
    dest = os.path.join(outdir, f"{label}.png")
    open(dest, "wb").write(data)
    mode = "RGBA" if data[25] == 6 else "RGB"  # PNG colour-type byte
    print(f"{label}: {time.time()-t0:.0f}s  {mode}  -> {dest}")


def main(cid, outdir):
    os.makedirs(outdir, exist_ok=True)
    base = json.load(open(os.path.join(HERE, "api", f"{cid}-pose-and-scene.api.json")))
    scene_id = find(base, "SCENE  ·")
    alpha_id = find(base, "TRANSPARENT PNG  ·")
    seed_id = find(base, "sampler")
    for scene in (False, True):
        for alpha in (False, True):
            g = json.loads(json.dumps(base))
            g[scene_id]["inputs"]["value"] = scene
            g[alpha_id]["inputs"]["value"] = alpha
            g[seed_id]["inputs"]["seed"] = 7
            run(g, f"{cid}-scene{int(scene)}-alpha{int(alpha)}", outdir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "seth",
         sys.argv[2] if len(sys.argv) > 2 else "/tmp/cast-smoke")
