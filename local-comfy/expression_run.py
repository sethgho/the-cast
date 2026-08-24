#!/usr/bin/env python3
"""Drive the expression app headlessly: a face in, Seth wearing its expression out.

    python3 expression_run.py <picture> [--seed N] [--padding N] [--out DIR]

<picture> is a path on this machine; it is uploaded to ComfyUI's input directory first, so the
app can be run against a file gpu-worker has never seen. Two files land in --out: the Seth render
and the face crop the model was shown, which is the first thing to look at when a run reads the
wrong expression.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import smoke_test as S  # noqa: E402


def upload(path):
    """POST the picture to /upload/image and return the name ComfyUI stored it under.

    subfolder is left empty and type defaults to "input", which is where LoadImage looks; a
    name collision is resolved by ComfyUI itself, hence reading the name back out of the reply
    instead of assuming basename(path).
    """
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode(),
        open(path, "rb").read(),
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue"
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(S.HOST + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["name"]


def fetch(image, dest):
    url = (f"/view?filename={image['filename']}&subfolder={image.get('subfolder','')}"
           f"&type={image.get('type','output')}")
    with urllib.request.urlopen(S.HOST + url, timeout=300) as r:
        open(dest, "wb").write(r.read())


def run(picture, seed, padding, outdir):
    graph = json.load(open(os.path.join(HERE, "api", "seth-expression.api.json")))
    graph[S.find(graph, "▶ 1 · SOURCE FACE")]["inputs"]["image"] = upload(picture)
    graph[S.find(graph, "▶ 2 · FACE CROP PADDING")]["inputs"]["padding"] = padding
    graph[S.find(graph, "▶ 3 · SEED")]["inputs"]["seed"] = seed

    t0 = time.time()
    pid = S.api("/prompt", {"prompt": graph, "client_id": "cast-expression"})["prompt_id"]
    while True:
        hist = S.api(f"/history/{pid}")
        if pid in hist:
            break
        time.sleep(3)
    status = hist[pid]["status"]
    if status.get("status_str") != "success":
        print(f"FAILED — {json.dumps(status)[:600]}")
        return 1

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(picture))[0]
    outputs = hist[pid]["outputs"]
    for title, suffix in (("RESULT", "seth"), ("FACE READ", "face")):
        dest = os.path.join(outdir, f"{stem}-{suffix}.png")
        fetch(outputs[S.find(graph, title)]["images"][0], dest)
        print(f"{time.time()-t0:.0f}s  -> {dest}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("picture")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--padding", type=int, default=48)
    p.add_argument("--out", default="/tmp/cast-expression")
    a = p.parse_args()
    return run(a.picture, a.seed, a.padding, a.out)


if __name__ == "__main__":
    sys.exit(main())
