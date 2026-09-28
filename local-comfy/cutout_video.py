#!/usr/bin/env python3
"""Cut the background out of every frame of a clip, in ComfyUI, and save RGBA PNG frames.

    python3 cutout_video.py <clip-in-comfy-input> <out-dir>

BiRefNet per frame, the same matting node the cast apps use for transparent PNGs.
JoinImageWithAlpha inverts its mask, hence the InvertMask -- see build_extras.cutout_chain.
The frames come back as a batch and are fetched one by one; packing them into a GIF or an
emoji is container work, done afterwards.
"""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S  # noqa: E402
from build_workflows import BGREMOVAL  # noqa: E402

clip, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

g = {
 "1": {"class_type": "LoadVideo", "inputs": {"file": clip}},
 "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
 "3": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": BGREMOVAL}},
 "4": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["3", 0], "image": ["2", 0]}},
 "5": {"class_type": "InvertMask", "inputs": {"mask": ["4", 0]}},
 "6": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["2", 0], "alpha": ["5", 0]}},
 "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "cutout/frame", "images": ["6", 0]}},
}
for n, d in g.items():
    d.setdefault("_meta", {"title": n})

t0 = time.time()
r = S.api("/prompt", {"prompt": g, "client_id": "cutout"})
if "prompt_id" not in r:
    raise SystemExit("REJECTED " + json.dumps(r)[:600])
pid = r["prompt_id"]
while True:
    h = S.api(f"/history/{pid}")
    if pid in h:
        break
    time.sleep(3)
st = h[pid]["status"]
if st.get("status_str") != "success":
    raise SystemExit("FAILED " + json.dumps(st)[:600])
imgs = h[pid]["outputs"]["9"]["images"]
for i, o in enumerate(imgs):
    q = f"/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output"
    with urllib.request.urlopen(S.HOST + q, timeout=120) as rr:
        open(f"{out}/f{i:03d}.png", "wb").write(rr.read())
print(f"{len(imgs)} frames in {time.time()-t0:.0f}s -> {out}")
