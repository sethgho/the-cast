#!/usr/bin/env python3
"""Splice a two-shot dialogue meme from halves rendered with their own refs, then finish it.

    python3 stepbros.py <out-name> <half-a> <half-b>

Wan Animate takes one reference per render, and a shot/reverse-shot puts a different face to
camera after the cut, so each shot is its own render. This joins seth.mp4 and drive.mp4 of the
halves into <out-name> and runs the finishing stages (cutout, post, emoji, compare, full, deliver).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sethmoji as S  # noqa: E402

out, a, b = sys.argv[1:4]
dd = S.d(out)
os.makedirs(dd, exist_ok=True)
for f in ("seth.mp4", "drive.mp4"):
    S.sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{S.d(a)}/{f}", "-i", f"{S.d(b)}/{f}",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "12", f"{dd}/{f}")
spec = json.load(open(f"{S.d(a)}/spec.json"))
spec["frames"] += json.load(open(f"{S.d(b)}/spec.json"))["frames"]
json.dump(spec, open(f"{dd}/spec.json", "w"))
S.sh("scp", "-q", f"{dd}/drive.mp4", f"{S.GPU}:comfyui/input/{S.PREFIX}-{out}-drive.mp4")
for st in ("cutout", "post", "emoji", "compare", "full", "deliver"):
    getattr(S, st)(out)
