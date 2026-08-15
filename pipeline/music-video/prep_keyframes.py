#!/usr/bin/env python3
"""Crop nano-banana-pro's inherited paper border, then resize for H3.

The style block's period language summons an aged-paper margin around the art no
matter how firmly the prompt forbids one — documented in
keyframes-from-character-sheets.md. It is a low-variance margin, so it is found by
row/column standard deviation rather than by a fixed inset, which would be wrong
the moment the margin width changes.
"""
import sys

import numpy as np
from PIL import Image

TARGET = (1088, 608)   # landscape 0.66MP — the both-ends conditioning ceiling


def content_box(im, thresh=14.0):
    g = np.asarray(im.convert("L"), dtype=np.float32)
    rows = g.std(axis=1)
    cols = g.std(axis=0)
    ys = np.where(rows > thresh)[0]
    xs = np.where(cols > thresh)[0]
    if len(ys) == 0 or len(xs) == 0:
        return (0, 0, im.width, im.height)
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)


def prep(src, dst):
    im = Image.open(src).convert("RGB")
    box = content_box(im)
    im = im.crop(box)

    # Cover-fit to the target aspect, then resize. Letterboxing would hand H3 bars
    # to animate; a centre crop keeps the frame full-bleed.
    tw, th = TARGET
    want = tw / th
    have = im.width / im.height
    if have > want:
        w = int(im.height * want)
        im = im.crop(((im.width - w) // 2, 0, (im.width - w) // 2 + w, im.height))
    elif have < want:
        h = int(im.width / want)
        im = im.crop((0, (im.height - h) // 2, im.width, (im.height - h) // 2 + h))

    im = im.resize(TARGET, Image.LANCZOS)
    im.save(dst)
    return box, im.size


if __name__ == "__main__":
    for p in sys.argv[1:]:
        out = p.replace(".png", "_prep.png")
        box, size = prep(p, out)
        print(f"{p}: crop {box} -> {size} -> {out}")
