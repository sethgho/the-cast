#!/usr/bin/env python3
"""Storyboard previz: one still per shot, before spending twenty minutes on video.

    python3 build_previz.py            # writes workflows/previz.json + api/previz.api.json

An H3 storyboard costs 17-21 minutes to see. A Qwen still of the same shot costs ~45 seconds, so a
six-shot board previews in four minutes -- which is what a storyboard is FOR. It will not tell you
whether the cuts land or whether a beat reads in motion; it tells you framing, staging, who is in
frame and whether the joke is legible. That is most of what goes wrong.

Same wiring as the action-selfie app, for the same measured reason: the character's canonical
plate is the EDIT TARGET and the shot travels as WORDS. Editing the plate keeps the character;
describing the moment gets the moment. Feeding a shot's prose against a photograph or against
another character merges them.

Two fields: PLATE (which character) and SHOT (the prose from the storyboard, camera included).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import Graph, STYLE_LOCK, NEGATIVE, STEPS, BLUE, GREEN, GREY  # noqa: E402
from build_extras import engine, note  # noqa: E402
from build_transition import TRAITS  # noqa: E402

W, H, SEED = 1152, 648, 7
# Copies of cast/<id>/assets/poses/standing-neutral.png. See build_supper.py.
PLATES = {"seth": "canon-seth.png", "wilson": "canon-wilson.png"}

HOW_TO = """# Storyboard previz

**One still per shot, ~45s each, before committing to a 20-minute render.**

| Control | What it does |
|---|---|
| **PLATE** | The character whose canonical drawing is being edited. One character per still. |
| **WHO** | That character's canon description. Leave it alone unless the cast changes. |
| **SHOT** | The shot's prose straight out of the storyboard — camera position included. |

It shows framing, staging, who is in frame and whether the joke reads. It does **not** show
whether H3 will cut, or whether a beat lands in motion. Those need the real render.

If two characters must share a frame, previz them separately first: one reference is one subject,
and a second character in the same edit merges both into one creature.
"""


def build():
    g = Graph()
    note(g, "HOW TO USE — storyboard previz", HOW_TO)
    e = engine(g, -60, -180, want_bg=False)

    plate = g.add("LoadImage", "▶ 1 · PLATE", (-60, 60), (400, 320),
                  {"image": PLATES["seth"], "upload": "image"},
                  outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], color=BLUE)
    g.app_input(plate, "image", "upload")

    who = g.add("PrimitiveStringMultiline", "▶ 2 · WHO", (420, 60), (460, 200),
                {"value": ("Image 1 is the drawing to edit, and the character in image 1 is the "
                           f"only character in the finished picture. Redraw him exactly as he is: "
                           f"{TRAITS['seth']}.")},
                outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(who, "value")
    shot = g.add("PrimitiveStringMultiline", "▶ 3 · SHOT", (420, 290), (460, 300),
                 {"value": "He sits on a sofa at night, mid-bite of a crisp, lit by a television "
                           "just out of frame to the left."},
                 outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(shot, "value")
    style = g.add("PrimitiveStringMultiline", "STYLE", (420, 620), (460, 200),
                  {"value": STYLE_LOCK}, outputs=[("STRING", "STRING")], color=GREEN)
    tail = g.add("PrimitiveStringMultiline", "RESTATE", (420, 850), (460, 160),
                 {"value": ("Above all, the character in the finished drawing is the character "
                            "from image 1, unchanged, and he is doing exactly what the shot "
                            "description says.")},
                 outputs=[("STRING", "STRING")], color=GREEN)

    j = who
    for i, nxt in enumerate((shot, style, tail)):
        j = g.add("StringConcatenate", f"join-{i+1}", (920, 60 + i * 60), (380, 110),
                  {"delimiter": " "},
                  links={"string_a": (j, 0, "STRING", True), "string_b": (nxt, 0, "STRING", True)},
                  outputs=[("STRING", "STRING")], collapsed=True)

    latent = g.add("EmptySD3LatentImage", "canvas", (920, 260), (380, 130),
                   {"width": W, "height": H, "batch_size": 1},
                   outputs=[("LATENT", "LATENT")], collapsed=True)
    pos = g.add("TextEncodeQwenImageEditPlus", "positive", (920, 320), (420, 120),
                links={"prompt": (j, 0, "STRING", True), "clip": (e["clip"], 0, "CLIP", False),
                       "vae": (e["vae"], 0, "VAE", False), "image1": (plate, 0, "IMAGE", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    neg = g.add("TextEncodeQwenImageEditPlus", "negative", (920, 380), (420, 120),
                {"prompt": NEGATIVE}, links={"clip": (e["clip"], 0, "CLIP", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    ks = g.add("KSampler", "▶ 4 · SEED", (920, 440), (400, 280),
               {"seed": SEED, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
               links={"model": (e["lora"], 0, "MODEL", False),
                      "positive": (pos, 0, "CONDITIONING", False),
                      "negative": (neg, 0, "CONDITIONING", False),
                      "latent_image": (latent, 0, "LATENT", False)},
               outputs=[("LATENT", "LATENT")], color=GREEN)
    g.app_input(ks, "seed", "steps")
    dec = g.add("VAEDecode", "decode", (920, 740), (300, 60),
                links={"samples": (ks, 0, "LATENT", False), "vae": (e["vae"], 0, "VAE", False)},
                outputs=[("IMAGE", "IMAGE")], collapsed=True)
    out = g.add("SaveImage", "RESULT", (1380, 60), (620, 700),
                {"filename_prefix": "previz/shot"},
                links={"images": (dec, 0, "IMAGE", False)}, color=GREY)
    g.app_output(out)
    return g


if __name__ == "__main__":
    g = build()
    for path, blob in ((os.path.join(HERE, "workflows", "previz.json"), g.to_ui()),
                       (os.path.join(HERE, "workflows", "previz.app.json"), g.to_ui()),
                       (os.path.join(HERE, "api", "previz.api.json"), g.to_api())):
        json.dump(blob, open(path, "w"), indent=1)
    print("wrote previz")
