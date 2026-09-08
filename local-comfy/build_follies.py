#!/usr/bin/env python3
"""The Follies style LoRA, as three ComfyUI apps.

    python3 build_follies.py        # workflows/follies-{portrait,invent,repose}.json + api/

The style is a WEIGHT now, not a paragraph. `f0llie5` at strength 1.2 reproduces what the written
STYLE_LOCK block used to do, measured side by side on Seth's canon plate and near
indistinguishable -- so every prompt here spends its budget on pose, action and scene instead.

  portrait  A photograph in, that person out as a cast member. The instruction is FIXED and
            subject-blind: it never names a feature, garment, age or mood, so no agent has to
            look at the photo and decide what it sees.
  invent    A new character from words alone, no reference.
  repose    A canon plate re-posed. Identity from the plate, action from a phrase.

Trained 2026-09-07 on the repo's own 64 training images across four body plans, captioned for
pose and framing only. Bake-off: http://wilson/follies-lora/ and http://wilson/follies-scenarios/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import (  # noqa: E402
    Graph, WIDGET_TYPES, NEGATIVE, STEPS, BLUE, GREEN, GREY,
)

# app_input() writes a slot for EVERY widget on a node, so both of these need a type registered
# even though only strength_model is promoted into the form.
WIDGET_TYPES.setdefault(("LoraLoaderModelOnly", "lora_name"), "COMBO")
WIDGET_TYPES.setdefault(("LoraLoaderModelOnly", "strength_model"), "FLOAT")
from build_extras import note  # noqa: E402
from build_transition import TRAITS  # noqa: E402

LORA, STRENGTH, SIZE, SEED = "follies-final.safetensors", 1.2, 1024, 7
UNET = "qwen-image-edit-2511-Q4_K_S.gguf"
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"

# The two clauses that turn a restyle into a reimagining. Both describe the DRAWING, never the
# person, so the portrait app stays subject-blind.
#
# PROPORTION is lifted from the cast repo's own assets.yaml, which records a whole pose library
# inheriting realistic proportions from its reference until this was stated outright.
PROPORTION = ("Draw him as a cartoon character in this style's own proportions, not the "
              "photograph's: a large rounded head about one fifth of his standing height, small "
              "simple facial features, chunky simplified limbs and hands. He is not a "
              "realistically proportioned man. This is a new drawing of him, not a traced or "
              "filtered photograph.")
# FULL FIGURE is what actually does the work. A head-and-shoulders ask leaves the model the
# photo's own framing to copy and tips into true caricature with an oversized head -- observed
# twice, on two different subjects. Head-to-feet forces it to draw a body that was never in the
# picture, so it has to invent rather than trace.
PORTRAIT = ("f0llie5, image 1 is a photograph of a person. Draw that person as a full-body "
            "cartoon character standing at rest, arms at his sides, seen from the front, head to "
            f"feet in frame. {PROPORTION} Keep who he is recognisable: the same hair, the same "
            "beard or lack of one, the same colouring and the same clothing. Drawn on aged paper.")

INVENT = ("f0llie5, an anthropomorphic giraffe in a well-cut suit made of deep purple velvet, "
          "standing at rest facing the viewer, on a plain flat background.")
REPOSE = (f"f0llie5, image 1 is the drawing to edit and the man in image 1 is the only character "
          f"in the finished picture. Redraw him exactly as he is: {TRAITS['seth']}. He is "
          f"throwing a straight right hook, gloves up, weight on his front foot, on a plain flat "
          f"background.")

HOW_TO = """# Follies style

**The house style as a LoRA.** `f0llie5` at strength **1.2**. Do not write style words as well -
the weight already does it, and the prompt budget is better spent on pose and scene.

| Control | What it does |
|---|---|
| **THE PICTURE** | Pose, action, scene. No style words. |
| **STRENGTH** | 1.2 ships. 0.8 lets generic colour back in; 1.5 goes grey and muddy and stretches the proportions. |
| **SEED / STEPS** | 8 steps, cfg 1.0 - the Lightning LoRA underneath requires it. |

**Photograph in?** Two clauses are load-bearing and both are already in the prompt box: ask for
the **full figure head to feet**, and state the **proportions** explicitly. Without them you get a
restyled photo rather than a character - and a head-and-shoulders crop tips into true caricature
with an oversized head, measured on two different subjects.

**What it learned:** palette, aged paper, ground, ink treatment. **What it did not:** the canon's
varied-weight line and caricature build, which are still Qwen's own. That is a data limit - 64
images at rank 32 - not a settings one.
"""


def build(kind):
    g = Graph()
    note(g, f"HOW TO USE — Follies style ({kind})", HOW_TO)

    unet = g.add("UnetLoaderGGUF", "diffusion model", (-60, -180), (400, 90), {"unet_name": UNET},
                 outputs=[("MODEL", "MODEL")], color=GREY, collapsed=True)
    light = g.add("LoraLoaderModelOnly", "Lightning LoRA (8 steps)", (-60, -130), (400, 110),
                  {"lora_name": LIGHTNING, "strength_model": 1.0},
                  links={"model": (unet, 0, "MODEL", False)},
                  outputs=[("MODEL", "MODEL")], color=GREY, collapsed=True)
    style = g.add("LoraLoaderModelOnly", "▶ STRENGTH — the Follies style", (-60, -80), (420, 130),
                  {"lora_name": LORA, "strength_model": STRENGTH},
                  links={"model": (light, 0, "MODEL", False)},
                  outputs=[("MODEL", "MODEL")], color=GREEN)
    g.app_input(style, "strength_model")
    clip = g.add("CLIPLoader", "text encoder", (-60, 60), (400, 130),
                 {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image",
                  "device": "default"}, outputs=[("CLIP", "CLIP")], color=GREY, collapsed=True)
    vae = g.add("VAELoader", "VAE", (-60, 110), (400, 80), {"vae_name": "qwen_image_vae.safetensors"},
                outputs=[("VAE", "VAE")], color=GREY, collapsed=True)

    plate, default = None, {"portrait": PORTRAIT, "invent": INVENT, "repose": REPOSE}[kind]
    if kind != "invent":
        img = {"portrait": "subject-in.png", "repose": "canon-seth.png"}[kind]
        label = "▶ 1 · THE PHOTOGRAPH" if kind == "portrait" else "▶ 1 · THE PLATE"
        plate = g.add("LoadImage", label, (-60, 200), (400, 320), {"image": img, "upload": "image"},
                      outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], color=BLUE)
        g.app_input(plate, "image", "upload")

    txt = g.add("PrimitiveStringMultiline", "▶ 2 · THE PICTURE", (420, 60), (500, 420),
                {"value": default}, outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(txt, "value")

    enc = {"prompt": (txt, 0, "STRING", True), "clip": (clip, 0, "CLIP", False),
           "vae": (vae, 0, "VAE", False)}
    if plate is not None:
        enc["image1"] = (plate, 0, "IMAGE", False)
    pos = g.add("TextEncodeQwenImageEditPlus", "positive", (960, 60), (420, 120), links=enc,
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    neg = g.add("TextEncodeQwenImageEditPlus", "negative", (960, 120), (420, 120),
                {"prompt": NEGATIVE + ", photograph, photorealistic skin, traced photo"},
                links={"clip": (clip, 0, "CLIP", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    lat = g.add("EmptySD3LatentImage", "canvas", (960, 180), (380, 130),
                {"width": SIZE, "height": SIZE, "batch_size": 1},
                outputs=[("LATENT", "LATENT")], collapsed=True)
    ks = g.add("KSampler", "▶ 3 · SEED / STEPS", (960, 240), (400, 280),
               {"seed": SEED, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
               links={"model": (style, 0, "MODEL", False), "positive": (pos, 0, "CONDITIONING", False),
                      "negative": (neg, 0, "CONDITIONING", False),
                      "latent_image": (lat, 0, "LATENT", False)},
               outputs=[("LATENT", "LATENT")], color=GREEN)
    g.app_input(ks, "seed", "steps")
    dec = g.add("VAEDecode", "decode", (960, 540), (300, 60),
                links={"samples": (ks, 0, "LATENT", False), "vae": (vae, 0, "VAE", False)},
                outputs=[("IMAGE", "IMAGE")], collapsed=True)
    out = g.add("SaveImage", "RESULT", (1420, 60), (620, 620),
                {"filename_prefix": f"follies/{kind}"},
                links={"images": (dec, 0, "IMAGE", False)}, color=GREY)
    g.app_output(out)
    return g


if __name__ == "__main__":
    for kind in ("portrait", "invent", "repose"):
        g = build(kind)
        stem = f"follies-{kind}"
        for path, blob in ((os.path.join(HERE, "workflows", f"{stem}.json"), g.to_ui()),
                           (os.path.join(HERE, "workflows", f"{stem}.app.json"), g.to_ui()),
                           (os.path.join(HERE, "api", f"{stem}.api.json"), g.to_api())):
            json.dump(blob, open(path, "w"), indent=1)
        print("wrote", stem)
