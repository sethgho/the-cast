#!/usr/bin/env python3
"""Grow the scene-plate library by editing an existing plate.

    python3 make_plates.py                 # renders every plate in NEW_PLATES
    python3 make_plates.py dressing-room   # just one

A plate is an empty location in the house style, no characters in it — the
workflows load one as `image2` when SCENE is on. Editing an existing plate keeps
the ink, the paper and the palette consistent; a from-scratch text-to-image pass
does not.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S  # noqa: E402
from build_workflows import STYLE_LOCK, UNET, LORA, CLIP, VAE, STEPS  # noqa: E402

SOURCE = "cast-scene-empty-stage.png"
LEAD_IN = (
    "Redraw the empty location in image 1 as a different empty location, in exactly the same ink, "
    "halftone shading, paper tone and muted sepia palette. There are no people, no animals and no "
    "characters anywhere in it — the place is empty and waiting. The location is:"
)

NEW_PLATES = {
    "dressing-room": (
        "a cramped backstage dressing room: a mirror ringed with bare bulbs, a stool, a costume rail "
        "of hanging coats, a cluttered dressing table with pots and brushes, wallpaper peeling at one "
        "seam."
    ),
    "back-alley": (
        "a narrow brick back alley behind the theatre at night, a single lamp over the stage door, "
        "stacked crates, a fire escape zig-zagging up the wall, wet cobbles catching the light."
    ),
    "machine-room": (
        "a low machine room of humming cabinets and patch panels, cables looped along the wall, a "
        "single caged bulb overhead, a workbench with a tin of tools at one end."
    ),
    "parlour": (
        "a small front parlour: a wingback armchair, a side table with a lamp, a framed picture over "
        "a cold fireplace, a rug worn thin in the middle, lace at the window."
    ),
}


def graph(source, prompt, seed):
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": LORA, "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "ref": {"class_type": "LoadImage", "inputs": {"image": source}},
        "pos": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["ref", 0],
                           "prompt": f"{LEAD_IN} {prompt} {STYLE_LOCK}"}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "prompt": ""}},
        "latent": {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "sampler": {"class_type": "KSampler",
                    "inputs": {"model": ["lora", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                               "latent_image": ["latent", 0], "seed": seed, "steps": STEPS,
                               "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                               "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "RESULT": {"class_type": "SaveImage",
                   "inputs": {"images": ["decode", 0], "filename_prefix": "cast/plate"},
                   "_meta": {"title": "RESULT"}},
    }


def main(names):
    out = os.path.join(HERE, "plates")
    for i, name in enumerate(names):
        g = graph(SOURCE, NEW_PLATES[name], 100 + i)
        for nid, node in g.items():
            node.setdefault("_meta", {"title": nid})
        S.run(g, f"cast-scene-{name}", out)


if __name__ == "__main__":
    main(sys.argv[1:] or list(NEW_PLATES))
