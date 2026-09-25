#!/usr/bin/env python3
"""The action-selfie app: a written moment in, a Follies scene of Seth out.

    python3 build_selfie.py        # writes api/seth-selfie.api.json

## Why this graph and not seth-scene

seth-scene drew Seth's head from the webcam face, froze those pixels on the sheet and painted the
room around them. Measured against ten real captures it fails the thing it exists for: three
visibly different moments came back as three near-identical portraits, because at webcam
resolution the expression signal is too weak to move the plate. It also costs ~360s, most of it
the harmonise pass that exists only to hide the freeze.

This graph carries the moment in WORDS instead of pixels. The caller reads the capture into a
description of pose, gesture, wardrobe and light; that text plus the scene drives a single edit
off Seth's canonical plate. Nothing is frozen, so nothing needs harmonising, and the character is
cast Seth by construction rather than by hope. Measured: full character AND the right moment, at
about a sixth of the cost.

## The contract with the caller

Node titles, asserted by the provider at load. The caller rewrites every text node per run, so
the wording here is only a default:

  PLATE, IDENTITY, POSE, SCENE, STYLE, RESTATE, POSITIVE, NEGATIVE, SAMPLER
  RESULT   the 1920x832 wallpaper
  SUBJECT  where Seth's head is in that same render, as SAM 3 boxes (JSON text)

SUBJECT runs on the OUTPUT rather than the input: the scene is composed freely, so there is no
fixed head box to crop. It is found by SAM 3 prompted with his hair, not by a face detector: a
face detector finds A face, and with co-stars in the scene MediaPipe cropped Cadbury in three
renders out of three. Nobody else in the troupe has long wavy hair, so the prompt picks Seth,
and the box it returns is his head. The caller does the cropping, where clamping the square
inside the frame and falling back when nothing is found are one line each.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from build_workflows import STYLE_LOCK, SETH_LOOK, NEGATIVE, STEPS  # noqa: E402

PLATE = "cast-seth-headshot-neutral.png"
W, H = 1920, 832          # 21:9 at 1.6MP, the same sheet seth-scene delivered
SUBJECT_MODEL = "sam3.1_multiplex_fp16.safetensors"   # Comfy-Org/sam3.1, models/checkpoints
SUBJECT_PROMPT = "long wavy hair"
SEED = 7

# The prompt is assembled in the graph from four string nodes so the caller can rewrite three of
# them without touching the fourth. Order matters: the laws say whatever must survive goes last,
# and what must survive is that this is Seth.
IDENTITY = (f"Image 1 is the drawing to edit, and the man in image 1 is the only character in the "
            f"finished picture. Redraw him exactly as he is: {SETH_LOOK}.")
RESTATE = ("Above all, the man in the finished drawing is the man from image 1 — the same long "
           "wavy hair, the same bushy handlebar moustache, the same dark v-neck t-shirt — and he "
           "is doing exactly what the description of his pose says.")

g = {
 "2":  {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}, "_meta": {"title": "ENGINE"}},
 "3":  {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}, "_meta": {"title": "LIGHTNING"}},
 "4":  {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}, "_meta": {"title": "CLIP"}},
 "5":  {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}, "_meta": {"title": "VAE"}},

 "6":  {"class_type": "LoadImage", "inputs": {"image": PLATE}, "_meta": {"title": "PLATE"}},

 "10": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": IDENTITY}, "_meta": {"title": "IDENTITY"}},
 "11": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "He sits at his desk, hands on the keyboard, looking at the screen."}, "_meta": {"title": "POSE"}},
 "12": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "An ordinary evening at the workbench."}, "_meta": {"title": "SCENE"}},
 "13": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": STYLE_LOCK}, "_meta": {"title": "STYLE"}},
 "14": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": RESTATE}, "_meta": {"title": "RESTATE"}},

 "15": {"class_type": "StringConcatenate", "inputs": {"delimiter": " ", "string_a": ["10", 0], "string_b": ["11", 0]}, "_meta": {"title": "join-1"}},
 "16": {"class_type": "StringConcatenate", "inputs": {"delimiter": " ", "string_a": ["15", 0], "string_b": ["12", 0]}, "_meta": {"title": "join-2"}},
 "17": {"class_type": "StringConcatenate", "inputs": {"delimiter": " ", "string_a": ["16", 0], "string_b": ["13", 0]}, "_meta": {"title": "join-3"}},
 "18": {"class_type": "StringConcatenate", "inputs": {"delimiter": " ", "string_a": ["17", 0], "string_b": ["14", 0]}, "_meta": {"title": "join-4"}},

 "19": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}, "_meta": {"title": "SHEET"}},
 "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": ["18", 0], "clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0]}, "_meta": {"title": "POSITIVE"}},
 "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE, "clip": ["4", 0]}, "_meta": {"title": "NEGATIVE"}},
 "22": {"class_type": "KSampler", "inputs": {"seed": SEED, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["3", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["19", 0]}, "_meta": {"title": "SAMPLER"}},
 "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}, "_meta": {"title": "DECODE"}},

 # Seth's head is found, not assumed. See the docstring.
 "30": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SUBJECT_MODEL}, "_meta": {"title": "subject-model"}},
 "31": {"class_type": "CLIPTextEncode", "inputs": {"text": SUBJECT_PROMPT, "clip": ["30", 1]}, "_meta": {"title": "subject-prompt"}},
 "32": {"class_type": "SAM3_Detect", "inputs": {"threshold": 0.5, "refine_iterations": 0, "individual_masks": True, "model": ["30", 0], "image": ["23", 0], "conditioning": ["31", 0]}, "_meta": {"title": "find-subject"}},

 "40": {"class_type": "SaveImage", "inputs": {"filename_prefix": "cast/seth-selfie", "images": ["23", 0]}, "_meta": {"title": "RESULT"}},
 "41": {"class_type": "PreviewAny", "inputs": {"source": ["32", 1]}, "_meta": {"title": "SUBJECT"}},
}

out = os.path.join(HERE, "api", "seth-selfie.api.json")
json.dump(g, open(out, "w"), indent=1)
print("wrote", out, len(g), "nodes")
