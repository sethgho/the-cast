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

Four node titles, asserted by the provider at load:

  POSE    what the man is doing, read off the webcam still
  SCENE   the invented situation around him
  STYLE   the house style directive
  RESULT  the 1920x832 wallpaper
  AVATAR  a square crop of the face from that same render

AVATAR is why MediaPipe runs on the OUTPUT here rather than the input: the scene is composed
freely, so unlike seth-scene there is no fixed head box to crop. The graph finds the head it
actually drew and emits it, which is exact where a constant would be a guess.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from build_workflows import STYLE_LOCK, SETH_LOOK, NEGATIVE, STEPS  # noqa: E402

PLATE = "cast-seth-headshot-neutral.png"
W, H = 1920, 832          # 21:9 at 1.6MP, the same sheet seth-scene delivered
AVATAR = 768
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

 # The avatar is found, not assumed. See the docstring.
 "30": {"class_type": "LoadMediaPipeFaceLandmarker", "inputs": {"model_name": "mediapipe_face_fp32.safetensors"}, "_meta": {"title": "face-model"}},
 "31": {"class_type": "MediaPipeFaceLandmarker", "inputs": {"detector_variant": "both", "num_faces": 1, "min_confidence": 0.1, "missing_frame_fallback": "empty", "face_detection_model": ["30", 0], "image": ["23", 0]}, "_meta": {"title": "find-face"}},
 "32": {"class_type": "CropByBBoxes", "inputs": {"output_width": AVATAR, "output_height": AVATAR, "padding": 96, "keep_aspect": "pad", "image": ["23", 0], "bboxes": ["31", 1]}, "_meta": {"title": "crop-face"}},

 "40": {"class_type": "SaveImage", "inputs": {"filename_prefix": "cast/seth-selfie", "images": ["23", 0]}, "_meta": {"title": "RESULT"}},
 "41": {"class_type": "SaveImage", "inputs": {"filename_prefix": "cast/seth-selfie-avatar", "images": ["32", 0]}, "_meta": {"title": "AVATAR"}},
}

out = os.path.join(HERE, "api", "seth-selfie.api.json")
json.dump(g, open(out, "w"), indent=1)
print("wrote", out, len(g), "nodes")
