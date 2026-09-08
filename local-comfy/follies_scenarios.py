#!/usr/bin/env python3
"""Three uses of the Follies style LoRA, one engine.

    python3 follies_scenarios.py

  portrait   Any photo in, the same person out in the house style. The prompt is FIXED and says
             nothing about who is in the picture -- no description written by an agent, no
             per-image tuning. That is only possible because the style lives in the weight now;
             before the LoRA the prompt had to carry a paragraph of style, and describing the
             subject was how it stayed on model.
  invent     A brand new character from words alone, no reference at all.
  action     A canon plate re-posed. Identity from the plate, action from the words.
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S
from build_workflows import NEGATIVE, STEPS
from build_transition import TRAITS

SEED, LORA, STRENGTH = 7, "follies-final.safetensors", 1.2
OUT = "/home/wilson/scratch/follies-scenarios"

# ---- 1. the fixed, subject-blind instruction -------------------------------------------------
# It never names a feature, a garment, an age or a mood. Everything the picture is stays in the
# picture; the only thing added is the trigger.
PORTRAIT = ("f0llie5, redraw the photograph in image 1 as a drawn cartoon character. Keep the "
            "same person: the same face, the same hair, the same beard or lack of one, the same "
            "build, the same clothing, the same pose and the same expression, and keep them in "
            "the same position in the frame at the same size.")

RUNS = [
    ("1-portrait", "subject-in.png", PORTRAIT, 1024, 1024),
    ("2-giraffe", None,
     "f0llie5, an anthropomorphic giraffe in a well-cut suit made of deep purple velvet, standing "
     "at rest facing the viewer, holding a small ticker-tape printout in one hoof, a coin pressed "
     "into his lapel like a pin, on a plain flat background.", 1024, 1024),
]
SPORTS = {
    "3-boxing": "throwing a straight right hook, gloves up, weight on his front foot",
    "3-tennis": "mid-swing with a tennis racket, reaching wide for a forehand",
    "3-bowling": "mid-delivery with a bowling ball, arm swung back and one leg crossed behind",
    "3-swimming": "mid-freestyle stroke, one arm out of the water, seen from the side",
}
for name, act in SPORTS.items():
    RUNS.append((name, "canon-seth.png",
                 f"f0llie5, image 1 is the drawing to edit and the man in image 1 is the only "
                 f"character in the finished picture. Redraw him exactly as he is: "
                 f"{TRAITS['seth']}. He is {act}, on a plain flat background.", 1024, 1024))


def run(name, plate, prompt, w, h):
    dest = f"{OUT}/{name}.png"
    if os.path.exists(dest):
        print(name, "cached"); return
    g = {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": LORA, "strength_model": STRENGTH, "model": ["3", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE, "clip": ["4", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
    }
    enc = {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0]}
    if plate:
        g["6"] = {"class_type": "LoadImage", "inputs": {"image": plate}}
        enc["image1"] = ["6", 0]
    g["20"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": enc}
    g["22"] = {"class_type": "KSampler", "inputs": {"seed": SEED, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}}
    g["99"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"follies/{name}", "images": ["23", 0]}}
    for n, d in g.items(): d.setdefault("_meta", {"title": n})
    r = S.api("/prompt", {"prompt": g, "client_id": "scenarios"})
    if "prompt_id" not in r: print(name, "REJECTED", json.dumps(r)[:400]); return
    pid = r["prompt_id"]
    while True:
        h_ = S.api(f"/history/{pid}")
        if pid in h_: break
        time.sleep(4)
    if h_[pid]["status"].get("status_str") != "success": print(name, "FAILED"); return
    o = h_[pid]["outputs"]["99"]["images"][0]
    with urllib.request.urlopen(S.HOST + f"/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output", timeout=300) as rr:
        open(dest, "wb").write(rr.read())
    print(name, "->", dest)


os.makedirs(OUT, exist_ok=True)
for a in RUNS: run(*a)
print("DONE")
