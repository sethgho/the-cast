#!/usr/bin/env python3
"""Judge the Follies style LoRA: one prompt, one seed, five checkpoints.

    python3 follies_bakeoff.py

The LoRA was trained on Qwen-Image's transformer. Qwen-Image-Edit is that same transformer with
extra conditioning, and the LoRA's keys are the plain `diffusion_model.transformer_blocks.*` set,
so it applies to the edit model we already run -- no second 20B download, and if it holds up it
plugs straight into previz, the selfie app and the storyboards.

No reference image and no style words in the prompt. That is the whole test: if the look has to
be described, the LoRA did not learn it. `f0llie5` is the trigger the captions withheld the style
from.
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S

W, H, SEED, STEPS = 1024, 1024, 7, 8
# Deliberately NOT one of the cast. A style LoRA has to make a character it has never seen.
PROMPT = ("f0llie5, a stout cartoon fire chief with a big walrus moustache and a peaked helmet, "
          "standing at rest with his arms at his sides, seen from the front, on a plain flat "
          "background.")
NEG = "photorealistic, 3d render, blurry, deformed, text, watermark"
RUNS = [("control", None), ("250", "follies-000000250.safetensors"),
        ("750", "follies-000000750.safetensors"), ("1250", "follies-000001250.safetensors"),
        ("2000", "follies-final.safetensors")]

# Strength sweep. A style that is NEARLY there often sharpens above 1.0 -- the learned direction
# is right and simply under-applied. Past ~1.5 it usually stops being a style and starts being an
# artefact, so the ladder stops there.
SWEEP = [("2000@0.8", "follies-final.safetensors", 0.8),
         ("2000@1.2", "follies-final.safetensors", 1.2),
         ("2000@1.5", "follies-final.safetensors", 1.5),
         ("1250@1.2", "follies-000001250.safetensors", 1.2)]


def graph(lora, strength=1.0):
    g = {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEG, "clip": ["4", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
    }
    # The Lightning LoRA rides underneath so 8 steps at cfg 1.0 stays valid; the style LoRA
    # stacks on top of it.
    g["3"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}}
    model = ["3", 0]
    if lora:
        g["30"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": lora, "strength_model": strength, "model": ["3", 0]}}
        model = ["30", 0]
    g["20"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": PROMPT, "clip": ["4", 0], "vae": ["5", 0]}}
    g["22"] = {"class_type": "KSampler", "inputs": {"seed": SEED, "steps": STEPS, "cfg": 1.0,
               "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": model,
               "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}}
    return g


out = "/home/wilson/scratch/follies-bakeoff"
os.makedirs(out, exist_ok=True)
for name, lora, strength in [(n, l, 1.0) for n, l in RUNS] + SWEEP:
    dest = f"{out}/{name}.png"
    if os.path.exists(dest):
        print(name, "cached"); continue
    g = graph(lora, strength)
    g["99"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"follies/{name}", "images": ["23", 0]}}
    for n, d in g.items(): d.setdefault("_meta", {"title": n})
    t0 = time.time()
    r = S.api("/prompt", {"prompt": g, "client_id": "bakeoff"})
    if "prompt_id" not in r:
        print(name, "REJECTED", json.dumps(r)[:400]); continue
    pid = r["prompt_id"]
    while True:
        h = S.api(f"/history/{pid}")
        if pid in h: break
        time.sleep(4)
    st = h[pid]["status"]
    if st.get("status_str") != "success":
        print(name, "FAILED", json.dumps(st)[:400]); continue
    o = h[pid]["outputs"]["99"]["images"][0]
    q = f"/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output"
    with urllib.request.urlopen(S.HOST + q, timeout=300) as rr:
        open(dest, "wb").write(rr.read())
    print(f"{name}: {time.time()-t0:.0f}s -> {dest}")
print("DONE")
