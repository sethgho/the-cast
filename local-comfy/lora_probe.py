"""Probe a MiniMax H3 style LoRA against one sprite clip.

    python3 lora_probe.py <cid> <move> <lora|none> <strength>

Same graph as hires_sprite.py, plus a LoraLoaderModelOnly between the UNET and EasyCache, and the
trigger word in front of the prompt. Writes to /tmp/probe-<cid>-<move>-<tag>.mp4 so nothing here
can overwrite a real clip.
"""
import json, sys, time, os, subprocess
sys.path.insert(0, '/home/wilson/dev/the-cast/local-comfy')
import smoke_test as S
import repaint_cells as RC
from build_sprite import (SPRITE_LOCK, MOVES, WHO_LEAD, KEY_MAGENTA, STAGE_RESTATE, LENGTH,
                          CLIP_SEED)
from build_transition import TRAITS, SOUND_LOCK, H3_UNET, H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE

def brief(cid, recipe):
    """Copy of hires_sprite.brief — that module runs its render at import, so it cannot be imported."""
    path = RC.manifest_path(cid)
    man = json.load(open(path)) if os.path.exists(path) else {}
    for tag in man.get("tags", []):
        if tag["recipe"] == recipe:
            return tag["recipe_text"], man["trait"], tag["cyclic"]
    return MOVES[recipe], TRAITS[cid], RC.is_cyclic(recipe)

cid, move, lora, strength = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
SIZE, STEPS = 832, 20
TRIGGER = "gulliv3r, " if lora != "none" else ""

RECIPE_TEXT, TRAIT, CYCLIC = brief(cid, move)
prompt = f"{TRIGGER}{SPRITE_LOCK} {RECIPE_TEXT}. {WHO_LEAD} {TRAIT}. {STAGE_RESTATE} {SOUND_LOCK}"
tag = "none" if lora == "none" else f"{lora}-{strength}"
name = f"probe-{cid}-{move}-{tag}"

model = ["6", 0]
g = {"6": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}}}
if lora != "none":
    g["lora"] = {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["6", 0], "lora_name": f"studio1939-{lora}.safetensors",
                            "strength_model": strength}}
    model = ["lora", 0]
g.update({
 "50": {"class_type": "EasyCache", "inputs": {"model": model, "reuse_threshold": 0.2, "start_percent": 0.15, "end_percent": 0.95, "verbose": True}},
 "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}},
 "11": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
 "24": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
 "screen": {"class_type": "EmptyImage", "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1, "color": KEY_MAGENTA}},
 "plate": {"class_type": "LoadImage", "inputs": {"image": f"cast-cutout-{cid}.png"}},
 "fit": {"class_type": "ImageScale", "inputs": {"image": ["plate", 0], "upscale_method": "lanczos", "width": SIZE, "height": SIZE, "crop": "disabled"}},
 "alpha": {"class_type": "InvertMask", "inputs": {"mask": ["plate", 1]}},
 "aimg": {"class_type": "MaskToImage", "inputs": {"mask": ["alpha", 0]}},
 "afit": {"class_type": "ImageScale", "inputs": {"image": ["aimg", 0], "upscale_method": "lanczos", "width": SIZE, "height": SIZE, "crop": "disabled"}},
 "amask": {"class_type": "ImageToMask", "inputs": {"image": ["afit", 0], "channel": "red"}},
 "first": {"class_type": "ImageCompositeMasked", "inputs": {"destination": ["screen", 0], "source": ["fit", 0], "mask": ["amask", 0], "x": 0, "y": 0, "resize_source": False}},
 "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["13", 0], "vae": ["11", 0], "prompt": prompt,
         "width": SIZE, "height": SIZE, "length": LENGTH, "first_frame": ["first", 0], **({} if CYCLIC else {"last_frame": ["first", 0]})}},
 "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": CLIP_SEED}},
 "16": {"class_type": "BasicGuider", "inputs": {"model": ["50", 0], "conditioning": ["104", 0]}},
 "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
 "9": {"class_type": "BasicScheduler", "inputs": {"model": ["50", 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}},
 "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0], "sigmas": ["9", 0], "latent_image": ["104", 1]}},
 "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
 "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
 "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": 24, "bit_depth": 8}},
 "92": {"class_type": "SaveVideo", "inputs": {"video": ["91", 0], "filename_prefix": f"video/{name}", "format": "auto", "codec": "auto"}, "_meta": {"title": "RESULT"}},
})
for n, d in g.items(): d.setdefault("_meta", {"title": n})
t0 = time.time()
pid = S.api("/prompt", {"prompt": g, "client_id": "probe"})["prompt_id"]
while True:
    h = S.api(f"/history/{pid}")
    if pid in h: break
    time.sleep(10)
st = h[pid]['status']
if st.get('status_str') != 'success':
    print("FAILED", json.dumps(st)[:800]); sys.exit(1)
o = h[pid]['outputs']['92']['images'][0]
raw = f"/tmp/{name}.mp4"
subprocess.run(["curl", "-s", f"{S.HOST}/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output", "-o", raw], check=True)
print(f"{name}: {time.time()-t0:.0f}s -> {raw}")
