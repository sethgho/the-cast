"""Seth's identity from a reference plate, the era's look from STUDIO 1939.

    python3 lora_ref.py [seed] [strength]

Text-to-video invents a stranger and first-frame conditioning makes the LoRA fight a drawing it
wants to repaint. MiniMaxH3ReferenceToVideo is the third door: it takes the plate as an IDENTITY
reference rather than frame zero, so the LoRA is free to paint its own scene around a face it has
been told to keep. That needs the ref2va weights, not the fl2va ones used everywhere else here.

EasyCache stays OFF. With it on it reused 10 of 20 steps and every character melted mid-clip.
"""
import json, sys, time, os, subprocess
sys.path.insert(0, '/home/wilson/dev/the-cast/local-comfy')
import smoke_test as S
from build_transition import H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE

UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
W, H, LENGTH, STEPS, FPS = 1280, 720, 124, 20, 24
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1939
STRENGTH = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
PLATE = os.environ.get("PLATE", "staged-seth.png")

PROMPT = (
    "gulliv3r, A golden-age hand-painted animated feature. The man in the reference image — long "
    "wavy chestnut hair to his shoulders, a big bushy brown moustache, a dark olive v-neck "
    "t-shirt, grey trousers and black-and-white sneakers — stands in a cluttered clockmaker's "
    "workshop. Behind him a wall of brass pendulum clocks in dark walnut cases, a leaded window "
    "throwing a warm butter-yellow shaft of afternoon light across a workbench of scattered cogs "
    "and springs, dust motes drifting in the beam. He holds a gold pocket watch up to his ear, "
    "listens, and slowly breaks into a warm smile. He does not move otherwise. Rich saturated "
    "inks, soft airbrushed cel shading, deep painted background. The camera is locked off and "
    "does not move. Audio: a warm orchestral woodwind theme, layered clock ticking and a soft "
    "chime. Not photographic, not 3D, not modern digital vector art, no text, no watermark."
)

MODEL = "lora" if STRENGTH > 0 else "6"
if STRENGTH <= 0:
    PROMPT = PROMPT.replace("gulliv3r, ", "")

g = {
 "6": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
 **({"lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["6", 0], "lora_name": "studio1939-strong.safetensors", "strength_model": STRENGTH}}} if STRENGTH > 0 else {}),
 "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}},
 "11": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
 "24": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
 "plate": {"class_type": "LoadImage", "inputs": {"image": PLATE}},
 "104": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["13", 0], "vae": ["11", 0],
         "audio_vae": ["24", 0], "prompt": PROMPT, "width": W, "height": H, "length": LENGTH,
         "ref_image_size": "max", "ref_images": {"ref_image_0": ["plate", 0]}}},
 "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
 "16": {"class_type": "BasicGuider", "inputs": {"model": [MODEL, 0], "conditioning": ["104", 0]}},
 "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
 "9": {"class_type": "BasicScheduler", "inputs": {"model": [MODEL, 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}},
 "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0], "sigmas": ["9", 0], "latent_image": ["104", 1]}},
 "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
 "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
 "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": FPS, "bit_depth": 8}},
 "92": {"class_type": "SaveVideo", "inputs": {"video": ["91", 0], "filename_prefix": f"video/sethref-{SEED}-{STRENGTH}", "format": "auto", "codec": "auto"}, "_meta": {"title": "RESULT"}},
}
for n, d in g.items(): d.setdefault("_meta", {"title": n})
t0 = time.time()
r = S.api("/prompt", {"prompt": g, "client_id": "sethref"})
if "prompt_id" not in r:
    print("REJECTED", json.dumps(r)[:1200]); sys.exit(1)
pid = r["prompt_id"]
while True:
    h = S.api(f"/history/{pid}")
    if pid in h: break
    time.sleep(10)
st = h[pid]['status']
if st.get('status_str') != 'success':
    print("FAILED", json.dumps(st)[:1200]); sys.exit(1)
o = h[pid]['outputs']['92']['images'][0]
raw = f"/tmp/sethref-{SEED}-{STRENGTH}.mp4"
subprocess.run(["curl", "-s", f"{S.HOST}/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output", "-o", raw], check=True)
print(f"seed {SEED} @ {STRENGTH}: {time.time()-t0:.0f}s -> {raw}")
