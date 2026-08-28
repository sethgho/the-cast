"""STUDIO 1939 used the way its card asks: pure text-to-video, trigger first, strength 1.0.

    python3 lora_native.py [seed] [strength]

Nothing of ours is imposed here -- no first frame, no character plate, no sepia, no magenta. The
card asks for 16:9, prompt expansion off, exhaustive art direction that names every colour and
ends with what must not appear, so that is exactly what this sends. 1280x720 is 0.92MP, just
inside the measured 0.94MP ceiling at ~5s.
"""
import json, sys, time, os, subprocess
sys.path.insert(0, '/home/wilson/dev/the-cast/local-comfy')
import smoke_test as S
from build_transition import H3_UNET, H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE

W, H, LENGTH, STEPS, FPS = 1280, 720, 121, 20, 24
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1939
STRENGTH = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
# EasyCache reuses a step's residual when the change rate stays under a threshold. A LoRA moves
# that profile, and every melted render so far had it skipping 10 of 20 steps, so it is off by
# default here and the render costs ~2x.
CACHE = os.environ.get("EASYCACHE", "0") == "1"
SRC = "50" if CACHE else "lora"

PROMPT = (
    "gulliv3r, A golden-age hand-painted animated feature. A short round-bellied man with long "
    "wavy chestnut hair to his shoulders and an enormous bushy brown handlebar moustache stands in "
    "a cluttered clockmaker's workshop. He wears a rust-red waistcoat over a cream linen shirt with "
    "rolled sleeves, olive-brown trousers and scuffed brown boots. Behind him a wall of brass "
    "pendulum clocks in dark walnut cases ticks and sways, a leaded window throws a warm butter-"
    "yellow shaft of afternoon light across a workbench of scattered cogs, springs and tiny screws, "
    "and dust motes drift through the beam. He holds a gold pocket watch up to his ear and "
    "listens, then slowly breaks into a warm smile. He does not move otherwise. "
    "Rich saturated inks, soft airbrushed cel shading, deep painted background. The camera is "
    "locked off and does not move. Audio: a warm orchestral woodwind theme, layered clock ticking, "
    "a soft chime, and the man's quiet contented hum. Not photographic, not 3D, not modern digital "
    "vector art, no text, no captions, no watermark."
)

g = {
 "6": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}},
 "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["6", 0], "lora_name": "studio1939-strong.safetensors", "strength_model": STRENGTH}},
 "50": {"class_type": "EasyCache", "inputs": {"model": ["lora", 0], "reuse_threshold": 0.2, "start_percent": 0.15, "end_percent": 0.95, "verbose": False}},
 "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}},
 "11": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
 "24": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
 "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["13", 0], "vae": ["11", 0], "prompt": PROMPT,
         "width": W, "height": H, "length": LENGTH}},
 "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
 "16": {"class_type": "BasicGuider", "inputs": {"model": [SRC, 0], "conditioning": ["104", 0]}},
 "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
 "9": {"class_type": "BasicScheduler", "inputs": {"model": [SRC, 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}},
 "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0], "sigmas": ["9", 0], "latent_image": ["104", 1]}},
 "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
 "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
 "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": FPS, "bit_depth": 8}},
 "92": {"class_type": "SaveVideo", "inputs": {"video": ["91", 0], "filename_prefix": f"video/lora1939-{SEED}-{STRENGTH}-c{int(CACHE)}", "format": "auto", "codec": "auto"}, "_meta": {"title": "RESULT"}},
}
for n, d in g.items(): d.setdefault("_meta", {"title": n})
t0 = time.time()
pid = S.api("/prompt", {"prompt": g, "client_id": "lora1939"})["prompt_id"]
while True:
    h = S.api(f"/history/{pid}")
    if pid in h: break
    time.sleep(10)
st = h[pid]['status']
if st.get('status_str') != 'success':
    print("FAILED", json.dumps(st)[:900]); sys.exit(1)
o = h[pid]['outputs']['92']['images'][0]
raw = f"/tmp/lora1939-{SEED}-{STRENGTH}-c{int(CACHE)}.mp4"
subprocess.run(["curl", "-s", f"{S.HOST}/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output", "-o", raw], check=True)
print(f"seed {SEED} @ {STRENGTH}: {time.time()-t0:.0f}s -> {raw}")
