"""One plain vertical H3 clip of Seth doing a vaudeville turn, with native audio.

    python3 vaudeville_clip.py [seed]

Nothing from the sprite pipeline applies here: no magenta key, no sprite lock, no repaint. The
first frame is the existing rubber-hose Seth plate, which is already 736x1280 on a flat sepia
ground -- 0.94MP, the measured ceiling at ~5s. STUDIO 1939 rides at strength 0.6 on the strong
(rank 64) file, the only setting that held Seth's face in the probe.
"""
import json, sys, time, os, subprocess
sys.path.insert(0, '/home/wilson/dev/the-cast/local-comfy')
import smoke_test as S
from build_transition import H3_UNET, H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE

W, H, LENGTH, STEPS, FPS = 736, 1280, 121, 20, 24
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 77
STRENGTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
CACHE = os.environ.get("EASYCACHE", "1") == "1"
PIN = os.environ.get("PIN", "1") == "1"
LORA = "studio1939-strong.safetensors"

# Seth dissolved at 0.6 with an unpinned end: the LoRA repaints him a little more each frame and
# nothing pulls him back. Pinning the last frame to the same plate makes the clip a one-shot that
# must return to a known drawing, which is the sprite pipeline's own rule for non-cycles.

# The LoRA card asks for the trigger first and then exhaustive art direction, so the prompt names
# every element rather than leaning on a style word.
PROMPT = (
    "gulliv3r, A 1930s vaudeville stage. A long-haired man with a bushy brown moustache, a dark "
    "brown v-neck t-shirt, tan trousers and white-and-black sneakers stands centre stage in the "
    "warm amber pool of a single footlight. Heavy crimson velvet curtains hang behind him and a "
    "worn wooden stage floor runs beneath. He grins, sweeps one arm wide in a big theatrical "
    "presenting flourish, tips an imaginary hat, then leans in, waggles his eyebrows, doffs the hat "
    "with a flourish, and finishes with a deep bow. Rubber-hose cartoon "
    "animation, bouncy elastic limbs, confident black ink outlines, soft halftone shading, warm "
    "sepia and cream tones, dust motes drifting in the light beam. The camera does not move. "
    "Audio: a tinny upright piano playing a bright ragtime vamp, a snare-and-woodblock rimshot, "
    "shoes tapping on wood, and a small delighted audience laughing and applauding."
)

MODEL = "lora" if STRENGTH > 0 else "6"
if STRENGTH <= 0:
    PROMPT = PROMPT.replace("gulliv3r, ", "")

g = {
 "6": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}},
 **({"lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["6", 0], "lora_name": LORA, "strength_model": STRENGTH}}} if STRENGTH > 0 else {}),
 "50": {"class_type": "EasyCache", "inputs": {"model": [MODEL, 0], "reuse_threshold": 0.2, "start_percent": 0.15, "end_percent": 0.95, "verbose": False}},
 "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}},
 "11": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
 "24": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
 "plate": {"class_type": "LoadImage", "inputs": {"image": os.environ.get("PLATE", "rubberhose-kf-seth-3q.png")}},
 "fit": {"class_type": "ImageScale", "inputs": {"image": ["plate", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}},
 "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["13", 0], "vae": ["11", 0], "prompt": PROMPT,
         "width": W, "height": H, "length": LENGTH, "first_frame": ["fit", 0], **({"last_frame": ["fit", 0]} if PIN else {})}},
 "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
 "16": {"class_type": "BasicGuider", "inputs": {"model": ["50" if CACHE else MODEL, 0], "conditioning": ["104", 0]}},
 "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
 "9": {"class_type": "BasicScheduler", "inputs": {"model": ["50" if CACHE else MODEL, 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}},
 "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0], "sigmas": ["9", 0], "latent_image": ["104", 1]}},
 "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
 "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
 "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": FPS, "bit_depth": 8}},
 "92": {"class_type": "SaveVideo", "inputs": {"video": ["91", 0], "filename_prefix": f"video/seth-vaudeville-{SEED}-{STRENGTH}", "format": "auto", "codec": "auto"}, "_meta": {"title": "RESULT"}},
}
for n, d in g.items(): d.setdefault("_meta", {"title": n})
t0 = time.time()
pid = S.api("/prompt", {"prompt": g, "client_id": "vaudeville"})["prompt_id"]
while True:
    h = S.api(f"/history/{pid}")
    if pid in h: break
    time.sleep(10)
st = h[pid]['status']
if st.get('status_str') != 'success':
    print("FAILED", json.dumps(st)[:900]); sys.exit(1)
o = h[pid]['outputs']['92']['images'][0]
raw = f"/tmp/seth-vaudeville-{SEED}-{STRENGTH}.mp4"
subprocess.run(["curl", "-s", f"{S.HOST}/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output", "-o", raw], check=True)
print(f"seed {SEED}: {time.time()-t0:.0f}s -> {raw}")
