#!/usr/bin/env python3
"""Seth as Kombucha Girl: four keyframes, three H3 transitions, one reaction GIF.

    python3 kombucha.py keys     # the four stills
    python3 kombucha.py clips    # H3 between each pair
    python3 kombucha.py gif      # stitch and encode

The meme is four beats in one continuous take: the sip, the grimace, the ponder, the grudging
"hm, okay". The h3-storyboard skill measured that a single shot cannot hold more than two or
three facial beats -- pack them in and H3 averages them into a frozen face. So the beats are
drawn as keyframes and H3 only ever animates ONE transition per clip, first frame to last frame.

Keyframes carry the likeness. They come off the canon headshot through Qwen-Image-Edit with the
Follies LoRA, and K2-K4 are each edited from K1 rather than from each other: the cast repo's
assets.yaml measured chained edits at RMSE 39.3 against 85.3 for independent ones, and editing
from a fixed parent stops drift accumulating across the chain.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S  # noqa: E402
from build_workflows import NEGATIVE, STEPS  # noqa: E402
from build_transition import H3_UNET, H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE  # noqa: E402

OUT = "/home/wilson/scratch/kombucha"
SIZE = 832
SEED = int(os.environ.get("KSEED", "7"))

WHO = ("a cartoon man with long wavy shoulder-length brown hair and a full bushy handlebar "
       "moustache, in a very dark v-neck t-shirt")
FRAME = ("Chest-up, facing the viewer, centred, the same framing and the same plain background "
         "as image 1. He holds a small clear glass bottle of amber kombucha in his right hand.")

# The BASE is not a beat. It is a neutral plate with the bottle held low, so the mouth is free.
# The first attempt drew the sip first and edited the other three off it: the bottle stayed at his
# lips in every frame, covering exactly the part of the face the meme is about, and only the
# eyebrows moved. A prompt cannot beat the plate -- so the plate has to already be right.
BASE = (f"f0llie5, image 1 is the drawing to edit, and the man in image 1 is the only character "
        f"in the finished picture. Redraw him exactly as he is: {WHO}. Show him from the chest up, "
        f"facing the viewer, centred, on a plain background. He holds a small clear glass bottle "
        f"of amber kombucha in his right hand at chest height, well below his chin, so his whole "
        f"face is clear. His face is completely neutral: mouth closed and straight, eyebrows "
        f"level, eyes open looking at the viewer.")

# Edits of K1. Each changes the face and the bottle position and nothing else. The KEEP line is
# the last clause because whatever must survive goes last.
KEEP = ("Everything else stays exactly as it is in image 1: the same man, the same hair, the same "
        "moustache, the same t-shirt, the same bottle, the same framing, the same background. His expression is exactly as described above, not as it is in image 1.")
# Rubber-hose faces are BIG. Subtle wording came back as mild worry. Each feature is named at
# both ends, the lesson from the expression app: an open mouth of unstated shape defaults to the
# plate's own.
EDITS = {
    "k1-sip": ("He has raised the bottle to his mouth and is taking a sip from it, lips on the "
               "rim, eyes open and wide, eyebrows level, curious."),
    "k2-grimace": ("The bottle is back down at chest height, away from his face. His face is "
                   "screwed up in an exaggerated cartoon grimace of disgust: eyes squeezed "
                   "tightly shut into two curved creases, eyebrows crushed down together, nose "
                   "wrinkled up, mouth stretched wide and turned sharply down at both corners "
                   "with gritted teeth showing, head pulled back and chin tucked."),
    "k3-ponder": ("The bottle is at chest height, away from his face. His head is tilted to one "
                  "side and his pupils point up into the top corner of his eyes, looking up and "
                  "away, not at the viewer. His lips are pushed together and screwed over to one "
                  "side of his face, one cheek puffed, eyebrows uneven, as if weighing up the "
                  "aftertaste."),
    "k4-okay": ("The bottle is at chest height, away from his face. He is impressed despite "
                "himself: his lower lip is pushed out in a big exaggerated pout, the corners of "
                "his mouth pulled down, his eyebrows raised high, his eyes half-lidded and "
                "looking straight at the viewer, and his head nodding slowly forward, the classic "
                "'hm, not bad' face."),
}

# One transition per clip. Prompt describes the MOTION between the two keyframes only.
TRANSITIONS = [
    ("t1", "k1-sip", "k2-grimace",
     "He lowers the bottle from his lips and his face instantly screws up in disgust, eyes "
     "squeezing shut, nose wrinkling, mouth pulling down, head jerking back."),
    ("t2", "k2-grimace", "k3-ponder",
     "The grimace slowly melts away. His eyes drift up and to one side and his lips purse as he "
     "thinks about the taste."),
    ("t3", "k3-ponder", "k4-okay",
     "He comes out of his thought, looks straight ahead, raises his eyebrows and gives a small "
     "grudging approving nod with a slight closed-mouth smile."),
]
STYLE_CLIP = ("1933 rubber-hose cartoon animation, confident ink outlines, soft halftone shading, "
              "warm sepia on aged paper. The camera is locked off and does not move. The man does "
              "not look away from the viewer except where described. Overall soundscape: "
              "silence. Non-diegetic music: none.")
CLIP_LEN = 39   # 17n+5: 1.6s at 24fps, one beat
CLIP_W = CLIP_H = 640


def upload(path, name):
    b = uuid.uuid4().hex
    body = b"".join([
        f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode(),
        open(path, "rb").read(),
        f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n".encode()])
    req = urllib.request.Request(S.HOST + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["name"]


def submit(g, save_node, kind, dest):
    for n, d in g.items():
        d.setdefault("_meta", {"title": n})
    r = S.api("/prompt", {"prompt": g, "client_id": "kombucha"})
    if "prompt_id" not in r:
        raise SystemExit("REJECTED " + json.dumps(r)[:500])
    pid = r["prompt_id"]
    while True:
        h = S.api(f"/history/{pid}")
        if pid in h:
            break
        time.sleep(5)
    st = h[pid]["status"]
    if st.get("status_str") != "success":
        raise SystemExit("FAILED " + json.dumps(st)[:500])
    o = h[pid]["outputs"][save_node][kind][0]
    q = f"/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output"
    with urllib.request.urlopen(S.HOST + q, timeout=600) as rr:
        open(dest, "wb").write(rr.read())


def still(plate, prompt, dest, prefix):
    g = {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "follies-final.safetensors", "strength_model": 1.2, "model": ["3", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "6": {"class_type": "LoadImage", "inputs": {"image": plate}},
     "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1}},
     "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0]}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE, "clip": ["4", 0]}},
     "22": {"class_type": "KSampler", "inputs": {"seed": SEED, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
     "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"kombucha/{prefix}", "images": ["23", 0]}},
    }
    submit(g, "99", "images", dest)


def keys():
    os.makedirs(OUT, exist_ok=True)
    base = f"{OUT}/base.png"
    if not os.path.exists(base):
        still("canon-seth-headshot.png", BASE, base, "base")
        print("base done")
    base_name = upload(base, "kombucha-base.png")
    for name, change in EDITS.items():
        dest = f"{OUT}/{name}.png" if SEED == 7 else f"{OUT}/{name}-s{SEED}.png"
        if os.path.exists(dest):
            print(name, "cached")
            continue
        prompt = (f"f0llie5, image 1 is the drawing to edit. Change only his expression and where "
                  f"the bottle is. {change} {FRAME} {KEEP}")
        still(base_name, prompt, dest, name)
        print(name, "done")


def clip(first, last, motion, dest, prefix):
    g = {
     "6": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}},
     "7": {"class_type": "MiniMaxH3TurboLoRA", "inputs": {"model": ["6", 0], "lora_name": "minimax_h3_turbo_4step.safetensors", "strength": 1.0, "low_vram": True}},
     "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}},
     "11": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
     "24": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
     "a": {"class_type": "LoadImage", "inputs": {"image": first}},
     "b": {"class_type": "LoadImage", "inputs": {"image": last}},
     "af": {"class_type": "ImageScale", "inputs": {"image": ["a", 0], "upscale_method": "lanczos", "width": CLIP_W, "height": CLIP_H, "crop": "center"}},
     "bf": {"class_type": "ImageScale", "inputs": {"image": ["b", 0], "upscale_method": "lanczos", "width": CLIP_W, "height": CLIP_H, "crop": "center"}},
     "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["13", 0], "vae": ["11", 0], "prompt": f"{motion} {STYLE_CLIP}",
             "width": CLIP_W, "height": CLIP_H, "length": CLIP_LEN, "first_frame": ["af", 0], "last_frame": ["bf", 0]}},
     "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
     "16": {"class_type": "BasicGuider", "inputs": {"model": ["7", 0], "conditioning": ["104", 0]}},
     "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
     "9": {"class_type": "BasicScheduler", "inputs": {"model": ["7", 0], "scheduler": "simple", "steps": 12, "denoise": 1.0}},
     "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0], "sigmas": ["9", 0], "latent_image": ["104", 1]}},
     "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
     "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
     "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": 24, "bit_depth": 8}},
     "92": {"class_type": "SaveVideo", "inputs": {"video": ["91", 0], "filename_prefix": f"video/kombucha-{prefix}", "format": "auto", "codec": "auto"}},
    }
    submit(g, "92", "images", dest)


def clips():
    names = {}
    for k in ("k1-sip", "k2-grimace", "k3-ponder", "k4-okay"):
        names[k] = upload(f"{OUT}/{k}.png", f"kombucha-{k}.png")
    for tid, a, b, motion in TRANSITIONS:
        dest = f"{OUT}/{tid}.mp4"
        if os.path.exists(dest):
            print(tid, "cached")
            continue
        t0 = time.time()
        clip(names[a], names[b], motion, dest, tid)
        print(f"{tid} {a}->{b} {time.time()-t0:.0f}s")


def gif():
    """Holds on the grimace and the final nod are the whole joke, so they get real dwell time."""
    # The grimace gets its own 0.5s hold: without it the disgust is a blur on the way to the ponder.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{OUT}/t1.mp4", "-an", "-vf",
                    "tpad=stop_mode=clone:stop_duration=0.5", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", f"{OUT}/t1-hold.mp4"], check=True)
    parts = f"{OUT}/concat.txt"
    with open(parts, "w") as f:
        for t in ("t1-hold", "t2", "t3"):
            f.write(f"file '{OUT}/{t}.mp4'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", parts,
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{OUT}/body.mp4"], check=True)
    # Hold the last frame 1.2s, then a palette-optimised GIF at 15fps and 480px.
    vf = ("tpad=stop_mode=clone:stop_duration=1.2,fps=15,scale=480:-1:flags=lanczos,"
          "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{OUT}/body.mp4", "-vf", vf,
                    "-loop", "0", f"{OUT}/seth-kombucha.gif"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{OUT}/body.mp4", "-vf",
                    "tpad=stop_mode=clone:stop_duration=1.2", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-movflags", "+faststart", f"{OUT}/seth-kombucha.mp4"], check=True)
    print("gif", os.path.getsize(f"{OUT}/seth-kombucha.gif") // 1024, "KB")


if __name__ == "__main__":
    {"keys": keys, "clips": clips, "gif": gif}[sys.argv[1]]()
