#!/usr/bin/env python3
"""Kombucha Girl, pose-exact: the original clip's own frames as edit targets, Seth swapped in.

    python3 kombucha_ref.py keys    # five beat frames of the meme, redrawn with Seth in them
    python3 kombucha_ref.py clips   # H3 between each pair
    python3 kombucha_ref.py gif

kombucha.py invented the beats. This version takes them from the real clip. The meme frame is
image 1, the EDIT TARGET, which is what keeps her exact head angle, gaze, framing and the room
behind her -- the same property the sprite repaint runs on. Seth arrives as image 2, an identity
reference only.

Image 2 is the canon NEUTRAL expression, not the headshot. The headshot smiles, and the
expression app measured that the reference's own face beats the prompt: asked to scowl, it
returned a puzzled grin. A resting face gives her grimace somewhere to land.

Source: the widely shared Tenor copy of the 2019 clip, 624x640, 53 frames at ~14fps.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S  # noqa: E402
from build_workflows import NEGATIVE, STEPS  # noqa: E402
from kombucha import upload, submit, clip  # noqa: E402

OUT = "/home/wilson/scratch/kombucha/ref"
SEED = int(os.environ.get("KSEED", "7"))
W, H = 816, 832          # the source is 624x640; keep its aspect so the edit target is not warped
BEATS = [0, 12, 20, 32, 50]

WHO = ("a cartoon man with long wavy shoulder-length brown hair and a full bushy handlebar "
       "moustache, in a very dark charcoal v-neck t-shirt")

# ONE image. The first attempt passed Seth's canon plate as image 2 and it won every frame
# outright: the same front-facing neutral face in the same framing, her pose and her room gone.
# With two images Qwen redrew the reference instead of the target. So the meme frame is the only
# image, and Seth travels as words plus the Follies LoRA -- which was trained on him.
# Per-beat faces. "Make exactly the face in the photograph" softened every expression, and the
# grimace -- the whole meme -- came back as mild worry. A cartoon face has to be BIGGER than the
# photographed one, so each beat is named, exaggerated, with both ends of each feature stated.
FACES = {
    0: "looking down and off to the side, mouth closed and flat, quietly tasting",
    12: ("screwed up in a huge exaggerated grimace of disgust: mouth pulled hard down at both "
         "corners with the lower lip pushed up, eyebrows knitted together and raised in the "
         "middle, nose wrinkled, eyes narrowed"),
    20: ("head pulled back and turned, eyes slid hard to the side, lips pressed flat together, "
         "deeply sceptical"),
    32: ("eyes wide open and turned up and to the side, eyebrows raised high, mouth closed, "
         "reconsidering"),
    50: ("looking down and to the side, one corner of his mouth tucked in a small grudging "
         "smirk, eyebrows slightly raised, conceding it is not bad"),
}


def prompt_for(n):
    return ("f0llie5, image 1 is a photograph to redraw as a cartoon. This is a repaint: the head "
            "position, the tilt of the head, the direction the eyes are looking, the framing and "
            "the room behind -- the ceiling, the light fitting, the doors -- are already exactly "
            f"right and must not change. Redraw the person as {WHO}. His face is {FACES[n]}. "
            "Draw the whole picture, the room included, in warm sepia ink on aged paper.")


def keyframe(frame_name, dest, prefix, prompt):
    g = {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "follies-final.safetensors", "strength_model": 1.2, "model": ["3", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "6": {"class_type": "LoadImage", "inputs": {"image": frame_name}},
     "6s": {"class_type": "ImageScale", "inputs": {"image": ["6", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}},
     "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
     "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0], "image1": ["6s", 0]}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE + ", photograph, woman, blonde hair", "clip": ["4", 0]}},
     "22": {"class_type": "KSampler", "inputs": {"seed": SEED, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
     "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"kombucha-ref/{prefix}", "images": ["23", 0]}},
    }
    submit(g, "99", "images", dest)


def keys():
    for n in BEATS:
        dest = f"{OUT}/seth-f{n}.png" if SEED == 7 else f"{OUT}/seth-f{n}-s{SEED}.png"
        if os.path.exists(dest):
            print(n, "cached")
            continue
        frame = upload(f"{OUT}/f{n}.png", f"kombucha-src-f{n}.png")
        t0 = time.time()
        keyframe(frame, dest, f"f{n}", prompt_for(n))
        print(f"frame {n}: {time.time()-t0:.0f}s")


# Motion between consecutive beats, read off the original clip.
MOTION = {
    (0, 12): "He lifts his head toward the viewer and his whole face crumples into a grimace of "
             "disgust, mouth pulling down at both corners, brows knitting.",
    (12, 20): "The grimace drops away. He pulls his head back and turns it to the side, eyes "
              "sliding sideways, lips pressing together, reconsidering.",
    (20, 32): "He keeps his head turned and his eyes drift up and wider, as if tasting the "
              "aftertaste and thinking it over.",
    (32, 50): "He tips his head down and glances down to the side with a small grudging "
              "approving smirk and a slight nod.",
}


def clips():
    names = {n: upload(f"{OUT}/seth-f{n}.png", f"kombucha-seth-f{n}.png") for n in BEATS}
    for a, b in zip(BEATS, BEATS[1:]):
        dest = f"{OUT}/t{a}-{b}.mp4"
        if os.path.exists(dest):
            print(a, b, "cached")
            continue
        t0 = time.time()
        clip(names[a], names[b], MOTION[(a, b)], dest, f"ref-t{a}-{b}")
        print(f"t{a}-{b}: {time.time()-t0:.0f}s")


def gif():
    parts = f"{OUT}/concat.txt"
    with open(parts, "w") as f:
        for a, b in zip(BEATS, BEATS[1:]):
            f.write(f"file '{OUT}/t{a}-{b}.mp4'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", parts,
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{OUT}/body.mp4"], check=True)
    vf = ("tpad=stop_mode=clone:stop_duration=1.0,fps=15,scale=480:-1:flags=lanczos,"
          "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{OUT}/body.mp4", "-vf", vf,
                    "-loop", "0", f"{OUT}/seth-kombucha-ref.gif"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{OUT}/body.mp4", "-vf",
                    "tpad=stop_mode=clone:stop_duration=1.0", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-movflags", "+faststart", f"{OUT}/seth-kombucha-ref.mp4"], check=True)
    print("gif", os.path.getsize(f"{OUT}/seth-kombucha-ref.gif") // 1024, "KB")


if __name__ == "__main__":
    {"keys": keys, "clips": clips, "gif": gif}[sys.argv[1]]()
