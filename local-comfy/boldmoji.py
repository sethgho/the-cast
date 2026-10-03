#!/usr/bin/env python3
"""Boldmojis: the sethmoji pipeline for "Bold", a sponsored comic character (@boldleonidas).

    python3 boldmoji.py <stage|all> [names]   # stages as sethmoji, plus hires

Same stages and emoji packing as sethmoji.py. No style LoRA: the look comes from the character
sheet itself, passed as image 2 of the reference edit (front + 3/4 views, text cropped off).
The sheet lives only on gpu-worker (input/boldmoji-ref.png) and in scratch -- it is the sponsor's
art, so never commit it.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sethmoji as B  # noqa: E402

SETH_MEMES = dict(B.MEMES)
B.ROOT = "/home/wilson/scratch/boldmoji"
B.WHO = "bold"
B.PREFIX = "boldmoji"
B.MAC_DIR = "Documents/avatars/boldmojis"

BOLD = ("a small chunky cartoon character drawn in soft graphite pencil with crosshatched texture, "
        "greyscale only. He has a big smooth round bald head with no hair, no ears and no nose, two "
        "huge hollow oval eyes that are EMPTY plain white ovals with a dark outline and NO pupils, NO irises, nothing inside them, and a "
        "small simple mouth. He wears a thick padded jacket with a tall stand-up collar, open over a "
        "horizontally striped shirt, chunky trousers and rounded boots. His hands are smooth rounded "
        "mitten nubs with no fingers")
NEG = ("colour, sepia, realistic, photograph, human face, hair, nose, ears, pupils, fingers, "
       "blurry, deformed, extra limbs, watermark, text")

B.MEMES = {
    "kombucha": dict(pick=0, seed=21, caption=0.0, top=0.62, strengths=(1.0, 1.0), sheet="boldmoji-head.png",
                     frame="A CLOSE-UP exactly like image 1: only his big head and the top of his collar and shoulders are visible, cut off at the chest by the bottom edge, his head filling the left half of the picture and turned to his left; NOT a full-body view, no legs, no feet. Behind him the white ceiling and the white doors of the room as in image 1.",
                     face="head turned three-quarters away to the side, the oval eyes looking off to the side, a small flat closed mouth, a neutral expression, NOT smiling",
                     scene="In a bedroom, wearing his padded jacket",
                     action="He sips kombucha, grimaces in disgust, reconsiders, then decides he actually likes it"),
    "andy-zoom": dict(pick=1, seed=7, caption=0.0, top=1.0,
                      frame="He is SITTING at the meeting table exactly where the man sits in image 1, seen from the waist up, one mitten hand raised in front of him at table height, with the dark back and shoulder of another person in the left foreground and the dark computer monitor in the right foreground, exactly as in image 1. NOT a full-body view, no legs, no feet.",
                      face="mouth wide open in a big round O gasp of pure excitement, the oval eyes stretched huge",
                      scene="In an office meeting room, wearing his padded jacket",
                      action="He gasps with wide-eyed excitement as the camera zooms in fast on his face"),
}


# The rest of the library, derived from Seth's entries: same clip, cut, crop and post effect;
# the face line loses anything Bold does not have. Close-ups take the head-and-collar sheet.
FROM_SETH = ["blink", "nod", "jim-look", "thats-me", "nooo", "lost", "cheers", "mind-blown",
             "slow-clap", "mic-drop", "its-happening", "popcorn", "salty", "elaine", "yes-you",
             "ok", "van-door", "hotline-bling", "rickroll", "carlton", "ear-cup", "chest-thump",
             "chefkiss", "pour-one-out", "thats-bait", "ron-dance", "ron-trash", "ron-eyeroll",
             "ron-nope", "ron-smile", "ron-snakejuice", "jack-salute"]
CLOSE_UPS = {"blink", "nod", "jim-look", "ok", "ron-eyeroll"}
NOT_BOLD = ("moustache", "nose", "hair", "beard", "teeth", "clean-shaven", "eyebrow", "lips", "cheek")


def _bolden(text):
    keep = [c for c in text.split(", ") if not any(w in c.lower() for w in NOT_BOLD)]
    return ", ".join(keep) or "an expression matching the person in the photograph"


for _n in FROM_SETH:
    _m = dict(SETH_MEMES[_n])
    _m["face"] = _bolden(_m.get("face", "making the same expression as the person in the photograph"))
    if "extra" in _m:
        _m["extra"] = _bolden(_m["extra"])
    _m.setdefault("seed", 7)
    if _n in CLOSE_UPS:
        _m["sheet"] = "boldmoji-head.png"
    B.MEMES[_n] = _m

def qwen_edit(images, prompt, W, H, prefix, dest, seed=7):
    g = {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
     "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0]}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEG, "clip": ["4", 0]}},
     "22": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": B.STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["3", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
     "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"boldmoji/{prefix}", "images": ["23", 0]}},
    }
    for k, img in enumerate(images, 1):
        g[f"i{k}"] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g["20"]["inputs"][f"image{k}"] = [f"i{k}", 0]
    return B.submit(g, "99", "images", dest)


def ref(name):
    m, dd = B.MEMES[name], B.d(name)
    spec = json.load(open(f"{dd}/spec.json"))
    B.sh("scp", "-q", f"{dd}/frame0.png", f"{B.GPU}:comfyui/input/boldmoji-{name}-f0.png")
    prompt = ("Image 1 is a frame from a video. Image 2 is a character sheet. Redraw image 1 with the "
              "person replaced by the character from image 2, exactly as he is drawn there: "
              f"{BOLD}. Everything about the shot stays exactly as in image 1: the pose, the tilt and "
              "turn of the head, where he is looking, the arms and anything held in the hands, the "
              "room, the camera angle, the crop and the size of the head in the frame. His face: "
              f"{m['face']}. {'He also wears ' + m['clothing'] + '. ' if 'clothing' in m else ''}"
              f"{m.get('extra', '')} {m.get('frame', '')} Draw the WHOLE picture, the room included, in the "
              "same soft greyscale graphite pencil style as image 2. No text, no lettering.")
    # close-ups take the sheet's head-and-collar crops; the full-body views make a close-up
    # come back as a standing figure
    qwen_edit([f"boldmoji-{name}-f0.png", m.get("sheet", "boldmoji-ref.png")], prompt, spec["W"], spec["H"],
              f"{name}-ref", f"{dd}/ref.png", seed=m.get("seed", 7))
    B.sh("scp", "-q", f"{dd}/ref.png", f"{B.GPU}:comfyui/input/boldmoji-{name}-ref.png")
    print(f"{name} ref")


def animate(name):
    m, dd = B.MEMES[name], B.d(name)
    spec = json.load(open(f"{dd}/spec.json"))
    a = argparse.Namespace(
        ref=f"boldmoji-{name}-ref.png", drive=f"boldmoji-{name}-drive.mp4",
        prompt=f"{BOLD}. {m['scene']}, the whole picture in the same greyscale pencil style. {m['action']}.",
        pose_prompt=f"{m['action'].replace('He ', 'A person ')}. Static camera.",
        out=f"boldmoji/{name}", model="wan_animate_2_distill_int8_convrot.safetensors",
        lora=None, width=spec["W"], height=spec["H"], length=spec["frames"], fps=B.FPS, steps=6,
        # Bold has no nose and no pupils: at pose 2.0 / ref 0.6 the driving face paints a human
        # nose and pupils on him, so a close-up meme can hold him tighter with `strengths`
        seed=77, cache="cpu", pose_strength=m.get("strengths", (2.0, 0.6))[0],
        ref_strength=m.get("strengths", (2.0, 0.6))[1])
    urllib.request.urlopen(urllib.request.Request(
        B.S.HOST + "/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
        {"Content-Type": "application/json"}), timeout=60).read()
    t0 = time.time()
    B.submit(B.animate_graph(a), "21", "images", f"{dd}/seth.mp4")
    print(f"{name} animate {time.time()-t0:.0f}s")


def hires(name):
    """RealESRGAN x4, then back down to 2x."""
    dd = B.d(name)
    B.sh("scp", "-q", f"{dd}/seth.mp4", f"{B.GPU}:comfyui/input/boldmoji-{name}-seth.mp4")
    g = {
     "1": {"class_type": "LoadVideo", "inputs": {"file": f"boldmoji-{name}-seth.mp4"}},
     "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
     "3": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x4plus.pth"}},
     "4": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["3", 0], "image": ["2", 0]}},
     "5": {"class_type": "ImageScaleBy", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "scale_by": 0.5}},
     "6": {"class_type": "CreateVideo", "inputs": {"images": ["5", 0], "fps": float(B.FPS)}},
     "7": {"class_type": "SaveVideo", "inputs": {"video": ["6", 0], "filename_prefix": f"boldmoji/{name}-hires", "format": "mp4", "codec": "h264"}},
    }
    B.submit(g, "7", "images", f"{dd}/{name}-bold-hires.mp4")
    print(f"{name} hires")


B.ref, B.animate, B.hires = ref, animate, hires
B.STAGES = ["prep", "ref", "animate", "cutout", "post", "emoji", "compare", "full", "hires", "deliver"]
B.DONE["hires"] = "{n}-bold-hires.mp4"

if __name__ == "__main__":
    stage, names = sys.argv[1], sys.argv[2:] or list(B.MEMES)
    if stage == "all":
        B.run_all(names)
    else:
        for n in names:
            getattr(B, stage)(n)
