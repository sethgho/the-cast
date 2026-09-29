#!/usr/bin/env python3
"""Jenmojis: the sethmoji pipeline for Jen, photorealistic.

    python3 jenmoji.py <stage|all> [names]   # stages as sethmoji, plus hires

Same stages and emoji packing as sethmoji.py; what changes is who and how she is drawn. No style
at all: the reference frame is the meme's first frame with her put into it, likeness taken from
two of her photographs (images 2 and 3 of the Qwen edit). A children's-book style was tried
first and dropped -- stylised, it stopped looking like her. Her photographs live only on
gpu-worker (input/jenmoji-photo-*.png) and in scratch -- never commit them.
"""
import argparse
import os
import sys
import time
import urllib.request
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sethmoji as B  # noqa: E402

B.ROOT = "/home/wilson/scratch/jenmoji"
B.WHO = "jen"
B.PREFIX = "jenmoji"
B.MAC_DIR = "Documents/avatars/jenmojis"

JEN = ("dark brown hair pulled loosely back with long side-swept bangs across her forehead, "
       "light blue-grey eyes, a slim face with defined cheekbones, faint freckles and small gold "
       "hoop earrings")
PHOTOS = ["jenmoji-photo-a.png", "jenmoji-photo-b.png"]
NEG = ("cartoon, illustration, painting, drawing, anime, 3d render, plastic skin, airbrushed, "
       "blurry, deformed, extra limbs, watermark, text")

B.MEMES = {
    "kombucha": dict(pick=0, seed=7, caption=0.0, top=0.62,
                     frame="The camera is as far away as in image 1: her head and shoulders fill only the left two thirds of the picture, and the white ceiling and the white doors of the room are clearly visible behind her.",
                     face="head turned three-quarters away to the side, eyes looking off to the side, mouth closed, a flat neutral expression, NOT smiling",
                     scene="In a bedroom, keeping the same clothing",
                     action="She sips kombucha, grimaces in disgust, reconsiders, then decides "
                            "she actually likes it"),
}


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
     "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"jenmoji/{prefix}", "images": ["23", 0]}},
    }
    for k, img in enumerate(images, 1):
        g[f"i{k}"] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g["20"]["inputs"][f"image{k}"] = [f"i{k}", 0]
    return B.submit(g, "99", "images", dest)


def ref(name):
    m, dd = B.MEMES[name], B.d(name)
    spec = json.load(open(f"{dd}/spec.json"))
    B.sh("scp", "-q", f"{dd}/frame0.png", f"{B.GPU}:comfyui/input/jenmoji-{name}-f0.png")
    prompt = ("Image 1 is a photograph. Replace the woman in image 1 with the woman in images 2 "
              "and 3: it must be recognisably the same real person, with her exact face, eyes, "
              f"nose, jaw, skin, hair and bangs ({JEN}). Everything else in image 1 is already "
              "exactly right and must not change: the pose, the tilt and turn of the head, the "
              "direction the eyes are looking, the clothing, the room, the lighting, the camera "
              "angle, the crop and the size of the head in the frame. Her expression: "
              f"{m['face']}. {m.get('frame', '')} A natural, unretouched phone-camera "
              "photograph, photorealistic, with real skin texture. There is no text anywhere.")
    qwen_edit([f"jenmoji-{name}-f0.png"] + PHOTOS, prompt, spec["W"], spec["H"], f"{name}-ref",
              f"{dd}/ref.png", seed=m.get("seed", 7))
    B.sh("scp", "-q", f"{dd}/ref.png", f"{B.GPU}:comfyui/input/jenmoji-{name}-ref.png")
    print(f"{name} ref")


def animate(name):
    m, dd = B.MEMES[name], B.d(name)
    spec = json.load(open(f"{dd}/spec.json"))
    a = argparse.Namespace(
        ref=f"jenmoji-{name}-ref.png", drive=f"jenmoji-{name}-drive.mp4",
        prompt=f"A photorealistic video of a woman with {JEN}. {m['scene']}. {m['action']}.",
        pose_prompt=f"{m['action'].replace('She ', 'A person ')}. Static camera.",
        out=f"jenmoji/{name}", model="wan_animate_2_distill_int8_convrot.safetensors",
        lora=None, width=spec["W"], height=spec["H"], length=spec["frames"], fps=B.FPS, steps=6,
        # a real face deforms at the defaults, unlike a cartoon one; full reference
        # strength holds her likeness through the grimace
        seed=77, cache="cpu", pose_strength=1.0, ref_strength=1.0)
    urllib.request.urlopen(urllib.request.Request(
        B.S.HOST + "/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
        {"Content-Type": "application/json"}), timeout=60).read()
    t0 = time.time()
    B.submit(B.animate_graph(a), "21", "images", f"{dd}/seth.mp4")
    print(f"{name} animate {time.time()-t0:.0f}s")


def hires(name):
    """RealESRGAN x4, then back down to 2x: the render is ~0.4MP, the 12GB ceiling for Wan
    Animate, and this is the version to watch full screen."""
    dd = B.d(name)
    g = {
     "1": {"class_type": "LoadVideo", "inputs": {"file": f"jenmoji-{name}-seth.mp4"}},
     "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
     "3": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x4plus.pth"}},
     "4": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["3", 0], "image": ["2", 0]}},
     "5": {"class_type": "ImageScaleBy", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "scale_by": 0.5}},
     "6": {"class_type": "CreateVideo", "inputs": {"images": ["5", 0], "fps": float(B.FPS)}},
     "7": {"class_type": "SaveVideo", "inputs": {"video": ["6", 0], "filename_prefix": f"jenmoji/{name}-hires", "format": "mp4", "codec": "h264"}},
    }
    B.submit(g, "7", "images", f"{dd}/{name}-jen-hires.mp4")
    print(f"{name} hires")


B.ref, B.animate, B.hires = ref, animate, hires
B.STAGES = ["prep", "ref", "animate", "cutout", "post", "emoji", "compare", "hires"]
B.DONE["hires"] = "{n}-jen-hires.mp4"

if __name__ == "__main__":
    stage, names = sys.argv[1], sys.argv[2:] or list(B.MEMES)
    if stage == "all":
        B.run_all(names)
    else:
        for n in names:
            getattr(B, stage)(n)
