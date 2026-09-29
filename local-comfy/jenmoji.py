#!/usr/bin/env python3
"""Jenmojis: the sethmoji pipeline for Jen, in a children's-book illustration style.

    python3 jenmoji.py style                 # Jen's portrait in the style, to settle the look
    python3 jenmoji.py <stage|all> [names]   # stages as sethmoji, plus hires

Same stages and emoji packing as sethmoji.py; what changes is who and how she is drawn. There
is no style LoRA here: the look is carried by words alone, on the plain Qwen edit model. Her
photograph lives only on gpu-worker (input/jenmoji-photo.png) and in scratch -- never commit it.
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

STYLE = ("a classic children's picture-book illustration: soft watercolour and gouache washes on "
         "textured paper, gentle coloured-pencil linework, a warm storybook palette, simplified "
         "rounded shapes and rosy cheeks, whimsical and friendly")
JEN = ("dark brown shoulder-length hair pulled loosely back, long side-swept curtain bangs over "
       "her forehead, light blue-grey eyes, a few faint freckles and small gold hoop earrings")

B.MEMES = {
    "kombucha": dict(pick=0, seed=7, caption=0.0, top=0.62,
                     frame="The camera is as far away as in image 1: her head and shoulders fill only the left two thirds of the picture, and the white ceiling, the smoke detector and the white doors of the room are clearly visible behind her.",
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
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": B.NEGATIVE + ", photograph, photorealistic, 3d render, text", "clip": ["4", 0]}},
     "22": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": B.STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["3", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
     "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"jenmoji/{prefix}", "images": ["23", 0]}},
    }
    for k, img in enumerate(images, 1):
        g[f"i{k}"] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g["20"]["inputs"][f"image{k}"] = [f"i{k}", 0]
    return B.submit(g, "99", "images", dest)


def style():
    out = f"{B.ROOT}/_style"
    for seed in (7, 21):
        qwen_edit(["jenmoji-photo.png"],
                  f"Redraw this photograph as {STYLE}. Keep her likeness exactly: the same "
                  "hair and bangs, the same eyes, the same big smile, the same earrings and the "
                  "same rust-orange t-shirt. Head and shoulders, on a plain soft cream "
                  "background. This is a painted illustration, not a filtered photograph.",
                  768, 1024, f"style-{seed}", f"{out}/jen-style-{seed}.png", seed=seed)
        print("style", seed)


def ref(name):
    m, dd = B.MEMES[name], B.d(name)
    spec = json.load(open(f"{dd}/spec.json"))
    B.sh("scp", "-q", f"{dd}/frame0.png", f"{B.GPU}:comfyui/input/jenmoji-{name}-f0.png")
    B.sh("scp", "-q", f"{B.ROOT}/_style/jen-style.png", f"{B.GPU}:comfyui/input/jenmoji-style.png")
    prompt = (f"Image 1 is a photograph to repaint as {STYLE}, in exactly the style of image 2. "
              "This is a repaint: the pose, the "
              "tilt of the head, the direction the eyes are looking, the hands and anything held "
              "in them, the clothing, the framing and the background are already exactly right "
              "and must not change, including the crop and the size of the head in the frame. "
              "Replace only the woman's face and hair with the woman from image 2: her face, "
              f"her hair and bangs and her earrings ({JEN}), seen from the angle of image 1. "
              "Her face: "
              f"{m['face']}. Keep the clothing exactly as in the photograph. Paint the whole "
              "picture, the background included, in the same storybook style. There is no text "
              f"anywhere in the picture. {m.get('frame', '')} Soft watercolour washes and "
              "gentle pencil lines, like image 2, not a comic book.")
    qwen_edit([f"jenmoji-{name}-f0.png", "jenmoji-style.png"], prompt, spec["W"], spec["H"], f"{name}-ref",
              f"{dd}/ref.png", seed=m.get("seed", 7))
    B.sh("scp", "-q", f"{dd}/ref.png", f"{B.GPU}:comfyui/input/jenmoji-{name}-ref.png")
    print(f"{name} ref")


def animate(name):
    m, dd = B.MEMES[name], B.d(name)
    spec = json.load(open(f"{dd}/spec.json"))
    a = argparse.Namespace(
        ref=f"jenmoji-{name}-ref.png", drive=f"jenmoji-{name}-drive.mp4",
        prompt=f"A woman with {JEN}, drawn as {STYLE}. {m['scene']}, all painted in the same "
               f"storybook style. {m['action']}.",
        pose_prompt=f"{m['action'].replace('She ', 'A person ')}. Static camera.",
        out=f"jenmoji/{name}", model="wan_animate_2_distill_int8_convrot.safetensors",
        lora=None, width=spec["W"], height=spec["H"], length=spec["frames"], fps=B.FPS, steps=6,
        seed=77, cache="cpu", pose_strength=2.0, ref_strength=0.6)
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
    if stage == "style":
        style()
    elif stage == "all":
        B.run_all(names)
    else:
        for n in names:
            getattr(B, stage)(n)
