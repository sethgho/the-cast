#!/usr/bin/env python3
"""Add a cast member to site/art/hero.png without redrawing anyone already on stage.

    python3 add_to_hero.py   # Swifton, 2026-10-17

A plain Qwen edit of the whole hero ("keep these four, add a fifth") REDRAWS the company:
measured, it produced two Cadburys and lost Wilson. So the hero is widened instead:

  1. split the stage at a clear column (x=888, between Seth's can and Wilson's lantern)
     and open a 480px gap - narrower clipped the coin he insists on holding out sideways;
  2. fill the gap with tiled real curtain above and a heavily blurred copy of the floor
     below (mirrored tiles leave a zigzag the model copies; a smear becomes a dark void);
  3. inpaint ONLY the gap (VAEEncode + SetLatentNoiseMask, denoise 1.0) with the
     character's plate as image 2, then paste original pixels back everywhere else, so
     the other performers are byte-for-byte untouched;
  4. the tiled swag survives the inpaint, so replace it with the best-matching 480px swag
     segment from the original curtain, masking the orange flame crest out of the patch.
"""
import json, os, sys, time, urllib.request
import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S
import swifton_canon as C
from build_workflows import NEGATIVE, STEPS
from expression_run import upload

HERO = os.path.join(HERE, "..", "site", "art", "hero.png")
OUT = C.OUT
X, G, FLOOR, SWAG, SEED = 888, 480, 430, 112, 7
PROMPT = ("f0llie5, image 1 is a vaudeville stage with an empty gap in the middle between two performers. Fill the gap "
          "with the griffon-phoenix from image 2, standing on the same stage floor in front of the same curtain, taking a "
          "showman's bow toward the audience. He is NO TALLER than the long-haired man beside him and his feet rest on "
          "the floorboards at the same height as theirs. The wooden stage floorboards run continuously across, matching "
          "the boards on either side; the sepia curtain and its draped top swag continue smoothly across behind him, "
          "matching both sides exactly. He holds the burning W coin up in front of his chest, fully in view: "
          f"{C.TRAITS}. Drawn on aged paper.")


def widen(im):
    W, H = im.size
    band = im.crop((1138, 0, 1178, FLOOR))
    fill = Image.new("RGB", (G, H))
    for x in range(0, G, 40):
        fill.paste(band, (x, 0))
    fill.paste(im.crop((X - 60, FLOOR, X + 60, H)).resize((G, H - FLOOR)).filter(ImageFilter.GaussianBlur(18)), (0, FLOOR))
    wide = Image.new("RGB", (W + G, H))
    wide.paste(im.crop((0, 0, X, H)), (0, 0)); wide.paste(fill, (X, 0)); wide.paste(im.crop((X, 0, W, H)), (X + G, 0))
    return wide


def inpaint(crop_path, mask_path):
    g = {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": C.LORA, "strength_model": C.STRENGTH, "model": ["3", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "6": {"class_type": "LoadImage", "inputs": {"image": upload(crop_path)}},
     "7": {"class_type": "LoadImage", "inputs": {"image": upload(f"{OUT}/plate2-s42.png")}},
     "8": {"class_type": "LoadImageMask", "inputs": {"image": upload(mask_path), "channel": "red"}},
     "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["5", 0]}},
     "13": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["12", 0], "mask": ["8", 0]}},
     "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": PROMPT, "clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0], "image2": ["7", 0]}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE, "clip": ["4", 0]}},
     "22": {"class_type": "KSampler", "inputs": {"seed": SEED, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["13", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
     "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": "swifton/hero-inpaint", "images": ["23", 0]}},
    }
    pid = S.api("/prompt", {"prompt": g, "client_id": "hero"})["prompt_id"]
    while pid not in (h := S.api(f"/history/{pid}")):
        time.sleep(4)
    o = h[pid]["outputs"]["99"]["images"][0]
    dest = f"{OUT}/hero-inpaint.png"
    urllib.request.urlretrieve(S.HOST + f"/view?filename={o['filename']}&subfolder={o['subfolder']}&type=output", dest)
    return Image.open(dest).convert("RGB")


def reswag(orig, fin):
    left = np.array(fin.crop((X - 4, 0, X, SWAG))).astype(int).mean(axis=1)
    right = np.array(fin.crop((X + G, 0, X + G + 4, SWAG))).astype(int).mean(axis=1)
    def cost(x):
        seg = np.array(orig.crop((x, 0, x + G, SWAG))).astype(int)
        return ((seg[:, :4].mean(axis=1) - left) ** 2).sum() + ((seg[:, -4:].mean(axis=1) - right) ** 2).sum()
    bx = min(range(150, orig.width - G - 150, 4), key=cost)
    f = np.array(fin.crop((X, 0, X + G, SWAG))).astype(int)
    flame = (f[..., 0] > 150) & (f[..., 0] - f[..., 2] > 70) & (f[..., 1] > 60)
    fl = Image.fromarray((flame * 255).astype("uint8")).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(2))
    ramp = np.ones((SWAG, 1)); ramp[SWAG - 14:, 0] = np.linspace(1, 0, 14)
    mask = Image.fromarray(((255 - np.array(fl).astype(int)) * ramp).clip(0, 255).astype("uint8"))
    fin.paste(orig.crop((bx, 0, bx + G, SWAG)), (X, 0), mask)
    return fin


if __name__ == "__main__":
    orig = Image.open(HERO).convert("RGB")
    wide = widen(orig); H = wide.height; L = X - 240
    crop = Image.new("RGB", (960, 608)); crop.paste(wide.crop((L, 0, L + 960, H)), (0, 0)); crop.save(f"{OUT}/hero-gap-crop.png")
    m = Image.new("L", (960, 608), 0); m.paste(255, (238, 0, 240 + G + 2, 608)); m.save(f"{OUT}/hero-gap-mask.png")
    inp = inpaint(f"{OUT}/hero-gap-crop.png", f"{OUT}/hero-gap-mask.png").resize((960, 608)).crop((0, 0, 960, H))
    region = wide.crop((L, 0, L + 960, H))
    region.paste(inp, (0, 0), m.crop((0, 0, 960, H)).filter(ImageFilter.GaussianBlur(3)))
    wide.paste(region, (L, 0))
    reswag(orig, wide).save(f"{OUT}/hero-with-swifton.png")
    print(f"{OUT}/hero-with-swifton.png", wide.size)
