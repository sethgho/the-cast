#!/usr/bin/env python3
"""Huff's canon, drawn locally with the Follies style LoRA.

    python3 huff_canon.py views      # turnaround, expressions, props and headshot from the base plate
    python3 huff_canon.py title      # Swifton's blanked sheet layout, retitled HUFF
    python3 huff_canon.py cut        # BiRefNet mattes for every view, expression and prop

Huff replaced Swifton on 2026-11-11 (Seth: "Swifton never caught on"). He is the Wolf of Wall
Street meme played as a 1933 cartoon: a wolf in a pinstripe suit, and the troupe's grift hunter
because it takes a wolf to catch a wolf. His name is the middle of $WHUFFIE.

The base plate is Seth's pick from the concept renders: the pen held up like a product, the round
W lapel pin on his left lapel. The pin's W is the model's own serif letter; a pasted-in copy of the
real Whuffie mark looked photoshopped and was rejected, so do not composite the logo in.

Every later view is an edit FROM the base plate, never an independent generation, per the cast
rule: independent generation re-stages the character.
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S
from build_workflows import BGREMOVAL, NEGATIVE, STEPS
from expression_run import upload

LORA, STRENGTH = "follies-final.safetensors", 1.2
ANGLES_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
OUT = "/home/wilson/scratch/huff"
BASE = f"{OUT}/base.png"
HEAD = f"{OUT}/head-base.png"      # the approved head-and-shoulders render, pin on the lapel
BLANK = "/home/wilson/scratch/swifton/sheet-blank-s7.png"

TRAITS = ("a lean, slim grey timber wolf standing upright like a man, with a long narrow muzzle, the "
          "fur on top of his head slicked straight back into a glossy black combed pompadour, narrow "
          "hooded eyes and a huge confident toothy salesman's grin, wearing a sharp double-breasted "
          "pinstripe suit with wide lapels, a mustard silk tie, a white pocket square, a round gold "
          "lapel pin with a letter W on his left lapel and a chunky gold wristwatch, his grey tail "
          "hanging behind him")
EDIT = ("f0llie5, image 1 is the drawing to edit and the wolf in image 1 is the only character in "
        "the finished picture. Redraw him exactly as he is, same design, same suit, same lapel pin, "
        "same pompadour: ")
GREY = "on a plain flat mid-grey background, nothing else in the frame."

VIEWS = {
    "view-front": "standing at rest facing the viewer, holding a fountain pen up beside his face, full body head to feet",
    "view-back": "standing at rest seen directly from behind, his tail hanging down, full body head to feet",
}
# The in-between angles need fal's multiple-angles LoRA. Asked for in words ("three-quarter view",
# "rotate the camera 45 degrees"), the edit only knows front, profile and back: the 3/4 views came
# back as a front and a back, and the grin went flat (Seth caught it, 2026-11-11). Each entry is
# (azimuth, seed); seeds are the measured picks, all facing the viewer's left like the old sheet.
# front-left ignored the LoRA on seeds 7 and 42 and stayed a front view.
ANGLED = {
    "view-3q-front": ("front-left quarter view", 99),
    "view-side": ("right side view", 7),
    "view-3q-back": ("back-right quarter view", 7),
}
# Expressions are edits of HEAD, not BASE: edited from the full-body plate, all four came back as
# the plate's own pose and grin, ignoring "head and shoulders only" (measured 2026-11-11).
EXPRESSIONS = {
    "expr-smug": "smug and self-satisfied, a closed-mouth sly smirk, one eyebrow raised high, eyes half closed",
    "expr-pitching": "mid-pitch, mouth wide open talking fast, eyebrows raised high and eager, eyes wide",
    "expr-outraged": "outraged and indignant, shouting with his mouth wide open, ears pinned flat back, brows furrowed down hard",
    "expr-triumphant": "laughing triumphantly, head tilted back, eyes squeezed shut, mouth wide open",
}
PROPS = {
    "prop-pen": "Show only his black and gold fountain pen, tilted diagonally, alone and centred, with no character in the frame",
    "prop-pin": "Show only his round gold lapel pin stamped with a bold letter W, large and centred, with no character in the frame",
}


def submit(g, prefix):
    r = S.api("/prompt", {"prompt": g, "client_id": "huff"})
    if "prompt_id" not in r:
        print(prefix, "REJECTED", json.dumps(r)[:400]); return None
    pid = r["prompt_id"]
    while pid not in (hist := S.api(f"/history/{pid}")):
        time.sleep(4)
    if hist[pid]["status"].get("status_str") != "success":
        print(prefix, "FAILED"); return None
    return hist[pid]["outputs"]["99"]["images"][0]


def fetch(o, dest):
    url = S.HOST + f"/view?filename={o['filename']}&subfolder={o.get('subfolder', '')}&type=output"
    with urllib.request.urlopen(url, timeout=300) as rr:
        open(dest, "wb").write(rr.read())


def run(name, prompt, image1=None, seed=7, w=1024, h=1024, angles=0.0):
    dest = f"{OUT}/{name}.png"
    if os.path.exists(dest):
        print(name, "cached"); return
    g = {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": LORA, "strength_model": STRENGTH, "model": ["3", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
     "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0]}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE, "clip": ["4", 0]}},
     "22": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
     "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"huff/{name}", "images": ["23", 0]}},
    }
    if angles:
        # fal's multiple-angles LoRA for 2511, prompted "<sks> <azimuth> <elevation> <distance>".
        g["31"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": ANGLES_LORA, "strength_model": angles, "model": ["30", 0]}}
        g["22"]["inputs"]["model"] = ["31", 0]
    if image1:
        g["6"] = {"class_type": "LoadImage", "inputs": {"image": upload(image1)}}
        g["20"]["inputs"]["image1"] = ["6", 0]
    t = time.time()
    o = submit(g, name)
    if o:
        fetch(o, dest)
        print(name, f"{time.time() - t:.0f}s", flush=True)


def views():
    for name, act in VIEWS.items():
        run(name, f"{EDIT}{TRAITS}. He is {act}, {GREY}", BASE)
    for name, (azimuth, seed) in ANGLED.items():
        run(name, f"<sks> {azimuth} eye-level shot wide shot. f0llie5, the same wolf in the pinstripe "
            f"suit, full body head to feet, {GREY}", BASE, seed=seed, angles=1.0)
    for name, act in EXPRESSIONS.items():
        run(name, f"{EDIT}{TRAITS}. Keep the same head-and-shoulders framing; change only his face: "
            f"he is {act}. {GREY}", HEAD)
    run("prop-pin", f"f0llie5, image 1 shows a wolf character and his props. {PROPS['prop-pin']}, {GREY}", BASE)
    # The pen as an edit of BASE kept the whole wolf in frame; drawn from words alone it isolates.
    run("prop-pen", f"f0llie5, {PROPS['prop-pen']}, {GREY}")
    run("headshot", f"{EDIT}{TRAITS}. A head-and-shoulders portrait in three-quarter view, the "
        "lapel pin clearly visible, flashing his toothy salesman's grin at the viewer, in front of a "
        "softly lit vaudeville stage curtain.", HEAD)


def title():
    run("sheet-blank", "f0llie5, image 1 is a blank character sheet. Keep everything exactly as it "
        "is, the paper, the border, the ribbon banner, every label and every empty panel. Change "
        "only the word on the ribbon banner: it now reads HUFF in the same bold black serif "
        "lettering, centred on the ribbon.", BLANK, w=1536, h=1024)


def cut():
    """RGBA cutouts: BiRefNet's mask from ComfyUI, joined with the render here."""
    from PIL import Image
    os.makedirs(f"{OUT}/cut", exist_ok=True)
    for name in [*VIEWS, *ANGLED, *EXPRESSIONS, *PROPS]:
        dest = f"{OUT}/cut/{name}.png"
        if os.path.exists(dest):
            continue
        g = {
         "1": {"class_type": "LoadImage", "inputs": {"image": upload(f"{OUT}/{name}.png")}},
         "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": BGREMOVAL}},
         "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
         "4": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
         "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"huff/mask-{name}", "images": ["4", 0]}},
        }
        o = submit(g, name)
        if not o:
            continue
        fetch(o, f"{OUT}/cut/mask-{name}.png")
        im = Image.open(f"{OUT}/{name}.png").convert("RGB")
        m = Image.open(f"{OUT}/cut/mask-{name}.png").convert("L").resize(im.size)
        im.putalpha(m)
        im.crop(m.getbbox()).save(dest)
        print("cut", name, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"views": views, "title": title, "cut": cut}[sys.argv[1]]()
