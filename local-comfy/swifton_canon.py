#!/usr/bin/env python3
"""Swifton's canon, drawn locally with the Follies style LoRA.

    python3 swifton_canon.py plate      # full-figure base plates, several seeds
    python3 swifton_canon.py views      # turnaround + expressions from the chosen plate
    python3 swifton_canon.py headshot   # square portrait for avatars

Swifton is the SwiftUI agent who fights grifters: a griffon-phoenix who burns $WHUFFIE coins
and keeps an ace up his sleeve. His design starts from Seth's logo avatar (image 1 of the plate
run), which is a full-colour vector mark; the strip is sepia with Ake's ochre as its only accent,
so the fire is drawn in ink here, never in red.

Every later view is an edit FROM the chosen base plate, never an independent generation, per the
cast rule: independent generation re-stages the character.
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S
from build_workflows import NEGATIVE, STEPS
from expression_run import upload

LORA, STRENGTH = "follies-final.safetensors", 1.2
OUT = "/home/wilson/scratch/swifton"

TRAITS = ("an upright anthropomorphic griffon-phoenix: a fierce eagle's head with a white face, "
          "a hooked beak and one stern narrowed eye, a crest of long flame-shaped feathers swept "
          "back from his brow, broad wings whose feather tips curl up into licks of flame, a "
          "lion's lower body with sturdy hind legs and a long tail ending in a flame tuft. He "
          "wears a card sharp's striped waistcoat and rolled shirtsleeves held by sleeve "
          "garters, and the corner of an ace of spades peeks out of his left cuff. In one taloned "
          "hand he holds up a large gold coin stamped with a bold letter W, curling up in flames")

PROPORTION = ("Draw him in this style's own cartoon proportions: a large rounded head about one "
              "fifth of his standing height, small simple features, chunky simplified limbs. This "
              "is a new drawing, not a traced or filtered copy of image 1.")


def run(name, prompt, image1=None, seed=7, w=1024, h=1024):
    dest = f"{OUT}/{name}.png"
    if os.path.exists(dest):
        print(name, "cached"); return dest
    g = {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": LORA, "strength_model": STRENGTH, "model": ["3", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE, "clip": ["4", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
    }
    enc = {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0]}
    if image1:
        g["6"] = {"class_type": "LoadImage", "inputs": {"image": upload(image1)}}
        enc["image1"] = ["6", 0]
    g["20"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": enc}
    g["22"] = {"class_type": "KSampler", "inputs": {"seed": seed, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}}
    g["99"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"swifton/{name}", "images": ["23", 0]}}
    r = S.api("/prompt", {"prompt": g, "client_id": "swifton"})
    if "prompt_id" not in r:
        print(name, "REJECTED", json.dumps(r)[:400]); return None
    pid = r["prompt_id"]
    while True:
        hist = S.api(f"/history/{pid}")
        if pid in hist: break
        time.sleep(4)
    if hist[pid]["status"].get("status_str") != "success":
        print(name, "FAILED"); return None
    o = hist[pid]["outputs"]["99"]["images"][0]
    with urllib.request.urlopen(S.HOST + f"/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output", timeout=300) as rr:
        open(dest, "wb").write(rr.read())
    print(name, "->", dest)
    return dest


def plate():
    src = f"{OUT}/source-logo.png"
    for seed in (7, 21, 42, 99):
        run(f"plate-s{seed}",
            f"f0llie5, image 1 is a colour logo of a fiery griffon. Draw that creature as a new "
            f"full-body cartoon character for the cast: {TRAITS}. He stands at rest facing the "
            f"viewer, head to feet in frame, chest out like a showman. {PROPORTION} Drawn on a "
            f"plain flat mid-grey background.", image1=src, seed=seed)




BASE = f"{OUT}/plate2-s42.png"          # Seth's pick, 2026-10-16: option A, orange flames kept
EDIT = ("f0llie5, image 1 is the drawing to edit and the griffon in image 1 is the only character "
        "in the finished picture. Redraw him exactly as he is, same design, same colours, same "
        "waistcoat, same orange flames: ")
GREY = "on a plain flat mid-grey background, nothing else in the frame."

VIEWS = {
    "view-front": "standing at rest facing the viewer, wings half open, holding the burning W coin up in one hand, full body head to feet",
    "view-3q-front": "standing at rest in three-quarter view turned to the viewer's left, wings folded, holding the burning W coin, full body head to feet",
    "view-side": "standing at rest in strict side profile facing the viewer's left, wings folded, the burning coin held in front of him, full body head to feet",
    "view-3q-back": "standing at rest in three-quarter rear view turned away from the viewer, wings folded, his flame-tufted tail toward us, full body head to feet",
    "view-back": "standing at rest seen directly from behind, wings folded, flame-tufted tail hanging down, full body head to feet",
}
EXPRESSIONS = {
    "expr-smug": "a smug, self-satisfied grin, one brow raised, head and shoulders only, facing the viewer",
    "expr-furious": "furious, glaring at a grifter, beak open in a shout, crest flaring higher, head and shoulders only, facing the viewer",
    "expr-sly": "a sly wink, flashing the ace of spades from his cuff beside his face, head and shoulders only, facing the viewer",
    "expr-triumphant": "triumphant laughter, head thrown back, holding the burning W coin aloft, head and shoulders only, facing the viewer",
}
PROPS = {
    "prop-coin": "Show only his gold coin stamped with a bold letter W, curling up in orange flames, alone and centred, with no character in the frame",
    "prop-ace": "Show only a single ace of spades playing card, slightly tilted, alone and centred, with no character in the frame",
}


def views():
    for name, act in {**VIEWS, **EXPRESSIONS}.items():
        run(name, f"{EDIT}{TRAITS}. He is {act}, {GREY}", image1=BASE, seed=7)
    for name, what in PROPS.items():
        run(name, f"f0llie5, image 1 shows a griffon character and his props. {what}, {GREY}",
            image1=BASE, seed=7)


def headshot():
    run("headshot", f"{EDIT}{TRAITS}. A head-and-shoulders portrait in three-quarter view, his "
        "flame crest and the tops of his wings in frame, the burning W coin held up beside his "
        "beak, in front of a softly lit vaudeville stage curtain.", image1=BASE, seed=7)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"plate": plate, "views": views, "headshot": headshot}[sys.argv[1]]()
