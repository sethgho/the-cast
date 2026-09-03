#!/usr/bin/env python3
"""The Follies Supper: thirteen cast members in Leonardo's composition, three ways.

    python3 build_supper.py            # writes all variants

Thirteen figures across 1920x832 puts each face at roughly sixty pixels. Identity is therefore
the whole problem, and the three variants differ ONLY in how they try to hold it:

  a-text     No reference at all. Every character described in words, one clause each. The
             control: whatever consistency appears is the prompt's alone.
  b-sheet    The four canon plates stitched into one contact sheet and handed in as image1. One
             reference carrying four subjects -- which the prompting laws say is exactly the
             dangerous case (a second character in image2 once merged Wilson and Ake into a
             single creature). Worth testing because a SHEET is not the same as a second image:
             the characters are already separated from each other inside it.
  c-h3ref    MiniMaxH3ReferenceToVideo with the four plates as four SEPARATE references, one
             frame pulled out. The only mechanism here actually designed for multiple subjects,
             and the only different model on the box.

Nine cast members are invented to fill the table, and they are worded identically in all three so
the comparison is about the mechanism rather than the description.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import (  # noqa: E402
    Graph, WIDGETS, WIDGET_TYPES, STYLE_LOCK, NEGATIVE, STEPS, BLUE, GREEN, GREY,
)
from build_extras import engine, note  # noqa: E402
from build_transition import H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE  # noqa: E402

W, H, SEED = 1920, 832, 7
# THE CANON LIVES IN THE REPO, NOT IN ComfyUI's INPUT FOLDER.
# cast/<id>/assets/poses/standing-neutral.png is the reference, published at
# https://cast.sethgholson.com/<id>/assets.json with provenance in cast/<id>/assets.yaml.
# canon-*.png in ~/comfyui/input are copies of those files and nothing else; re-copy them when
# the repo moves. The old rubberhose-kf-*.png plates were a local invention that drifted off
# model and are retired to input/_retired.
PLATES = ["canon-seth.png", "canon-wilson.png",
          "canon-cadbury.png", "canon-ake.png"]

WIDGETS.setdefault("ImageStitch",
                   ["direction", "match_image_size", "spacing_width", "spacing_color"])
for _w, _t in (("direction", "COMBO"), ("match_image_size", "BOOLEAN"),
               ("spacing_width", "INT"), ("spacing_color", "COMBO")):
    WIDGET_TYPES.setdefault(("ImageStitch", _w), _t)
for _n, _ws in (("UNETLoader", ["unet_name", "weight_dtype"]),
                ("MiniMaxH3TurboLoRA", ["lora_name", "strength", "low_vram"]),
                ("MiniMaxH3ReferenceToVideo",
                 ["prompt", "width", "height", "length", "ref_image_size"]),
                ("RandomNoise", ["noise_seed"]), ("KSamplerSelect", ["sampler_name"]),
                ("BasicScheduler", ["scheduler", "steps", "denoise"]),
                ("BasicGuider", []), ("SamplerCustomAdvanced", []),
                ("ImageFromBatch", ["batch_index", "length"])):
    WIDGETS.setdefault(_n, _ws)
for _k, _t in ((("MiniMaxH3ReferenceToVideo", "length"), "INT"),
               (("MiniMaxH3ReferenceToVideo", "width"), "INT"),
               (("MiniMaxH3ReferenceToVideo", "height"), "INT"),
               (("MiniMaxH3ReferenceToVideo", "ref_image_size"), "COMBO"),
               (("MiniMaxH3ReferenceToVideo", "prompt"), "STRING"),
               (("RandomNoise", "noise_seed"), "INT"),
               (("BasicScheduler", "steps"), "INT")):
    WIDGET_TYPES.setdefault(_k, _t)

# ---------------------------------------------------------------- the table
#
# Four canon members and nine invented ones, seated left to right. Leonardo groups the twelve into
# four clusters of three flanking the centre, so they are written in those clusters: the model
# holds a group of three far better than a list of twelve, and the clusters are what make the
# composition read as the painting rather than as a row of people.
CENTRE = ("At the exact centre, alone and framed by the middle window, sits SETH: a lanky cartoon "
          "man with long wavy shoulder-length brown hair and a full bushy handlebar moustache in a "
          "dark v-neck t-shirt, arms open and palms down on the table, calm.")
GROUPS = [
    ("Far left group of three: OLD MAN CRANK, a bent stagehand in patched overalls clutching a "
     "mop; MOPSY, a hobo clown in a moth-eaten overcoat with a squeezebox on his knee; and "
     "SERGEANT WHISTLE, a bird-legged comic policeman swamped by an oversized custodian helmet."),
    ("Inner left group of three: CADBURY, a thin composed butler in a black tailcoat with slicked "
     "hair, a pencil moustache and a brass clockwork monocle, half-risen with a napkin; AKE, a "
     "plump goldfish in a spherical glass fishbowl on a small wheeled wooden dolly wearing a "
     "little dark cap, the bowl up on the table; and MISS PEARL BELLWEATHER, a tall soprano in a "
     "beaded gown with her hair in a tight marcel wave, one hand to her throat."),
    ("Inner right group of three: WILSON, a living picket-fence panel — the fence panel is his "
     "body — with a floppy bucket hat on top and two cartoon eyes floating in the shadow under its "
     "brim, leaning in stiffly because he has no waist; THE GREAT BOMBARDO, a stout strongman in a "
     "leopard-print singlet with a waxed moustache, both fists on the table; and DOC FENNIMORE, a "
     "patent-medicine barker in a loud checked suit and bowler hat, mid-gesture."),
    ("Far right group of three: LULU PINWHEEL, a plate-spinner in bloomers and boots with three "
     "plates still turning on sticks; VERA FANDANGO, a dancer in a fringed dress holding "
     "castanets; and RUFUS TANGLE, a lanky contortionist in a striped union suit with his legs "
     "folded up impossibly beside him."),
]
COMPOSITION = (
    "A wide horizontal painting recreating the composition of Leonardo's Last Supper. Thirteen "
    "characters sit along one long table that runs straight across the picture, parallel to the "
    "viewer, covered in a white cloth with bread, tin cups and a scatter of playing cards. Behind "
    "them a plain plaster wall with three tall arched windows, the middle one directly behind the "
    "central figure, opening onto a pale evening sky. Deep one-point perspective: the ceiling "
    "beams and the side walls converge on the central figure's head. Everyone is seated or leaning "
    "along the far side of the table, nobody in front of it, all thirteen visible and none "
    "overlapping another's face.")
TAIL = ("Thirteen characters exactly, evenly spread across the full width of the frame. Every face "
        "is drawn clearly and each character is distinct from the others.")

PROMPT = " ".join([COMPOSITION, CENTRE] + GROUPS + [STYLE_LOCK, TAIL])
NEG = NEGATIVE + ", extra people, crowd, duplicated faces, merged characters, empty seats"

HOW_TO = """# The Follies Supper

**Thirteen cast members in Leonardo's composition, 1920&times;832.**

At this width each face is about sixty pixels, so **identity is the whole problem** and the three
variants differ only in how they try to hold it.

| Variant | Mechanism |
|---|---|
| **a-text** | No reference. Everything in words. The control. |
| **b-sheet** | The four canon plates stitched into one contact sheet, handed in as `image1`. |
| **c-h3ref** | H3 Ref2VA with the four plates as four separate references, one frame pulled out. |

The cast is written in **four groups of three flanking the centre**, which is how Leonardo groups
the twelve. That is not decoration: the model holds a group of three far better than a list of
twelve, and the clusters are what make it read as the painting instead of a row of people.

Nine of the thirteen are invented to fill the table. They are worded identically across all three
variants, so the comparison is about the mechanism and not the description.
"""


def qwen_variant(sheet):
    g = Graph()
    note(g, f"HOW TO USE — the Follies Supper ({'b-sheet' if sheet else 'a-text'})", HOW_TO)
    e = engine(g, -60, -180, want_bg=False)

    img1 = None
    if sheet:
        # Stitched in-graph rather than prepared offline, so the sheet's layout is a widget.
        prev = None
        for i, pl in enumerate(PLATES):
            n = g.add("LoadImage", f"plate {i+1}", (-60, 60 + i * 200), (380, 180),
                      {"image": pl, "upload": "image"},
                      outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], color=BLUE, collapsed=True)
            if prev is None:
                prev = n
                continue
            prev = g.add("ImageStitch", f"stitch {i}", (360, 60 + i * 120), (380, 200),
                         {"direction": "right", "match_image_size": True, "spacing_width": 16,
                          "spacing_color": "white"},
                         links={"image1": (prev, 0, "IMAGE", False), "image2": (n, 0, "IMAGE", False)},
                         outputs=[("IMAGE", "IMAGE")], collapsed=True)
        img1 = g.add("ImageScale", "▶ REFERENCE SHEET", (760, 60), (400, 170),
                     {"upscale_method": "lanczos", "width": 1536, "height": 512, "crop": "disabled"},
                     links={"image": (prev, 0, "IMAGE", False)},
                     outputs=[("IMAGE", "IMAGE")], color=BLUE)
        g.app_output(g.add("PreviewImage", "the sheet the model sees", (760, 260), (420, 300),
                           links={"images": (img1, 0, "IMAGE", False)}, color=GREY))

    prompt_txt = PROMPT
    if sheet:
        prompt_txt = ("Image 1 is a reference sheet showing four of the characters side by side; "
                      "draw those four exactly as they appear there and place them in the scene "
                      "described below. Do not copy the sheet's layout. " + PROMPT)
    pnode = g.add("PrimitiveStringMultiline", "▶ THE PICTURE", (1240, 60), (500, 600),
                  {"value": prompt_txt}, outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(pnode, "value")

    links = {"prompt": (pnode, 0, "STRING", True), "clip": (e["clip"], 0, "CLIP", False),
             "vae": (e["vae"], 0, "VAE", False)}
    if img1 is not None:
        links["image1"] = (img1, 0, "IMAGE", False)
    pos = g.add("TextEncodeQwenImageEditPlus", "positive", (1780, 60), (420, 120), links=links,
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    neg = g.add("TextEncodeQwenImageEditPlus", "negative", (1780, 120), (420, 120), {"prompt": NEG},
                links={"clip": (e["clip"], 0, "CLIP", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    lat = g.add("EmptySD3LatentImage", "canvas", (1780, 180), (380, 130),
                {"width": W, "height": H, "batch_size": 1},
                outputs=[("LATENT", "LATENT")], collapsed=True)
    ks = g.add("KSampler", "▶ SEED / STEPS", (1780, 240), (400, 280),
               {"seed": SEED, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
               links={"model": (e["lora"], 0, "MODEL", False),
                      "positive": (pos, 0, "CONDITIONING", False),
                      "negative": (neg, 0, "CONDITIONING", False),
                      "latent_image": (lat, 0, "LATENT", False)},
               outputs=[("LATENT", "LATENT")], color=GREEN)
    g.app_input(ks, "seed", "steps")
    dec = g.add("VAEDecode", "decode", (1780, 540), (300, 60),
                links={"samples": (ks, 0, "LATENT", False), "vae": (e["vae"], 0, "VAE", False)},
                outputs=[("IMAGE", "IMAGE")], collapsed=True)
    out = g.add("SaveImage", "RESULT", (2240, 60), (620, 620),
                {"filename_prefix": f"supper/{'b-sheet' if sheet else 'a-text'}"},
                links={"images": (dec, 0, "IMAGE", False)}, color=GREY)
    g.app_output(out)
    return g


def h3_variant(wide=False):
    """Four separate references, a very short clip, one frame kept. H3 is the only engine here
    designed to take more than one subject reference at a time."""
    g = Graph()
    note(g, "HOW TO USE — the Follies Supper (c-h3ref)", HOW_TO)
    unet = g.add("UNETLoader", "H3 reference model", (-60, -180), (420, 110),
                 {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                  "weight_dtype": "default"},
                 outputs=[("MODEL", "MODEL")], color=GREY, collapsed=True)
    turbo = g.add("MiniMaxH3TurboLoRA", "turbo LoRA", (-60, -130), (420, 140),
                  {"lora_name": "minimax_h3_turbo_4step.safetensors", "strength": 1.0,
                   "low_vram": True},
                  links={"model": (unet, 0, "MODEL", False)},
                  outputs=[("MODEL", "MODEL")], color=GREY, collapsed=True)
    clip = g.add("CLIPLoader", "text encoder", (-60, -80), (420, 130),
                 {"clip_name": H3_CLIP, "type": "minimax", "device": "default"},
                 outputs=[("CLIP", "CLIP")], color=GREY, collapsed=True)
    vvae = g.add("VAELoader", "video VAE", (-60, -30), (420, 80), {"vae_name": H3_VIDEO_VAE},
                 outputs=[("VAE", "VAE")], color=GREY, collapsed=True)
    avae = g.add("VAELoader", "audio VAE", (-60, 20), (420, 80), {"vae_name": H3_AUDIO_VAE},
                 outputs=[("VAE", "VAE")], color=GREY, collapsed=True)

    plate_nodes = []
    for i, pl in enumerate(PLATES):
        n = g.add("LoadImage", f"▶ REFERENCE {i+1}", (-60, 100 + i * 200), (380, 180),
                  {"image": pl, "upload": "image"},
                  outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], color=BLUE, collapsed=True)
        plate_nodes.append(n)

    txt = ("A completely still tableau, nobody moves and the camera does not move. " + PROMPT)
    pnode = g.add("PrimitiveStringMultiline", "▶ THE PICTURE", (420, 60), (500, 600),
                  {"value": txt}, outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(pnode, "value")
    cond = g.add("MiniMaxH3ReferenceToVideo", "▶ SIZE", (960, 60), (440, 280),
                 {"prompt": "", "width": 1920 if wide else 1152,
                  "height": 832 if wide else 640, "length": 22, "ref_image_size": "max"},
                 links={"prompt": (pnode, 0, "STRING", True), "clip": (clip, 0, "CLIP", False),
                        "vae": (vvae, 0, "VAE", False), "audio_vae": (avae, 0, "VAE", False)},
                 outputs=[("positive", "CONDITIONING"), ("LATENT", "LATENT")], color=BLUE)
    g.app_input(cond, "length", "width", "height", "ref_image_size")
    cond["inputs"].append({"name": "ref_images", "type": "COMFY_AUTOGROW_V3", "link": None})

    noise = g.add("RandomNoise", "▶ SEED", (960, 360), (400, 130), {"noise_seed": SEED},
                  outputs=[("NOISE", "NOISE")], color=GREEN)
    guider = g.add("BasicGuider", "guider", (960, 500), (320, 80),
                   links={"model": (turbo, 0, "MODEL", False),
                          "conditioning": (cond, 0, "CONDITIONING", False)},
                   outputs=[("GUIDER", "GUIDER")], collapsed=True)
    samp = g.add("KSamplerSelect", "sampler", (960, 560), (320, 80),
                 {"sampler_name": "res_multistep"}, outputs=[("SAMPLER", "SAMPLER")], collapsed=True)
    sched = g.add("BasicScheduler", "▶ STEPS", (960, 620), (400, 160),
                  {"scheduler": "simple", "steps": 12, "denoise": 1.0},
                  links={"model": (turbo, 0, "MODEL", False)},
                  outputs=[("SIGMAS", "SIGMAS")], color=GREEN)
    adv = g.add("SamplerCustomAdvanced", "sample", (960, 800), (340, 100),
                links={"noise": (noise, 0, "NOISE", False), "guider": (guider, 0, "GUIDER", False),
                       "sampler": (samp, 0, "SAMPLER", False), "sigmas": (sched, 0, "SIGMAS", False),
                       "latent_image": (cond, 1, "LATENT", False)},
                outputs=[("output", "LATENT"), ("denoised_output", "LATENT")], collapsed=True)
    dec = g.add("VAEDecode", "decode", (1420, 60), (300, 60),
                links={"samples": (adv, 0, "LATENT", False), "vae": (vvae, 0, "VAE", False)},
                outputs=[("IMAGE", "IMAGE")], collapsed=True)
    pick = g.add("ImageFromBatch", "keep one frame", (1420, 130), (380, 140),
                 {"batch_index": 10, "length": 1},
                 links={"image": (dec, 0, "IMAGE", False)},
                 outputs=[("IMAGE", "IMAGE")], color=GREEN)
    out = g.add("SaveImage", "RESULT", (1420, 300), (620, 480),
                {"filename_prefix": "supper/d-h3wide" if wide else "supper/c-h3ref"},
                links={"images": (pick, 0, "IMAGE", False)}, color=GREY)
    g.app_output(out)
    return g, plate_nodes


if __name__ == "__main__":
    builds = {"supper-a-text": (qwen_variant, False), "supper-b-sheet": (qwen_variant, True)}
    for stem, (fn, arg) in builds.items():
        g = fn(arg)
        for path, blob in ((os.path.join(HERE, "workflows", f"{stem}.json"), g.to_ui()),
                           (os.path.join(HERE, "workflows", f"{stem}.app.json"), g.to_ui()),
                           (os.path.join(HERE, "api", f"{stem}.api.json"), g.to_api())):
            json.dump(blob, open(path, "w"), indent=1)
        print("wrote", stem)

    for wide, stem in ((False, "supper-c-h3ref"), (True, "supper-d-h3wide")):
      g, plates = h3_variant(wide)
      ui, api = g.to_ui(), g.to_api()
      loads = [k for k, v in api.items() if v["class_type"] == "LoadImage"]
      for n in api.values():
          if n["class_type"] == "MiniMaxH3ReferenceToVideo":
              n["inputs"]["ref_images"] = {f"ref_image_{i}": [k, 0] for i, k in enumerate(loads)}
      for path, blob in ((os.path.join(HERE, "workflows", f"{stem}.json"), ui),
                         (os.path.join(HERE, "workflows", f"{stem}.app.json"), ui),
                         (os.path.join(HERE, "api", f"{stem}.api.json"), api)):
          json.dump(blob, open(path, "w"), indent=1)
      print("wrote", stem)
