#!/usr/bin/env python3
"""Three more local-ComfyUI apps for the cast, off the same Graph builder.

    prop-maker        a single prop in the house style, from words only (no reference image)
    style-transfer    any photo or sketch redrawn in the house style
    hover-keyframes   frame B of a small motion, matched to frame A, for MiniMax H3

    python3 build_extras.py            # writes all three
    python3 build_extras.py props      # just one

Character apps live in build_workflows.py; this module reuses its Graph, its style lock and its
node tables. Everything the two files share is imported, never copied.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import (  # noqa: E402
    Graph, WIDGETS, WIDGET_TYPES, STYLE_LOCK, NEGATIVE,
    UNET, LORA, CLIP, VAE, BGREMOVAL, STEPS,
    BLUE, GREEN, GREY, ENGINE, GROUP_LOCK, GROUP_SHOT, GROUP_OUT, GROUP_ENGINE,
)

# nodes these apps need that the character apps do not (widget order from /object_info)
WIDGETS.setdefault("ImageStitch", ["direction", "match_image_size", "spacing_width", "spacing_color"])
WIDGET_TYPES.setdefault(("ImageStitch", "direction"), "COMBO")

# ---------------------------------------------------------------- locked prompts

# Text-to-image drifts to modern printing — bright orange leather, cool grey card — unless the
# two-ink rule is spelled out. Stated positively: what the inks ARE, not what they are not.
PROP_LOCK = (
    "A single object, drawn as a prop for a 1933 newspaper comic strip: one object alone, centred "
    "and filling most of the frame, resting on a flat mid-grey backdrop of one even tone, with the "
    "whole object inside the frame. It is drawn in two muted inks only — a warm dark sepia brown "
    "for the outlines and halftone shading, and the pale cream of aged newsprint for the light "
    "areas — so every colour in the picture is a desaturated brown, cream or grey, as faded as "
    "old printing. The object is:"
)

# "Redraw this in the house style" gets a filtered photograph: Qwen-Edit is structurally faithful,
# so a real person stays a real person however the style is described. REPLACING the person is the
# instruction that redesigns them — same seat, same pose, same size, new body plan.
RESTYLE_LOCK = (
    "Replace every person in image 1 with a 1933 Fleischer-era rubber-hose cartoon character in "
    "exactly the same place, the same pose and the same size in the frame: a big round head much "
    "larger than a real head on a small rounded body, huge white cartoon eyes with round dark "
    "pupils, a little button nose, a wide simple curved mouth, bendy noodle arms and legs with no "
    "elbows or knees, and oversized rounded cartoon hands. Each one keeps whatever makes them "
    "recognisable — their hairstyle, the cut and colour of their clothes, whatever they are "
    "carrying, and only the facial features they already have in image 1. The head is about a "
    "third of the whole figure's height and the arms and legs are simple bendy tubes. Redraw the room and every object around them in the same hand: simple bold cartoon "
    "shapes, confident smooth outlines of even weight, shaded only with soft halftone dots, in warm "
    "sepia duotone on aged grey-brown newsprint, every colour muted and desaturated as faded "
    "printing. Keep the layout and the perspective of image 1 — who is where, and what is behind "
    "them. The picture shows:"
)

PALETTE_RESTATE = (
    "Finally, the finished picture is printed in two muted inks only: a warm dark sepia brown for "
    "the outlines and the halftone shading, and the pale cream of aged newsprint for the light "
    "areas. Every colour in it — clothes, books, walls, skin — is a desaturated brown, cream or "
    "grey, as faded as old newspaper printing."
)

KEEP_LEAD_IN = ("These things stay recognisable through the redraw, drawn as cartoon shapes rather than copied:")

KEYFRAME_LOCK = (
    "Redraw the picture in image 1 as the next frame of the same drawing, for a short looping "
    "animation. Keep the character's identity, costume, colour, size, position in the frame and "
    "distance from the camera exactly as they are in image 1, and keep the background exactly as "
    "it is, the same flat even tone, so the two frames stack on each other and everything holds "
    "still except the one movement described. Every feature of the character stays exactly as image "
    "1 draws it, gaining nothing new. The movement is:"
)
HOLD_LEAD_IN = "Everything else holds:"

# ---------------------------------------------------------------- shared pieces

BLANK = "  "


def engine(g, x, y, dy=46, want_bg=True):
    """The loaders every app shares, collapsed. Returns a dict of handles."""
    unet = g.add("UnetLoaderGGUF", "diffusion model", (x, y), (400, 80), {"unet_name": UNET},
                 outputs=[("MODEL", "MODEL")], color=ENGINE, collapsed=True)
    lora = g.add("LoraLoaderModelOnly", "Lightning LoRA (8 steps)", (x, y + dy), (400, 110),
                 {"lora_name": LORA, "strength_model": 1.0},
                 links={"model": (unet, 0, "MODEL", False)},
                 outputs=[("MODEL", "MODEL")], color=ENGINE, collapsed=True)
    clip = g.add("CLIPLoader", "text encoder", (x, y + 2 * dy), (400, 130),
                 {"clip_name": CLIP, "type": "qwen_image", "device": "default"},
                 outputs=[("CLIP", "CLIP")], color=ENGINE, collapsed=True)
    vae = g.add("VAELoader", "VAE", (x, y + 3 * dy), (400, 80), {"vae_name": VAE},
                outputs=[("VAE", "VAE")], color=ENGINE, collapsed=True)
    out = {"lora": lora, "clip": clip, "vae": vae}
    if want_bg:
        out["bg"] = g.add("LoadBackgroundRemovalModel", "BiRefNet (for TRANSPARENT PNG)",
                          (x, y + 4 * dy), (400, 80), {"bg_removal_name": BGREMOVAL},
                          outputs=[("BACKGROUND_REMOVAL", "BACKGROUND_REMOVAL")], color=ENGINE,
                          collapsed=True)
    return out


def cutout_chain(g, decode, bgmodel, toggle, x, y, dy=46):
    """decode -> optional RGBA. JoinImageWithAlpha inverts its mask, hence the InvertMask."""
    cut = g.add("RemoveBackground", "cut the subject out", (x, y), (380, 90),
                links={"bg_removal_model": (bgmodel, 0, "BACKGROUND_REMOVAL", False),
                       "image": (decode, 0, "IMAGE", False)},
                outputs=[("MASK", "MASK")], collapsed=True)
    inv = g.add("InvertMask", "flip to ComfyUI mask convention", (x, y + dy), (380, 80),
                links={"mask": (cut, 0, "MASK", False)},
                outputs=[("MASK", "MASK")], collapsed=True)
    rgba = g.add("JoinImageWithAlpha", "make it RGBA", (x, y + 2 * dy), (380, 90),
                 links={"image": (decode, 0, "IMAGE", False), "alpha": (inv, 0, "MASK", False)},
                 outputs=[("IMAGE", "IMAGE")], collapsed=True)
    return g.add("ComfySwitchNode", "transparent switch", (x, y + 3 * dy), (380, 130),
                 {"switch": False},
                 links={"on_false": (decode, 0, "IMAGE", False),
                        "on_true": (rgba, 0, "IMAGE", False),
                        "switch": (toggle, 0, "BOOLEAN", True)},
                 outputs=[("output", "IMAGE")], collapsed=True)


def note(g, title, text):
    g.add("MarkdownNote", title, (-560, -620), (480, 900), {"text": text})


# ---------------------------------------------------------------- 1 · prop maker

PROP_HOW_TO = """# Prop maker

**One object, drawn in the house style, from words alone.**

No reference image at all — this app is text-to-image, so the two-ink rule has to be stated in
the lock or the palette comes back as modern printing (bright orange leather on cool grey card).

| Control | What it does |
|---|---|
| **1 · WHAT THE PROP IS** | The object, in a phrase. "a brass mechanical adding machine spooling paper tape". |
| **2 · TRANSPARENT PNG** | Cuts the object out and saves RGBA — how you get a prop you can drop into a panel. |
| **3 · WIDTH / HEIGHT** | 1024x1024 by default. |
| **4 · SEED + STEPS** | Re-roll here. 8 steps ships. |

The cast's canon props are listed in `cast/<id>/character.yaml` under `props:` — the ledger, the
adding machine, the rack of tiny humming servers, the FEW cap, the fountain pen.

Diegetic lettering works: ask for a crate stencilled SLOPSHOP or a cap lettered FEW and put the
exact string in quotes.
"""


def build_props():
    g = Graph()
    note(g, "HOW TO USE — prop maker", PROP_HOW_TO)

    X, y = -40, -620
    what = g.add("PrimitiveStringMultiline", "▶ 1 · WHAT THE PROP IS (TYPE HERE)",
                 (X, y), (460, 300),
                 {"value": "a leather-bound double-entry ledger, closed, lying flat, its cover "
                           "scuffed at the corners."},
                 outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(what, "value", value="1 · WHAT THE PROP IS")
    y += 330
    alpha = g.add("PrimitiveBoolean", "▶ 2 · TRANSPARENT PNG   on = cut the object out",
                  (X, y), (460, 80), {"value": False},
                  outputs=[("BOOLEAN", "BOOLEAN")], color=GREY)
    g.app_input(alpha, "value", value="2 · TRANSPARENT PNG — cut the object out")
    y += 110
    latent = g.add("EmptySD3LatentImage", "▶ 3 · OUTPUT SIZE", (X, y), (460, 130),
                   {"width": 1024, "height": 1024, "batch_size": 1},
                   outputs=[("LATENT", "LATENT")], color=GREY)
    g.app_input(latent, "width", "height", width="3 · WIDTH", height="3 · HEIGHT")
    g.group("① THE PROP · everything you touch is in this column",
            (X - 30, -690, 520, y + 300), GROUP_SHOT)

    LX, LY, DY = 1240, -620, 46
    prop_lock = g.add("PrimitiveStringMultiline", "PROP LOCK — house prop rules",
                      (LX, LY), (430, 300), {"value": PROP_LOCK},
                      outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    style_lock = g.add("PrimitiveStringMultiline", "STYLE LOCK — house style",
                       (LX, LY + DY), (430, 300), {"value": STYLE_LOCK},
                       outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    cat = g.add("StringConcatenate", "prop lock + what the prop is", (LX, LY + 2 * DY), (330, 150),
                {"delimiter": " "},
                links={"string_a": (prop_lock, 0, "STRING", True),
                       "string_b": (what, 0, "STRING", True)},
                outputs=[("STRING", "STRING")], collapsed=True)
    prompt = g.add("StringConcatenate", "FINAL PROMPT — expand to read what the model got",
                   (LX, LY + 3 * DY), (330, 150), {"delimiter": " "},
                   links={"string_a": (cat, 0, "STRING", True),
                          "string_b": (style_lock, 0, "STRING", True)},
                   outputs=[("STRING", "STRING")], collapsed=True)
    g.group("② PROP & STYLE LOCK · leave alone", (LX - 30, LY - 70, 480, 4 * DY + 90), GROUP_LOCK)

    E = engine(g, LX, LY + 5 * DY)
    # No image1: this is text-to-image. TextEncodeQwenImageEditPlus is happy with clip + prompt.
    pos = g.add("TextEncodeQwenImageEditPlus", "encode (words only, no reference image)",
                (LX, LY + 10 * DY), (400, 130), {"prompt": ""},
                links={"clip": (E["clip"], 0, "CLIP", False),
                       "prompt": (prompt, 0, "STRING", True)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    neg = g.add("TextEncodeQwenImageEditPlus", "negative (only bites when cfg > 1)",
                (LX, LY + 11 * DY), (400, 130), {"prompt": NEGATIVE},
                links={"clip": (E["clip"], 0, "CLIP", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], color=ENGINE, collapsed=True)
    g.group("engine · don't touch", (LX - 30, LY + 5 * DY - 70, 480, 7 * DY + 90), GROUP_ENGINE)

    sampler = g.add("KSampler", "▶ 4 · SEED + STEPS", (520, 640), (460, 270),
                    {"seed": 1, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                     "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
                    links={"model": (E["lora"], 0, "MODEL", False),
                           "positive": (pos, 0, "CONDITIONING", False),
                           "negative": (neg, 0, "CONDITIONING", False),
                           "latent_image": (latent, 0, "LATENT", False)},
                    outputs=[("LATENT", "LATENT")])
    g.app_input(sampler, "seed", "steps", seed="4 · SEED", steps="4 · STEPS (8 ships)")
    decode = g.add("VAEDecode", "decode", (520, 950), (400, 80),
                   links={"samples": (sampler, 0, "LATENT", False),
                          "vae": (E["vae"], 0, "VAE", False)},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)
    final = cutout_chain(g, decode, E["bg"], alpha, 520, 1000)
    g.app_output(g.add("SaveImage", "RESULT", (520, -620), (620, 1200),
                       {"filename_prefix": "cast/prop"},
                       links={"images": (final, 0, "IMAGE", False)}))
    g.group("③ RESULT", (490, -690, 680, 1930), GROUP_OUT)
    return g


# ---------------------------------------------------------------- 2 · style transfer

STYLE_HOW_TO = """# Style transfer

**Any photo or sketch in, a house-style drawing out.**

Point it at a photo of the real fence, the real rack, a real room, and get something that sits
next to the cast without describing it from scratch.

| Control | What it does |
|---|---|
| **1 · YOUR PICTURE** | Upload the photo. Anything ComfyUI can load. |
| **2 · WHAT IT IS** | Say what the picture shows. The model reads the image, but naming the subject stops it inventing. |
| | *People are **replaced** with rubber-hose cartoon characters, not merely re-inked — that is the only phrasing that redesigns a real face.* |
| **3 · KEEP** | The parts that must survive — layout, specific objects, the lettering on a sign. |
| **4 · TRANSPARENT PNG** | Cuts the subject out. Useful when the photo is of one object. |
| **5 · WIDTH / HEIGHT** | Match the source aspect or the composition gets squeezed. |
| **6 · SEED + STEPS** | Re-roll here. |

It is one reference image and words — never add a second image hoping for style, Qwen reads every
reference as a subject and will merge them.
"""


def build_style():
    g = Graph()
    note(g, "HOW TO USE — style transfer", STYLE_HOW_TO)

    X, y = -40, -620
    pic = g.add("LoadImage", "▶ 1 · YOUR PICTURE — the photo to redraw",
                (X, y), (460, 400), {"image": "cast-scene-source.png", "upload": "image"},
                outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    g.app_input(pic, "image", image="1 · YOUR PICTURE")
    y += 430
    what = g.add("PrimitiveStringMultiline", "▶ 2 · WHAT IT IS (TYPE HERE)",
                 (X, y), (460, 220), {"value": "a rack of computer servers in a machine room."},
                 outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(what, "value", value="2 · WHAT IT IS")
    y += 250
    keep = g.add("PrimitiveStringMultiline", "▶ 3 · KEEP (TYPE HERE)",
                 (X, y), (460, 220),
                 {"value": "the layout of the room, the number of machines, and where the light "
                           "falls."},
                 outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(keep, "value", value="3 · KEEP — what must survive the redraw")
    y += 250
    alpha = g.add("PrimitiveBoolean", "▶ 4 · TRANSPARENT PNG   on = cut the subject out",
                  (X, y), (460, 80), {"value": False},
                  outputs=[("BOOLEAN", "BOOLEAN")], color=GREY)
    g.app_input(alpha, "value", value="4 · TRANSPARENT PNG — cut the subject out")
    y += 110
    latent = g.add("EmptySD3LatentImage", "▶ 5 · OUTPUT SIZE", (X, y), (460, 130),
                   {"width": 1024, "height": 1024, "batch_size": 1},
                   outputs=[("LATENT", "LATENT")], color=GREY)
    g.app_input(latent, "width", "height", width="5 · WIDTH", height="5 · HEIGHT")
    g.group("① YOUR PICTURE · everything you touch is in this column",
            (X - 30, -690, 520, y + 300), GROUP_SHOT)

    LX, LY, DY = 1240, -620, 46
    restyle_lock = g.add("PrimitiveStringMultiline", "RESTYLE LOCK — how the redraw is asked for",
                         (LX, LY), (430, 300), {"value": RESTYLE_LOCK},
                         outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    keep_lead = g.add("PrimitiveStringMultiline", "KEEP LEAD-IN — leave alone",
                      (LX, LY + DY), (430, 190), {"value": KEEP_LEAD_IN},
                      outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    style_lock = g.add("PrimitiveStringMultiline", "STYLE LOCK — house style",
                       (LX, LY + 2 * DY), (430, 300), {"value": STYLE_LOCK},
                       outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    cat1 = g.add("StringConcatenate", "restyle lock + what it is", (LX, LY + 3 * DY), (330, 150),
                 {"delimiter": " "},
                 links={"string_a": (restyle_lock, 0, "STRING", True),
                        "string_b": (what, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    cat2 = g.add("StringConcatenate", "+ keep lead-in", (LX, LY + 4 * DY), (330, 150),
                 {"delimiter": " "},
                 links={"string_a": (cat1, 0, "STRING", True),
                        "string_b": (keep_lead, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    cat3 = g.add("StringConcatenate", "+ keep", (LX, LY + 5 * DY), (330, 150), {"delimiter": " "},
                 links={"string_a": (cat2, 0, "STRING", True),
                        "string_b": (keep, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    cat4 = g.add("StringConcatenate", "+ style lock", (LX, LY + 6 * DY), (330, 150),
                 {"delimiter": " "},
                 links={"string_a": (cat3, 0, "STRING", True),
                        "string_b": (style_lock, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    # The character redesign eats the prompt's weight and the source photo's colour wins, so the
    # two-ink rule is restated at the very end — same trick as the expression restatement.
    palette = g.add("PrimitiveStringMultiline", "PALETTE RESTATEMENT — leave alone",
                    (LX, LY + 7 * DY), (430, 190), {"value": PALETTE_RESTATE},
                    outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    prompt = g.add("StringConcatenate", "FINAL PROMPT — expand to read what the model got",
                   (LX, LY + 8 * DY), (330, 150), {"delimiter": " "},
                   links={"string_a": (cat4, 0, "STRING", True),
                          "string_b": (palette, 0, "STRING", True)},
                   outputs=[("STRING", "STRING")], collapsed=True)
    g.group("② RESTYLE & STYLE LOCK · leave alone", (LX - 30, LY - 70, 480, 9 * DY + 90), GROUP_LOCK)

    E = engine(g, LX, LY + 8 * DY)
    pos = g.add("TextEncodeQwenImageEditPlus", "encode (your picture + the prompt)",
                (LX, LY + 13 * DY), (400, 130), {"prompt": ""},
                links={"clip": (E["clip"], 0, "CLIP", False),
                       "vae": (E["vae"], 0, "VAE", False),
                       "image1": (pic, 0, "IMAGE", False),
                       "prompt": (prompt, 0, "STRING", True)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    neg = g.add("TextEncodeQwenImageEditPlus", "negative (only bites when cfg > 1)",
                (LX, LY + 14 * DY), (400, 130), {"prompt": NEGATIVE},
                links={"clip": (E["clip"], 0, "CLIP", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], color=ENGINE, collapsed=True)
    g.group("engine · don't touch", (LX - 30, LY + 8 * DY - 70, 480, 7 * DY + 90), GROUP_ENGINE)

    sampler = g.add("KSampler", "▶ 6 · SEED + STEPS", (520, 640), (460, 270),
                    {"seed": 1, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                     "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
                    links={"model": (E["lora"], 0, "MODEL", False),
                           "positive": (pos, 0, "CONDITIONING", False),
                           "negative": (neg, 0, "CONDITIONING", False),
                           "latent_image": (latent, 0, "LATENT", False)},
                    outputs=[("LATENT", "LATENT")])
    g.app_input(sampler, "seed", "steps", seed="6 · SEED", steps="6 · STEPS (8 ships)")
    decode = g.add("VAEDecode", "decode", (520, 950), (400, 80),
                   links={"samples": (sampler, 0, "LATENT", False),
                          "vae": (E["vae"], 0, "VAE", False)},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)
    final = cutout_chain(g, decode, E["bg"], alpha, 520, 1000)
    g.app_output(g.add("SaveImage", "RESULT", (520, -620), (620, 1200),
                       {"filename_prefix": "cast/restyle"},
                       links={"images": (final, 0, "IMAGE", False)}))
    g.group("③ RESULT", (490, -690, 680, 1930), GROUP_OUT)
    return g


# ---------------------------------------------------------------- 3 · hover keyframes

KEYFRAME_HOW_TO = """# Hover-loop keyframes

**Frame B of a small motion, matched to frame A, for MiniMax H3 image-to-video.**

H3 interpolates between two stills. If the two frames disagree about scale, crop or background,
the interpolation swims. So this app generates frame B *from* frame A and holds everything else.

| Control | What it does |
|---|---|
| **1 · FRAME A** | The frame you already have — a pose render or a cutout. |
| **2 · THE MOTION** | The one thing that changes. Small beats loop best: a hat tip, a fin flap, a shrug. |
| **3 · HOLD** | What must not move. Leave the default unless you have a reason. |
| **4 · SEED + STEPS** | Re-roll here. |

Two outputs: **RESULT** is frame B on its own, **PAIR** is A and B side by side so you can check
the match before feeding them to H3. If the pair drifts in size or position, re-roll the seed
before touching the prompt.

Feed the two frames to the `minimax-h3` skill's first-last-frame graph on this same box.
"""


def build_keyframes():
    g = Graph()
    note(g, "HOW TO USE — hover keyframes", KEYFRAME_HOW_TO)

    X, y = -40, -620
    frame_a = g.add("LoadImage", "▶ 1 · FRAME A — the frame you already have",
                    (X, y), (460, 400), {"image": "cast-cutout-wilson.png", "upload": "image"},
                    outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    g.app_input(frame_a, "image", image="1 · FRAME A")
    y += 430
    motion = g.add("PrimitiveStringMultiline", "▶ 2 · THE MOTION (TYPE HERE)",
                   (X, y), (460, 240),
                   {"value": "he lifts the bucket hat an inch off the picket points in a small "
                             "greeting, and both eyes crinkle up."},
                   outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(motion, "value", value="2 · THE MOTION")
    y += 270
    hold = g.add("PrimitiveStringMultiline", "▶ 3 · HOLD (leave the default unless you must)",
                 (X, y), (460, 220),
                 {"value": "the same framing, the same scale, the same position in the frame, the "
                           "same colours, and a background of exactly the same flat even tone as "
                           "image 1 — the two frames stack on each other."},
                 outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(hold, "value", value="3 · HOLD — what must not move")
    y += 250
    latent = g.add("EmptySD3LatentImage", "output size (match frame A)", (X, y), (460, 130),
                   {"width": 1024, "height": 1024, "batch_size": 1},
                   outputs=[("LATENT", "LATENT")], color=GREY, collapsed=True)
    g.group("① THE MOTION · everything you touch is in this column",
            (X - 30, -690, 520, y + 250), GROUP_SHOT)

    LX, LY, DY = 1240, -620, 46
    kf_lock = g.add("PrimitiveStringMultiline", "KEYFRAME LOCK — how the next frame is asked for",
                    (LX, LY), (430, 300), {"value": KEYFRAME_LOCK},
                    outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    hold_lead = g.add("PrimitiveStringMultiline", "HOLD LEAD-IN — leave alone",
                      (LX, LY + DY), (430, 190), {"value": HOLD_LEAD_IN},
                      outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    style_lock = g.add("PrimitiveStringMultiline", "STYLE LOCK — house style",
                       (LX, LY + 2 * DY), (430, 300), {"value": STYLE_LOCK},
                       outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    cat1 = g.add("StringConcatenate", "keyframe lock + the motion", (LX, LY + 3 * DY), (330, 150),
                 {"delimiter": " "},
                 links={"string_a": (kf_lock, 0, "STRING", True),
                        "string_b": (motion, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    cat2 = g.add("StringConcatenate", "+ hold lead-in", (LX, LY + 4 * DY), (330, 150),
                 {"delimiter": " "},
                 links={"string_a": (cat1, 0, "STRING", True),
                        "string_b": (hold_lead, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    cat3 = g.add("StringConcatenate", "+ hold", (LX, LY + 5 * DY), (330, 150), {"delimiter": " "},
                 links={"string_a": (cat2, 0, "STRING", True),
                        "string_b": (hold, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    prompt = g.add("StringConcatenate", "FINAL PROMPT — expand to read what the model got",
                   (LX, LY + 6 * DY), (330, 150), {"delimiter": " "},
                   links={"string_a": (cat3, 0, "STRING", True),
                          "string_b": (style_lock, 0, "STRING", True)},
                   outputs=[("STRING", "STRING")], collapsed=True)
    g.group("② KEYFRAME & STYLE LOCK · leave alone", (LX - 30, LY - 70, 480, 7 * DY + 90),
            GROUP_LOCK)

    E = engine(g, LX, LY + 8 * DY, want_bg=False)
    pos = g.add("TextEncodeQwenImageEditPlus", "encode (frame A + the prompt)",
                (LX, LY + 12 * DY), (400, 130), {"prompt": ""},
                links={"clip": (E["clip"], 0, "CLIP", False),
                       "vae": (E["vae"], 0, "VAE", False),
                       "image1": (frame_a, 0, "IMAGE", False),
                       "prompt": (prompt, 0, "STRING", True)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    neg = g.add("TextEncodeQwenImageEditPlus", "negative (only bites when cfg > 1)",
                (LX, LY + 13 * DY), (400, 130), {"prompt": NEGATIVE},
                links={"clip": (E["clip"], 0, "CLIP", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], color=ENGINE, collapsed=True)
    g.group("engine · don't touch", (LX - 30, LY + 8 * DY - 70, 480, 6 * DY + 90), GROUP_ENGINE)

    sampler = g.add("KSampler", "▶ 4 · SEED + STEPS", (520, 640), (460, 270),
                    {"seed": 1, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                     "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
                    links={"model": (E["lora"], 0, "MODEL", False),
                           "positive": (pos, 0, "CONDITIONING", False),
                           "negative": (neg, 0, "CONDITIONING", False),
                           "latent_image": (latent, 0, "LATENT", False)},
                    outputs=[("LATENT", "LATENT")])
    g.app_input(sampler, "seed", "steps", seed="4 · SEED", steps="4 · STEPS (8 ships)")
    frame_b = g.add("VAEDecode", "decode frame B", (520, 950), (400, 80),
                    links={"samples": (sampler, 0, "LATENT", False),
                           "vae": (E["vae"], 0, "VAE", False)},
                    outputs=[("IMAGE", "IMAGE")], collapsed=True)
    pair = g.add("ImageStitch", "A and B side by side", (520, 1000), (400, 180),
                 {"direction": "right", "match_image_size": True, "spacing_width": 8,
                  "spacing_color": "white"},
                 links={"image1": (frame_a, 0, "IMAGE", False),
                        "image2": (frame_b, 0, "IMAGE", False)},
                 outputs=[("IMAGE", "IMAGE")], collapsed=True)
    g.app_output(g.add("SaveImage", "RESULT", (520, -620), (620, 700),
                       {"filename_prefix": "cast/keyframe-b"},
                       links={"images": (frame_b, 0, "IMAGE", False)}))
    g.app_output(g.add("SaveImage", "PAIR", (520, 100), (620, 500),
                       {"filename_prefix": "cast/keyframe-pair"},
                       links={"images": (pair, 0, "IMAGE", False)}))
    g.group("③ RESULT · frame B, and the pair to check it against",
            (490, -690, 680, 1930), GROUP_OUT)
    return g


# ---------------------------------------------------------------- output

APPS = {
    "props": ("prop-maker", build_props),
    "style": ("style-transfer", build_style),
    "keyframes": ("hover-keyframes", build_keyframes),
}


def main(argv):
    for key in (argv or list(APPS)):
        name, builder = APPS[key]
        g = builder()
        ui = g.to_ui()
        for path, blob in ((os.path.join(HERE, "workflows", f"{name}.json"), ui),
                           (os.path.join(HERE, "workflows", f"{name}.app.json"), ui),
                           (os.path.join(HERE, "api", f"{name}.api.json"), g.to_api())):
            json.dump(blob, open(path, "w"), indent=1)
            print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
