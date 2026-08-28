#!/usr/bin/env python3
"""Mode 2 sandboxes: a photograph's SHAPE, rendered as abstract pattern.

    python3 build_abstract.py     # writes workflows/abstract-{1..4}*.json and api/

One workflow per round of the probe, so each dead end stays walkable instead of being described
second-hand. The structure maps are derived IN THE GRAPH here -- the probe built them on the CPU
with PIL, which meant every threshold was a code edit. Canny, ImageBlur, ImageQuantize and
ImageBlend cover all four encodings between them, so every number is a widget now.

What the rounds found, in order:

  1  encodings    edges hold the shape but stay a recoloured tracing; depth is abstract but the
                  shape is gone; contours land in between and are the reason mode 2 looks viable.
  2  materials    a CLOSED palette per vocabulary, and rainbow/neon in the negative. Fixes the
                  tie-dye problem and exposes the real one: the material never arrives, because a
                  hard line map reads as line art to fill in.
  3  loosen       a tonal-mass map, or the raw photo. The material becomes real and the shape
                  leaves. Both failure modes are one problem.
  4  strength     fade the map toward flat grey. Barely moves: an edit at denoise 1.0 is faithful
                  to whatever lines it is given, so there is no dial here to find.

The conclusion the four of them add up to: image-to-image takes structure AND content from the
same picture, so it cannot take shape from the frame and material from the prompt. That
separation is what a ControlNet does, and it has the strength dial round 4 tried to fake.

Everything runs at cfg 1.0 / denoise 1.0 because the Lightning LoRA is trained for it; a partial
denoise here returns mush (see build_scene.py).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import (  # noqa: E402
    Graph, WIDGETS, WIDGET_TYPES, STEPS, BLUE, GREEN, GREY,
    GROUP_LOCK, GROUP_SHOT, GROUP_OUT,
)
from build_extras import engine, note  # noqa: E402

# Nodes only these apps use. Widget order is straight out of /object_info.
WIDGETS.setdefault("Canny", ["low_threshold", "high_threshold"])
WIDGETS.setdefault("ImageBlur", ["blur_radius", "sigma"])
WIDGETS.setdefault("ImageQuantize", ["colors", "dither"])
WIDGETS.setdefault("ImageBlend", ["blend_factor", "blend_mode"])
WIDGETS.setdefault("ImageInvert", [])
for _n, _w, _t in (("Canny", "low_threshold", "FLOAT"), ("Canny", "high_threshold", "FLOAT"),
                   ("ImageBlur", "blur_radius", "INT"), ("ImageBlur", "sigma", "FLOAT"),
                   ("ImageQuantize", "colors", "INT"), ("ImageQuantize", "dither", "COMBO"),
                   ("ImageBlend", "blend_factor", "FLOAT"), ("ImageBlend", "blend_mode", "COMBO"),
                   ("EmptyImage", "color", "INT"), ("ImageScale", "width", "INT"),
                   ("ImageScale", "height", "INT")):
    WIDGET_TYPES.setdefault((_n, _w), _t)

W, H = 1152, 648
FRAME = "f2.jpg"          # any capture; ~/.sag/action-selfies/*.source.jpg on Seth's Mac
GREYFIELD = 0x808080

KEEP = ("Follow the light and dark shapes of image 1 exactly: every edge, mass and boundary in the "
        "finished picture sits precisely where it does in image 1, at the same scale and in the "
        "same place in the frame. Nothing is drawn outside those shapes and nothing is moved.")
DROP = ("Do not draw a person, a face, a room, furniture or any recognisable object. The finished "
        "picture is pure abstract pattern.")

# An open palette collapses to tie-dye every time, so each vocabulary names a real medium AND
# closes its palette. Swap one of these into the STYLE box.
VOCABULARIES = {
 "chladni": "Redraw image 1 as a Chladni figure: fine pale sand scattered on a black steel plate, "
            "vibrated into sharp nodal ridges and swept bare in between. Photographed from directly "
            "above in raking light, so the grains cast tiny shadows and the bare steel is dark and "
            "slightly scuffed. Monochrome: bone-white sand, gunmetal plate, nothing else.",
 "ferrofluid": "Redraw image 1 as ferrofluid on glass: glossy black magnetic liquid pulled into dense "
            "fields of sharp spikes and smooth beading pools, wet and mirror-bright, lit by one hard "
            "studio light from the left. Almost entirely black on black, readable only by specular "
            "highlights and reflection. No colour at all.",
 "agate":   "Redraw image 1 as a cut and polished agate slab: concentric mineral banding in rust, "
            "ochre, cream and smoky grey, with crystalline druzy pockets where the bands close, "
            "veined in white quartz. Lit from behind so the thin bands glow. Earth tones only.",
 "schlieren": "Redraw image 1 as a schlieren photograph of moving air: shockwaves and thermal plumes "
            "as smooth grey density gradients with knife-edge dark and light fringes on a flat grey "
            "field. A laboratory image, greyscale only, no colour whatsoever.",
 "kirlian": "Redraw image 1 as a Kirlian corona-discharge photograph: fine electric filaments "
            "branching off every boundary into a black photographic ground, violet-white at the core "
            "fading to deep indigo, with film grain and a slight halation bloom. Violet and white on "
            "black, nothing else.",
 "suminagashi": "Redraw image 1 as suminagashi marbling: black sumi ink floated on still water and "
            "drawn into fine concentric feathered rings, printed onto damp cream washi paper with "
            "visible fibre and deckle. Ink black and one muted indigo on undyed paper.",
}
NEGATIVE = ("a person, a face, a room, furniture, text, watermark, blurry, muddy, "
            "rainbow, tie-dye, neon, psychedelic, oversaturated, garish")


def source(g, x, y):
    """The capture, cover-scaled to the working size. Every round starts here."""
    load = g.add("LoadImage", "▶ 1 · SOURCE PHOTO", (x, y), (400, 320), {"image": FRAME, "upload": "image"},
                 outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], color=BLUE)
    g.app_input(load, "image", "upload")
    fit = g.add("ImageScale", "fit to 1152x648", (x, y + 360), (380, 130),
                {"upscale_method": "lanczos", "width": W, "height": H, "crop": "center"},
                links={"image": (load, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")],
                color=BLUE, collapsed=True)
    return fit


def sampler(g, e, structure, style_text, x, y, title):
    """map -> abstract picture. The map is the EDIT TARGET, which is what preserves geometry."""
    style = g.add("PrimitiveStringMultiline", "▶ STYLE — the material", (x, y), (460, 220),
                  {"value": style_text}, outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(style, "value")
    pos = g.add("TextEncodeQwenImageEditPlus", "positive", (x, y + 260), (420, 120),
                links={"prompt": (style, 0, "STRING", True), "clip": (e["clip"], 0, "CLIP", False),
                       "vae": (e["vae"], 0, "VAE", False), "image1": (structure, 0, "IMAGE", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    neg = g.add("TextEncodeQwenImageEditPlus", "negative", (x, y + 310), (420, 120),
                {"prompt": NEGATIVE}, links={"clip": (e["clip"], 0, "CLIP", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    latent = g.add("EmptySD3LatentImage", "canvas", (x, y + 360), (380, 130),
                   {"width": W, "height": H, "batch_size": 1},
                   outputs=[("LATENT", "LATENT")], collapsed=True)
    ks = g.add("KSampler", "▶ SEED / STEPS", (x, y + 410), (400, 280),
               {"seed": 7, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
               links={"model": (e["lora"], 0, "MODEL", False), "positive": (pos, 0, "CONDITIONING", False),
                      "negative": (neg, 0, "CONDITIONING", False), "latent_image": (latent, 0, "LATENT", False)},
               outputs=[("LATENT", "LATENT")], color=GREEN)
    g.app_input(ks, "seed", "steps")
    dec = g.add("VAEDecode", "decode", (x, y + 710), (300, 60),
                links={"samples": (ks, 0, "LATENT", False), "vae": (e["vae"], 0, "VAE", False)},
                outputs=[("IMAGE", "IMAGE")], collapsed=True)
    out = g.add("SaveImage", "RESULT", (x + 460, y), (620, 700), {"filename_prefix": f"cast/{title}"},
                links={"images": (dec, 0, "IMAGE", False)}, color=GREY)
    g.app_output(out)
    return out


def preview(g, node, title, pos):
    p = g.add("PreviewImage", title, pos, (420, 400), links={"images": (node, 0, "IMAGE", False)},
              color=GREY)
    g.app_output(p)
    return p


# ---------------------------------------------------------------- round 1

R1 = """# Round 1 — three structure encodings

**Which reduction of the photo keeps its shape without keeping its subject?**

Three encoders run in parallel off one photo. All three previews render every time, so you can see
what the model is being handed. **Only CONTOURS is wired to the sampler** — drag a different
encoder's output into `positive.image1` to try the others.

| Control | What it does |
|---|---|
| **EDGES · low/high threshold** | Canny sensitivity. Lower = more lines, more clutter. |
| **DEPTH · blur radius / sigma** | How far the luminance is smeared. Big numbers destroy structure. |
| **CONTOURS · colors** | Posterisation levels before edge-finding: the band count. **This is the knob that matters** — fewer bands, more abstract. |

**What it found:** edges hold the shape but come back a recoloured tracing, and the face is still
legible. Depth is properly abstract and the shape is completely gone — soft masses give a
diffusion edit nothing to grip. Contours land in between, and are the reason mode 2 looks viable.
"""


def build_r1():
    g = Graph()
    note(g, "HOW TO USE — round 1 · encodings", R1)
    e = engine(g, -60, -180, want_bg=False)
    src = source(g, -60, 60)

    # edges: where the picture changes
    eb = g.add("ImageBlur", "soften first (Canny is noisy on video)", (420, 60), (380, 130),
               {"blur_radius": 2, "sigma": 1.2}, links={"image": (src, 0, "IMAGE", False)},
               outputs=[("IMAGE", "IMAGE")], collapsed=True)
    edges = g.add("Canny", "▶ EDGES · threshold", (420, 120), (400, 130),
                  {"low_threshold": 0.2, "high_threshold": 0.5},
                  links={"image": (eb, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], color=BLUE)
    g.app_input(edges, "low_threshold", "high_threshold")
    edges_i = g.add("ImageInvert", "black on white", (420, 260), (300, 60),
                    links={"image": (edges, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")],
                    collapsed=True)
    preview(g, edges_i, "EDGES map", (860, 60))

    # depth: near/far masses only
    depth = g.add("ImageBlur", "▶ DEPTH · blur", (420, 480), (400, 130), {"blur_radius": 24, "sigma": 8.0},
                  links={"image": (src, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], color=BLUE)
    g.app_input(depth, "blur_radius", "sigma")
    preview(g, depth, "DEPTH map", (860, 480))

    # contours: iso-luminance bands
    cb = g.add("ImageBlur", "soften", (420, 900), (380, 130), {"blur_radius": 8, "sigma": 4.0},
               links={"image": (src, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], collapsed=True)
    quant = g.add("ImageQuantize", "▶ CONTOURS · band count", (420, 960), (400, 130),
                  {"colors": 6, "dither": "none"}, links={"image": (cb, 0, "IMAGE", False)},
                  outputs=[("IMAGE", "IMAGE")], color=BLUE)
    g.app_input(quant, "colors")
    cedge = g.add("Canny", "band boundaries", (420, 1100), (380, 130),
                  {"low_threshold": 0.05, "high_threshold": 0.15},
                  links={"image": (quant, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], collapsed=True)
    contours = g.add("ImageInvert", "black on white", (420, 1160), (300, 60),
                     links={"image": (cedge, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")],
                     collapsed=True)
    preview(g, contours, "CONTOURS map — the one that is wired up", (860, 900))

    sampler(g, e, contours, VOCABULARIES["kirlian"], 1340, 60, "abstract-r1")
    g.group("1 · the photo", [-100, 0, 900, 420], "#3f5159")
    g.group("2 · three encodings", [400, 40, 880, 1300], "#3f5159")
    return g


# ---------------------------------------------------------------- round 2

R2 = """# Round 2 — material vocabularies

**Contours map, six closed palettes.** The round-1 prompts were all neon-on-black with rainbow
hues: one aesthetic in three hats. Every vocabulary here names a real physical medium and closes
its palette, and `rainbow, tie-dye, neon, psychedelic` sit in the negative. An open palette
collapses to tie-dye every time.

Paste one of these into **STYLE** (the full text of each is in `build_abstract.py`):

| id | what it is |
|---|---|
| **chladni** | bone-white sand on a vibrating gunmetal plate, raking light |
| **ferrofluid** | glossy black magnetic spikes, one hard light, no colour |
| **agate** | banded mineral slab lit from behind, earth tones |
| **schlieren** | greyscale air-density photograph, knife-edge fringes |
| **kirlian** | violet-white corona discharge on black film |
| **suminagashi** | sumi ink on water, printed to cream washi |

**What it found:** the palettes stop being tie-dye, and the *material still never arrives* — all
six come back as the same contour drawing recoloured. Hard black lines on white read to the model
as line art to fill in. That is what round 3 goes after.
"""


def build_r2():
    g = Graph()
    note(g, "HOW TO USE — round 2 · materials", R2)
    e = engine(g, -60, -180, want_bg=False)
    src = source(g, -60, 60)
    cb = g.add("ImageBlur", "soften", (420, 60), (380, 130), {"blur_radius": 8, "sigma": 4.0},
               links={"image": (src, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], collapsed=True)
    quant = g.add("ImageQuantize", "▶ BAND COUNT", (420, 120), (400, 130), {"colors": 6, "dither": "none"},
                  links={"image": (cb, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], color=BLUE)
    g.app_input(quant, "colors")
    cedge = g.add("Canny", "band boundaries", (420, 260), (380, 130),
                  {"low_threshold": 0.05, "high_threshold": 0.15},
                  links={"image": (quant, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], collapsed=True)
    contours = g.add("ImageInvert", "black on white", (420, 320), (300, 60),
                     links={"image": (cedge, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")],
                     collapsed=True)
    preview(g, contours, "the map the model is handed", (860, 60))
    sampler(g, e, contours, VOCABULARIES["ferrofluid"], 1340, 60, "abstract-r2")
    g.group("1 · the photo", [-100, 0, 900, 420], "#3f5159")
    g.group("2 · contours", [400, 40, 880, 460], "#3f5159")
    return g


# ---------------------------------------------------------------- round 3

R3 = """# Round 3 — loosen the map

**Same materials, a map with no outlines.** Round 2's hard lines were being filled in rather than
interpreted, so this replaces them with tonal MASSES: blur and posterise, and skip the edge-find
entirely. The raw photo is previewed alongside — drag it into `positive.image1` to try the other
extreme, no map at all.

| Control | What it does |
|---|---|
| **MASSES · blur** | How much detail is thrown away before posterising. |
| **MASSES · colors** | How many grey levels survive. 4–6 is the useful range. |

**What it found:** the material becomes real — actual ferrofluid spikes, actual sand on a plate,
actual polished agate — and the shape leaves. The model renders a generic instance of the medium
centred in the frame. Together with round 2 that is one problem from both ends: image-to-image
takes structure and content from the same picture, so it cannot take shape from the photo and
material from the prompt.
"""


def build_r3():
    g = Graph()
    note(g, "HOW TO USE — round 3 · loosen", R3)
    e = engine(g, -60, -180, want_bg=False)
    src = source(g, -60, 60)
    mb = g.add("ImageBlur", "▶ MASSES · blur", (420, 60), (400, 130), {"blur_radius": 8, "sigma": 4.0},
               links={"image": (src, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], color=BLUE)
    g.app_input(mb, "blur_radius", "sigma")
    masses = g.add("ImageQuantize", "▶ MASSES · grey levels", (420, 200), (400, 130),
                   {"colors": 5, "dither": "none"}, links={"image": (mb, 0, "IMAGE", False)},
                   outputs=[("IMAGE", "IMAGE")], color=BLUE)
    g.app_input(masses, "colors")
    preview(g, masses, "MASSES map — the one that is wired up", (860, 60))
    preview(g, src, "the raw photo — the other extreme", (860, 500))
    sampler(g, e, masses, VOCABULARIES["ferrofluid"], 1340, 60, "abstract-r3")
    g.group("1 · the photo", [-100, 0, 900, 420], "#3f5159")
    g.group("2 · tonal masses", [400, 40, 880, 340], "#3f5159")
    return g


# ---------------------------------------------------------------- round 4

R4 = """# Round 4 — a strength dial that is not there

**Fade the contour map toward flat grey and watch nothing happen.** `MAP STRENGTH` blends the map
against a neutral field: 1.0 is the full map, 0.0 is featureless grey.

| Control | What it does |
|---|---|
| **MAP STRENGTH** | Blend factor. 1.0 = full map, 0.35 and 0.6 are the values that were measured. |
| **BAND COUNT** | As round 2. |

**What it found:** barely moves. A Qwen edit at denoise 1.0 is faithful to whatever lines it is
given, and lowering their contrast does not lower their authority — so there is no dial between
"recoloured drawing" and "generic lump of material" to be found this way.

**Why lowering denoise is not the answer either:** the Lightning LoRA is trained for cfg 1.0 at
full denoise, and a partial denoise returns mush. To get a real strength dial the structure has to
be conditioned separately from the content, which is what a ControlNet does.
"""


def build_r4():
    g = Graph()
    note(g, "HOW TO USE — round 4 · strength", R4)
    e = engine(g, -60, -180, want_bg=False)
    src = source(g, -60, 60)
    cb = g.add("ImageBlur", "soften", (420, 60), (380, 130), {"blur_radius": 8, "sigma": 4.0},
               links={"image": (src, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], collapsed=True)
    quant = g.add("ImageQuantize", "▶ BAND COUNT", (420, 120), (400, 130), {"colors": 6, "dither": "none"},
                  links={"image": (cb, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], color=BLUE)
    g.app_input(quant, "colors")
    cedge = g.add("Canny", "band boundaries", (420, 260), (380, 130),
                  {"low_threshold": 0.05, "high_threshold": 0.15},
                  links={"image": (quant, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")], collapsed=True)
    contours = g.add("ImageInvert", "black on white", (420, 320), (300, 60),
                     links={"image": (cedge, 0, "IMAGE", False)}, outputs=[("IMAGE", "IMAGE")],
                     collapsed=True)
    flat = g.add("EmptyImage", "flat grey field", (420, 400), (380, 170),
                 {"width": W, "height": H, "batch_size": 1, "color": GREYFIELD},
                 outputs=[("IMAGE", "IMAGE")], collapsed=True)
    faded = g.add("ImageBlend", "▶ MAP STRENGTH", (420, 470), (400, 170),
                  {"blend_factor": 0.6, "blend_mode": "normal"},
                  links={"image1": (flat, 0, "IMAGE", False), "image2": (contours, 0, "IMAGE", False)},
                  outputs=[("IMAGE", "IMAGE")], color=BLUE)
    g.app_input(faded, "blend_factor")
    preview(g, faded, "the faded map", (860, 60))
    sampler(g, e, faded, VOCABULARIES["ferrofluid"], 1340, 60, "abstract-r4")
    g.group("1 · the photo", [-100, 0, 900, 420], "#3f5159")
    g.group("2 · contours, faded", [400, 40, 880, 640], "#3f5159")
    return g


BUILDS = {
    "abstract-1-encodings": build_r1,
    "abstract-2-materials": build_r2,
    "abstract-3-loosen": build_r3,
    "abstract-4-strength": build_r4,
}

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "workflows"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "api"), exist_ok=True)
    for stem, fn in BUILDS.items():
        g = fn()
        for path, blob in ((os.path.join(HERE, "workflows", f"{stem}.json"), g.to_ui()),
                           (os.path.join(HERE, "workflows", f"{stem}.app.json"), g.to_ui()),
                           (os.path.join(HERE, "api", f"{stem}.api.json"), g.to_api())):
            json.dump(blob, open(path, "w"), indent=1)
        print("wrote", stem)
