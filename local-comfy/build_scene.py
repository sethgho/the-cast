#!/usr/bin/env python3
"""The scene app: a picture of a face in, a wide cartoon scene of Seth out.

    python3 build_scene.py

Writes workflows/seth-scene.json, workflows/seth-scene.app.json and api/seth-scene.api.json.

Two stages on one graph. Stage one is `build_expression.expression_stage` unchanged — the
proven "any face in, Seth's headshot wearing that expression out" block, imported rather than
copied so its prompt has exactly one home. Stage two pastes that headshot onto a wide sheet of
blank paper and asks the model to draw the room around it.

## Why the head is held by pixels and not by words

Qwen will not keep a face across a re-render; asked for Seth in a room it draws a man who looks
a little like Seth. So the head is composited into the output canvas first and its pixels are
frozen with SetLatentNoiseMask, exactly the way build_workflows.build_duo freezes its two
figures. Only the paper around him is painted.

## The panel border, and what actually fixed it

The first version of this froze the head's **bounding box**: a hard-edged rectangle of held
pixels in the middle of a repainted sheet. The model read that rectangle as a comic panel and
inked a border around it, then gave the room its own separate panel next to it — the same
failure `build_duo` records. Three changes, all needed:

  1. The frozen region is the head's **silhouette**, from BiRefNet, not its box. There is no
     rectangle in the latent for the model to trace, and the headshot's own backdrop never
     reaches the canvas, so there is no second sheet of paper inside the picture.
  2. The prompt states, positively and last, that the picture is one continuous drawing whose
     room passes behind the man to all four edges. It never says "no border": naming a border
     draws one, which is why an earlier attempt that added "no panel borders" made it worse.
  3. The silhouette boundary is a ramp, not a step — a hard mask edge is itself an edge the
     sampler can decide to draw.

Measured, all three at the same seed and scene:

  - box mask, hard edge: the room in its own inked panel, the head on its own torn paper
    beside it, a hard vertical seam between them.
  - silhouette + prompt, grown by 8px and blurred by 25: no border and no seam, but a bright
    cream halo around him that the model finished into a **spotlight cone** up to the ceiling.
    The halo is the ramp itself: a half-held pixel keeps half the blank canvas, and the blank
    canvas is cream, so a wide ramp bakes a cream outline into the drawing.
  - silhouette + prompt, **eroded** by 4px and blurred by 9: clean. Putting the ramp INSIDE his
    outline means the held pixels that get half-painted are his own dark ink, not the paper, so
    the blend has nothing bright to leave behind. Hence GROW is negative and small.

## The head box

The paste is at a fixed, known place, so the caller can crop a square avatar out of the wide
image without generating twice. HEAD_BOX below is that place, and scene_run.py writes it beside
every render. Both offsets and the size are multiples of 8 so the box lands on whole latent
cells — an off-grid paste smears the held pixels by half a cell at the seam.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import (  # noqa: E402
    Graph, WIDGETS, STYLE_LOCK, NEGATIVE, STEPS,
    BLUE, GREEN, GREY, GROUP_LOCK, GROUP_SHOT, GROUP_OUT, GROUP_ENGINE,
)
from build_extras import engine, note  # noqa: E402
from build_expression import expression_stage  # noqa: E402

# Nodes only this app uses, in /object_info widget order. Neither is promoted into the form,
# but the builder still needs the order to lay widgets_values out positionally.
WIDGETS.setdefault("GrowMask", ["expand", "tapered_corners"])
WIDGETS.setdefault("ImageBlur", ["blur_radius", "sigma"])

# ---------------------------------------------------------------- geometry

OUT_W, OUT_H = 1920, 832        # 21:9 at 1.6MP. Measured: the pair of stages peaks at
                                # 11,417 MiB of the 12,288 MiB card, so this is the ceiling.
HEAD = 768                      # 92% of the height, bottom-aligned
HEAD_X = 920                    # centres the head on 68% of the width
HEAD_Y = OUT_H - HEAD
HEAD_BOX = {"x": HEAD_X, "y": HEAD_Y, "w": HEAD, "h": HEAD}

# The boundary between held and painted pixels. Negative GROW erodes: the ramp lands inside
# his ink outline, which is what stopped the cream halo described above. Keep the ramp near one
# latent cell (8px) — SetLatentNoiseMask works at 1/8 scale, so a wider blur buys no smoothness,
# only halo.
GROW = -4
FEATHER = 9
FEATHER_SIGMA = 5.0

# Aged cream, sampled off a finished render. The canvas is fully repainted, but a mid-grey or
# white sheet biases the first sampling steps toward a modern palette.
PAPER = 0xDEC7A0

# ---------------------------------------------------------------- the prompt

# Stated as a drawing that is half finished, because that is what the latent actually is: his
# pixels are frozen and everything else is blank paper. "Finish this drawing" is a job Qwen does
# well; "put this man in a room" invites it to redraw him.
SCENE_LOCK = (
    "Image 1 is one wide drawing that is only half finished. The man is already drawn and is "
    "complete: leave his face, hair, moustache, shirt and expression exactly as they are. "
    "Everything else in image 1 is blank paper. Finish the drawing by drawing the place he is "
    "in onto that blank paper, in the same ink and the same halftone, so that it meets his "
    "outline and continues behind him: the floor runs under and behind him, the wall and its "
    "things stand behind his shoulders and behind his hair, one light falls across the whole "
    "width of the picture, and his own soft shadow lies on the surface behind him. He is the "
    "only person here; every other shape is furniture, scenery or a prop. The place he is in is:"
)

# Law 4, whatever must not be lost goes last — and here that is the continuity, because the
# pixel freeze already holds the man. Every clause is a positive statement of what IS true.
# An earlier draft that ended with "no panel borders, no frame, no seam" produced a bordered
# panel every time: the words border and frame are things the model knows how to draw.
CONTINUITY = (
    "Above all this is a single continuous illustration on one unbroken sheet of paper. The "
    "room reaches the left edge, the right edge, the top edge and the bottom edge of the "
    "picture and is cut off only by those edges. The scenery passes behind the man's head, his "
    "hair and his shoulders and comes out the other side, the same floor and the same wall "
    "surround him on every side, and the whole width of the picture is one room drawn in one "
    "perspective under one light, with the same paper texture and the same two inks everywhere."
)

# Tried and rejected: a further clause that "the shading and the halftone of that room cover the
# paper all the way out to the four edges". A night scene still leaves a pale margin at the top
# corners — the cream canvas biases the outer paper light and the model rationalises it as a near
# wall — and the clause changed that only marginally while making the prompt longer, which law 4
# says costs the sentences before it. A dark scene described as filling the frame ("a kitchen at
# night lit only by the lamp over the sink, the darkness reaching every corner") does more.
DEFAULT_SCENE = ("a cluttered home-office desk seen from across the room, deep stacks of "
                 "monitors and tangled cables, a gooseneck lamp leaning in from the left.")

HOW_TO = """# Seth in a scene, from any face

**Drop in a picture of a face and describe a place. Get a 1920x832 cartoon of Seth standing in
that place, wearing that picture's expression.**

Two stages run back to back. First the expression app draws Seth's headshot with the source
face's expression. Then that headshot is pasted onto a wide blank sheet and the model draws the
room around it — the head's own pixels never go through the second sampler, so the face you saw
in stage one is the face in the wide image, to the pixel.

| Control | What it does |
|---|---|
| **1 · SOURCE PICTURE** | Any picture with a face in it. Its expression is what Seth wears. |
| **2 · THE SCENE** | Where he is. One or two sentences of place, not of action. |
| **3 · SEED** | Re-roll. Drives both stages, so the same seed gives the same face and the same room. |

Two outputs: **RESULT** is the wide image, **HEADSHOT** is what stage one drew — check that one
first when the face is wrong, because stage two cannot have touched it.

## The head sits in a known box

Always at x={x}, y={y}, {w}x{h} in the 1920x832 output. Crop that square and you have the
avatar, with no second generation. `scene_run.py` writes it to JSON beside every render.

## Describe a place, not an event

The man is already drawn and cannot move. "Seth throwing a wrench" gives him a room with a
wrench lying in it. Say what is around him.

## Speed

8 steps, cfg 1.0, denoise 1.0 in both stages. Measured on gpu-worker: 174s for the first run
after the weights are unloaded, 126s with the model resident, and 57s when only the scene text or
the seed changed and ComfyUI still has stage one cached. Peak 11,417 MiB of the 12GB card.
""".format(**HEAD_BOX)


def build():
    g = Graph()
    note(g, "HOW TO USE — Seth in a scene", HOW_TO)

    # BiRefNet is wanted here: it is what turns the headshot into a silhouette.
    E = engine(g, 2500, -160, want_bg=True)

    # One seed for both samplers: two fields would let the face change when only the room was
    # meant to be re-rolled, and the form is meant to have three controls, not four.
    seed = g.add("PrimitiveInt", "▶ 3 · SEED", (-40, 900), (460, 80), {"value": 1},
                 outputs=[("INT", "INT")], color=GREY)
    st = expression_stage(g, E, seed_src=seed)
    head = st["image"]

    # --- the form -----------------------------------------------------
    scene = g.add("PrimitiveStringMultiline", "▶ 2 · THE SCENE (TYPE HERE)", (-40, 1020),
                  (460, 260), {"value": DEFAULT_SCENE},
                  outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(st["source"], "image", image="1 · SOURCE PICTURE")
    g.app_input(scene, "value", value="2 · THE SCENE — where he is")
    g.app_input(seed, "value", value="3 · SEED")
    g.group("① YOUR PICTURE AND YOUR SCENE · everything you touch",
            (-70, -690, 520, 2050), GROUP_SHOT)

    # --- lay the head out in pixels ---------------------------------------
    CX, CY, DY = 560, 1100, 46
    cut = g.add("RemoveBackground", "Seth's silhouette (BiRefNet)", (CX, CY), (340, 90),
                links={"bg_removal_model": (E["bg"], 0, "BACKGROUND_REMOVAL", False),
                       "image": (head, 0, "IMAGE", False)},
                outputs=[("MASK", "MASK")], collapsed=True)
    sil = g.add("MaskToImage", "silhouette as an image", (CX, CY + DY), (340, 80),
                links={"mask": (cut, 0, "MASK", False)},
                outputs=[("IMAGE", "IMAGE")], collapsed=True)
    head_s = g.add("ImageScale", "head at scene scale", (CX, CY + 2 * DY), (340, 150),
                   {"upscale_method": "lanczos", "width": HEAD, "height": HEAD,
                    "crop": "disabled"},
                   links={"image": (head, 0, "IMAGE", False)},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)
    sil_s = g.add("ImageScale", "silhouette at scene scale", (CX, CY + 3 * DY), (340, 150),
                  {"upscale_method": "lanczos", "width": HEAD, "height": HEAD,
                   "crop": "disabled"},
                  links={"image": (sil, 0, "IMAGE", False)},
                  outputs=[("IMAGE", "IMAGE")], collapsed=True)
    sil_mask = g.add("ImageToMask", "silhouette as a mask", (CX, CY + 4 * DY), (340, 90),
                     {"channel": "red"},
                     links={"image": (sil_s, 0, "IMAGE", False)},
                     outputs=[("MASK", "MASK")], collapsed=True)
    paper = g.add("EmptyImage", "the blank sheet", (CX, CY + 5 * DY), (340, 150),
                  {"width": OUT_W, "height": OUT_H, "batch_size": 1, "color": PAPER},
                  outputs=[("IMAGE", "IMAGE")], collapsed=True)
    # Composited through the silhouette, not as a square: the backdrop the headshot was drawn
    # on never reaches the canvas, so there is no second sheet of paper inside the picture.
    comp = g.add("ImageCompositeMasked", "paste Seth onto the sheet", (CX, CY + 6 * DY),
                 (340, 150), {"x": HEAD_X, "y": HEAD_Y, "resize_source": False},
                 links={"destination": (paper, 0, "IMAGE", False),
                        "source": (head_s, 0, "IMAGE", False),
                        "mask": (sil_mask, 0, "MASK", False)},
                 outputs=[("IMAGE", "IMAGE")], collapsed=True)
    black = g.add("EmptyImage", "black sheet for the frozen-pixel mask", (CX, CY + 7 * DY),
                  (340, 150), {"width": OUT_W, "height": OUT_H, "batch_size": 1, "color": 0},
                  outputs=[("IMAGE", "IMAGE")], collapsed=True)
    keep_img = g.add("ImageCompositeMasked", "stamp the silhouette at the same place",
                     (CX, CY + 8 * DY), (340, 150),
                     {"x": HEAD_X, "y": HEAD_Y, "resize_source": False},
                     links={"destination": (black, 0, "IMAGE", False),
                            "source": (sil_s, 0, "IMAGE", False),
                            "mask": (sil_mask, 0, "MASK", False)},
                     outputs=[("IMAGE", "IMAGE")], collapsed=True)
    keep = g.add("ImageToMask", "what must not be repainted", (CX, CY + 9 * DY), (340, 90),
                 {"channel": "red"},
                 links={"image": (keep_img, 0, "IMAGE", False)},
                 outputs=[("MASK", "MASK")], collapsed=True)
    grown = g.add("GrowMask", "hold his ink outline too", (CX, CY + 10 * DY), (340, 110),
                  {"expand": GROW, "tapered_corners": True},
                  links={"mask": (keep, 0, "MASK", False)},
                  outputs=[("MASK", "MASK")], collapsed=True)
    grown_img = g.add("MaskToImage", "grown mask as an image", (CX, CY + 11 * DY), (340, 80),
                      links={"mask": (grown, 0, "MASK", False)},
                      outputs=[("IMAGE", "IMAGE")], collapsed=True)
    # The feather is the second half of the border fix. A step from held to painted is an edge,
    # and an edge is something this model happily inks; a ramp is not.
    soft_img = g.add("ImageBlur", "feather the boundary", (CX, CY + 12 * DY), (340, 130),
                     {"blur_radius": FEATHER, "sigma": FEATHER_SIGMA},
                     links={"image": (grown_img, 0, "IMAGE", False)},
                     outputs=[("IMAGE", "IMAGE")], collapsed=True)
    soft = g.add("ImageToMask", "feathered keep-mask", (CX, CY + 13 * DY), (340, 90),
                 {"channel": "red"},
                 links={"image": (soft_img, 0, "IMAGE", False)},
                 outputs=[("MASK", "MASK")], collapsed=True)
    paint = g.add("InvertMask", "everything except him", (CX, CY + 14 * DY), (340, 80),
                  links={"mask": (soft, 0, "MASK", False)},
                  outputs=[("MASK", "MASK")], collapsed=True)
    g.group("② LAYOUT · pixels, not prompting", (CX - 30, CY - 70, 400, 15 * DY + 90), GROUP_OUT)

    # --- the scene prompt -------------------------------------------------
    LX, LY = 1020, 1100
    lock = g.add("PrimitiveStringMultiline", "SCENE LOCK — how the room is asked for",
                 (LX, LY), (430, 300), {"value": SCENE_LOCK},
                 outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    style = g.add("PrimitiveStringMultiline", "STYLE LOCK — house style", (LX, LY + DY),
                  (430, 300), {"value": STYLE_LOCK},
                  outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    cont = g.add("PrimitiveStringMultiline", "CONTINUITY — one unbroken drawing, stated last",
                 (LX, LY + 2 * DY), (430, 300), {"value": CONTINUITY},
                 outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    cat1 = g.add("StringConcatenate", "scene lock + the scene", (LX, LY + 3 * DY), (330, 150),
                 {"delimiter": " "},
                 links={"string_a": (lock, 0, "STRING", True),
                        "string_b": (scene, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    cat2 = g.add("StringConcatenate", "+ style lock", (LX, LY + 4 * DY), (330, 150),
                 {"delimiter": " "},
                 links={"string_a": (cat1, 0, "STRING", True),
                        "string_b": (style, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    prompt = g.add("StringConcatenate", "FINAL PROMPT — expand to read what the model got",
                   (LX, LY + 5 * DY), (330, 150), {"delimiter": " "},
                   links={"string_a": (cat2, 0, "STRING", True),
                          "string_b": (cont, 0, "STRING", True)},
                   outputs=[("STRING", "STRING")], collapsed=True)
    g.group("③ SCENE & STYLE LOCK · leave alone", (LX - 30, LY - 70, 480, 6 * DY + 90),
            GROUP_LOCK)

    # --- sample the paper around him --------------------------------------
    EX, EY = 1500, 1100
    pos = g.add("TextEncodeQwenImageEditPlus", "encode (the half-finished sheet + the prompt)",
                (EX, EY), (400, 150), {"prompt": ""},
                links={"clip": (E["clip"], 0, "CLIP", False),
                       "vae": (E["vae"], 0, "VAE", False),
                       "image1": (comp, 0, "IMAGE", False),
                       "prompt": (prompt, 0, "STRING", True)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    neg = g.add("TextEncodeQwenImageEditPlus", "negative (only bites when cfg > 1)",
                (EX, EY + DY), (400, 130), {"prompt": NEGATIVE},
                links={"clip": (E["clip"], 0, "CLIP", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    latent = g.add("VAEEncode", "the half-finished sheet as latent", (EX, EY + 2 * DY),
                   (400, 80),
                   links={"pixels": (comp, 0, "IMAGE", False), "vae": (E["vae"], 0, "VAE", False)},
                   outputs=[("LATENT", "LATENT")], collapsed=True)
    masked = g.add("SetLatentNoiseMask", "freeze him, paint the rest", (EX, EY + 3 * DY),
                   (400, 80),
                   links={"samples": (latent, 0, "LATENT", False),
                          "mask": (paint, 0, "MASK", False)},
                   outputs=[("LATENT", "LATENT")], collapsed=True)
    g.group("engine · don't touch", (EX - 30, EY - 70, 480, 4 * DY + 90), GROUP_ENGINE)

    sampler = g.add("KSampler", "draw the room", (EX, EY + 5 * DY), (400, 270),
                    {"seed": 1, "control_after_generate": "fixed", "steps": STEPS, "cfg": 1.0,
                     "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
                    links={"model": (E["lora"], 0, "MODEL", False),
                           "positive": (pos, 0, "CONDITIONING", False),
                           "negative": (neg, 0, "CONDITIONING", False),
                           "latent_image": (masked, 0, "LATENT", False),
                           "seed": (seed, 0, "INT", True)},
                    outputs=[("LATENT", "LATENT")], collapsed=True)
    decode = g.add("VAEDecode", "decode", (EX, EY + 6 * DY), (400, 80),
                   links={"samples": (sampler, 0, "LATENT", False),
                          "vae": (E["vae"], 0, "VAE", False)},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)

    # Titled RESULT, not SCENE: scene_run finds nodes by title prefix and "SCENE LOCK"
    # matched first, which fetched the wrong node's output.
    g.app_output(g.add("SaveImage", "RESULT — the wide scene", (2000, 1100), (960, 500),
                       {"filename_prefix": "cast/seth-scene"},
                       links={"images": (decode, 0, "IMAGE", False)}))
    g.app_output(g.add("SaveImage", "HEADSHOT — what stage one drew", (2000, 1640), (400, 460),
                       {"filename_prefix": "cast/seth-scene-head"},
                       links={"images": (head, 0, "IMAGE", False)}, color=GREEN))
    g.group("④ RESULT · the wide scene, and the headshot inside it",
            (1970, 1030, 1020, 1100), GROUP_OUT)
    return g


def main():
    g = build()
    ui = g.to_ui()
    for path, blob in ((os.path.join(HERE, "workflows", "seth-scene.json"), ui),
                       (os.path.join(HERE, "workflows", "seth-scene.app.json"), ui),
                       (os.path.join(HERE, "api", "seth-scene.api.json"), g.to_api())):
        json.dump(blob, open(path, "w"), indent=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
