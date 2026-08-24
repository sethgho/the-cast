#!/usr/bin/env python3
"""The expression app: any face in, Seth wearing that face's expression out.

    python3 build_expression.py

Writes workflows/seth-expression.json, workflows/seth-expression.app.json and
api/seth-expression.api.json off the same Graph builder as every other cast app.

## The wiring, and why it is this way round

Two images reach the sampler, which the prompting laws say is dangerous: a reference is an
IDENTITY reference, and one reference means one subject — a second character wired into
`image2` once merged Wilson and Ake into a single creature. So the images are not peers here:

  image1 = Seth's canonical headshot   -> the EDIT TARGET. The graph is a repaint of it.
  image2 = the source face, cropped    -> consulted for ONE property, the expression.

Making the thing that must survive the edit target is what keeps Seth as Seth; the prompt then
gives image2 exactly one job and says so twice, at the front and at the very end. The source is
cropped to the face first for the same reason: a whole photograph carries a body, a room and a
palette that all read as things to draw, while a face crop carries little except the expression.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import (  # noqa: E402
    Graph, WIDGETS, WIDGET_TYPES, STYLE_LOCK, NEGATIVE, SETH_LOOK, STEPS,
    BLUE, GREEN, GREY, GROUP_LOCK, GROUP_SHOT, GROUP_OUT, GROUP_ENGINE,
)
from build_extras import engine, note  # noqa: E402

# Nodes only this app uses. Widget order is straight out of /object_info; `bboxes` is a
# forceInput socket, not a widget, so it takes no widgets_values slot.
WIDGETS.setdefault("LoadMediaPipeFaceLandmarker", ["model_name"])
WIDGETS.setdefault("MediaPipeFaceLandmarker",
                   ["detector_variant", "num_faces", "min_confidence", "missing_frame_fallback"])
WIDGETS.setdefault("CropByBBoxes", ["output_width", "output_height", "padding", "keep_aspect"])
for _w, _t in (("output_width", "INT"), ("output_height", "INT"), ("padding", "INT"),
               ("keep_aspect", "COMBO")):
    WIDGET_TYPES.setdefault(("CropByBBoxes", _w), _t)

FACE_MODEL = "mediapipe_face_fp32.safetensors"
# The plate the expression is painted ONTO, and it is deliberately the neutral one.
#
# The canonical headshot has a broad smile drawn into it, and our own law says a prompt cannot beat
# the plate: asked to scowl, the model kept the plate's cheerful brows and returned a puzzled grin,
# exactly the way Cadbury's serving tray survived being told his hands were empty. Repainting the
# plate ONCE to a resting face -- brows level, mouth closed and straight -- gives the expression
# somewhere to travel from. Regenerate it with `python3 build_expression.py --neutral-plate`.
SETH_PLATE = "cast-seth-headshot-neutral.png"
SIZE = 1024          # cast-seth-headshot.png is 1024x1024; the output matches it exactly
CROP_PADDING = 48    # px around the face bbox — enough to hold brows and chin

# ---------------------------------------------------------------- the prompt

# Image 1 is named as the edit target in the first clause. Everything about him is declared
# unchanged EXCEPT the expression, which is the only door left open for image 2 to walk through.
IDENTITY_LOCK = (
    "Image 1 is the drawing to edit, and the man in image 1 is the only character in the finished "
    f"picture. Redraw him exactly as he is: {SETH_LOOK}. Keep the shape of his face, his long wavy "
    "shoulder-length hair, his full bushy handlebar moustache, his dark v-neck t-shirt, the size "
    "and position of his head in the frame and the plain aged-paper background behind him "
    "identical to image 1, and keep every colour as muted and desaturated as image 1. One thing "
    "about him changes: his expression."
)

# Image 2 is given one job in plain words. Stated positively — what it IS for, and where the rest
# of it belongs — because naming the parts to ignore is what puts them in the picture.
#
# Both ends of every feature are named — brows up OR down, mouth corners up OR down — because
# asking only "how far the mouth opens, drawn big and clear" returned the same broad grin for all
# three test faces: the plate is smiling, so an open mouth of unstated shape defaults to the
# plate's. A sentence that named the moods instead ("a happy face makes him happy, an angry one
# makes him angry") made it worse, not better — it says "happy" twice, and law 3 holds even for a
# word used in a conditional. The features carry it; the mood words do not.
EXPRESSION_SOURCE = (
    "Image 2 is an expression chart: a photograph of a stranger, held up beside the drawing so "
    "that one thing can be read off it. Read from image 2 only the angle of the brows — raised, "
    "level, or pulled down and knotted — how wide the eyes are open or how tightly they are shut, "
    "where the gaze goes, and the shape of the mouth, both how far it opens and which way its "
    "corners turn. Give that same expression to the man in image 1, drawn in his own cartoon "
    "features and as strongly as the photograph shows it. The face, hair, skin, age and clothing "
    "of the photograph stay inside the photograph."
)

# Law 4: whatever must not be lost goes last. The headshot app ignored its expression field
# outright until the field was restated at the end of the chain. Both identity and expression are
# at risk here, so both are restated — but the EXPRESSION sentence is last, because it is the
# fragile one: the edit-target wiring already holds identity on its own, while a first draft that
# ended on identity returned the reference plate's own grin for a belly-laugh source.
RESTATEMENT = (
    "Above all: the man in the finished drawing is the man from image 1 — the same long wavy "
    "hair, the same bushy handlebar moustache, the same dark v-neck t-shirt. And his face is "
    "doing what the face in image 2 is doing: his brows at that angle, his eyes open exactly that "
    "wide, his mouth in exactly that shape, and the same mood as that face — his mouth turns down "
    "if that mouth turns down."
)

HOW_TO = """# Seth's expression, from any face

**Drop in a picture of a face. Get Seth's headshot wearing that face's expression.**

The picture is cropped to the face automatically and shown to the model at 1024x1024, the same
size as Seth's canonical headshot. The output is that headshot, repainted.

| Control | What it does |
|---|---|
| **1 · SOURCE FACE** | Any picture with a face in it. Photo or drawing, any size. |
| **2 · FACE CROP PADDING** | Pixels kept around the detected face. Raise it if the crop cuts the brows or chin off a small, off-centre face. |
| **3 · SEED** | Re-roll. Everything else is fixed. |

Two outputs: **RESULT** is Seth, **FACE READ** is the crop the model was actually shown — check
that one first when a run comes back with the wrong expression.

**No face in the picture?** `CropByBBoxes` returns the picture unchanged when detection finds
nothing, so the run still completes: the model is handed the whole picture as the expression
chart and reads whatever mood it can from it. Nothing crashes, but FACE READ shows you the
uncropped picture, which is how you tell that happened.

## Why Seth is image 1 and your picture is image 2

A reference image is an identity reference. Wired as peers, the stranger's face bleeds into
Seth's. So Seth's headshot is the **edit target** — the picture being repainted — and the source
is consulted for one property only, named at the front of the prompt and again at the very end.
Swapping the two inputs round is how you get a drawing of a stranger with Seth's hair.

## Speed

8 steps, cfg 1.0, denoise 1.0 — the Lightning LoRA tolerates nothing else. ~48s at 1024x1024.
"""


def build():
    g = Graph()
    note(g, "HOW TO USE — Seth's expression", HOW_TO)

    # --- 1 · the column you touch ----------------------------------------
    X, y = -40, -620
    source = g.add("LoadImage", "▶ 1 · SOURCE FACE — the expression to copy",
                   (X, y), (460, 400), {"image": SETH_PLATE, "upload": "image"},
                   outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    g.app_input(source, "image", image="1 · SOURCE FACE")
    y += 430

    # --- 2 · crop the source to its face ---------------------------------
    detector = g.add("LoadMediaPipeFaceLandmarker", "face detector", (X, y + 700), (400, 80),
                     {"model_name": FACE_MODEL},
                     outputs=[("FACE_DETECTION_MODEL", "FACE_DETECTION_MODEL")], collapsed=True)
    # 'both' runs the close-up and the far detector and keeps whichever found more faces. The
    # far one is what catches a small, off-centre face in a wide shot; the cost is one extra
    # detection pass on a 5MB model, invisible next to a 48s diffusion run.
    landmarks = g.add("MediaPipeFaceLandmarker", "find the face", (X, y + 750), (400, 180),
                      {"detector_variant": "both", "num_faces": 1, "min_confidence": 0.3,
                       "missing_frame_fallback": "empty"},
                      links={"face_detection_model": (detector, 0, "FACE_DETECTION_MODEL", False),
                             "image": (source, 0, "IMAGE", False)},
                      outputs=[("FACE_LANDMARKS", "FACE_LANDMARKS"),
                               ("BOUNDING_BOX", "BOUNDING_BOX")], collapsed=True)
    # keep_aspect="pad" rather than "stretch": stretching a non-square face box distorts the
    # very geometry — brow angle, mouth width — that this whole app exists to read.
    crop = g.add("CropByBBoxes", "▶ 2 · FACE CROP PADDING", (X, y), (460, 200),
                 {"output_width": SIZE, "output_height": SIZE, "padding": CROP_PADDING,
                  "keep_aspect": "pad"},
                 links={"image": (source, 0, "IMAGE", False),
                        "bboxes": (landmarks, 1, "BOUNDING_BOX", False)},
                 outputs=[("IMAGE", "IMAGE")], color=GREY)
    g.app_input(crop, "padding", padding="2 · FACE CROP PADDING (px around the face)")
    y += 230
    # CropByBBoxes hands back the ORIGINAL image, at its original size, when detection finds no
    # face. This scale is what makes that path safe: the expression chart is always 1024x1024,
    # whether it is a face crop or a whole picture that had no face in it.
    face = g.add("ImageScale", "expression chart at headshot size", (X, y + 800), (400, 150),
                 {"upscale_method": "lanczos", "width": SIZE, "height": SIZE, "crop": "disabled"},
                 links={"image": (crop, 0, "IMAGE", False)},
                 outputs=[("IMAGE", "IMAGE")], collapsed=True)

    latent = g.add("EmptySD3LatentImage", "output size — matches Seth's headshot",
                   (X, y + 850), (460, 130),
                   {"width": SIZE, "height": SIZE, "batch_size": 1},
                   outputs=[("LATENT", "LATENT")], color=GREY, collapsed=True)
    g.group("① YOUR PICTURE · everything you touch is in this column",
            (X - 30, -690, 520, y + 300), GROUP_SHOT)

    # --- 3 · the locks ----------------------------------------------------
    LX, LY, DY = 1240, -620, 46
    identity = g.add("PrimitiveStringMultiline", "IDENTITY LOCK — who Seth is",
                     (LX, LY), (430, 300), {"value": IDENTITY_LOCK},
                     outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    expression = g.add("PrimitiveStringMultiline", "EXPRESSION SOURCE — the one job image 2 has",
                       (LX, LY + DY), (430, 300), {"value": EXPRESSION_SOURCE},
                       outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    style = g.add("PrimitiveStringMultiline", "STYLE LOCK — house style",
                  (LX, LY + 2 * DY), (430, 300), {"value": STYLE_LOCK},
                  outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    restate = g.add("PrimitiveStringMultiline", "RESTATEMENT — identity and expression, last",
                    (LX, LY + 3 * DY), (430, 220), {"value": RESTATEMENT},
                    outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    plate = g.add("LoadImage", "SETH'S HEADSHOT — the edit target (image 1)",
                  (LX, LY + 4 * DY), (430, 400), {"image": SETH_PLATE, "upload": "image"},
                  outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], color=BLUE, collapsed=True)

    cat1 = g.add("StringConcatenate", "identity + expression source", (LX, LY + 5 * DY), (330, 150),
                 {"delimiter": " "},
                 links={"string_a": (identity, 0, "STRING", True),
                        "string_b": (expression, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    cat2 = g.add("StringConcatenate", "+ style lock", (LX, LY + 6 * DY), (330, 150),
                 {"delimiter": " "},
                 links={"string_a": (cat1, 0, "STRING", True),
                        "string_b": (style, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    prompt = g.add("StringConcatenate", "FINAL PROMPT — expand to read what the model got",
                   (LX, LY + 7 * DY), (330, 150), {"delimiter": " "},
                   links={"string_a": (cat2, 0, "STRING", True),
                          "string_b": (restate, 0, "STRING", True)},
                   outputs=[("STRING", "STRING")], collapsed=True)
    g.group("② SETH & STYLE LOCK · expand only to re-canonise him",
            (LX - 30, LY - 70, 480, 8 * DY + 90), GROUP_LOCK)

    # --- 4 · engine -------------------------------------------------------
    EY = LY + 8 * DY
    E = engine(g, LX, EY, want_bg=False)
    pos = g.add("TextEncodeQwenImageEditPlus", "encode (headshot + face crop + the prompt)",
                (LX, EY + 4 * DY), (400, 150), {"prompt": ""},
                links={"clip": (E["clip"], 0, "CLIP", False),
                       "vae": (E["vae"], 0, "VAE", False),
                       "image1": (plate, 0, "IMAGE", False),
                       "image2": (face, 0, "IMAGE", False),
                       "prompt": (prompt, 0, "STRING", True)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    neg = g.add("TextEncodeQwenImageEditPlus", "negative (only bites when cfg > 1)",
                (LX, EY + 5 * DY), (400, 130), {"prompt": NEGATIVE},
                links={"clip": (E["clip"], 0, "CLIP", False)},
                outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    g.group("engine · don't touch", (LX - 30, EY - 70, 480, 7 * DY + 90), GROUP_ENGINE)

    # --- 5 · sample and save ---------------------------------------------
    sampler = g.add("KSampler", "▶ 3 · SEED", (520, 640), (460, 270),
                    {"seed": 1, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                     "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
                    links={"model": (E["lora"], 0, "MODEL", False),
                           "positive": (pos, 0, "CONDITIONING", False),
                           "negative": (neg, 0, "CONDITIONING", False),
                           "latent_image": (latent, 0, "LATENT", False)},
                    outputs=[("LATENT", "LATENT")])
    g.app_input(sampler, "seed", seed="3 · SEED")
    decode = g.add("VAEDecode", "decode", (520, 950), (400, 80),
                   links={"samples": (sampler, 0, "LATENT", False),
                          "vae": (E["vae"], 0, "VAE", False)},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)
    g.app_output(g.add("SaveImage", "RESULT", (520, -620), (620, 1200),
                       {"filename_prefix": "cast/seth-expression"},
                       links={"images": (decode, 0, "IMAGE", False)}))
    g.app_output(g.add("SaveImage", "FACE READ — what the model was shown as image 2",
                       (1180, -620), (400, 460), {"filename_prefix": "cast/seth-expression-face"},
                       links={"images": (face, 0, "IMAGE", False)}, color=GREEN))
    g.group("③ RESULT · Seth, and the face he was read from",
            (490, -690, 1120, 1930), GROUP_OUT)
    return g


def main():
    g = build()
    ui = g.to_ui()
    for path, blob in ((os.path.join(HERE, "workflows", "seth-expression.json"), ui),
                       (os.path.join(HERE, "workflows", "seth-expression.app.json"), ui),
                       (os.path.join(HERE, "api", "seth-expression.api.json"), g.to_api())):
        json.dump(blob, open(path, "w"), indent=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
