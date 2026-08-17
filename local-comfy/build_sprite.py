#!/usr/bin/env python3
"""sprite-clip — one cast member performing one move, on a flat key colour, for spriting.

    python3 build_sprite.py

The clip is not the deliverable. `sprite_sheet.py` turns it into a normalised PNG atlas plus
JSON, which is what web animation, Phaser, Pixi and Godot actually want. This app's whole job is
to produce a clip that a packer can key and align without fighting it:

- **A flat magenta stage.** Matting a keyed colour is exact and identical frame to frame, where
  BiRefNet on aged-paper texture wobbles a pixel or two per frame — and that wobble IS sprite
  jitter.
- **A locked camera and a fixed ground line.** Every frame has to be the same shot of the same
  stage, so the packer can align on the character's own feet instead of guessing.
- **Side-on staging.** Sprite moves read from the side; a three-quarter turn makes the silhouette
  change width mid-cycle, which no alignment can fix.

The character trait line is the same mechanism as `headshot-transition`: Ake has no legs, so
"walk cycle" reads as rolling for him.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import Graph, WIDGETS, WIDGET_TYPES, BLUE, GREEN, GREY, ENGINE  # noqa: E402
from build_workflows import GROUP_LOCK, GROUP_SHOT, GROUP_OUT, GROUP_ENGINE  # noqa: E402
from build_transition import (  # noqa: E402
    TRAITS, SOUND_LOCK, H3_UNET, H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE,
)

WIDGETS.setdefault("EmptyImage", ["width", "height", "batch_size", "color"])
WIDGET_TYPES.setdefault(("EmptyImage", "width"), "INT")

SIZE, STEPS, LENGTH, FPS = 512, 10, 61, 24
KEY_MAGENTA = 0xFF00FF

# ---------------------------------------------------------------- the locks

SPRITE_LOCK = (
    "A sprite animation reference for a 1933 rubber-hose cartoon character, drawn with confident "
    "smooth ink outlines of even weight and soft halftone shading, in the muted sepia inks of aged "
    "newsprint. The character is alone on a completely flat, evenly filled bright magenta screen "
    "that fills every part of the picture behind and around him — the magenta is a blank keying "
    "screen, so it stays one single solid colour with nothing drawn on it, no floor, no horizon, "
    "no shadow and no scenery. The camera is locked off and never moves, pans or zooms: one fixed "
    "wide side-on view, the character seen from his side, his whole body inside the frame with "
    "clear space above his head and below his feet, his feet staying on the same invisible ground "
    "line for the whole clip, and his size in the frame never changing. He performs one single "
    "move, cleanly and completely, and every drawing of him stays exactly the character in the "
    "first frame. The move is:"
)

WHO_LEAD = "The character is"

# Restated at the very END of the prompt. A long move line pushes the staging clause out of
# effective range and H3 quietly swaps the keying screen for aged paper — which the packer then
# cannot key at all. Whatever must not be lost goes last.
STAGE_RESTATE = (
    "Throughout the whole clip the background behind him stays one completely flat, evenly filled "
    "bright magenta screen of a single solid colour, covering every part of the picture behind and "
    "around him, with no paper texture, no floor, no shadow and nothing else drawn on it."
)

# ---------------------------------------------------------------- the move library

MOVES = {
    "idle-breathe": (
        "a gentle idle: he stands square, breathing, weight shifting a little from one side to the "
        "other, and returns exactly to the pose he started in"),
    "roll-cycle": (
        "he rolls steadily to the right at a constant speed on his wheels, the wheels turning and "
        "the body rocking slightly with the roll, ending on the pose he began with so the cycle "
        "loops — for a character with legs, read this as a smooth walk"),
    "walk-cycle": (
        "a full walk cycle in place, striding steadily to the right through contact, down, passing "
        "and up positions, ending on the same pose he began with so the cycle loops"),
    "run-cycle": (
        "a fast run cycle in place, leaning forward, arms pumping, both feet leaving the ground at "
        "the stretch, ending on the pose he began with so the cycle loops"),
    "jump": (
        "a standing jump: he crouches, springs up with arms rising, hangs at the top, and lands "
        "back on the ground line absorbing the impact"),
    "crouch": (
        "he ducks down into a low crouch, holds it a beat, and rises back to standing"),
    "punch": (
        "he winds up, throws one straight punch out to the right, holds the extension a beat, and "
        "pulls the arm back to his guard"),
    "kick": (
        "he plants one foot, swings the other leg up in a high kick to the right, holds the "
        "extension, and sets the foot back down"),
    "block": (
        # H3 read the first version as walking forward with his hands up. "Braced" and "guard"
        # both imply advancing, so the planted feet have to be stated as the loudest thing in
        # the line, twice, before the arms are mentioned at all.
        "he stands still on one spot with both feet planted flat on the ground and never takes a "
        "step, never walks and never moves across the screen; standing exactly where he is, he "
        "crosses both forearms up in front of his face and chest to shield himself, hunches his "
        "shoulders and tucks his head down behind his arms, holds that braced position, then "
        "lowers his arms back to his sides, his feet staying in the same place the whole time"),
    "wave": (
        "he raises one arm and waves broadly twice, then lowers the arm back to his side"),
    "take-a-bow": (
        "he bows deeply from the waist with one arm sweeping across himself, holds the bow, and "
        "straightens back up"),
    "surprise-jolt": (
        "he jolts upright in surprise, whole body popping up off the ground line for a beat with "
        "limbs flung out, then drops back down and settles"),
    "faint": (
        "he wobbles, stiffens, and topples over backwards out of the standing pose, landing flat"),
}

DEFAULT_MOVE = "walk-cycle"

HOW_TO = """# Sprite clip

**One character, one move, on a flat magenta screen — the raw material for a sprite sheet.**

The clip is not the deliverable. Run it through `sprite_sheet.py` and you get a normalised PNG
atlas, a JSON descriptor, a CSS `steps()` snippet and an animated preview:

```
python3 sprite_sheet.py <clip.mp4> <name> --frames 8
```

| Control | What it does |
|---|---|
| **1 · CHARACTER PLATE** | A transparent cutout of the character. It is composited onto the magenta screen for you. |
| **2 · WHO HE IS** | The trait line — how this character moves. Ake has no legs; his line says the dolly rolls him. |
| **3 · THE MOVE** | Paste one of the recipes below. |
| **4 · SEED + STEPS** | Re-roll here. 10 steps is the working default. |

## Why magenta

Keying a flat colour is exact and identical on every frame. Matting the same character off aged
paper with BiRefNet wobbles a pixel or two per frame, and that wobble *is* sprite jitter. The
packer keys the magenta, finds each frame's true bounding box, and aligns every cell on the
character's own feet.

## The moves

{moves}

## The trait lines

{traits}
"""


def build():
    g = Graph()
    moves = "\n".join(f"- **`{k}`** — {v}." for k, v in MOVES.items())
    traits = "\n".join(f"- **{k}** — {v}." for k, v in TRAITS.items())
    g.add("MarkdownNote", "HOW TO USE — sprite clip", (-560, -620), (520, 1100),
          {"text": HOW_TO.format(moves=moves, traits=traits)})

    X, y = -40, -620
    plate = g.add("LoadImage", "▶ 1 · CHARACTER PLATE — a transparent cutout",
                  (X, y), (460, 400), {"image": "cast-cutout-seth.png", "upload": "image"},
                  outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    g.app_input(plate, "image", image="1 · CHARACTER PLATE (transparent PNG)")
    y += 430
    who = g.add("PrimitiveStringMultiline", "▶ 2 · WHO HE IS — and how he moves",
                (X, y), (460, 240), {"value": TRAITS["seth"]},
                outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(who, "value", value="2 · WHO HE IS — and how he moves")
    y += 270
    move = g.add("PrimitiveStringMultiline", "▶ 3 · THE MOVE (paste a recipe)",
                 (X, y), (460, 260), {"value": MOVES[DEFAULT_MOVE]},
                 outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(move, "value", value="3 · THE MOVE")
    g.group("① THE MOVE · everything you touch is in this column",
            (X - 30, -690, 520, y + 350), GROUP_SHOT)

    # --- the magenta stage -------------------------------------------------
    CX, CY, DY = 560, -620, 46
    screen = g.add("EmptyImage", "flat magenta keying screen", (CX, CY), (330, 150),
                   {"width": SIZE, "height": SIZE, "batch_size": 1, "color": KEY_MAGENTA},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)
    fitted = g.add("ImageScale", "fit the plate to the canvas", (CX, CY + DY), (330, 150),
                   {"upscale_method": "lanczos", "width": SIZE, "height": SIZE, "crop": "disabled"},
                   links={"image": (plate, 0, "IMAGE", False)},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)
    alpha = g.add("InvertMask", "plate alpha", (CX, CY + 2 * DY), (330, 80),
                  links={"mask": (plate, 1, "MASK", False)},
                  outputs=[("MASK", "MASK")], collapsed=True)
    alpha_img = g.add("MaskToImage", "alpha as image", (CX, CY + 3 * DY), (330, 80),
                      links={"mask": (alpha, 0, "MASK", False)},
                      outputs=[("IMAGE", "IMAGE")], collapsed=True)
    alpha_fit = g.add("ImageScale", "fit the alpha the same way", (CX, CY + 4 * DY), (330, 150),
                      {"upscale_method": "lanczos", "width": SIZE, "height": SIZE,
                       "crop": "disabled"},
                      links={"image": (alpha_img, 0, "IMAGE", False)},
                      outputs=[("IMAGE", "IMAGE")], collapsed=True)
    fit_mask = g.add("ImageToMask", "back to a mask", (CX, CY + 5 * DY), (330, 90),
                     {"channel": "red"},
                     links={"image": (alpha_fit, 0, "IMAGE", False)},
                     outputs=[("MASK", "MASK")], collapsed=True)
    first_frame = g.add("ImageCompositeMasked", "the character on magenta",
                        (CX, CY + 6 * DY), (330, 150), {"x": 0, "y": 0, "resize_source": False},
                        links={"destination": (screen, 0, "IMAGE", False),
                               "source": (fitted, 0, "IMAGE", False),
                               "mask": (fit_mask, 0, "MASK", False)},
                        outputs=[("IMAGE", "IMAGE")], collapsed=True)
    g.add("PreviewImage", "frame 0, before H3 sees it", (CX, CY + 7 * DY + 20), (330, 330),
          links={"images": (first_frame, 0, "IMAGE", False)})
    g.group("② THE MAGENTA STAGE · built for you", (CX - 30, CY - 70, 400, 8 * DY + 90), GROUP_OUT)

    # --- locked prompt -----------------------------------------------------
    LX, LY = 1040, -620
    sprite_lock = g.add("PrimitiveStringMultiline", "SPRITE LOCK — stage, camera and framing",
                        (LX, LY), (430, 300), {"value": SPRITE_LOCK},
                        outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    who_lead = g.add("PrimitiveStringMultiline", "WHO LEAD-IN", (LX, LY + DY), (430, 120),
                     {"value": WHO_LEAD}, outputs=[("STRING", "STRING")], color=BLUE,
                     collapsed=True)
    sound_lock = g.add("PrimitiveStringMultiline", "SOUND LOCK — H3 needs this even when muted",
                       (LX, LY + 2 * DY), (430, 190), {"value": SOUND_LOCK},
                       outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)

    def cat(title, a, b, row):
        return g.add("StringConcatenate", title, (LX, LY + row * DY), (330, 150), {"delimiter": " "},
                     links={"string_a": (a, 0, "STRING", True), "string_b": (b, 0, "STRING", True)},
                     outputs=[("STRING", "STRING")], collapsed=True)

    acc = cat("sprite lock + the move", sprite_lock, move, 3)
    acc = cat("+ who lead-in", acc, who_lead, 4)
    acc = cat("+ who he is", acc, who, 5)
    prompt = cat("FINAL PROMPT — expand to read what H3 got", acc, sound_lock, 6)
    g.group("③ SPRITE & SOUND LOCK · leave alone", (LX - 30, LY - 70, 480, 7 * DY + 90), GROUP_LOCK)

    # --- engine ------------------------------------------------------------
    EX, EY = LX, LY + 8 * DY
    unet = g.add("UNETLoader", "MiniMax H3 (video)", (EX, EY), (400, 110),
                 {"unet_name": H3_UNET, "weight_dtype": "default"},
                 outputs=[("MODEL", "MODEL")], color=ENGINE, collapsed=True)
    cache = g.add("EasyCache", "EasyCache", (EX, EY + DY), (400, 160),
                  {"reuse_threshold": 0.2, "start_percent": 0.15, "end_percent": 0.95,
                   "verbose": True},
                  links={"model": (unet, 0, "MODEL", False)},
                  outputs=[("MODEL", "MODEL")], color=ENGINE, collapsed=True)
    clip = g.add("CLIPLoader", "H3 text encoder", (EX, EY + 2 * DY), (400, 130),
                 {"clip_name": H3_CLIP, "type": "minimax", "device": "default"},
                 outputs=[("CLIP", "CLIP")], color=ENGINE, collapsed=True)
    vvae = g.add("VAELoader", "video VAE", (EX, EY + 3 * DY), (400, 80),
                 {"vae_name": H3_VIDEO_VAE}, outputs=[("VAE", "VAE")], color=ENGINE, collapsed=True)
    avae = g.add("VAELoader", "audio VAE", (EX, EY + 4 * DY), (400, 80),
                 {"vae_name": H3_AUDIO_VAE}, outputs=[("VAE", "VAE")], color=ENGINE, collapsed=True)
    h3 = g.add("MiniMaxH3ImageToVideo", "H3 — the magenta frame pinned first",
               (EX, EY + 5 * DY), (400, 200),
               {"prompt": "", "width": SIZE, "height": SIZE, "length": LENGTH},
               links={"clip": (clip, 0, "CLIP", False),
                      "vae": (vvae, 0, "VAE", False),
                      "prompt": (prompt, 0, "STRING", True),
                      "first_frame": (first_frame, 0, "IMAGE", False)},
               outputs=[("CONDITIONING", "CONDITIONING"), ("LATENT", "LATENT")], collapsed=True)
    guider = g.add("BasicGuider", "guider", (EX, EY + 6 * DY), (400, 80),
                   links={"model": (cache, 0, "MODEL", False),
                          "conditioning": (h3, 0, "CONDITIONING", False)},
                   outputs=[("GUIDER", "GUIDER")], color=ENGINE, collapsed=True)
    sampler_sel = g.add("KSamplerSelect", "sampler", (EX, EY + 7 * DY), (400, 90),
                        {"sampler_name": "res_multistep"},
                        outputs=[("SAMPLER", "SAMPLER")], color=ENGINE, collapsed=True)
    g.group("engine · don't touch", (EX - 30, EY - 70, 480, 8 * DY + 90), GROUP_ENGINE)

    noise = g.add("RandomNoise", "▶ 4 · SEED", (560, 700), (460, 110), {"noise_seed": 11},
                  outputs=[("NOISE", "NOISE")], color=GREY)
    g.app_input(noise, "noise_seed", noise_seed="4 · SEED")
    sched = g.add("BasicScheduler", "▶ 4 · STEPS", (560, 840), (460, 160),
                  {"scheduler": "simple", "steps": STEPS, "denoise": 1.0},
                  links={"model": (cache, 0, "MODEL", False)},
                  outputs=[("SIGMAS", "SIGMAS")], color=GREY)
    g.app_input(sched, "steps", steps="4 · STEPS (10 is the working default)")
    sample = g.add("SamplerCustomAdvanced", "sample", (560, 1030), (400, 130),
                   links={"noise": (noise, 0, "NOISE", False),
                          "guider": (guider, 0, "GUIDER", False),
                          "sampler": (sampler_sel, 0, "SAMPLER", False),
                          "sigmas": (sched, 0, "SIGMAS", False),
                          "latent_image": (h3, 1, "LATENT", False)},
                   outputs=[("LATENT", "LATENT"), ("LATENT", "LATENT")], collapsed=True)
    frames = g.add("VAEDecode", "decode frames", (560, 1080), (400, 80),
                   links={"samples": (sample, 0, "LATENT", False),
                          "vae": (vvae, 0, "VAE", False)},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)
    audio = g.add("VAEDecodeAudio", "decode audio", (560, 1130), (400, 80),
                  links={"samples": (sample, 0, "LATENT", False),
                         "vae": (avae, 0, "VAE", False)},
                  outputs=[("AUDIO", "AUDIO")], collapsed=True)
    video = g.add("CreateVideo", "make the clip", (560, 1180), (400, 130),
                  {"fps": FPS, "bit_depth": 8},
                  links={"images": (frames, 0, "IMAGE", False),
                         "audio": (audio, 0, "AUDIO", False)},
                  outputs=[("VIDEO", "VIDEO")], collapsed=True)
    g.app_output(g.add("SaveVideo", "RESULT", (1560, -620), (620, 900),
                       {"filename_prefix": "video/sprite", "format": "auto", "codec": "auto"},
                       links={"video": (video, 0, "VIDEO", False)}))
    g.group("④ RESULT · the clip to pack", (1530, -690, 680, 1000), GROUP_OUT)
    return g


def main():
    g = build()
    ui = g.to_ui()
    for path, blob in ((os.path.join(HERE, "workflows", "sprite-clip.json"), ui),
                       (os.path.join(HERE, "workflows", "sprite-clip.app.json"), ui),
                       (os.path.join(HERE, "api", "sprite-clip.api.json"), g.to_api()),
                       (os.path.join(HERE, "moves.json"), {"moves": MOVES, "traits": TRAITS})):
        json.dump(blob, open(path, "w"), indent=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
