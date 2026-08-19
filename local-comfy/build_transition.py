#!/usr/bin/env python3
"""headshot-transition — one cast member trades places with another, as video.

    python3 build_transition.py

Writes workflows/headshot-transition.json (+ .app.json) and api/headshot-transition.api.json,
plus transitions.json so scripts can pick a recipe by name.

MiniMax H3 with `first_frame` = headshot A and `last_frame` = headshot B. H3 has to *invent* the
handover in between, which is exactly the job. The two things that make it work for any pair:

1. **A trait line per character.** Ake has no legs — he is a goldfish in a bowl on a dolly — so
   "walks out of frame" is nonsense for him. Each character carries one sentence saying how they
   move, and the prompt says it out loud. Adding a fifth cast member is one dict entry.
2. **The transition is written for two anonymous characters**, never for a specific pair. The
   recipes say "the character who is leaving" and "the character who is arriving", so any recipe
   works with any two headshots.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import Graph, WIDGETS, WIDGET_TYPES, BLUE, GREEN, GREY, ENGINE  # noqa: E402
from build_workflows import GROUP_LOCK, GROUP_SHOT, GROUP_OUT, GROUP_ENGINE  # noqa: E402

# H3 nodes, widget order straight from /object_info
WIDGETS.update({
    "UNETLoader": ["unet_name", "weight_dtype"],
    "EasyCache": ["reuse_threshold", "start_percent", "end_percent", "verbose"],
    "MiniMaxH3ImageToVideo": ["prompt", "width", "height", "length"],
    "RandomNoise": ["noise_seed"],
    "KSamplerSelect": ["sampler_name"],
    "BasicScheduler": ["scheduler", "steps", "denoise"],
    "SamplerCustomAdvanced": [],
    "BasicGuider": [],
    "VAEDecodeAudio": [],
    "CreateVideo": ["fps", "bit_depth"],
    "SaveVideo": ["filename_prefix", "format", "codec"],
})
WIDGET_TYPES.update({
    ("RandomNoise", "noise_seed"): "INT",
    ("BasicScheduler", "steps"): "INT",
    ("BasicScheduler", "scheduler"): "COMBO",
    ("BasicScheduler", "denoise"): "FLOAT",
    ("MiniMaxH3ImageToVideo", "width"): "INT",
    ("MiniMaxH3ImageToVideo", "height"): "INT",
    ("MiniMaxH3ImageToVideo", "length"): "INT",
})

H3_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
SIZE, STEPS, LENGTH, FPS = 480, 10, 61, 24

# ---------------------------------------------------------------- the locks

SHOT_LOCK = (
    "A short clip from a 1933 newspaper comic strip, drawn in Fleischer-era rubber-hose cartoon "
    "style: confident smooth ink outlines of even weight, soft halftone dot shading, warm sepia "
    "duotone on aged grey-brown newsprint, every colour muted and desaturated as faded printing. "
    "The camera never moves and the framing never changes: one fixed head-and-shoulders frame with "
    "the same flat backdrop throughout, like a portrait slot that two characters take turns "
    "standing in. The clip begins on the character in the first frame and ends on the character in "
    "the last frame, and only ever those two characters appear."
)

WHO_A_LEAD = "The character who leaves is"
WHO_B_LEAD = "The character who arrives is"
TRANSITION_LEAD = "The handover happens like this:"

# H3 is a joint video+audio model. A prompt with no sound clause leaves the audio head to invent
# garbled human vocalisations, which also degrades the picture.
SOUND_LOCK = (
    "SOUND: the clatter and whoosh of the handover itself, a vaudeville woodblock knock, quiet and "
    "close. The only sounds are the practical noises the action itself makes. "
    # H3 scores a vaudeville-looking clip unprompted — it hears "1933 cartoon" and reaches for a
    # piano. Harmless in a muted sprite clip, but the transition app ships its audio, and a two
    # second sting of music under every handover is unusable. A negation is safe here: this clause
    # is about sound, so it cannot summon anything into the picture.
    "NEVER any music, melody, score, song, piano, orchestra or musical tones of any kind. "
    "No voices, no speech, no singing, no humming, no breathing, no human vocal sounds of "
    "any kind. No text, no lettering, no captions, no subtitles, no title cards."
)

# ---------------------------------------------------------------- the cast's traits
#
# One sentence per character, saying how that character MOVES. This is the whole
# non-humanoid story: Ake cannot walk, so every recipe reads as rolling for him.

TRAITS = {
    "seth": ("a lanky cartoon man with long wavy shoulder-length brown hair and a bushy handlebar "
             "moustache, in a very dark v-neck t-shirt; he moves on two legs and can walk, run, "
             "duck and jump"),
    "wilson": ("a living picket-fence panel — the fence panel IS his body — with two short stubby "
               "metal legs in heavy black boots, a floppy bucket hat on top and two cartoon eyes "
               "floating in the shadow beneath its brim; he stumps stiffly about on those short "
               "boots, upright as a plank, and never bends at a waist because he has none"),
    "cadbury": ("a thin, composed butler in a black tailcoat with slicked hair, a pencil moustache "
                "and a brass clockwork monocle on the eye at the viewer's right; he walks smoothly "
                "and unhurried, and can bow or sidestep"),
    "ake": ("a plump goldfish inside a spherical glass fishbowl mounted on a small wheeled wooden "
            "dolly, wearing a little dark cap; he has no legs and never walks — the dolly rolls "
            "him along, he arrives and leaves by rolling, and the water sloshes in the bowl as he "
            "moves"),
}

# ---------------------------------------------------------------- the transition library
#
# Written for two anonymous characters, so any recipe fits any pair.

TRANSITIONS = {
    "walk-across": (
        "the character who leaves turns and walks out of frame past the left edge, and as the frame "
        "empties the character who arrives walks in from the right edge and stops in exactly the "
        "same spot, settling into the portrait"),
    "slide-swap": (
        "both characters slide sideways as if standing on a moving belt — the one who leaves glides "
        "out past the left edge while the one who arrives glides in from the right — and the "
        "arriving character comes to rest square in the middle of the frame"),
    "curtain-wipe": (
        "a heavy stage curtain sweeps across the frame from one side, hiding everything for a "
        "moment, and when it sweeps back the character who arrives is standing in the portrait "
        "instead"),
    "iris-wipe": (
        "the picture irises down — a circle of black closing in from the edges until it pinches "
        "shut on the character who leaves — then irises open again from a point to reveal the "
        "character who arrives standing in the same spot"),
    "spin-swap": (
        "the whole portrait spins on the spot like a flipped coin, blurring as it turns, and when it "
        "slows and settles the character who arrives is the one facing forward"),
    "trapdoor": (
        "a trapdoor bangs open under the character who leaves and drops him straight down out of "
        "frame with a puff of dust; a beat later the same trapdoor flips up and the character who "
        "arrives rises up through it into the portrait"),
    "vaudeville-hook": (
        "a long crooked shepherd's hook reaches in from the side of the frame, catches the character "
        "who leaves and yanks him out sideways; the character who arrives strolls in from the other "
        "side and takes the spot"),
    "puff-of-smoke": (
        "the character who leaves vanishes in a sudden puff of smoke, the smoke swells and curls to "
        "fill the frame, and as it thins and drifts away the character who arrives is standing "
        "there in his place"),
    "anvil-drop": (
        "a heavy iron anvil drops into frame on top of the character who leaves with a cloud of "
        "dust, and when the dust settles it is the character who arrives standing there, unbothered"),
    "page-turn": (
        "the whole picture is a sheet of newsprint that peels up at one corner and turns like a page, "
        "and the page underneath is the same portrait with the character who arrives in it"),
    "cannon-launch": (
        "the character who leaves is fired out of frame like a circus cannonball, leaving a little "
        "smoke ring behind, and the character who arrives drops into the empty spot from above and "
        "bounces once before settling"),
    "squash-and-stretch": (
        "the character who leaves squashes down into a rubbery blob, the blob wobbles, and then it "
        "stretches back up into the character who arrives, who shakes himself straight"),
}

DEFAULT = "walk-across"

HOW_TO = """# Headshot transition

**Two cast headshots in, a clip of one trading places with the other out.**

MiniMax H3 pins the first frame to headshot A and the last frame to headshot B, and invents the
handover in between. 480x480, 10 steps, 61 frames — about **50 seconds a clip**.

| Control | What it does |
|---|---|
| **1 · HEADSHOT A** | The character who starts in frame. |
| **2 · WHO A IS** | One sentence on who he is and **how he moves**. Paste from the list below. |
| **3 · HEADSHOT B** | The character who ends in frame. |
| **4 · WHO B IS** | Same for the arriving character. |
| **5 · THE TRANSITION** | Paste one of the twelve recipes below. |
| **6 · SEED + STEPS** | Re-roll here. 10 steps is the working default. |

## Why the trait lines exist

Ake is a goldfish in a bowl on a wheeled dolly. "Walks out of frame" is nonsense for him, and H3
will either give him legs or ignore the instruction. His trait line says he has no legs and rolls,
so every recipe reads as rolling. Any new character is one more line.

## The transition recipes

{recipes}

## The trait lines

{traits}
"""


def build():
    g = Graph()
    recipes = "\n".join(f"- **`{k}`** — {v}." for k, v in TRANSITIONS.items())
    traits = "\n".join(f"- **{k}** — {v}." for k, v in TRAITS.items())
    g.add("MarkdownNote", "HOW TO USE — headshot transition", (-560, -620), (520, 1100),
          {"text": HOW_TO.format(recipes=recipes, traits=traits)})

    X, y = -40, -620
    frame_a = g.add("LoadImage", "▶ 1 · HEADSHOT A — who starts in frame",
                    (X, y), (460, 400), {"image": "cast-seth-headshot.png", "upload": "image"},
                    outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    g.app_input(frame_a, "image", image="1 · HEADSHOT A — who starts in frame")
    y += 430
    who_a = g.add("PrimitiveStringMultiline", "▶ 2 · WHO A IS — and how he moves",
                  (X, y), (460, 220), {"value": TRAITS["seth"]},
                  outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(who_a, "value", value="2 · WHO A IS — and how he moves")
    y += 250
    frame_b = g.add("LoadImage", "▶ 3 · HEADSHOT B — who ends in frame",
                    (X, y), (460, 400), {"image": "cast-wilson-headshot.png", "upload": "image"},
                    outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    g.app_input(frame_b, "image", image="3 · HEADSHOT B — who ends in frame")
    y += 430
    who_b = g.add("PrimitiveStringMultiline", "▶ 4 · WHO B IS — and how he moves",
                  (X, y), (460, 220), {"value": TRAITS["wilson"]},
                  outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(who_b, "value", value="4 · WHO B IS — and how he moves")
    y += 250
    transition = g.add("PrimitiveStringMultiline", "▶ 5 · THE TRANSITION (paste a recipe)",
                       (X, y), (460, 260), {"value": TRANSITIONS[DEFAULT]},
                       outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(transition, "value", value="5 · THE TRANSITION")
    g.group("① THE HANDOVER · everything you touch is in this column",
            (X - 30, -690, 520, y + 350), GROUP_SHOT)

    # --- locked prompt ----------------------------------------------------
    LX, LY, DY = 1240, -620, 46
    shot_lock = g.add("PrimitiveStringMultiline", "SHOT LOCK — style, camera and framing",
                      (LX, LY), (430, 300), {"value": SHOT_LOCK},
                      outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    a_lead = g.add("PrimitiveStringMultiline", "WHO-A LEAD-IN", (LX, LY + DY), (430, 120),
                   {"value": WHO_A_LEAD}, outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    b_lead = g.add("PrimitiveStringMultiline", "WHO-B LEAD-IN", (LX, LY + 2 * DY), (430, 120),
                   {"value": WHO_B_LEAD}, outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    t_lead = g.add("PrimitiveStringMultiline", "TRANSITION LEAD-IN", (LX, LY + 3 * DY), (430, 120),
                   {"value": TRANSITION_LEAD}, outputs=[("STRING", "STRING")], color=BLUE,
                   collapsed=True)
    sound_lock = g.add("PrimitiveStringMultiline", "SOUND LOCK — H3 needs this even when muted",
                       (LX, LY + 4 * DY), (430, 190), {"value": SOUND_LOCK},
                       outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)

    def cat(title, a, b, row):
        return g.add("StringConcatenate", title, (LX, LY + row * DY), (330, 150), {"delimiter": " "},
                     links={"string_a": (a, 0, "STRING", True), "string_b": (b, 0, "STRING", True)},
                     outputs=[("STRING", "STRING")], collapsed=True)

    acc = cat("shot lock + who-A lead", shot_lock, a_lead, 5)
    acc = cat("+ who A is", acc, who_a, 6)
    acc = cat("+ who-B lead", acc, b_lead, 7)
    acc = cat("+ who B is", acc, who_b, 8)
    acc = cat("+ transition lead", acc, t_lead, 9)
    acc = cat("+ the transition", acc, transition, 10)
    prompt = cat("FINAL PROMPT — expand to read what H3 got", acc, sound_lock, 11)
    g.group("② SHOT & SOUND LOCK · leave alone", (LX - 30, LY - 70, 480, 12 * DY + 90), GROUP_LOCK)

    # --- engine -----------------------------------------------------------
    EX, EY = LX, LY + 13 * DY
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
    scale_a = g.add("ImageScale", "scale headshot A to the canvas", (EX, EY + 5 * DY), (400, 150),
                    {"upscale_method": "lanczos", "width": SIZE, "height": SIZE, "crop": "disabled"},
                    links={"image": (frame_a, 0, "IMAGE", False)},
                    outputs=[("IMAGE", "IMAGE")], collapsed=True)
    scale_b = g.add("ImageScale", "scale headshot B to the canvas", (EX, EY + 6 * DY), (400, 150),
                    {"upscale_method": "lanczos", "width": SIZE, "height": SIZE, "crop": "disabled"},
                    links={"image": (frame_b, 0, "IMAGE", False)},
                    outputs=[("IMAGE", "IMAGE")], collapsed=True)
    h3 = g.add("MiniMaxH3ImageToVideo", "H3 — A pinned first, B pinned last",
               (EX, EY + 7 * DY), (400, 200),
               {"prompt": "", "width": SIZE, "height": SIZE, "length": LENGTH},
               links={"clip": (clip, 0, "CLIP", False),
                      "vae": (vvae, 0, "VAE", False),
                      "prompt": (prompt, 0, "STRING", True),
                      "first_frame": (scale_a, 0, "IMAGE", False),
                      "last_frame": (scale_b, 0, "IMAGE", False)},
               outputs=[("CONDITIONING", "CONDITIONING"), ("LATENT", "LATENT")], collapsed=True)
    guider = g.add("BasicGuider", "guider", (EX, EY + 8 * DY), (400, 80),
                   links={"model": (cache, 0, "MODEL", False),
                          "conditioning": (h3, 0, "CONDITIONING", False)},
                   outputs=[("GUIDER", "GUIDER")], color=ENGINE, collapsed=True)
    sampler_sel = g.add("KSamplerSelect", "sampler", (EX, EY + 9 * DY), (400, 90),
                        {"sampler_name": "res_multistep"},
                        outputs=[("SAMPLER", "SAMPLER")], color=ENGINE, collapsed=True)
    g.group("engine · don't touch", (EX - 30, EY - 70, 480, 10 * DY + 90), GROUP_ENGINE)

    noise = g.add("RandomNoise", "▶ 6 · SEED", (520, 640), (460, 110), {"noise_seed": 11},
                  outputs=[("NOISE", "NOISE")], color=GREY)
    g.app_input(noise, "noise_seed", noise_seed="6 · SEED")
    sched = g.add("BasicScheduler", "▶ 6 · STEPS", (520, 780), (460, 160),
                  {"scheduler": "simple", "steps": STEPS, "denoise": 1.0},
                  links={"model": (cache, 0, "MODEL", False)},
                  outputs=[("SIGMAS", "SIGMAS")], color=GREY)
    g.app_input(sched, "steps", steps="6 · STEPS (10 is the working default)")
    sample = g.add("SamplerCustomAdvanced", "sample", (520, 970), (400, 130),
                   links={"noise": (noise, 0, "NOISE", False),
                          "guider": (guider, 0, "GUIDER", False),
                          "sampler": (sampler_sel, 0, "SAMPLER", False),
                          "sigmas": (sched, 0, "SIGMAS", False),
                          "latent_image": (h3, 1, "LATENT", False)},
                   outputs=[("LATENT", "LATENT"), ("LATENT", "LATENT")], collapsed=True)
    frames = g.add("VAEDecode", "decode frames", (520, 1020), (400, 80),
                   links={"samples": (sample, 0, "LATENT", False),
                          "vae": (vvae, 0, "VAE", False)},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)
    audio = g.add("VAEDecodeAudio", "decode audio", (520, 1070), (400, 80),
                  links={"samples": (sample, 0, "LATENT", False),
                         "vae": (avae, 0, "VAE", False)},
                  outputs=[("AUDIO", "AUDIO")], collapsed=True)
    video = g.add("CreateVideo", "make the video", (520, 1120), (400, 130),
                  {"fps": FPS, "bit_depth": 8},
                  links={"images": (frames, 0, "IMAGE", False),
                         "audio": (audio, 0, "AUDIO", False)},
                  outputs=[("VIDEO", "VIDEO")], collapsed=True)
    g.app_output(g.add("SaveVideo", "RESULT", (520, -620), (620, 1200),
                       {"filename_prefix": "video/transition", "format": "auto", "codec": "auto"},
                       links={"video": (video, 0, "VIDEO", False)}))
    g.group("③ RESULT · the clip", (490, -690, 680, 1930), GROUP_OUT)
    return g


def main():
    g = build()
    ui = g.to_ui()
    for path, blob in ((os.path.join(HERE, "workflows", "headshot-transition.json"), ui),
                       (os.path.join(HERE, "workflows", "headshot-transition.app.json"), ui),
                       (os.path.join(HERE, "api", "headshot-transition.api.json"), g.to_api()),
                       (os.path.join(HERE, "transitions.json"),
                        {"transitions": TRANSITIONS, "traits": TRAITS})):
        json.dump(blob, open(path, "w"), indent=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
