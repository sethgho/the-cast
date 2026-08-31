#!/usr/bin/env python3
"""H3 storyboard: three short shots in one clip, one emotional beat each.

    python3 build_storyboard.py    # workflows/h3-storyboard.json + api/

Built to the method in the `h3-storyboard` skill, whose central finding is measured and
counter-intuitive: **a shot cannot hold more than two or three facial beats.** Pack nine beats
into one seven-second close-up and H3 averages them into a frozen face -- 37-42 dB PSNR at the
emotional peak, no error, no warning. Split the same beats across three 2-3 second shots and it
drops to 22-23 dB and every beat lands. Splitting is the mechanism; everything else is trim.

Three shot fields, concatenated in order, so the structure is visible in the form instead of
buried in one prose blob. The rules each field is written to:

  camera first    If what the character looks at is off-screen, H3 has them look at the LENS.
                  Fixing that by adding "he turns to face camera" fights the story and loses.
                  Move the CAMERA to where he is looking, then forbid looking at the lens.
  one beat        One thing happens per shot. Beats that follow a spoken line go in the NEXT
                  shot -- H3 treats a whole shot as "this person is talking" and will not close
                  the mouth partway through for a blink or a swallow.
  dialogue steals A <d> line makes H3 reallocate frames TOWARD that shot and away from the one
                  after it -- measured 55 -> 84 frames on the line's shot, 56 -> 42 on the next,
                  at which point a background character vanished entirely. So the line goes on
                  the LAST shot here: the squeeze then has nowhere to land.
  tail slack      H3 degrades into noise blocks in the final 1.2-1.7s. The frame count buys
                  1.3-1.5s of slack past the last beat, and the tail still gets checked.

LENGTH is on H3's 17n+5 grid. 209 = 8.71s, the skill's budget for "action, reaction, settle";
a clip WITH cuts nominally wants 243+, but 243 at 0.74MP does not fit our 12GB card.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import Graph, WIDGETS, WIDGET_TYPES, BLUE, GREEN, GREY  # noqa: E402
from build_extras import note  # noqa: E402
from build_transition import H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE  # noqa: E402

UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
W, H, LENGTH, STEPS, FPS, SEED = 1152, 640, 209, 12, 24, 7
PLATE = "rubberhose-kf-seth.png"

for _n, _ws in (("UNETLoader", ["unet_name", "weight_dtype"]),
                ("MiniMaxH3TurboLoRA", ["lora_name", "strength", "low_vram"]),
                ("MiniMaxH3ReferenceToVideo",
                 ["prompt", "width", "height", "length", "ref_image_size"]),
                ("RandomNoise", ["noise_seed"]),
                ("KSamplerSelect", ["sampler_name"]),
                ("BasicScheduler", ["scheduler", "steps", "denoise"]),
                ("BasicGuider", []), ("SamplerCustomAdvanced", []), ("VAEDecodeAudio", []),
                ("CreateVideo", ["fps", "bit_depth"]),
                ("SaveVideo", ["filename_prefix", "format", "codec"])):
    WIDGETS.setdefault(_n, _ws)
for _k, _t in ((("MiniMaxH3ReferenceToVideo", "length"), "INT"),
               (("MiniMaxH3ReferenceToVideo", "width"), "INT"),
               (("MiniMaxH3ReferenceToVideo", "height"), "INT"),
               (("MiniMaxH3ReferenceToVideo", "ref_image_size"), "COMBO"),
               (("MiniMaxH3ReferenceToVideo", "prompt"), "STRING"),
               (("RandomNoise", "noise_seed"), "INT"),
               (("BasicScheduler", "steps"), "INT"),
               (("BasicScheduler", "scheduler"), "COMBO"),
               (("BasicScheduler", "denoise"), "FLOAT")):
    WIDGET_TYPES.setdefault(_k, _t)

LOOK = ("1933 rubber-hose cartoon animation, confident black ink outlines, soft halftone shading, "
        "warm sepia and cream on aged newsprint. A lanky cartoon man with long wavy shoulder-length "
        "brown hair and a full bushy handlebar moustache, in a dark v-neck t-shirt, at a cluttered "
        "desk in a small workshop.")

# Camera sits BESIDE the screen for every shot, because the screen is what he looks at and it is
# not in frame -- so looking at it reads as looking just off the lens rather than into it.
SHOT1 = ("Shot 1, two and a half seconds. The camera is beside the monitor, at screen height, "
         "looking back at him across the desk; the screen itself is out of frame. He is slumped, "
         "watching it. His eyes widen and his eyebrows shoot up as the light on his face turns "
         "hard red. He does not look into the lens.")
SHOT2 = ("Cut. Shot 2, two and a half seconds, a closer low angle from the desk surface looking "
         "up at him. One thing happens: his shoulders drop, he lets out a long breath and drags "
         "one hand down over his face. He does not look into the lens.")
SHOT3 = ("Cut. Shot 3, three and a half seconds, back to the wide shot beside the monitor. The "
         "red light on his face turns green. He sits up, and a broad delighted grin spreads "
         "across his face. He (S1) says: <d>[English] Oh, thank goodness.</d> He does not look "
         "into the lens.")
SOUND = ("Overall soundscape: a quiet workshop, a desk fan, one soft electronic chime. "
         "Non-diegetic music: none.")

HOW_TO = """# H3 storyboard — three shots, one beat each

**The rule this exists for: a shot cannot hold more than two or three facial beats.**
Nine beats in one seven-second close-up renders a frozen face — 37–42 dB PSNR at the emotional
peak, and no error to tell you. The same beats across three 2–3 second shots measure 22–23 dB and
every one of them lands. Splitting is the mechanism; everything else below is trim.

| Control | What it does |
|---|---|
| **LOOK** | Style and who the character is. Same every shot. |
| **SHOT 1 / 2 / 3** | One beat each. Start each with `Cut.` and its length in seconds. |
| **SOUND** | `Overall soundscape:` and `Non-diegetic music:` — H3's documented audio fields. |
| **REFERENCE** | Identity reference, not frame zero. `ref_image_size: max` costs time and buys likeness. |
| **LENGTH** | H3's 17n+5 grid. 209 = 8.71s. Leave 1.3–1.5s of slack past the last beat. |

**Camera before expression.** If what he looks at is off-screen, H3 has him look at the lens.
Do not fix that by adding "he turns to face camera" — that fights the story and the model picks
the story. Move the *camera* to where he is looking, then forbid looking at the lens explicitly;
"look beside the lens" and "look at the lens" are near-identical to the model.

**Dialogue steals frames.** A `<d>` line pulls frames toward its shot and away from the one after
it — 55→84 on the line's shot, 56→42 on the next, and at 42 a background character vanished
outright. Put the line on the LAST shot so the squeeze has nowhere to land.

**Beats after a spoken line go in the next shot.** H3 treats a whole shot as "this person is
talking" and will not close the mouth partway through for a blink or a swallow.

**Check the tail.** H3 degrades into noise blocks in the final 1.2–1.7s, and it is easy to miss
when scrubbing a dark scene. Pull the last frames and look.
"""


# ---------------------------------------------------------------- storyboards
#
# A storyboard is (width, height, length, look, [shots], sound). Shot count is free: the method
# rations BEATS, not shots, so four short shots is as valid as three.

STORYBOARDS = {
    # (w, h, length, look, [shots], sound, [reference plates])
    "deploy": (1152, 640, 209, LOOK, [SHOT1, SHOT2, SHOT3], SOUND, [PLATE]),
}

# The Office: cold open, escalation, deadpan hold, confessional button.
#
# Two deliberate inversions of the camera rule. In the scene shots the family are OFF-SCREEN
# voices, so he looks off-lens naturally and only ONE character is ever in frame -- which also
# dodges the one-reference-one-subject trap that merges two characters into one creature. In the
# confessional the lens look is the whole joke, so it is asked for explicitly instead of forbidden.
#
# Both <d> lines sit as late as they can: the family's complaint opens shot 1 (nothing before it
# to squeeze) and the punchline is the last shot (nothing after it to squeeze). The middle two
# shots are silent on purpose.
OFFICE_LOOK = ("1933 rubber-hose cartoon animation in the style of a modern mockumentary, "
               "confident black ink outlines, soft halftone shading, warm sepia and cream on aged "
               "newsprint, handheld documentary camera with a slight drift. A lanky cartoon man "
               "with long wavy shoulder-length brown hair and a full bushy handlebar moustache, in "
               "a dark v-neck t-shirt.")
OFFICE = [
    # 1 — cold open. Complaint from off-screen, so the camera sits beside the television.
    ("Shot 1, three seconds. A family living room at night. The camera is beside the television, "
     "at seated height, looking back across the room at the man on the sofa; the television is out "
     "of frame and its light flickers on his face. He is mid-bite of a crisp. A woman (S2) says "
     "from off-screen, exasperated: <d>[English] Seth. It is in German. Again.</d> He stops "
     "chewing. He does not look into the lens."),
    # 2 — escalation, one physical beat, no dialogue.
    ("Cut. Shot 2, two and a half seconds, closer, still beside the television. One thing happens: "
     "he raises both hands in a helpless shrug, palms open, eyebrows high, and holds it. He does "
     "not look into the lens."),
    # 3 — the hold. The Office's actual engine: nothing happens for a beat too long.
    ("Cut. Shot 3, two and a half seconds, the same closer angle. Nobody speaks. His hands come "
     "slowly down. His face settles into a flat, cornered stare directed just past the camera, "
     "and holds there. He does not look into the lens."),
    # 4 — confessional button. Here the lens look is the joke, so it is asked for.
    ("Cut. Shot 4, four seconds. A talking-head confessional: he sits alone against a plain wall "
     "in even flat light, framed centre, and looks straight down the barrel of the lens directly "
     "into the camera. He holds the look for a moment, entirely deadpan, then gives one small "
     "shrug. He (S1) says, flatly, straight to camera: <d>[English] That is Wilson's department.</d>"),
]
OFFICE_SOUND = ("Overall soundscape: a muffled television in another language, a room tone, one "
                "crisp packet rustle. Non-diegetic music: none.")

STORYBOARDS["office"] = (1024, 576, 294, OFFICE_LOOK, OFFICE, OFFICE_SOUND, [PLATE])


# "French Again": the family finds the shows dubbed wrong, Seth blames Wilson without a word, and
# Wilson is caught mid-portrait of his server rack posed like the drawing scene in Titanic.
#
# Two fixes came out of the Qwen previz (http://wilson/previz-french/), and neither would have
# shown up in the text:
#   - "a family living room" invented four extra people on the sofa. The room is now stated EMPTY.
#   - "the rack reclines on the chaise" drew it standing beside the couch every time. It is now
#     spelled out as lying on its back along the chaise the way a person lies on a couch.
# The previz's other three failures were Qwen re-posing the character for a shot whose only
# content is a small movement, which is a single-image-edit problem H3 does not have.
FRENCH_LOOK = ("1933 rubber-hose cartoon animation in the style of a modern mockumentary, "
               "confident black ink outlines, soft halftone shading, warm sepia and cream on aged "
               "newsprint, handheld documentary camera with a slight drift.")
FRENCH = [
    ("Shot 1, two and a half seconds. An empty family living room at night; he is the only person "
     "in the room. A lanky cartoon man with long wavy shoulder-length brown hair and a bushy "
     "handlebar moustache, in a dark v-neck t-shirt, sits alone on the sofa. The camera is beside "
     "the television at seated height, looking back at him; the television is out of frame and its "
     "light flickers on his face. He is mid-bite of a crisp. A woman (S2) says from off-screen, "
     "exasperated: <d>[English] Seth, dateline is in French again!</d> He stops chewing. He does "
     "not look into the lens."),
    ("Cut. Shot 2, two and a half seconds, closer on the man, same side. One thing happens: his "
     "face screws up into a hard wince, eyes squeezed shut, teeth bared, shoulders hunching up "
     "around his ears. He does not look into the lens."),
    ("Cut. Shot 3, three seconds, the same closer angle on the man. Nobody speaks. The wince "
     "drains away and his face settles into flat, unimpressed annoyance. Then, slowly and "
     "deliberately, keeping his head still, he swivels his eyes all the way to the left and holds "
     "them there, glaring off past the edge of the frame. He does not look into the lens."),
    ("Cut. Shot 4, three seconds. A small studio room, a completely different place. A living "
     "picket-fence panel — the fence panel is his body — with two short stubby legs in heavy black "
     "boots, a floppy bucket hat on top and two cartoon eyes floating in the shadow under its "
     "brim, stands in profile at a wooden easel with a paintbrush raised. Beside him a tall black "
     "server rack lies on its back along a red velvet chaise longue, stretched out full length the "
     "way a person lies down on a couch, one thick cable draped over it, its row of small red "
     "status lights glowing. He turns his head to look back over his shoulder toward the left edge "
     "of the frame."),
    ("Cut. Shot 5, two seconds, closer on the picket-fence character at the easel, still turned to "
     "look off to the left. One thing happens: he blinks, three slow deliberate blinks in a row."),
    ("Cut. Shot 6, two seconds, a close shot of the tall black server rack still lying on its back "
     "along the red velvet chaise longue. Its row of small red status lights flares bright and "
     "hot, glowing much more strongly and pulsing, washing red light across the velvet."),
]
FRENCH_SOUND = ("Overall soundscape: a muffled television speaking French in another room, quiet "
                "room tone, one crisp packet rustle, a soft electrical hum. Non-diegetic music: "
                "none.")

STORYBOARDS["french-again"] = (1024, 576, 345, FRENCH_LOOK, FRENCH, FRENCH_SOUND,
                               ["rubberhose-kf-seth.png", "rubberhose-kf-wilson.png"])


def build(name="deploy"):
    w, h, length, look, shots, sound, plates = STORYBOARDS[name]
    g = Graph()
    note(g, f"HOW TO USE — H3 storyboard ({name})", HOW_TO)

    unet = g.add("UNETLoader", "H3 reference model (ref2va)", (-60, -180), (420, 110),
                 {"unet_name": UNET, "weight_dtype": "default"},
                 outputs=[("MODEL", "MODEL")], color=GREY, collapsed=True)
    turbo = g.add("MiniMaxH3TurboLoRA", "turbo LoRA (12 steps)", (-60, -130), (420, 140),
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

    # One reference per character. They never share a frame in these boards, which is the safe
    # way to use more than one: a second character inside a single shot merges both into one
    # creature. Ref2VA accepts up to nine.
    plate_nodes = []
    for i, pl in enumerate(plates):
        n = g.add("LoadImage", f"▶ REFERENCE {i+1}", (-60, 100 + i * 360), (400, 320),
                  {"image": pl, "upload": "image"},
                  outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], color=BLUE)
        g.app_input(n, "image", "upload")
        plate_nodes.append(n)
    plate = plate_nodes[0]

    fields = [("▶ LOOK", look)]
    fields += [(f"▶ SHOT {i+1}", t) for i, t in enumerate(shots)]
    fields += [("▶ SOUND", sound)]
    texts, y = [], 60
    for title, val in fields:
        n = g.add("PrimitiveStringMultiline", title, (420, y), (460, 220), {"value": val},
                  outputs=[("STRING", "STRING")], color=GREEN)
        g.app_input(n, "value")
        texts.append(n)
        y += 250

    joined = texts[0]
    for i, nxt in enumerate(texts[1:]):
        joined = g.add("StringConcatenate", f"join-{i+1}", (920, 60 + i * 60), (380, 110),
                       {"delimiter": " "},
                       links={"string_a": (joined, 0, "STRING", True),
                              "string_b": (nxt, 0, "STRING", True)},
                       outputs=[("STRING", "STRING")], collapsed=True)

    cond = g.add("MiniMaxH3ReferenceToVideo", "▶ LENGTH / SIZE", (1340, 60), (440, 280),
                 {"prompt": "", "width": w, "height": h, "length": length, "ref_image_size": "max"},
                 links={"prompt": (joined, 0, "STRING", True), "clip": (clip, 0, "CLIP", False),
                        "vae": (vvae, 0, "VAE", False), "audio_vae": (avae, 0, "VAE", False)},
                 outputs=[("positive", "CONDITIONING"), ("LATENT", "LATENT")], color=BLUE)
    g.app_input(cond, "length", "width", "height", "ref_image_size")
    # Autogrow: the reference arrives as ONE dict, not as a flat ref_image_0 key. Validation
    # accepts the flat form and execution then rejects it, so this only fails at run time.
    cond["inputs"].append({"name": "ref_images", "type": "COMFY_AUTOGROW_V3", "link": None})
    cond["_autogrow"] = {"ref_images": {f"ref_image_{i}": (n, 0)
                                        for i, n in enumerate(plate_nodes)}}

    noise = g.add("RandomNoise", "▶ SEED", (1340, 360), (400, 130), {"noise_seed": SEED},
                  outputs=[("NOISE", "NOISE")], color=GREEN)
    g.app_input(noise, "noise_seed")
    guider = g.add("BasicGuider", "guider", (1340, 500), (320, 80),
                   links={"model": (turbo, 0, "MODEL", False),
                          "conditioning": (cond, 0, "CONDITIONING", False)},
                   outputs=[("GUIDER", "GUIDER")], collapsed=True)
    samp = g.add("KSamplerSelect", "sampler", (1340, 560), (320, 80),
                 {"sampler_name": "res_multistep"}, outputs=[("SAMPLER", "SAMPLER")], collapsed=True)
    sched = g.add("BasicScheduler", "▶ STEPS", (1340, 620), (400, 160),
                  {"scheduler": "simple", "steps": STEPS, "denoise": 1.0},
                  links={"model": (turbo, 0, "MODEL", False)},
                  outputs=[("SIGMAS", "SIGMAS")], color=GREEN)
    g.app_input(sched, "steps")
    adv = g.add("SamplerCustomAdvanced", "sample", (1340, 800), (340, 100),
                links={"noise": (noise, 0, "NOISE", False), "guider": (guider, 0, "GUIDER", False),
                       "sampler": (samp, 0, "SAMPLER", False), "sigmas": (sched, 0, "SIGMAS", False),
                       "latent_image": (cond, 1, "LATENT", False)},
                outputs=[("output", "LATENT"), ("denoised_output", "LATENT")], collapsed=True)
    vid = g.add("VAEDecode", "decode video", (1780, 60), (300, 60),
                links={"samples": (adv, 0, "LATENT", False), "vae": (vvae, 0, "VAE", False)},
                outputs=[("IMAGE", "IMAGE")], collapsed=True)
    aud = g.add("VAEDecodeAudio", "decode audio", (1780, 120), (300, 60),
                links={"samples": (adv, 0, "LATENT", False), "vae": (avae, 0, "VAE", False)},
                outputs=[("AUDIO", "AUDIO")], collapsed=True)
    mk = g.add("CreateVideo", "mux", (1780, 180), (320, 110), {"fps": FPS, "bit_depth": 8},
               links={"images": (vid, 0, "IMAGE", False), "audio": (aud, 0, "AUDIO", False)},
               outputs=[("VIDEO", "VIDEO")], collapsed=True)
    out = g.add("SaveVideo", "RESULT", (1780, 300), (520, 420),
                {"filename_prefix": f"video/h3-{name}", "format": "auto", "codec": "auto"},
                links={"video": (mk, 0, "VIDEO", False)}, color=GREY)
    g.app_output(out)
    return g


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "deploy"
    g = build(name)
    ui, api = g.to_ui(), g.to_api()
    # to_api() cannot express an autogrow socket; patch it on after serialisation.
    for nid, n in api.items():
        if n["class_type"] == "MiniMaxH3ReferenceToVideo":
            loads = [k for k, v in api.items() if v["class_type"] == "LoadImage"]
            n["inputs"]["ref_images"] = {f"ref_image_{i}": [k, 0] for i, k in enumerate(loads)}
    for path, blob in ((os.path.join(HERE, "workflows", f"h3-storyboard-{name}.json"), ui),
                       (os.path.join(HERE, "workflows", f"h3-storyboard-{name}.app.json"), ui),
                       (os.path.join(HERE, "api", f"h3-storyboard-{name}.api.json"), api)):
        json.dump(blob, open(path, "w"), indent=1)
    print("wrote h3-storyboard-" + name)
