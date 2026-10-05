#!/usr/bin/env python3
"""Sethmojis: a famous reaction clip, re-performed by Seth, as a Slack emoji plus a comparison.

    python3 sethmoji.py all              # every meme, every stage, skipping what exists
    python3 sethmoji.py all blink nod    # some memes
    python3 sethmoji.py <stage> <name>   # one stage: prep ref animate cutout emoji compare deliver

Per meme, in order:
  prep      source clip -> caption band cropped off, 16fps, 4n+1 frames (the driving video)
  ref       frame 0 redrawn with Seth in it: a ONE-image Qwen edit + the Follies LoRA. The
            source frame is the edit target, so pose, props, clothes and set survive; only the
            head changes. Passing Seth's plate as a second image loses: it wins every frame.
  animate   Wan Animate 2 (stock video_wan_animate2 graph, run_animate2.build): ref + driving
            clip. pose 2.0 / reference 0.6 -- at the defaults a cartoon face stays neutral while
            the real one moves. Measured on Kombucha Girl.
  cutout    BiRefNet per frame -> RGBA frames
  emoji     square crop around the subject across ALL frames, 128px, one-bit alpha, halftone
            smoothed, frame rate stepped down until it fits Slack's 128KB
  compare   original and Seth side by side, as a GIF
  deliver   copy both to the Mac's ~/Documents/avatars/sethmojis/

Sources are Tenor's copies of each meme; see sethmoji_source.py.
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S  # noqa: E402
from build_workflows import BGREMOVAL, NEGATIVE, STEPS  # noqa: E402
from run_animate2 import build as animate_graph  # noqa: E402

ROOT = "/home/wilson/scratch/sethmoji"
GPU = "wilson@192.168.0.210"
MAC = "sethgho@100.64.185.78"
MAC_DIR = "Documents/avatars/sethmojis"
# who the set is of, and the prefix for files in ComfyUI's input folder; jenmoji.py swaps both
WHO = "seth"
PREFIX = "sethmoji"
FPS = 16
MAX_FRAMES = 81

SETH = "long wavy shoulder-length brown hair and a full bushy handlebar moustache"
CHAR = ("A 1930s rubber-hose cartoon man drawn in warm sepia ink with soft halftone shading on "
        f"aged paper, with {SETH} and a big elastic cartoon face that exaggerates every "
        "expression")

# pick: candidate index from sethmoji_source. caption: fraction of height to crop off the
# bottom (burned-in subtitles). top: how far down the frame the emoji crop may reach -- a
# head-only meme stops at the shoulders, a pointing one needs the arm.
MEMES = {
    "blink": dict(pick=0, caption=0.0, top=1.0,
                  face="an extreme close-up: his face fills the whole frame at exactly the same size and position as in the photograph, head turned three-quarters, eyes half-lidded, mouth closed, no body visible",
                  scene="Close up, in front of a softly blurred background",
                  action="He glances toward the viewer and blinks twice in disbelief"),
    "nod": dict(pick=1, caption=0.14, top=0.85,
                face="clean-shaven except for the handlebar moustache, NO beard, a calm knowing look",
                scene="Outdoors in snowy woods, wearing a heavy fur coat",
                action="He looks toward the viewer and gives one slow, knowing, approving nod"),
    "jim-look": dict(pick=0, caption=0.0, top=0.7,
                     face="completely deadpan, NOT smiling, mouth flat and closed, eyebrows slightly raised",
                     scene="In an office, wearing a white shirt and a tie",
                     action="He turns his head and stares straight into the camera, deadpan, "
                            "eyebrows slightly raised"),
    "side-eye": dict(pick=0, caption=0.0, top=0.85,
                     face="head tilted and turned slightly away exactly as in the photograph, eyes shifted hard to the side in a suspicious side-eye, lips pressed together, NOT smiling",
                     scene="Indoors, close up",
                     action="He turns his head slightly away and shifts his eyes sideways in "
                            "a suspicious side-eye"),
    "thats-me": dict(pick=0, caption=0.0, top=0.95,
                     scene="In a dark living room, sitting in an armchair with a drink",
                     action="He leans forward and points emphatically at something in front "
                            "of him with a big delighted grin"),
    "nooo": dict(pick=0, caption=0.2, top=0.95,
                 scene="In an office, wearing a suit and tie",
                 action="He throws his head back and his arms up, crying out no in anguish"),
    "lost": dict(pick=1, caption=0.0, top=0.6,
                 scene="On a plain white background, wearing a black suit, holding a jacket",
                 action="He turns around looking left and right, bewildered and lost"),
    "cheers": dict(pick=2, caption=0.0, top=0.85,
                   scene="At a lavish party at night with fireworks behind him, wearing a "
                         "black tuxedo",
                   action="He raises a champagne glass toward the viewer in a toast with a "
                          "charming smile"),
    # ---- batch 2 ----
    "mind-blown": dict(pick=2, caption=0.0, top=0.9, segments=[(0, 1.0), (3.7, 5.3)],
                       face="wide-eyed, mouth open in awe, wearing big 1980s aviator glasses with "
                            "large square clear lenses and thin gold wire frames",
                       scene="Close up, in front of a dark background, wearing big 1980s aviator "
                             "glasses and a black turtleneck sweater",
                       clothing="a black turtleneck sweater with a tall rolled collar",
                       # the splice cuts at render frame 18; the boom fills the 2.5s it removed
                       post=dict(node="MindBlownFX", freeze_frame=17, resume_frame=19,
                                 gap_frames=40, burst_x=0.34, burst_y=0.64, seed=7,
                                 orig_seconds=5.0, backdrop="0x141414"),
                       action="He presses his fingertips to his temples, then flings both hands "
                              "outward as his mind is blown, mouth wide open"),
    "slow-clap": dict(pick=2, caption=0.0, top=0.9, repeat=4,
                      face="stern, proud and unsmiling",
                      scene="In a dim theatre box, wearing a tuxedo and bow tie",
                      action="He claps slowly and deliberately, head held high"),
    "mic-drop": dict(pick=0, caption=0.0, top=0.85,
                     face="confident and self-satisfied",
                     scene="At a podium in a grand ballroom, wearing a black tuxedo",
                     action="He holds a microphone out at arm's length, then lets it drop with "
                            "a confident nod"),
    "its-happening": dict(pick=1, caption=0.0, top=0.95,
                          face="grinning with excitement",
                          scene="In front of a dark background, wearing a suit and tie",
                          action="He throws both hands up and waves them excitedly overhead",
                          # measured from the source: the caption is up on these drive frames.
                          # width 0.64 keeps it inside the square emoji crop.
                          post=dict(node="BlinkCaption", top_text="IT'S", bottom_text="HAPPENING",
                                    frames="0-4,10-15,21-26,32-37,43-49", size=0.17, width=0.64,
                                    margin=0.02)),
    "popcorn": dict(pick=0, caption=0.0, top=0.9,
                    face="a wide delighted grin, eyes fixed on something in front of him",
                    scene="In a dark cinema, wearing a red leather jacket",
                    action="He eats popcorn from his hand, grinning, eyes fixed on the screen"),
    "salty": dict(pick=0, caption=0.0, top=0.95,
                  face="cool and confident",
                  scene="In a restaurant, wearing a white t-shirt",
                  action="He raises his hand high and sprinkles salt down along his forearm "
                         "with a dramatic flourish"),
    # ---- batch 3 ----
    "yes-you": dict(pick=0, caption=0.0, top=0.9,
                    face="his head fills the top of the frame at the same size and position as in the photograph, tilted to one side, eyes closed, lips pursed and pushed out in a smug pout, NOT smiling",
                    scene="At a crowded outdoor event, wearing a blue suit, white shirt and red tie",
                    extra="Where the other man stands behind him in the photograph there is only a dark blurred shape: there is no second person.",
                    action="He sways with his eyes closed, then points straight at the viewer "
                           "and nods approvingly"),
    "ok": dict(pick=0, caption=0.0, top=0.9,
               face="unimpressed, mouth making a sarcastic 'oh okay', eyebrows raised, NOT smiling",
               scene="In front of a dark background, wearing a white blouse with a blue floral pattern",
               action="He says oh okay with a sarcastic eye roll, nods and gives a thumbs-up"),
    "van-door": dict(pick=0, caption=0.0, top=1.0,
                     face="a polite tight-lipped grin, looking straight at the viewer, wearing glasses",
                     scene="Sitting in the back seat of a grey van, while another man's arm in a suit sleeve pulls the sliding door shut",
                     extra="Keep the other man's arm in the suit sleeve and the van door exactly as they are.",
                     action="He grins at the viewer as the van's sliding door is pulled shut across him",
                     crop_top=0.0,
                     post=dict(node="SlidingDoorFX", start_frame=4, end_frame=27, end_x=0.0, seed=7)),
    # ---- batch 4 ----
    "hotline-bling": dict(pick=1, caption=0.0, top=0.95,
                          face="cool and smooth, lips pursed, NOT smiling",
                          scene="In front of a plain glowing blue wall, wearing a black hoodie",
                          action="He dances the Hotline Bling dance, swaying and waving his arms "
                                 "smoothly out to the sides"),
    "rickroll": dict(pick=1, caption=0.0, top=0.95,
                     face="singing earnestly with a small confident smile",
                     scene="On a plain white background, wearing a long dark trench coat over a striped shirt, a microphone on a stand in front of him",
                     action="He sways and swings his arms as he sings into the microphone"),
    "carlton": dict(pick=1, caption=0.0, top=1.0,
                    face="an ecstatic open-mouthed grin, fully committed to the dance",
                    scene="In a living room, wearing a red sweater and khaki trousers",
                    action="He does the Carlton dance, swinging his arms side to side in rhythm"),
    "ear-cup": dict(pick=0, caption=0.0, top=1.0,
                    face="head tilted back, eyes squinted, cupping his ear with a smug grin",
                    scene="In a wrestling ring at night, shirtless and oiled, in front of a steel cage",
                    action="He cups his hand to his ear and leans in to hear the crowd cheer louder"),
    "chest-thump": dict(pick=0, caption=0.0, top=0.9,
                        face="eyes half closed, humming with lips pressed together, deadpan",
                        scene="At a restaurant table by a big window over the city, wearing a dark pinstriped suit and red tie",
                        action="He hums and thumps his chest rhythmically with his fist"),
    # c3 is chefkiss2/c1 from sethmoji_source ("chefs kiss perfection"); the glitch wipe starts at 2.4s
    "chefkiss": dict(pick=3, caption=0.0, top=0.95, segments=[(0, 2.37)],
                     face="eyes closed in bliss, kissing his pinched fingertips, utterly satisfied",
                     scene="In front of a black background, wearing a tall white chef's hat and a white neckerchief",
                     extra="He is not wearing sunglasses: his eyes are visible.",
                     action="He pinches his fingertips together, kisses them and flings his hand open in a chef's kiss"),
    # c3 is pour-one-out2/c2 from sethmoji_source: Dr. Evil, "and one for my homies"
    "pour-one-out": dict(pick=3, caption=0.17, top=0.95,
                         face="solemn and sincere, eyes on the bottle, mouth slightly open mid-sentence, his head exactly where the bald man's head is, just right of centre, at the same size",
                         extra="Behind him there is only a plain, flat, pale paper background with nothing in it: no globe, no walls. The bottle stays in his hand at the lower left.",
                         # the pour is the point: frame face, arm and bottle, letterboxed square
                         box=(0.0, 0.0, 0.72, 1.0),
                         scene="In front of a plain pale background, wearing a pale grey high-collared jacket, holding a big malt liquor bottle",
                         action="He tips a big bottle of malt liquor and pours some out onto the floor for his fallen homies, a stream of beer splashing down"),
    "andy-zoom": dict(pick=1, caption=0.0, top=1.0,
                      face="mouth wide open in a gasp of pure excitement, eyebrows shot up, eyes huge",
                      scene="In an office meeting room, wearing a white shirt and a striped tie",
                      action="He gasps with wide-eyed excitement as the camera zooms in fast on his face"),
    # c0 is Giphy YBU0Wrys1Cs4U, a 6-frame cartoon loop -- repeat makes a clip of it
    "squidward": dict(pick=0, caption=0.0, top=1.0, repeat=8,
                      face="eyes half closed, smug and unbothered, mouth flat",
                      scene="In Squidward's green house with yellow round windows and a green wooden floor, wearing a brown short-sleeved shirt",
                      extra="He is a man with two legs, bent over with his knees bent and his hands on his knees, twerking, in exactly the place and size of the figure in the picture.",
                      action="He bends over with his hands on his knees and twerks, bouncing his hips up and down to the beat"),
    # rendered by hand: full-body cast Seth posed from squidward frame 0, then MiniMax H3 i2v
    # (both ends pinned, turbo 8 steps) for the twerk -- Wan and flow both undersold it
    "squidward-body": dict(pick=0, caption=0.0, top=1.0, face="", scene="", action=""),
    "thats-bait": dict(pick=0, seed=7, caption=0.0, top=0.95,
                       face="squinting knowingly at something off to the side, unimpressed, mouth closed, his head just right of centre and the same size as the man's head in the photograph",
                       extra="Keep the whole cab exactly as in the photograph: the red-haired woman in goggles on the left behind him, the pale bald man on the right, the dark metal cab walls and the bright window. Do not zoom in.",
                       scene="In the cab of a rusty war rig in the desert, wearing a battered leather jacket, a red-haired woman sitting behind him",
                       action="He glances aside, raises his wrapped hand to wipe his face, rolls his eyes and says that's bait",
                       # the source subtitles THAT'S at 2.4s and BAIT at 2.6s (16fps frames 38, 42)
                       post=dict(node="Captions", passes=[
                           dict(top="THAT'S", frames="38-52", size=0.2, width=0.7),
                           dict(bottom="BAIT", frames="42-44,47-49,52", size=0.24, width=0.6)])),
    "jack-salute": dict(seed=77, pick=0, caption=0.0, top=0.95,
                        face="a furious scrunched GRIMACE, NOT smiling, NOT surprised: his eyebrows crushed down hard into a deep V, his eyes squeezed almost shut into angry slits, his nose wrinkled up, his mouth clamped shut with the lower lip shoved up into a big frowning pout under the moustache, the corners of the mouth pulled DOWN, like a tough guy biting back emotion, his head at exactly the same place and size as the man's head in the photograph, his long hair falling over his forehead",
                        scene="In front of a green chalkboard in a classroom, wearing a black suit jacket over a patterned shirt",
                        extra="The background is the same as the photograph: a green chalkboard filling the whole wall behind him, paper notices pinned at the top left, and a red box and a poster at the lower left, all drawn in sepia ink. His hand is held flat at his brow in a salute exactly as in the photograph. He wears a black jacket over a patterned shirt with a little bow at the collar.",
                        action="He snaps a fierce salute from his brow and whips his arm down and across his chest, his long hair swinging and flopping wildly with the motion"),
    "top-kek": dict(seed=7, keep_bg=True, pick=2, caption=0.0, top=0.95, segments=[(0, 3.95)],
                    face="laughing so hard he is crying, eyes squeezed shut, mouth wide open in a huge gap-toothed laugh with a few teeth missing",
                    # ref.png is hand-finished: the ref edit draws full teeth, so the mouth was
                    # inpainted (masked Qwen edit) to the gap teeth. Rerunning `ref` loses them.
                    scene="In a TV studio in front of a bright pink and magenta backdrop, wearing a grey knit sweater",
                    extra="Full bleed: the bright pink and magenta studio backdrop fills the whole picture edge to edge, with NO border, NO frame and no paper margin. His head is in the LEFT half of the picture, tilted to his right and thrown forward exactly like the man's, at the same size.",
                    action="He laughs uncontrollably, rocking back and forth and wiping tears, wheezing with laughter"),
    # ---- ron swanson batch ----
    "ron-dance": dict(seed=31, pick=1, caption=0.0, top=0.95,
                      face="eyes half closed, lips pursed, utterly serious while dancing, his head and body at the left of the picture at exactly the same size as the man in the photograph",
                      extra="Keep the whole office from the photograph: the wooden bookshelves and doorway behind him and the man in the vest in the background. Do not zoom out.",
                      scene="In a government office, wearing a red polo shirt",
                      action="He dances with stiff deadpan intensity, pumping his fists and swaying his shoulders"),
    "ron-trash": dict(pick=2, caption=0.0, top=1.0,
                      face="grim and determined, jaw set",
                      scene="In an alley beside a green dumpster, wearing a blue button-up shirt and brown trousers",
                      action="He hoists a computer monitor and hurls it into the green dumpster, then walks away"),
    "ron-eyeroll": dict(seed=31, pick=1, caption=0.0, top=0.9,
                        face="completely deadpan, heavy-lidded eyes, eyebrows low and flat, mouth flat under the moustache, bored disdain, NOT worried, NOT wide-eyed, NOT smiling",
                        extra="Keep the window and the green outdoors behind him. No scribbles or motion marks.",
                        scene="In an office in front of a window, wearing a maroon polo shirt",
                        action="He stares flatly and slowly rolls his eyes in total disdain"),
    "ron-nope": dict(seed=7, pick=1, caption=0.0, top=0.95,
                     face="stern and unmoved, NOT smiling",
                     extra="Keep the whole office from the photograph: the wooden window frames, the doorway and the man in a blue shirt at the right edge.",
                     scene="In a government office, wearing a maroon polo shirt, hands on hips",
                     action="He stands with hands on his hips, turns to the camera and glares a flat refusal",
                     # the source flashes NO at 2.5s (16fps frame 40)
                     post=dict(node="Captions", passes=[dict(bottom="NO", frames="40-52", size=0.3, width=0.5)])),
    "ron-smile": dict(pick=1, caption=0.0, top=0.8,
                      face="a rare, small, genuine smile under the moustache",
                      scene="Sitting at a big wooden desk in a dark office, wearing a brown sweater",
                      action="He looks at the viewer and slowly breaks into a rare small smile"),
    "ron-snakejuice": dict(seed=7, pick=0, caption=0.0, top=0.9,
                           face="drunk, eyes squinting, laughing helplessly, his head tipped back, at the same size and place as the man in the photograph",
                           extra="Keep the dark crowded bar from the photograph: the big glowing orange-red sign behind him and the crowd on both sides.",
                           scene="At a crowded bar at night with glowing orange lights, wearing a dark shirt",
                           action="He sways drunkenly and laughs, giddy and loose, head lolling"),
    "elaine": dict(pick=0, caption=0.0, top=1.0,
                   face="fully committed to the dance, grinning",
                   scene="At a crowded office party, keeping the same clothing",
                   action="He dances wildly and awkwardly, jerking his arms and thumbs and "
                          "kicking his legs"),
}


def sh(*args, **kw):
    return subprocess.run(list(args), check=True, **kw)


def probe(path, key):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           f"stream={key}", "-of", "csv=p=0", path],
                          capture_output=True, text=True).stdout.strip()


def d(name):
    return f"{ROOT}/{name}"


def size_for(w, h):
    """~0.4MP at the source aspect, multiples of 16: the envelope Kombucha rendered in 165s."""
    ar = w / h
    H = int(round((409600 / ar) ** 0.5 / 16)) * 16
    W = int(round(H * ar / 16)) * 16
    return W, H


# ------------------------------------------------------------------ stages

def prep(name):
    m, dd = MEMES[name], d(name)
    src = f"{dd}/c{m['pick']}.mp4"
    w, h = map(int, probe(src, "width,height").split(","))
    keep = int(h * (1 - m["caption"])) // 2 * 2
    w2 = w // 2 * 2
    # segments: splice time ranges, e.g. mind-blown keeps the clean frames either side of the
    # VFX explosion that hides his head. repeat: loop a clip that is one natural cycle, e.g.
    # slow-clap's single 0.56s clap.
    segs = m.get("segments") or [(0, 999)]
    parts = "".join(f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS,crop={w2}:{keep}:0:0,fps={FPS}[s{i}];"
                    for i, (a, b) in enumerate(segs))
    parts += "".join(f"[s{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1[c]"
    if m.get("repeat", 1) > 1:
        parts += f";[c]loop=loop={m['repeat'] - 1}:size=1000:start=0[c2]"
        out_label = "[c2]"
    else:
        out_label = "[c]"
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", src, "-filter_complex", parts, "-map",
       out_label, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12",
       f"{dd}/drive-full.mp4")
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                            f"{dd}/drive-full.mp4"], capture_output=True, text=True).stdout.strip())
    n = min(n, MAX_FRAMES)
    n = (n - 1) // 4 * 4 + 1
    W, H = size_for(w2, keep)
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{dd}/drive-full.mp4", "-vf",
       f"scale={W}:{H}:flags=lanczos", "-frames:v", str(n), "-an", "-c:v", "libx264",
       "-pix_fmt", "yuv420p", "-crf", "12", f"{dd}/drive.mp4")
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{dd}/drive.mp4", "-frames:v", "1",
       f"{dd}/frame0.png")
    json.dump({"W": W, "H": H, "frames": n}, open(f"{dd}/spec.json", "w"))
    sh("scp", "-q", f"{dd}/drive.mp4", f"{GPU}:comfyui/input/{PREFIX}-{name}-drive.mp4")
    print(f"{name} prep {W}x{H} {n} frames")


def submit(g, node, kind, dest):
    for k, v in g.items():
        v.setdefault("_meta", {"title": k})
    r = S.api("/prompt", {"prompt": g, "client_id": "sethmoji"})
    if "prompt_id" not in r:
        raise SystemExit("REJECTED " + json.dumps(r)[:600])
    pid = r["prompt_id"]
    while True:
        h = S.api(f"/history/{pid}")
        if pid in h:
            break
        time.sleep(5)
    if h[pid]["status"].get("status_str") != "success":
        raise SystemExit("FAILED " + json.dumps(h[pid]["status"])[:600])
    outs = h[pid]["outputs"][node][kind]
    got = []
    for i, o in enumerate(outs):
        q = f"/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output"
        path = dest if len(outs) == 1 else dest.format(i=i)
        with urllib.request.urlopen(S.HOST + q, timeout=600) as rr:
            open(path, "wb").write(rr.read())
        got.append(path)
    return got


def ref(name):
    m, dd = MEMES[name], d(name)
    seed = m.get("seed", 7)
    spec = json.load(open(f"{dd}/spec.json"))
    sh("scp", "-q", f"{dd}/frame0.png", f"{GPU}:comfyui/input/{PREFIX}-{name}-f0.png")
    prompt = ("f0llie5, image 1 is a photograph to redraw as a cartoon. This is a repaint: the "
              "pose, the tilt of the head, the direction the eyes are looking, the hands and "
              "anything held in them, the clothing, the framing and the background are already "
              "exactly right and must not change, including the crop and the size of the head in the frame. Replace only the person's head, face and hair "
              f"with a cartoon man with {SETH}. His face: {m.get('face', 'making exactly the same expression as the person in the photograph, not smiling unless they are')}. "
              f"{'He wears ' + m['clothing'] + '. ' if 'clothing' in m else ''}{m.get('extra', '') + ' ' if 'extra' in m else ''}Keep the clothing "
              "exactly as in the photograph. Draw the whole picture, the background included, "
              "in warm sepia ink on aged paper. Remove any words, captions or lettering: there is no "
              "text anywhere in the picture.")
    W, H = spec["W"], spec["H"]
    g = {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "follies-final.safetensors", "strength_model": 1.2, "model": ["3", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "6": {"class_type": "LoadImage", "inputs": {"image": f"{PREFIX}-{name}-f0.png"}},
     "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
     "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0]}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE + ", photograph, text, subtitles", "clip": ["4", 0]}},
     "22": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
     "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"sethmoji/{name}-ref", "images": ["23", 0]}},
    }
    submit(g, "99", "images", f"{dd}/ref.png")
    sh("scp", "-q", f"{dd}/ref.png", f"{GPU}:comfyui/input/{PREFIX}-{name}-ref.png")
    print(f"{name} ref")


def animate(name):
    m, dd = MEMES[name], d(name)
    spec = json.load(open(f"{dd}/spec.json"))
    a = argparse.Namespace(
        ref=f"{PREFIX}-{name}-ref.png", drive=f"{PREFIX}-{name}-drive.mp4",
        prompt=f"{CHAR}. {m['scene']}, all drawn in the same sepia ink style. {m['action']}.",
        pose_prompt=f"{m['action'].replace('He ', 'A person ')}. Static camera.",
        out=f"sethmoji/{name}", model="wan_animate_2_distill_int8_convrot.safetensors",
        lora=None, width=spec["W"], height=spec["H"], length=spec["frames"], fps=FPS, steps=6,
        seed=77, cache="cpu", pose_strength=2.0, ref_strength=0.6)
    # /free answers with an empty body, so it cannot go through S.api (which parses JSON).
    urllib.request.urlopen(urllib.request.Request(
        S.HOST + "/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
        {"Content-Type": "application/json"}), timeout=60).read()
    t0 = time.time()
    submit(animate_graph(a), "21", "images", f"{dd}/seth.mp4")
    print(f"{name} animate {time.time()-t0:.0f}s")


def cutout(name):
    dd = d(name)
    sh("scp", "-q", f"{dd}/seth.mp4", f"{GPU}:comfyui/input/{PREFIX}-{name}-seth.mp4")
    g = {
     "1": {"class_type": "LoadVideo", "inputs": {"file": f"{PREFIX}-{name}-seth.mp4"}},
     "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
     "3": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": BGREMOVAL}},
     "4": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["3", 0], "image": ["2", 0]}},
     "5": {"class_type": "InvertMask", "inputs": {"mask": ["4", 0]}},
     "6": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["2", 0], "alpha": ["5", 0]}},
     "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"sethmoji/{name}-cut", "images": ["6", 0]}},
    }
    os.makedirs(f"{dd}/cut", exist_ok=True)
    for f in glob.glob(f"{dd}/cut/*.png"):
        os.remove(f)
    got = submit(g, "9", "images", f"{dd}/cut/f{{i:03d}}.png")
    print(f"{name} cutout {len(got)} frames")


def post_graph(name, p, cutout=False):
    load = lambda k, f: {k: {"class_type": "LoadVideo", "inputs": {"file": f}},
                         k + "c": {"class_type": "GetVideoComponents", "inputs": {"video": [k, 0]}}}
    g = {"m": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": BGREMOVAL}}}
    g.update(load("s", f"{PREFIX}-{name}-seth.mp4"))
    g["sm"] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["m", 0], "image": ["sc", 0]}}
    args = {k: v for k, v in p.items() if k not in ("node", "orig_seconds", "backdrop")}
    if p["node"] == "BlinkCaption":
        # Seth over the source's own background, with the source performer and caption painted out
        g.update(load("o", f"{PREFIX}-{name}-drive.mp4"))
        g["om"] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["m", 0], "image": ["oc", 0]}}
        g["ot"] = {"class_type": "BrightTextMask", "inputs": {"images": ["oc", 0], "threshold": 0.9, "top_band": 0.22, "bottom_band": 0.25, "grow": 7}}
        g["og0"] = {"class_type": "GrowMask", "inputs": {"mask": ["om", 0], "expand": 10, "tapered_corners": True}}
        g["og"] = {"class_type": "MaskComposite", "inputs": {"destination": ["og0", 0], "source": ["ot", 0], "x": 0, "y": 0, "operation": "add"}}
        g["bg"] = {"class_type": "FillMaskFromSurroundings", "inputs": {"images": ["oc", 0], "mask": ["og", 0]}}
        g["cm"] = {"class_type": "ImageCompositeMasked", "inputs": {"destination": ["bg", 0], "source": ["sc", 0], "x": 0, "y": 0, "resize_source": False, "mask": ["sm", 0]}}
        g["fx"] = {"class_type": "BlinkCaption", "inputs": {"images": ["cm", 0], **args}}
        g["out"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"sethmoji/{name}-post", "images": ["fx", 0]}}
    elif p["node"] == "Captions":
        # text only, over the render itself: one BlinkCaption pass per line, each on its own frames
        prev = ["sc", 0]
        if cutout:
            g["si"] = {"class_type": "InvertMask", "inputs": {"mask": ["sm", 0]}}
            g["sj"] = {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["sc", 0], "alpha": ["si", 0]}}
            prev = ["sj", 0]
        for k, c in enumerate(p["passes"]):
            g[f"cap{k}"] = {"class_type": "BlinkCaption", "inputs": {"images": prev, "top_text": c.get("top", ""), "bottom_text": c.get("bottom", ""), "frames": c["frames"], "size": c.get("size", 0.17), "width": c.get("width", 0.9), "margin": c.get("margin", 0.03)}}
            prev = [f"cap{k}", 0]
        g["out"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"sethmoji/{name}-post", "images": prev}}
    elif p["node"] == "SlidingDoorFX":
        g["fx"] = {"class_type": p["node"], "inputs": {"images": ["sc", 0], **args}}
        g["out"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"sethmoji/{name}-post", "images": ["fx", 0]}}
    else:
        g["fx"] = {"class_type": p["node"], "inputs": {"images": ["sc", 0], "alpha": ["sm", 0], **args}}
        g["iv"] = {"class_type": "InvertMask", "inputs": {"mask": ["fx", 1]}}
        g["ja"] = {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["fx", 0], "alpha": ["iv", 0]}}
        g["out"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"sethmoji/{name}-post", "images": ["ja", 0]}}
    return g


def post(name):
    """Per-meme finishing in ComfyUI (the sethmoji_fx node pack). Writes post/f###.png, which
    emoji prefers over cut/, and final.mp4 plus cmp-orig.mp4, which compare prefers."""
    m, dd = MEMES[name], d(name)
    p = m.get("post")
    if not p:
        return
    os.makedirs(f"{dd}/post", exist_ok=True)
    for f in glob.glob(f"{dd}/post/*.png"):
        os.remove(f)
    if p["node"] == "Captions":
        # full frames feed final.mp4 (compare, full); the captioned cutout feeds the emoji
        os.makedirs(f"{dd}/post_full", exist_ok=True)
        submit(post_graph(name, p), "out", "images", f"{dd}/post_full/f{{i:03d}}.png")
        got = submit(post_graph(name, p, cutout=True), "out", "images", f"{dd}/post/f{{i:03d}}.png")
        src_frames = f"{dd}/post_full/f%03d.png"
    else:
        got = submit(post_graph(name, p), "out", "images", f"{dd}/post/f{{i:03d}}.png")
        src_frames = f"{dd}/post/f%03d.png"
    bd = p.get("backdrop")
    vf = (f"color=c={bd}:s=16x16:r={FPS}[k];[k][0:v]scale2ref[k2][v];[k2][v]overlay=format=auto:shortest=1,format=yuv420p"
          if bd else "format=yuv420p")
    sh("ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", src_frames,
       "-filter_complex", vf, "-c:v", "libx264", "-crf", "14", f"{dd}/final.mp4")
    if p.get("orig_seconds"):
        # the render was spliced; compare against the whole original, beat for beat
        spec = json.load(open(f"{dd}/spec.json"))
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{dd}/c{m['pick']}.mp4", "-vf",
           f"fps={FPS},scale={spec['W']}:{spec['H']}:flags=lanczos", "-t", str(p["orig_seconds"]),
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", f"{dd}/cmp-orig.mp4")
    print(f"{name} post {len(got)} frames")


def emoji(name):
    from PIL import Image, ImageFilter
    m, dd = MEMES[name], d(name)
    src = "post" if os.path.exists(f"{dd}/post/f000.png") else "cut"
    frames = [Image.open(f).convert("RGBA") for f in sorted(glob.glob(f"{dd}/{src}/f*.png"))]
    W, H = frames[0].size
    lim = int(H * m["top"])
    x0, y0, x1, y1 = W, H, 0, 0
    for f in frames:
        bb = f.getchannel("A").crop((0, 0, W, lim)).point(lambda v: 255 if v > 128 else 0).getbbox()
        if bb:
            x0, y0, x1, y1 = min(x0, bb[0]), min(y0, bb[1]), max(x1, bb[2]), max(y1, bb[3])
    side = min(max(x1 - x0, y1 - y0) + 12, W, H)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = max(0, min(W - side, cx - side // 2))
    upper = max(0, min(H - side, cy - side // 2))
    if "crop_top" in m:
        upper = int(H * m["crop_top"])
    box = (left, upper, left + side, upper + side)
    if "box" in m:
        bx0, by0, bx1, by1 = m["box"]
        box = (int(W * bx0), int(H * by0), int(W * bx1), int(H * by1))
    if m.get("keep_bg"):
        # crop where the cutout says he is, but keep the rendered backdrop (opaque emoji)
        os.makedirs(f"{dd}/opaque", exist_ok=True)
        for f in glob.glob(f"{dd}/opaque/*.png"):
            os.remove(f)
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{dd}/seth.mp4", f"{dd}/opaque/f%03d.png")
        frames = [Image.open(f).convert("RGBA") for f in sorted(glob.glob(f"{dd}/opaque/f*.png"))][:len(frames)]
    tmp = f"{dd}/emo"
    os.makedirs(tmp, exist_ok=True)
    bw, bh = box[2] - box[0], box[3] - box[1]
    for i, f in enumerate(frames):
        im = f.crop(box)
        if bw != bh:
            sq = Image.new("RGBA", (max(bw, bh),) * 2, (0, 0, 0, 0))
            sq.paste(im, ((max(bw, bh) - bw) // 2, (max(bw, bh) - bh) // 2))
            im = sq
        a = im.getchannel("A")
        im = im.convert("RGB").filter(ImageFilter.MedianFilter(5)).convert("RGBA")
        im.putalpha(a)
        im = im.resize((128, 128), Image.LANCZOS)
        im.putalpha(im.getchannel("A").point(lambda v: 255 if v >= 128 else 0))
        im.save(f"{tmp}/{i:03d}.png")
    out = f"{dd}/{name}-{WHO}.gif"
    for fps, lossy, colours in ((16, 80, 64), (12, 80, 64), (10, 110, 48), (8, 110, 48),
                                (6, 140, 32), (5, 170, 32), (4, 200, 24)):
        raw = f"{dd}/emo-raw.gif"
        sh("ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/%03d.png",
           "-vf", f"fps={fps},split[a][b];[a]palettegen=max_colors={colours}:reserve_transparent=1[p];"
                  f"[b][p]paletteuse=dither=none:alpha_threshold=128",
           "-gifflags", "+transdiff", "-loop", "0", raw)
        sh("gifsicle", "-O3", f"--lossy={lossy}", raw, "-o", out, stderr=subprocess.DEVNULL)
        kb = os.path.getsize(out) / 1024
        if kb < 127:
            break
    print(f"{name} emoji {fps}fps {kb:.0f}KB crop {box}")


def compare(name):
    """Original | Seth. Long camera-move clips ran 10MB at 320px/16fps, so this steps down
    height and frame rate and lets gifsicle trim, aiming under 3MB so it posts anywhere."""
    dd = d(name)
    out = f"{dd}/{name}-{WHO}-comparison.gif"
    raw = f"{dd}/cmp-raw.gif"
    for height, fps, lossy in ((300, 12, 60), (260, 12, 90), (240, 10, 110), (200, 10, 140)):
        orig = f"{dd}/cmp-orig.mp4" if os.path.exists(f"{dd}/cmp-orig.mp4") else f"{dd}/drive.mp4"
        seth = f"{dd}/final.mp4" if os.path.exists(f"{dd}/final.mp4") else f"{dd}/seth.mp4"
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", orig, "-i", seth,
           "-filter_complex",
           f"[0:v]scale=-2:{height}:flags=lanczos[a];[1:v]scale=-2:{height}:flags=lanczos[b];"
           f"[a][b]hstack=inputs=2,fps={fps},split[x][y];[x]palettegen=max_colors=128[p];"
           "[y][p]paletteuse=dither=bayer:bayer_scale=4",
           "-loop", "0", raw)
        sh("gifsicle", "-O3", f"--lossy={lossy}", raw, "-o", out, stderr=subprocess.DEVNULL)
        if os.path.getsize(out) < 3 * 1024 * 1024:
            break
    print(f"{name} compare {height}px {fps}fps {os.path.getsize(out)//1024}KB")



def full(name):
    """The render alone at its native size, no original beside it: the version to post as a
    big reaction rather than an emoji. Steps down frame rate and lossiness to stay under 8MB."""
    dd = d(name)
    src = f"{dd}/final.mp4" if os.path.exists(f"{dd}/final.mp4") else f"{dd}/seth.mp4"
    out, raw = f"{dd}/{name}-{WHO}-full.gif", f"{dd}/full-raw.gif"
    for fps, lossy in ((16, 40), (16, 80), (12, 80), (12, 120), (10, 140)):
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", src, "-filter_complex",
           f"fps={fps},split[x][y];[x]palettegen=max_colors=192[p];"
           "[y][p]paletteuse=dither=bayer:bayer_scale=4", "-loop", "0", raw)
        sh("gifsicle", "-O3", f"--lossy={lossy}", raw, "-o", out, stderr=subprocess.DEVNULL)
        if os.path.getsize(out) < 8 * 1024 * 1024:
            break
    print(f"{name} full {fps}fps {os.path.getsize(out)//1024}KB")

def deliver(name):
    dd = d(name)
    sh("scp", "-q", f"{dd}/{name}-{WHO}.gif", f"{dd}/{name}-{WHO}-comparison.gif",
       f"{dd}/{name}-{WHO}-full.gif",
       f"{MAC}:{MAC_DIR}/")
    print(f"{name} delivered")


STAGES = ["prep", "ref", "animate", "cutout", "post", "emoji", "compare", "full", "deliver"]
DONE = {"prep": "spec.json", "ref": "ref.png", "animate": "seth.mp4", "cutout": "cut/f000.png", "post": "post/f000.png",
        "emoji": "{n}-{w}.gif", "compare": "{n}-{w}-comparison.gif", "full": "{n}-{w}-full.gif", "deliver": None}


def run_all(names):
    for name in names:
        for st in STAGES:
            marker = DONE[st]
            if marker and os.path.exists(f"{d(name)}/{marker.format(n=name, w=WHO)}"):
                continue
            try:
                globals()[st](name)
            except (SystemExit, subprocess.CalledProcessError) as e:
                print(f"{name} {st} FAILED: {e}")
                break


if __name__ == "__main__":
    stage, names = sys.argv[1], sys.argv[2:] or list(MEMES)
    if stage == "all":
        run_all(names)
    else:
        for n in names:
            globals()[stage](n)
