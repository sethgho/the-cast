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
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf",
       f"crop={w2}:{keep}:0:0,fps={FPS}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-crf", "12", f"{dd}/drive-full.mp4")
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
    sh("scp", "-q", f"{dd}/drive.mp4", f"{GPU}:comfyui/input/sethmoji-{name}-drive.mp4")
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
    spec = json.load(open(f"{dd}/spec.json"))
    sh("scp", "-q", f"{dd}/frame0.png", f"{GPU}:comfyui/input/sethmoji-{name}-f0.png")
    prompt = ("f0llie5, image 1 is a photograph to redraw as a cartoon. This is a repaint: the "
              "pose, the tilt of the head, the direction the eyes are looking, the hands and "
              "anything held in them, the clothing, the framing and the background are already "
              "exactly right and must not change, including the crop and the size of the head in the frame. Replace only the person's head, face and hair "
              f"with a cartoon man with {SETH}. His face: {m.get('face', 'making exactly the same expression as the person in the photograph, not smiling unless they are')}. Keep the clothing "
              "exactly as in the photograph. Draw the whole picture, the background included, "
              "in warm sepia ink on aged paper.")
    W, H = spec["W"], spec["H"]
    g = {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "follies-final.safetensors", "strength_model": 1.2, "model": ["3", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "6": {"class_type": "LoadImage", "inputs": {"image": f"sethmoji-{name}-f0.png"}},
     "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
     "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0]}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE + ", photograph, text, subtitles", "clip": ["4", 0]}},
     "22": {"class_type": "KSampler", "inputs": {"seed": 7, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
     "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"sethmoji/{name}-ref", "images": ["23", 0]}},
    }
    submit(g, "99", "images", f"{dd}/ref.png")
    sh("scp", "-q", f"{dd}/ref.png", f"{GPU}:comfyui/input/sethmoji-{name}-ref.png")
    print(f"{name} ref")


def animate(name):
    m, dd = MEMES[name], d(name)
    spec = json.load(open(f"{dd}/spec.json"))
    a = argparse.Namespace(
        ref=f"sethmoji-{name}-ref.png", drive=f"sethmoji-{name}-drive.mp4",
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
    sh("scp", "-q", f"{dd}/seth.mp4", f"{GPU}:comfyui/input/sethmoji-{name}-seth.mp4")
    g = {
     "1": {"class_type": "LoadVideo", "inputs": {"file": f"sethmoji-{name}-seth.mp4"}},
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


def emoji(name):
    from PIL import Image, ImageFilter
    m, dd = MEMES[name], d(name)
    frames = [Image.open(f).convert("RGBA") for f in sorted(glob.glob(f"{dd}/cut/f*.png"))]
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
    box = (left, upper, left + side, upper + side)
    tmp = f"{dd}/emo"
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        im = f.crop(box)
        a = im.getchannel("A")
        im = im.convert("RGB").filter(ImageFilter.MedianFilter(5)).convert("RGBA")
        im.putalpha(a)
        im = im.resize((128, 128), Image.LANCZOS)
        im.putalpha(im.getchannel("A").point(lambda v: 255 if v >= 128 else 0))
        im.save(f"{tmp}/{i:03d}.png")
    out = f"{dd}/{name}-seth.gif"
    for fps, lossy, colours in ((16, 80, 64), (12, 80, 64), (10, 110, 48), (8, 110, 48),
                                (6, 140, 32)):
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
    out = f"{dd}/{name}-seth-comparison.gif"
    raw = f"{dd}/cmp-raw.gif"
    for height, fps, lossy in ((300, 12, 60), (260, 12, 90), (240, 10, 110), (200, 10, 140)):
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{dd}/drive.mp4", "-i", f"{dd}/seth.mp4",
           "-filter_complex",
           f"[0:v]scale=-2:{height}:flags=lanczos[a];[1:v]scale=-2:{height}:flags=lanczos[b];"
           f"[a][b]hstack=inputs=2,fps={fps},split[x][y];[x]palettegen=max_colors=128[p];"
           "[y][p]paletteuse=dither=bayer:bayer_scale=4",
           "-loop", "0", raw)
        sh("gifsicle", "-O3", f"--lossy={lossy}", raw, "-o", out, stderr=subprocess.DEVNULL)
        if os.path.getsize(out) < 3 * 1024 * 1024:
            break
    print(f"{name} compare {height}px {fps}fps {os.path.getsize(out)//1024}KB")


def deliver(name):
    dd = d(name)
    sh("scp", "-q", f"{dd}/{name}-seth.gif", f"{dd}/{name}-seth-comparison.gif",
       f"{MAC}:{MAC_DIR}/")
    print(f"{name} delivered")


STAGES = ["prep", "ref", "animate", "cutout", "emoji", "compare", "deliver"]
DONE = {"prep": "spec.json", "ref": "ref.png", "animate": "seth.mp4", "cutout": "cut/f000.png",
        "emoji": "{n}-seth.gif", "compare": "{n}-seth-comparison.gif", "deliver": None}


def run_all(names):
    for name in names:
        for st in STAGES:
            marker = DONE[st]
            if marker and os.path.exists(f"{d(name)}/{marker.format(n=name)}"):
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
