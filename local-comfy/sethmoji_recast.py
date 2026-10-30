#!/usr/bin/env python3
"""Recast sethmojis, Body Double style: Seth (Follies style) replaces the performer; the rest of the shot stays real.

    python3 sethmoji_recast.py <stage> <name> [--seed N]

conform   source -> 16 fps, 4n+1 frames, model size (ffmpeg, as sethmoji.py prep)
detect    SAM 3.1 picks the performer in frame 0 (positive points on him) -> init mask
anchor    Qwen-Edit + Follies LoRA repaints frame 0 with Seth (one per --seed; look at them, pick one)
refs      graph: the anchor's Seth, masked and pasted onto the REAL frame 0 -> ref0, the identity reference
render    graph: SCAIL-2 replacement (ref0 = identity, source = motion), the cast tracked from Seth's own
          frame-0 outline, Seth pasted into the source frames inside the grown (source U cast) mask, and
          burned-in captions put back on top -> composite frames + RGBA cut frames
sethmoji  hand the result to sethmoji.py (seth.mp4 + cut/, keep_bg) for emoji/compare/full/deliver

Every image operation is a ComfyUI node (cast-local-comfy rule); this file only builds and queues graphs.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sethmoji as M  # noqa: E402

ROOT = "/home/wilson/scratch/higgs"
GPU = M.GPU
SAM = "sam3.1_multiplex_fp16.safetensors"
SCAIL = dict(dit="wan2.1_14B_SCAIL_2_int8_convrot.safetensors", dpo="wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
             lightx2v="Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
             umt5="umt5_xxl_fp8_e4m3fn_scaled.safetensors", clip_vision="clip_vision_h.safetensors",
             vae="Wan2_1_VAE_bf16.safetensors")

# points: fractions of the frame on the performer (head, chest). caption: BrightTextMask bottom_band holding burned-in text.
MEMES = {
    "chest-thump": dict(
        url="https://media.tenor.com/_oIo8GMrUQ8AAAPo/wolf-of-wall-street-matthew-mcconaughey.mp4",
        size=(992, 416), points=[(0.53, 0.22), (0.56, 0.75)], anchor_seed=7,
        who="the man in the dark pinstriped suit at the left of centre",
        wear="the same dark pinstriped suit, white shirt and red patterned tie",
        face="eyes half closed, humming with lips pressed together, deadpan",
        keep="the window, the city, the restaurant, the diners behind him, the glasses and the bowl on the table",
        prompt="in a dark pinstriped suit and red tie, sits at a restaurant table by a big window over the city. "
               "He hums with his lips pressed together and thumps his chest rhythmically with his fist."),
    "thats-the-rule": dict(
        url="https://media.tenor.com/VsjL6RW_H1sAAAPo/tucker-tucker-carlson.mp4",
        points=[(0.53, 0.22), (0.53, 0.55)], caption=0.22,
        who="the news anchor in the dark suit at the desk",
        wear="the same dark suit, light blue shirt, red and navy striped tie and white pocket square",
        face="smug and earnest, eyebrows raised, mouth open mid-word as he lectures the camera",
        keep="the blue world-map backdrop, the news desk and the white caption text at the bottom",
        prompt="in a dark suit and red striped tie, sits at a TV news desk in front of a blue world map. "
               "He lectures the camera smugly, nodding and tilting his head as he talks."),
}


class Graph(dict):
    def add(self, cls, **inputs):
        nid = str(len(self) + 1)
        self[nid] = {"class_type": cls, "inputs": inputs}
        return nid


class Job:
    def __init__(self, name):
        self.name, self.m = name, MEMES[name]
        self.dd, self.tag = f"{ROOT}/{name}", f"recast-{name}"
        os.makedirs(self.dd, exist_ok=True)
        spec = f"{self.dd}/spec.json"
        self.spec = json.load(open(spec)) if os.path.exists(spec) else None

    def p(self, rel):
        return f"{self.dd}/{rel}"

    def up(self, rel, name):
        M.sh("scp", "-q", self.p(rel), f"{GPU}:comfyui/input/{self.tag}-{name}")
        return f"{self.tag}-{name}"

    def inp(self, name):
        return f"{self.tag}-{name}"


def run(g, outs, dest):
    """Queue one graph; fetch every SaveImage in `outs` ({key: node}) into dest/<key>/NNN.png."""
    for k, v in g.items():
        v.setdefault("_meta", {"title": k})
    r = M.S.api("/prompt", {"prompt": g, "client_id": "recast"})
    if "prompt_id" not in r:
        raise SystemExit("REJECTED " + json.dumps(r)[:800])
    pid, t0 = r["prompt_id"], time.time()
    while pid not in (h := M.S.api(f"/history/{pid}")):
        time.sleep(5)
    if h[pid]["status"].get("status_str") != "success":
        raise SystemExit("FAILED " + json.dumps(h[pid]["status"])[:1500])
    for k, nid in outs.items():
        os.makedirs(f"{dest}/{k}", exist_ok=True)
        for i, im in enumerate(h[pid]["outputs"][nid]["images"]):
            q = f"/view?filename={im['filename']}&subfolder={im.get('subfolder', '')}&type=output"
            with urllib.request.urlopen(M.S.HOST + q, timeout=600) as rr:
                open(f"{dest}/{k}/{i:03d}.png", "wb").write(rr.read())
    return time.time() - t0


def conform(j):
    if not os.path.exists(j.p("src.mp4")):
        urllib.request.urlretrieve(j.m["url"], j.p("src.mp4"))
    w, h = map(int, M.probe(j.p("src.mp4"), "width,height").split(","))
    W, H = j.m.get("size") or M.size_for(w, h)
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-i", j.p("src.mp4"), "-vf", f"fps={M.FPS},scale={W}:{H}:flags=lanczos",
         "-frames:v", str(M.MAX_FRAMES), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", j.p("full.mp4"))
    n = int(M.sh("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                 "stream=nb_read_frames", "-of", "csv=p=0", j.p("full.mp4"), capture_output=True, text=True).stdout)
    n = (n - 1) // 4 * 4 + 1
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-i", j.p("full.mp4"), "-frames:v", str(n), "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "10", j.p("drive.mp4"))
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-i", j.p("drive.mp4"), "-frames:v", "1", j.p("f0.png"))
    j.spec = {"W": W, "H": H, "frames": n}
    json.dump(j.spec, open(j.p("spec.json"), "w"))
    j.up("drive.mp4", "drive.mp4")
    j.up("f0.png", "f0.png")
    print(j.name, "conform", W, H, n)


def detect_graph(image, prefix, points=None, text=None):
    g = Graph()
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM)
    kw = {"positive_coords": json.dumps(points)} if points else \
        {"conditioning": [g.add("CLIPTextEncode", clip=[sam, 1], text=text), 0]}
    det = g.add("SAM3_Detect", model=[sam, 0], image=[g.add("LoadImage", image=image), 0], threshold=0.5,
                refine_iterations=2, individual_masks=False, **kw)
    return g, g.add("SaveImage", images=[g.add("MaskToImage", mask=[det, 0]), 0], filename_prefix=prefix)


def detect(j):
    W, H = j.spec["W"], j.spec["H"]
    pts = [{"x": int(W * x), "y": int(H * y)} for x, y in j.m["points"]]
    g, out = detect_graph(j.inp("f0.png"), f"{j.tag}/init", points=pts)
    M.submit(g, out, "images", j.p("init_mask.png"))
    j.up("init_mask.png", "init.png")
    print(j.name, "detect", (np.asarray(Image.open(j.p("init_mask.png")).convert("L")) > 127).mean())


def anchor(j, seed):
    m, W, H = j.m, j.spec["W"], j.spec["H"]
    prompt = (f"f0llie5, image 1 is a photograph. Replace ONLY {m['who']} with a cartoon man with {M.SETH}, drawn "
              "in warm sepia ink with soft halftone shading like a 1930s rubber-hose cartoon. He keeps exactly the same "
              f"pose, the same size and place in the frame, the same tilt of the head, the same hands and {m['wear']}, "
              f"all drawn in the cartoon style. His face: {m['face']}. Everything else stays a real colour photograph "
              f"exactly as it is: {m['keep']}. Do not redraw the background.")
    g = {
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
        "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "follies-final.safetensors", "strength_model": 1.2, "model": ["3", 0]}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "6": {"class_type": "LoadImage", "inputs": {"image": j.inp("f0.png")}},
        "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0]}},
        "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": M.NEGATIVE, "clip": ["4", 0]}},
        "22": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": M.STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
        "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
        "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"{j.tag}/anchor{seed}", "images": ["23", 0]}},
    }
    M.submit(g, "99", "images", j.p(f"anchor{seed}.png"))
    print(j.name, "anchor", seed)


def soften(g, mask, grow, radius, sigma):
    """Grow a mask, then feather it (ImageBlur on the mask as an image)."""
    m = [g.add("GrowMask", mask=mask, expand=grow, tapered_corners=True), 0]
    blur = g.add("ImageBlur", image=[g.add("MaskToImage", mask=m), 0], blur_radius=radius, sigma=sigma)
    return [g.add("ImageToMask", image=[blur, 0], channel="red"), 0]


def refs(j, seed):
    """Anchor's Seth (SAM 'cartoon man' U the performer's mask) pasted onto the real frame 0."""
    shutil.copy(j.p(f"anchor{seed}.png"), j.p("anchor.png"))
    j.up("anchor.png", "anchor.png")
    g = Graph()
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM)
    anc = [g.add("LoadImage", image=j.inp("anchor.png")), 0]
    det = [g.add("SAM3_Detect", model=[sam, 0], image=anc, threshold=0.5, refine_iterations=2, individual_masks=False,
                 conditioning=[g.add("CLIPTextEncode", clip=[sam, 1], text="cartoon man"), 0]), 0]
    init = [g.add("LoadImageMask", image=j.inp("init.png"), channel="red"), 0]
    both = [g.add("MaskComposite", destination=det, source=init, x=0, y=0, operation="or"), 0]
    ref0 = g.add("ImageCompositeMasked", destination=[g.add("LoadImage", image=j.inp("f0.png")), 0], source=anc,
                 x=0, y=0, resize_source=False, mask=soften(g, both, 4, 5, 3.0))
    outs = {"ref0": g.add("SaveImage", images=[ref0, 0], filename_prefix=f"{j.tag}/ref0"),
            "amask": g.add("SaveImage", images=[g.add("MaskToImage", mask=det), 0], filename_prefix=f"{j.tag}/amask")}
    run(g, outs, j.p("refs"))
    shutil.copy(j.p("refs/ref0/000.png"), j.p("ref0.png"))
    j.up("ref0.png", "ref0.png")
    shutil.copy(j.p("refs/amask/000.png"), j.p("anchor_mask.png"))
    j.up("anchor_mask.png", "ref0-mask.png")
    print(j.name, "refs", seed)


def render(j, seed, steps=6, lx=0.8):
    W, H, n = j.spec["W"], j.spec["H"], j.spec["frames"]
    prompt = (f"A cartoon man drawn in warm sepia ink with soft halftone shading, with long wavy brown hair and a bushy "
              f"handlebar moustache, {j.m['prompt']}")
    g = Graph()
    model = [g.add("UNETLoader", unet_name=SCAIL["dit"], weight_dtype="default"), 0]
    model = [g.add("LoraLoaderModelOnly", model=model, lora_name=SCAIL["dpo"], strength_model=1.0), 0]
    model = [g.add("LoraLoaderModelOnly", model=model, lora_name=SCAIL["lightx2v"], strength_model=lx), 0]
    model = [g.add("ModelSamplingSD3", model=model, shift=5.0), 0]
    clip = g.add("CLIPLoader", clip_name=SCAIL["umt5"], type="wan", device="default")
    pos = g.add("CLIPTextEncode", clip=[clip, 0], text=prompt)
    neg = g.add("CLIPTextEncode", clip=[clip, 0], text="")
    vae = g.add("VAELoader", vae_name=SCAIL["vae"])
    ref0 = [g.add("LoadImage", image=j.inp("ref0.png")), 0]
    cv = g.add("CLIPVisionEncode", clip_vision=[g.add("CLIPVisionLoader", clip_name=SCAIL["clip_vision"]), 0],
               image=ref0, crop="none")
    frames = [g.add("GetVideoComponents", video=[g.add("LoadVideo", file=j.inp("drive.mp4")), 0]), 0]
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM)
    init = [g.add("LoadImageMask", image=j.inp("init.png"), channel="red"), 0]
    tr = g.add("SAM3_VideoTrack", images=frames, model=[sam, 0], initial_mask=init, detection_threshold=0.5,
               max_objects=1, detect_interval=1)
    colored = g.add("SCAIL2ColoredMask", driving_track_data=[tr, 0], object_indices="", sort_by="none",
                    replacement_mode=True,
                    ref_track_data=[g.add("LoadImageMask", image=j.inp("ref0-mask.png"), channel="red"), 0])
    scail = g.add("WanSCAILToVideo", positive=[pos, 0], negative=[neg, 0], vae=[vae, 0], width=W, height=H,
                  length=n, batch_size=1, pose_strength=1.0, pose_start=0.0, pose_end=1.0, video_frame_offset=0,
                  previous_frame_count=5, pose_video=frames, pose_video_mask=[colored, 0], replacement_mode=True,
                  reference_image=ref0, reference_image_mask=[colored, 1], clip_vision_output=[cv, 0])
    sample = g.add("SamplerCustom", model=model, add_noise=True, noise_seed=seed, cfg=1.0, positive=[scail, 0],
                   negative=[scail, 1], sampler=[g.add("KSamplerSelect", sampler_name="euler"), 0],
                   sigmas=[g.add("BasicScheduler", model=model, scheduler="simple", steps=steps, denoise=1.0), 0],
                   latent_image=[scail, 2])
    decoded = [g.add("VAEDecode", samples=[sample, 0], vae=[vae, 0]), 0]
    # the cast is tracked from Seth's own frame-0 outline: the performer's outline clips his long hair
    cast_tr = g.add("SAM3_VideoTrack", images=decoded, model=[sam, 0],
                    initial_mask=[g.add("LoadImageMask", image=j.inp("ref0-mask.png"), channel="red"), 0],
                    detection_threshold=0.5, max_objects=1, detect_interval=1)
    src_m = [g.add("SAM3_TrackToMask", track_data=[tr, 0], object_indices="0"), 0]
    cast_m = [g.add("SAM3_TrackToMask", track_data=[cast_tr, 0], object_indices="0"), 0]
    both = [g.add("MaskComposite", destination=src_m, source=cast_m, x=0, y=0, operation="or"), 0]
    comp = [g.add("ImageCompositeMasked", destination=frames, source=decoded, x=0, y=0, resize_source=False,
                  mask=soften(g, both, 2, 3, 1.5)), 0]
    if "caption" in j.m:
        # SCAIL-2 redraws the caption inside the mask a few pixels off; grow 7 covers that copy too
        text = [g.add("BrightTextMask", images=frames, threshold=0.84, top_band=0.0, bottom_band=j.m["caption"],
                      grow=7), 0]
        comp = [g.add("ImageCompositeMasked", destination=comp, source=frames, x=0, y=0, resize_source=False,
                      mask=text), 0]
    cut = g.add("JoinImageWithAlpha", image=comp, alpha=[g.add("InvertMask", mask=cast_m), 0])
    outs = {"comp": g.add("SaveImage", images=comp, filename_prefix=f"{j.tag}/comp"),
            "cut": g.add("SaveImage", images=[cut, 0], filename_prefix=f"{j.tag}/cut")}
    shutil.rmtree(j.p("out"), ignore_errors=True)
    t = run(g, outs, j.p("out"))
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14"]
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-framerate", str(M.FPS), "-i", j.p("out/comp/%03d.png"), *enc,
         j.p("recast.mp4"))
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-i", j.p("drive.mp4"), "-i", j.p("recast.mp4"), "-filter_complex",
         "[0:v][1:v]vstack,scale=640:-2", *enc, j.p("compare.mp4"))
    print(j.name, "render", seed, f"{t:.0f}s", j.p("recast.mp4"))


def to_sethmoji(j):
    """sethmoji.py's emoji/compare/full read seth.mp4 + cut/ RGBA frames (alpha only places the crop; keep_bg)."""
    sd = M.d(j.name)
    os.makedirs(sd, exist_ok=True)
    shutil.rmtree(f"{sd}/cut", ignore_errors=True)
    os.makedirs(f"{sd}/cut")
    for i, f in enumerate(sorted(glob.glob(j.p("out/cut/*.png")))):
        shutil.copy(f, f"{sd}/cut/f{i:03d}.png")
    for f in ("post", "opaque", "emo"):
        shutil.rmtree(f"{sd}/{f}", ignore_errors=True)
    for f in ("final.mp4", "cmp-orig.mp4"):
        if os.path.exists(f"{sd}/{f}"):
            os.remove(f"{sd}/{f}")
    shutil.copy(j.p("recast.mp4"), f"{sd}/seth.mp4")
    shutil.copy(j.p("drive.mp4"), f"{sd}/drive.mp4")
    print(j.name, "-> sethmoji", sd)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("name")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    job = Job(a.name)
    {"conform": conform, "detect": detect, "anchor": lambda j: anchor(j, a.seed), "refs": lambda j: refs(j, a.seed),
     "render": lambda j: render(j, a.seed), "sethmoji": to_sethmoji}[a.stage](job)
