#!/usr/bin/env python3
"""Recast sethmojis, Body Double style: Seth (Follies style) replaces the performer; the rest of the shot stays real.

    python3 sethmoji_recast.py <stage> <name> [--seed N]

conform   source -> 16 fps, 4n+1 frames, model size; the plate is the same frames (no restyle)
detect    SAM 3.1 picks the performer in frame 0 (positive points on him) -> init mask
anchor    Qwen-Edit + Follies LoRA repaints frame 0 with Seth (one per --seed; look at them, pick one)
refs      the chosen anchor's Seth, masked and pasted onto the REAL frame 0: the identity reference
segment   SCAIL-2 replacement mode: ref0 = identity, source = motion; output + per-frame source masks
retrack   cast masks tracked on the output from Seth's own frame-0 outline (a source mask clips long hair)
assemble  paste Seth into the plate inside dilated (source U cast) masks; burned-in captions go back on top
sethmoji  hand the result to sethmoji.py (seth.mp4 + RGBA cut frames, keep_bg) for emoji/compare/full/deliver
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
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sethmoji as M  # noqa: E402

ROOT = "/home/wilson/scratch/higgs"
GPU = M.GPU
SAM = "sam3.1_multiplex_fp16.safetensors"
SCAIL = dict(dit="wan2.1_14B_SCAIL_2_int8_convrot.safetensors", dpo="wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
             lightx2v="Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
             umt5="umt5_xxl_fp8_e4m3fn_scaled.safetensors", clip_vision="clip_vision_h.safetensors",
             vae="Wan2_1_VAE_bf16.safetensors")

# points: fractions of the frame on the performer (head, chest). caption: (y0, y1) band of burned-in text to restore.
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
        points=[(0.53, 0.22), (0.53, 0.55)], caption=(0.79, 0.93),
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
    shutil.rmtree(j.p("plate"), ignore_errors=True)
    os.makedirs(j.p("plate"))
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-i", j.p("drive.mp4"), j.p("plate/f%03d.png"))
    j.spec = {"W": W, "H": H, "frames": n}
    json.dump(j.spec, open(j.p("spec.json"), "w"))
    j.up("drive.mp4", "drive.mp4")
    j.up("plate/f001.png", "f0.png")
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


def mask_of(path, size):
    return Image.open(path).convert("L").resize(size)


def refs(j, seed):
    W, H = j.spec["W"], j.spec["H"]
    shutil.copy(j.p(f"anchor{seed}.png"), j.p("anchor.png"))
    j.up("anchor.png", "anchor.png")
    g, out = detect_graph(j.inp("anchor.png"), f"{j.tag}/amask", text="cartoon man")
    M.submit(g, out, "images", j.p("anchor_mask.png"))
    m = np.maximum(np.asarray(mask_of(j.p("anchor_mask.png"), (W, H))), np.asarray(mask_of(j.p("init_mask.png"), (W, H))))
    m = Image.fromarray(m).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(3))
    Image.composite(Image.open(j.p("anchor.png")).convert("RGB").resize((W, H)),
                    Image.open(j.p("plate/f001.png")).convert("RGB"), m).save(j.p("ref0.png"))
    j.up("ref0.png", "ref0.png")
    j.up("anchor_mask.png", "ref0-mask.png")
    print(j.name, "refs", seed)


def segment(j, seed, steps=6, lx=0.8):
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
    cv = g.add("CLIPVisionEncode", clip_vision=[g.add("CLIPVisionLoader", clip_name=SCAIL["clip_vision"]), 0],
               image=[g.add("LoadImage", image=j.inp("ref0.png")), 0], crop="none")
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
                  reference_image=[g.add("LoadImage", image=j.inp("ref0.png")), 0],
                  reference_image_mask=[colored, 1], clip_vision_output=[cv, 0])
    sample = g.add("SamplerCustom", model=model, add_noise=True, noise_seed=seed, cfg=1.0, positive=[scail, 0],
                   negative=[scail, 1], sampler=[g.add("KSamplerSelect", sampler_name="euler"), 0],
                   sigmas=[g.add("BasicScheduler", model=model, scheduler="simple", steps=steps, denoise=1.0), 0],
                   latent_image=[scail, 2])
    decoded = [g.add("VAEDecode", samples=[sample, 0], vae=[vae, 0]), 0]
    src_m = g.add("MaskToImage", mask=[g.add("SAM3_TrackToMask", track_data=[tr, 0], object_indices="0"), 0])
    outs = {"f": g.add("SaveImage", images=decoded, filename_prefix=f"{j.tag}/f"),
            "src": g.add("SaveImage", images=[src_m, 0], filename_prefix=f"{j.tag}/src")}
    shutil.rmtree(j.p("seg"), ignore_errors=True)
    t = run(g, outs, j.p("seg"))
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-framerate", str(M.FPS), "-i", j.p("seg/f/%03d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", j.p("raw.mp4"))
    print(j.name, "segment", seed, f"{t:.0f}s")


def retrack(j):
    j.up("raw.mp4", "raw.mp4")
    g = Graph()
    frames = [g.add("GetVideoComponents", video=[g.add("LoadVideo", file=j.inp("raw.mp4")), 0]), 0]
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM)
    init = [g.add("LoadImageMask", image=j.inp("ref0-mask.png"), channel="red"), 0]
    tr = g.add("SAM3_VideoTrack", images=frames, model=[sam, 0], initial_mask=init, detection_threshold=0.5,
               max_objects=1, detect_interval=1)
    out = g.add("SaveImage", images=[g.add("MaskToImage", mask=[g.add("SAM3_TrackToMask", track_data=[tr, 0], object_indices="0"), 0]), 0],
                filename_prefix=f"{j.tag}/cast")
    run(g, {"cast": out}, j.p("seg"))
    print(j.name, "retrack")


def caption_mask(plate, band):
    """Burned-in meme text: near-white pixels in the band, grown past their black outline: SCAIL-2 redraws the
    caption inside the mask a few pixels off, and a tight mask leaves that copy showing as doubled letters."""
    W, H = plate.size
    a = np.asarray(plate.convert("L"))
    m = np.zeros_like(a)
    y0, y1 = int(H * band[0]), int(H * band[1])
    m[y0:y1] = np.where(a[y0:y1] > 215, 255, 0)
    return Image.fromarray(m).filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(1.5))


def assemble(j):
    W, H, n = j.spec["W"], j.spec["H"], j.spec["frames"]
    shutil.rmtree(j.p("comp"), ignore_errors=True)
    os.makedirs(j.p("comp"))
    os.makedirs(j.p("alpha"), exist_ok=True)
    for i in range(n):
        plate = Image.open(j.p(f"plate/f{i + 1:03d}.png")).convert("RGB")
        gen = Image.open(j.p(f"seg/f/{i:03d}.png")).convert("RGB").resize((W, H))
        cast = mask_of(j.p(f"seg/cast/{i:03d}.png"), (W, H))
        m = np.maximum(np.asarray(mask_of(j.p(f"seg/src/{i:03d}.png"), (W, H))), np.asarray(cast))
        m = Image.fromarray(m).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5))
        out = Image.composite(gen, plate, m)
        if "caption" in j.m:
            out = Image.composite(plate, out, caption_mask(plate, j.m["caption"]))
        out.save(j.p(f"comp/f{i:03d}.png"))
        cast.save(j.p(f"alpha/{i:03d}.png"))
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14"]
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-framerate", str(M.FPS), "-i", j.p("comp/f%03d.png"), *enc, j.p("recast.mp4"))
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-i", j.p("drive.mp4"), "-i", j.p("recast.mp4"), "-filter_complex",
         "[0:v][1:v]vstack,scale=640:-2", *enc, j.p("compare.mp4"))
    print(j.name, "assemble", j.p("recast.mp4"))


def to_sethmoji(j):
    """sethmoji.py's emoji/compare/full read seth.mp4 + cut/ RGBA frames (alpha only places the crop; keep_bg)."""
    sd = M.d(j.name)
    os.makedirs(sd, exist_ok=True)
    shutil.rmtree(f"{sd}/cut", ignore_errors=True)
    os.makedirs(f"{sd}/cut")
    for i in range(j.spec["frames"]):
        im = Image.open(j.p(f"comp/f{i:03d}.png")).convert("RGBA")
        im.putalpha(Image.open(j.p(f"alpha/{i:03d}.png")).convert("L"))
        im.save(f"{sd}/cut/f{i:03d}.png")
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
     "segment": lambda j: segment(j, a.seed), "retrack": retrack, "assemble": assemble,
     "sethmoji": to_sethmoji}[a.stage](job)
