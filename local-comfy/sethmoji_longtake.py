#!/usr/bin/env python3
"""Long-take recast with the original audio: cartoon Seth through a whole edited clip, shot by shot.

    python3 sethmoji_longtake.py <stage> <name>     stages: conform anchor refs shot-anchor shot-refs render plate recomp assemble

The Body Double method beyond one 81-frame job (see the `comfy-recast` skill):
- the take is split at the source's own cuts (TAKES[name]["shots"]); a shot marked "keep" (a crowd cutaway)
  passes through untouched;
- each shot is tracked from its OWN first frame (SAM 3.1 detects "man"), so cuts never confuse the tracker;
- a shot longer than 81 frames is chained: 81-frame segments overlapping by 5, each conditioned on the last 5
  rendered frames of the one before (WanSCAILToVideo previous_frames), its track started from the previous
  segment's source mask at the overlap; the 5 overlap frames are dropped and colour-matched;
- one identity for the whole take: ref0 (the Qwen + Follies anchor pasted on the real first frame of shot 0);
- the original audio is muxed back over the result, frame-exact at 16 fps.
Every image step is a ComfyUI node; this file only cuts, queues graphs and concatenates.
"""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sethmoji as M  # noqa: E402
import sethmoji_recast as R  # noqa: E402

SEG, OVERLAP = 81, 5
STEP = SEG - OVERLAP
TAKES = {
    "ballmer": dict(
        url="https://www.youtube.com/watch?v=8fcSviC7cRM", start=29.4, end=53.5, size=(752, 560),
        # source cut times (s, absolute); "keep" = no performer to replace
        shots=[(29.4, 39.63), (39.63, 44.63), (44.63, 47.37), (47.37, 48.73, "keep"), (48.73, 50.57), (50.57, 53.5)],
        who="the bald man in the light blue shirt", face="mouth wide open mid-shout, eyes wide, wild and sweaty",
        closeups=[1, 2, 5],
        prompt="in a sweat-soaked light blue button-down shirt and khaki trousers, on a big conference stage, "
               "shouting and chanting with wild energy, clapping, stomping and pumping his fists"),
}
# anchor + refs reuse sethmoji_recast's MEMES machinery for shot 0
R.MEMES["ballmer"] = dict(
    url=TAKES["ballmer"]["url"], points=[(0.48, 0.20), (0.55, 0.45), (0.36, 0.35), (0.58, 0.80)],
    who="the bald man in the light blue shirt and khaki trousers on the stage",
    wear="the same sweaty light blue button-down shirt and khaki trousers",
    face="mouth wide open mid-shout, eyes wide, wild and sweaty with excitement",
    keep="the stage, the purple and blue set, the black backdrop and the lights",
    prompt=TAKES["ballmer"]["prompt"])


def batch_images(g, nodes):
    if len(nodes) == 1:
        return nodes[0]
    return [g.add("BatchImagesNode", **{f"images.image{i}": n for i, n in enumerate(nodes)}), 0]


def pad_4n1(n):
    n = max(n, 9)
    return n + (-(n - 1)) % 4


def plan(t):
    """Per shot: (first frame, frame count) at 16 fps from the take start, and its segments."""
    out, prev_end = [], 0
    for s in t["shots"]:
        end_f = round((s[1] - t["start"]) * M.FPS)
        n = end_f - prev_end
        segs, k = [], 0
        while True:
            a = k * STEP
            segs.append((a, min(SEG, n - a)))
            if a + SEG >= n:
                break
            k += 1
        out.append(dict(first=prev_end, frames=n, keep=len(s) > 2 and s[2] == "keep", segs=segs))
        prev_end = end_f
    return out


def conform(j, t):
    src = j.p("src57.mp4")
    if not os.path.exists(src):
        M.sh(os.path.expanduser("~/.local/bin/yt-dlp"), "-q", "--js-runtimes", "node", "-f",
             "bv*[height<=720]+ba/b[height<=720]", "--merge-output-format", "mp4", "-o", src, t["url"])
    W, H = t["size"]
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-ss", str(t["start"]), "-to", str(t["end"]), "-i", src,
         "-vf", f"fps={M.FPS},scale={W}:{H}:flags=lanczos", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "10", j.p("take.mp4"))
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-ss", str(t["start"]), "-to", str(t["end"]), "-i", src,
         "-vn", "-c:a", "aac", "-b:a", "192k", j.p("take.m4a"))
    shutil.rmtree(j.p("frames"), ignore_errors=True)
    os.makedirs(j.p("frames"))
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-i", j.p("take.mp4"), j.p("frames/%04d.png"))
    total = len(glob.glob(j.p("frames/*.png")))
    shots = plan(t)
    # every shot's segments as their own 4n+1 mp4 (last frame repeated to pad)
    for si, sh in enumerate(shots):
        for ki, (a, n) in enumerate(sh["segs"]):
            seg_dir = j.p(f"seg/s{si}k{ki}")
            shutil.rmtree(seg_dir, ignore_errors=True)
            os.makedirs(seg_dir)
            L = pad_4n1(n)
            for i in range(L):
                f = sh["first"] + a + min(i, n - 1) + 1
                f = min(f, total)
                shutil.copy(j.p(f"frames/{f:04d}.png"), f"{seg_dir}/{i:04d}.png")
            M.sh("ffmpeg", "-y", "-loglevel", "error", "-framerate", str(M.FPS), "-i", f"{seg_dir}/%04d.png",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", j.p(f"seg/s{si}k{ki}.mp4"))
            sh["segs"][ki] = (a, n, L)
            if not sh["keep"]:
                j.up(f"seg/s{si}k{ki}.mp4", f"s{si}k{ki}.mp4")
                if ki == 0:
                    j.up(f"seg/s{si}k{ki}/0000.png", f"s{si}f0.png")
    j.spec = {"W": W, "H": H, "frames": total, "shots": shots}
    json.dump(j.spec, open(j.p("spec.json"), "w"), indent=1)
    # shot 0's first frame is the anchor frame for recast's detect/anchor/refs
    shutil.copy(j.p("seg/s0k0/0000.png"), j.p("f0.png"))
    j.up("f0.png", "f0.png")
    print(j.name, "conform", W, H, total, "frames,", [(s["frames"], len(s["segs"]), s["keep"]) for s in shots])


def shot_anchor(j, t, si, seed):
    """A close-up needs its own identity reference: shot 0's ref0 shows Seth's face about 30 px tall, so a close-up
    rendered from it invents a face (Ballmer's last shot got a beard). Qwen + Follies repaints the performer on this
    shot's own first frame. One image only: the canon plate as image 2 wins over the pose (cast-style), and
    "no beard" summons one, so the bare chin is stated positively and last."""
    W, H = j.spec["W"], j.spec["H"]
    prompt = (f"f0llie5, image 1 is a photograph. This is a repaint: replace ONLY {t['who']} with a cartoon man "
              f"with {M.SETH}, drawn in warm sepia ink with soft halftone shading like a 1930s rubber-hose cartoon. "
              "He keeps exactly the same pose, the same size and place in the frame, the same tilt of the head and "
              "the same hands, and he wears a plain tan button-down shirt. Everything else stays a real colour "
              f"photograph exactly as it is. His face: {t['face']}. His chin, jaw, cheeks and neck are smooth and "
              "bare; the only hair on his face is the big handlebar moustache.")
    g = {
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
        "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "follies-final.safetensors", "strength_model": 1.2, "model": ["3", 0]}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "6": {"class_type": "LoadImage", "inputs": {"image": j.inp(f"s{si}f0.png")}},
        "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0]}},
        "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": M.NEGATIVE, "clip": ["4", 0]}},
        "22": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": M.STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
        "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
        "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"{j.tag}/s{si}anchor{seed}", "images": ["23", 0]}},
    }
    M.submit(g, "99", "images", j.p(f"s{si}anchor{seed}.png"))
    print(j.name, f"shot {si} anchor", seed)


def shot_refs(j, si, seed):
    """The shot's anchor, cast outline only (plus the performer's), pasted on its real first frame -> s<si>ref."""
    up = j.up(f"s{si}anchor{seed}.png", f"s{si}anchor.png")
    g = R.Graph()
    sam = g.add("CheckpointLoaderSimple", ckpt_name=R.SAM)

    def det(image, text):
        return [g.add("SAM3_Detect", model=[sam, 0], image=image, threshold=0.5, refine_iterations=2,
                      individual_masks=False,
                      conditioning=[g.add("CLIPTextEncode", clip=[sam, 1], text=text), 0]), 0]
    anc, f0 = [g.add("LoadImage", image=up), 0], [g.add("LoadImage", image=j.inp(f"s{si}f0.png")), 0]
    cast = det(anc, "cartoon man")
    union = [g.add("MaskComposite", destination=det(f0, "man"), source=cast, x=0, y=0, operation="or"), 0]
    ref = g.add("ImageCompositeMasked", destination=f0, source=anc, x=0, y=0, resize_source=False,
                mask=R.soften(g, union, 4, 5, 3.0))
    d = j.p(f"refs/s{si}")
    shutil.rmtree(d, ignore_errors=True)
    R.run(g, {"ref": g.add("SaveImage", images=[ref, 0], filename_prefix=f"{j.tag}/s{si}ref"),
              "mask": g.add("SaveImage", images=[g.add("MaskToImage", mask=cast), 0],
                            filename_prefix=f"{j.tag}/s{si}refmask")}, d)
    shutil.copy(f"{d}/ref/000.png", j.p(f"s{si}ref.png"))
    shutil.copy(f"{d}/mask/000.png", j.p(f"s{si}ref-mask.png"))
    j.up(f"s{si}ref.png", f"s{si}ref.png")
    j.up(f"s{si}ref-mask.png", f"s{si}ref-mask.png")
    print(j.name, f"shot {si} refs", seed)


def seg_graph(j, si, ki, L, prev_tail, init_name, prompt, seed, ref="ref0.png", ref_mask="ref-mask0.png"):
    g = R.Graph()
    S = R.SCAIL
    model = [g.add("UNETLoader", unet_name=S["dit"], weight_dtype="default"), 0]
    model = [g.add("LoraLoaderModelOnly", model=model, lora_name=S["dpo"], strength_model=1.0), 0]
    model = [g.add("LoraLoaderModelOnly", model=model, lora_name=S["lightx2v"], strength_model=0.8), 0]
    model = [g.add("ModelSamplingSD3", model=model, shift=5.0), 0]
    clip = g.add("CLIPLoader", clip_name=S["umt5"], type="wan", device="default")
    pos = g.add("CLIPTextEncode", clip=[clip, 0], text=prompt)
    neg = g.add("CLIPTextEncode", clip=[clip, 0], text="")
    vae = g.add("VAELoader", vae_name=S["vae"])
    ref0 = [g.add("LoadImage", image=j.inp(ref)), 0]
    cv = g.add("CLIPVisionEncode", clip_vision=[g.add("CLIPVisionLoader", clip_name=S["clip_vision"]), 0],
               image=ref0, crop="none")
    frames = [g.add("GetVideoComponents", video=[g.add("LoadVideo", file=j.inp(f"s{si}k{ki}.mp4")), 0]), 0]
    sam = g.add("CheckpointLoaderSimple", ckpt_name=R.SAM)
    if init_name:
        trk = dict(initial_mask=[g.add("LoadImageMask", image=j.inp(init_name), channel="red"), 0])
    else:  # a new shot: find the performer on its own first frame
        trk = dict(conditioning=[g.add("CLIPTextEncode", clip=[sam, 1], text="man"), 0])
    tr = g.add("SAM3_VideoTrack", images=frames, model=[sam, 0], detection_threshold=0.4, max_objects=1,
               detect_interval=1, **trk)
    colored = g.add("SCAIL2ColoredMask", driving_track_data=[tr, 0], object_indices="", sort_by="none",
                    replacement_mode=True,
                    ref_track_data=[g.add("LoadImageMask", image=j.inp(ref_mask), channel="red"), 0])
    extra = {}
    if prev_tail:
        extra["previous_frames"] = batch_images(g, [[g.add("LoadImage", image=f), 0] for f in prev_tail])
    scail = g.add("WanSCAILToVideo", positive=[pos, 0], negative=[neg, 0], vae=[vae, 0], width=j.spec["W"],
                  height=j.spec["H"], length=L, batch_size=1, pose_strength=1.0, pose_start=0.0, pose_end=1.0,
                  video_frame_offset=0, previous_frame_count=OVERLAP, pose_video=frames,
                  pose_video_mask=[colored, 0], replacement_mode=True, reference_image=ref0,
                  reference_image_mask=[colored, 1], clip_vision_output=[cv, 0], **extra)
    sample = g.add("SamplerCustom", model=model, add_noise=True, noise_seed=seed, cfg=1.0, positive=[scail, 0],
                   negative=[scail, 1], sampler=[g.add("KSamplerSelect", sampler_name="euler"), 0],
                   sigmas=[g.add("BasicScheduler", model=model, scheduler="simple", steps=6, denoise=1.0), 0],
                   latent_image=[scail, 2])
    decoded = [g.add("VAEDecode", samples=[sample, 0], vae=[vae, 0]), 0]
    if prev_tail:  # keep the lighting continuous across the join: match the re-rendered overlap to the last frame
        decoded = [g.add("ColorTransfer", image_target=decoded,
                         image_ref=[g.add("LoadImage", image=prev_tail[-1]), 0], method="reinhard_lab",
                         source_stats="target_frame", **{"source_stats.target_index": OVERLAP - 1},
                         strength=1.0), 0]
    cast_tr = g.add("SAM3_VideoTrack", images=decoded, model=[sam, 0], detection_threshold=0.4, max_objects=1,
                    detect_interval=1, conditioning=[g.add("CLIPTextEncode", clip=[sam, 1], text="cartoon man"), 0])
    src_m = [g.add("SAM3_TrackToMask", track_data=[tr, 0], object_indices=""), 0]
    cast_m = [g.add("SAM3_TrackToMask", track_data=[cast_tr, 0], object_indices=""), 0]
    both = [g.add("MaskComposite", destination=src_m, source=cast_m, x=0, y=0, operation="or"), 0]
    comp = g.add("ImageCompositeMasked", destination=frames, source=decoded, x=0, y=0, resize_source=False,
                 mask=R.soften(g, both, 2, 3, 1.5))
    outs = {"comp": g.add("SaveImage", images=[comp, 0], filename_prefix=f"{j.tag}/s{si}k{ki}/comp"),
            "raw": g.add("SaveImage", images=decoded, filename_prefix=f"{j.tag}/s{si}k{ki}/raw"),
            "src": g.add("SaveImage", images=[g.add("MaskToImage", mask=src_m), 0],
                         filename_prefix=f"{j.tag}/s{si}k{ki}/src")}
    return g, outs


def render(j, t, seed, only=None):
    shots = j.spec["shots"]
    for si, sh in enumerate(shots):
        if sh["keep"] or (only is not None and si not in only):
            continue
        prev_tail = None
        init = "init0.png" if si == 0 else None  # shot 0 starts from the hand-pointed detect mask
        for ki, (a, n, L) in enumerate(sh["segs"]):
            if ki:
                p = j.p(f"out/s{si}k{ki - 1}")
                # tail of the previous render (its frames STEP..STEP+4 are this segment's first 5) + handoff mask
                prev_tail = []
                for i in range(OVERLAP):
                    nm = f"s{si}k{ki}-prev{i}.png"
                    shutil.copy(f"{p}/raw/{STEP + i:03d}.png", j.p(nm))
                    prev_tail.append(j.up(nm, nm))
                shutil.copy(f"{p}/src/{STEP:03d}.png", j.p(f"s{si}k{ki}-init.png"))
                j.up(f"s{si}k{ki}-init.png", f"s{si}k{ki}-init.png")
                init = f"s{si}k{ki}-init.png"
            own = si in t.get("closeups", []) and os.path.exists(j.p(f"s{si}ref.png"))
            refs = (f"s{si}ref.png", f"s{si}ref-mask.png") if own else ("ref0.png", "ref-mask0.png")
            g, outs = seg_graph(j, si, ki, L, prev_tail, init, R_prompt(t), seed, *refs)
            out = j.p(f"out/s{si}k{ki}")
            for part in ("comp", "comp-v1", "raw", "src"):  # keep plate/: it depends only on the source
                shutil.rmtree(f"{out}/{part}", ignore_errors=True)
            secs = R.run(g, outs, out)
            print(j.name, f"shot {si} seg {ki}", f"{L} frames", f"{secs:.0f}s", flush=True)


PLATE_W, PLATE_H = 768, 576  # LTX wants multiples of 64; scaled back to the take size after decode
PLATE_GRAPH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab/openfield_effects/effect-empty.json")


def plate_graph(j, name, n):
    """LTX-2.5 + Clean-Plate IC-LoRA: the segment with the performer removed (the 'empty the scene' effect),
    patched to 0.43.2 core nodes like run_effects.py, saved as frames at the take size."""
    g = json.load(open(PLATE_GRAPH))
    for k, v in list(g.items()):
        c, i = v["class_type"], v["inputs"]
        if c == "LTXICLoRALoaderModelOnly":
            v["class_type"] = "LoraLoaderModelOnly"
            g[k + "_p"] = {"class_type": "GetICLoRAParameters", "inputs": {"iclora_model": [k, 0]}}
        if c == "VAEDecodeTiled":
            i.update(tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
        if c == "VHS_LoadVideo":
            i.update(video=name, force_rate=0, custom_width=PLATE_W, custom_height=PLATE_H, frame_load_cap=n)
        if c == "ImageScale":
            i.update(width=PLATE_W, height=PLATE_H)
        if c == "EmptyLTXVLatentVideo":
            i.update(width=PLATE_W, height=PLATE_H, length=n)
        if c == "LTXVEmptyLatentAudio":
            i.update(frames_number=n)
        if c == "LTXVConditioning":
            i.update(frame_rate=M.FPS)
    for k, v in list(g.items()):
        if v["class_type"] == "LTXAddVideoICLoRAGuide":
            i = v["inputs"]
            v["class_type"] = "LTXVAddGuide"
            v["inputs"] = {"positive": i["positive"], "negative": i["negative"], "vae": i["vae"], "latent": i["latent"],
                           "image": i["image"], "frame_idx": 0, "strength": 1,
                           "iclora_parameters": [i["latent_downscale_factor"][0] + "_p", 0]}
    del g["vid"], g["save"]
    g["fit"] = {"class_type": "ImageScale", "inputs": {"image": ["dv", 0], "upscale_method": "lanczos",
                                                        "width": j.spec["W"], "height": j.spec["H"], "crop": "disabled"}}
    g["out"] = {"class_type": "SaveImage", "inputs": {"images": ["fit", 0], "filename_prefix": f"{j.tag}/plate"}}
    return g


def plates(j, only=None):
    """A clean plate per segment. LTX needs 8n+1 frames, so the segment is padded with its last frame."""
    for si, sh in enumerate(j.spec["shots"]):
        if sh["keep"] or (only is not None and si not in only):
            continue
        for ki, (a, n, L) in enumerate(sh["segs"]):
            L8 = L + (-(L - 1)) % 8
            d = j.p(f"seg/s{si}k{ki}")
            M.sh("ffmpeg", "-y", "-loglevel", "error", "-framerate", str(M.FPS), "-i", f"{d}/%04d.png", "-vf",
                 f"tpad=stop_mode=clone:stop={L8 - L}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10",
                 j.p(f"seg/s{si}k{ki}-plate.mp4"))
            name = j.up(f"seg/s{si}k{ki}-plate.mp4", f"s{si}k{ki}-plate.mp4")
            M.urllib.request.urlopen(M.urllib.request.Request(  # /free has an empty body; S.api parses JSON
                M.S.HOST + "/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
                {"Content-Type": "application/json"}), timeout=60).read()
            out = j.p(f"out/s{si}k{ki}")
            shutil.rmtree(f"{out}/plate", ignore_errors=True)
            secs = R.run(plate_graph(j, name, L8), {"plate": "out"}, out)
            print(j.name, f"plate shot {si} seg {ki}", f"{L8} frames", f"{secs:.0f}s", flush=True)


def recomp(j, only=None):
    """Re-composite without the ghost: real frame outside, the clean plate inside the performer's grown outline,
    and the render on top inside the CAST's own tight outline only. Pasting the render's background over the
    performer's whole outline left a Steve-shaped patch (the render's background is a few levels off)."""
    for si, sh in enumerate(j.spec["shots"]):
        if sh["keep"] or (only is not None and si not in only):
            continue
        for ki, (a, n, L) in enumerate(sh["segs"]):
            o = j.p(f"out/s{si}k{ki}")
            if not os.path.isdir(f"{o}/comp-v1"):
                shutil.copytree(f"{o}/comp", f"{o}/comp-v1")
            names = {}
            for part in ("raw", "src", "plate"):
                pat = f"{o}/{part}/%03d.png"
                M.sh("ffmpeg", "-y", "-loglevel", "error", "-framerate", str(M.FPS), "-i", pat, "-frames:v", str(L),
                     "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "4", f"{o}/{part}.mp4")
                names[part] = j.up(f"out/s{si}k{ki}/{part}.mp4", f"s{si}k{ki}-{part}.mp4")
            g = R.Graph()

            def frames(nm):
                return [g.add("GetVideoComponents", video=[g.add("LoadVideo", file=nm), 0]), 0]
            src = frames(j.inp(f"s{si}k{ki}.mp4"))
            raw, plate = frames(names["raw"]), frames(names["plate"])
            src_m = [g.add("ImageToMask", image=frames(names["src"]), channel="red"), 0]
            sam = g.add("CheckpointLoaderSimple", ckpt_name=R.SAM)
            cast_tr = g.add("SAM3_VideoTrack", images=raw, model=[sam, 0], detection_threshold=0.4, max_objects=1,
                            detect_interval=1,
                            conditioning=[g.add("CLIPTextEncode", clip=[sam, 1], text="cartoon man"), 0])
            cast_m = [g.add("SAM3_TrackToMask", track_data=[cast_tr, 0], object_indices=""), 0]
            base = g.add("ImageCompositeMasked", destination=src, source=plate, x=0, y=0, resize_source=False,
                         mask=R.soften(g, src_m, 12, 8, 4.0))
            comp = g.add("ImageCompositeMasked", destination=[base, 0], source=raw, x=0, y=0, resize_source=False,
                         mask=R.soften(g, cast_m, 2, 3, 1.5))
            shutil.rmtree(f"{o}/comp", ignore_errors=True)
            secs = R.run(g, {"comp": g.add("SaveImage", images=[comp, 0], filename_prefix=f"{j.tag}/recomp")}, o)
            print(j.name, f"recomp shot {si} seg {ki}", f"{secs:.0f}s", flush=True)


def R_prompt(t):
    return ("A cartoon man drawn in warm sepia ink with soft halftone shading, with long wavy brown hair and a bushy "
            f"handlebar moustache, {t['prompt']}")


def assemble(j, t):
    shots = j.spec["shots"]
    dst = j.p("final")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    i = 0
    for si, sh in enumerate(shots):
        for ki, (a, n, L) in enumerate(sh["segs"]):
            keep_from = OVERLAP if ki else 0
            for f in range(keep_from, n):
                src = (j.p(f"seg/s{si}k{ki}/{f:04d}.png") if sh["keep"]
                       else j.p(f"out/s{si}k{ki}/comp/{f:03d}.png"))
                shutil.copy(src, f"{dst}/{i:04d}.png")
                i += 1
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16"]
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-framerate", str(M.FPS), "-i", f"{dst}/%04d.png", "-i",
         j.p("take.m4a"), *enc, "-c:a", "aac", "-shortest", "-movflags", "+faststart", j.p("longtake.mp4"))
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-i", j.p("take.mp4"), "-i", j.p("longtake.mp4"), "-filter_complex",
         "[0:v]scale=480:-2[a];[1:v]scale=480:-2[b];[a][b]hstack", "-map", "1:a", *enc, "-c:a", "aac",
         "-movflags", "+faststart", j.p("longtake-compare.mp4"))
    print(j.name, "assemble", i, "frames", "of", j.spec["frames"], j.p("longtake.mp4"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("name")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--shots", default="", help="comma list of shot indices to render (default all)")
    a = p.parse_args()
    job, take = R.Job(a.name), TAKES[a.name]
    if a.stage == "conform":
        conform(job, take)
    elif a.stage == "anchor":
        R.detect(job)
        R.anchor(job, a.seed)
    elif a.stage == "refs":
        R.refs(job, a.seed)
    elif a.stage == "shot-anchor":
        for si in take["closeups"]:
            shot_anchor(job, take, si, a.seed)
    elif a.stage == "shot-refs":
        for si in take["closeups"]:
            shot_refs(job, si, a.seed)
    elif a.stage == "render":
        render(job, take, a.seed, [int(x) for x in a.shots.split(",")] if a.shots else None)
    elif a.stage == "plate":
        plates(job, [int(x) for x in a.shots.split(",")] if a.shots else None)
    elif a.stage == "recomp":
        recomp(job, [int(x) for x in a.shots.split(",")] if a.shots else None)
    elif a.stage == "assemble":
        assemble(job, take)
