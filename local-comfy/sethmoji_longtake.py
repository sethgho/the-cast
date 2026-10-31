#!/usr/bin/env python3
"""Long-take recast with the original audio: cartoon Seth through a whole edited clip, shot by shot.

    python3 sethmoji_longtake.py <stage> <name>     stages: conform anchor render assemble

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


def seg_graph(j, si, ki, L, prev_tail, init_name, prompt, seed):
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
    ref0 = [g.add("LoadImage", image=j.inp("ref0.png")), 0]
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
                    ref_track_data=[g.add("LoadImageMask", image=j.inp("ref-mask0.png"), channel="red"), 0])
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
            g, outs = seg_graph(j, si, ki, L, prev_tail, init, R_prompt(t), seed)
            out = j.p(f"out/s{si}k{ki}")
            shutil.rmtree(out, ignore_errors=True)
            secs = R.run(g, outs, out)
            print(j.name, f"shot {si} seg {ki}", f"{L} frames", f"{secs:.0f}s", flush=True)


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
    elif a.stage == "render":
        render(job, take, a.seed, [int(x) for x in a.shots.split(",")] if a.shots else None)
    elif a.stage == "assemble":
        assemble(job, take)
