#!/usr/bin/env python3
"""Turn a keyframe pair into a seamless avatar-sized hover loop.

    python3 hover_loop.py <frameA.png> <frameB.png> <name> "<motion sentence>" [seed]

Frame A and frame B come from the `hover-keyframes` app. This renders the motion
between them with MiniMax H3 on gpu-worker (first_frame = A, last_frame = B), then
mirrors the clip so it plays A->B->A and cuts seamlessly on repeat.

Measured 2026-08-17: 480x480, 10 steps, 61 frames = **50s per clip**, ~400KB of mp4.
Avatar work does not need more; 1024 would OOM the 12GB card at this length.

Outputs, into --outdir (default /home/wilson/artifacts/cast-local-comfy):
    <name>-loop.mp4   320x320, muted, palindrome — what a web page embeds
    <name>-loop.webm  the same, VP9
The unmirrored clip with H3's own audio stays in ComfyUI's output/video/.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S  # noqa: E402

SIZE, STEPS, LENGTH, AVATAR = 480, 10, 61, 320

# H3 is a joint video+audio model: a prompt with no sound clause leaves the audio head
# unsupervised and it invents garbled human vocalisations. The loops are muted for the
# web, but the clause still has to be there or the visuals degrade too.
SOUND = (
    " SOUND: a soft creak of wood and a faint rustle of paper, quiet and close. "
    "No voices, no speech, no singing, no humming, no breathing, no human vocal sounds of any "
    "kind. No text, no lettering, no captions, no subtitles, no title cards."
)
HOLD = " The drawing style, the paper, the framing and the flat background never change."


def upload(path):
    sys.path.insert(0, "/home/wilson/scratch/local-cast")
    import run as R
    return R.upload(path)


def prep(src, dst, bg=(196, 190, 170)):
    """Flatten any alpha onto the plate tone and square it to SIZE."""
    from PIL import Image
    im = Image.open(src)
    if im.mode == "RGBA":
        flat = Image.new("RGB", im.size, bg)
        flat.paste(im, (0, 0), im.getchannel("A"))
        im = flat
    im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(dst)
    return dst


def graph(a_name, b_name, prompt, prefix, seed):
    return {
        "6": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "50": {"class_type": "EasyCache", "inputs": {
            "model": ["6", 0], "reuse_threshold": 0.2, "start_percent": 0.15, "end_percent": 0.95,
            "verbose": True}},
        "13": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax",
            "device": "default"}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "24": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "30": {"class_type": "LoadImage", "inputs": {"image": a_name}},
        "31": {"class_type": "ImageScale", "inputs": {
            "image": ["30", 0], "upscale_method": "lanczos", "width": SIZE, "height": SIZE,
            "crop": "disabled"}},
        "32": {"class_type": "LoadImage", "inputs": {"image": b_name}},
        "33": {"class_type": "ImageScale", "inputs": {
            "image": ["32", 0], "upscale_method": "lanczos", "width": SIZE, "height": SIZE,
            "crop": "disabled"}},
        "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["13", 0], "vae": ["11", 0], "prompt": prompt, "width": SIZE, "height": SIZE,
            "length": LENGTH, "first_frame": ["31", 0], "last_frame": ["33", 0]}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "16": {"class_type": "BasicGuider", "inputs": {"model": ["50", 0], "conditioning": ["104", 0]}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "BasicScheduler", "inputs": {
            "model": ["50", 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0], "sigmas": ["9", 0],
            "latent_image": ["104", 1]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo", "inputs": {
            "images": ["10", 0], "audio": ["23", 0], "fps": 24, "bit_depth": 8}},
        "92": {"class_type": "SaveVideo", "inputs": {
            "video": ["91", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"},
            "_meta": {"title": "RESULT"}},
    }


def render(frame_a, frame_b, name, motion, seed=11, outdir="/home/wilson/artifacts/cast-local-comfy"):
    a = upload(prep(frame_a, f"/tmp/{name}-a.png"))
    b = upload(prep(frame_b, f"/tmp/{name}-b.png"))
    wf = graph(a, b, motion + HOLD + SOUND, f"video/{name}", seed)
    for node in wf.values():
        node.setdefault("_meta", {"title": "node"})
    t0 = time.time()
    pid = S.api("/prompt", {"prompt": wf, "client_id": "hover-loop"})["prompt_id"]
    while True:
        hist = S.api(f"/history/{pid}")
        if pid in hist:
            break
        time.sleep(10)
    status = hist[pid]["status"]
    if status.get("status_str") != "success":
        raise SystemExit(f"{name}: FAILED — {json.dumps(status)[:400]}")
    out = hist[pid]["outputs"]["92"]["images"][0]
    raw = f"/tmp/{name}-raw.mp4"
    subprocess.run([
        "curl", "-s",
        f"{S.HOST}/view?filename={out['filename']}&subfolder={out.get('subfolder','')}&type=output",
        "-o", raw], check=True)

    os.makedirs(outdir, exist_ok=True)
    mp4 = os.path.join(outdir, f"{name}-loop.mp4")
    webm = os.path.join(outdir, f"{name}-loop.webm")
    # Mirror the clip: A->B then B->A, so the loop point is a real frame match, not a cut.
    subprocess.run([
        "ffmpeg", "-v", "error", "-i", raw, "-filter_complex",
        f"[0:v]scale={AVATAR}:{AVATAR},split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-y", mp4], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-c:v", "libvpx-vp9", "-b:v", "0",
                    "-crf", "34", "-an", "-y", webm], check=True)
    print(f"{name}: {time.time() - t0:.0f}s -> {mp4}")
    return mp4


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
           int(sys.argv[5]) if len(sys.argv) > 5 else 11)
