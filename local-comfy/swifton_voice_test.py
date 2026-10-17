#!/usr/bin/env python3
"""Swifton's voice, heard: three H3 image-to-video clips of his sample lines.

    python3 swifton_voice_test.py

Departs from a 16:9 portrait (dialogue is only intelligible at 16:9 on H3, measured
2026-08-28), puts each line in a <d> tag, and carries the h3_voice_clause from
cast/swifton/voice.yaml verbatim. Turbo LoRA at 8 steps, EasyCache off: EasyCache melts
faces on characters.
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S
from expression_run import upload

import yaml

OUT = "/home/wilson/scratch/swifton/voice"
FRAME = "/home/wilson/scratch/swifton/frame169.png"
W, H, FRAMES, STEPS = 1152, 640, 124, 8
VOICE = yaml.safe_load(open(os.path.join(HERE, "..", "cast", "swifton", "voice.yaml")))["speech"]["h3_voice_clause"].strip()

LINES = {
    "pick-a-card": ("He leans toward the viewer with a sly grin and fans the fingers of his free hand as if offering a deck of cards",
                    "Pick a card, friend. Any card. They're all yours anyway."),
    "whitepaper": ("He narrows one eye, taps the side of his beak knowingly, then points straight at the viewer",
                   "That's not a whitepaper. That's a tell."),
    "reputation": ("He holds the burning W coin up beside his face, watches it flare, then closes his talon around it with a satisfied nod",
                   "Watch closely. Now it's ash. Reputation matters."),
}


def prompt(action, line):
    return (f"A 1930s rubber-hose cartoon on aged sepia paper, a fixed camera on a vaudeville stage. "
            f"The griffon-phoenix card sharp (S1) with the orange flame crest performs continuously for the whole "
            f"shot and never holds still: {action}, his beak moving clearly in time with every word. "
            f"The griffon (S1) says: <d>[English] {line}</d> "
            f"Overall soundscape: a quiet empty theatre, the faint crackle of the flaming coin. "
            f"Non-diegetic music: none. {VOICE}")


def graph(image_name, text):
    return {
        "6":  {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "7":  {"class_type": "MiniMaxH3TurboLoRA", "inputs": {"model": ["6", 0], "lora_name": "minimax_h3_turbo_4step.safetensors", "strength": 1.0, "low_vram": True}},
        "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "24": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "30": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "31": {"class_type": "ImageScale", "inputs": {"image": ["30", 0], "upscale_method": "lanczos", "width": W, "height": H, "crop": "center"}},
        "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["13", 0], "vae": ["11", 0], "prompt": text,
                                                                 "width": W, "height": H, "length": FRAMES, "first_frame": ["31", 0]}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": 7}},
        "16": {"class_type": "BasicGuider", "inputs": {"model": ["7", 0], "conditioning": ["104", 0]}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9":  {"class_type": "BasicScheduler", "inputs": {"model": ["7", 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0], "sigmas": ["9", 0], "latent_image": ["104", 1]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": 24}},
        "92": {"class_type": "SaveVideo", "inputs": {"video": ["91", 0], "filename_prefix": "video/swifton_voice", "format": "auto", "codec": "auto"}},
    }


def run(name, action, line, image_name):
    dest = f"{OUT}/{name}-raw.mp4"
    if os.path.exists(dest):
        print(name, "cached"); return
    t0 = time.time()
    r = S.api("/prompt", {"prompt": graph(image_name, prompt(action, line)), "client_id": "swifton-voice"})
    if "prompt_id" not in r:
        print(name, "REJECTED", json.dumps(r)[:600]); return
    pid = r["prompt_id"]
    while True:
        hist = S.api(f"/history/{pid}")
        if pid in hist: break
        time.sleep(10)
    if hist[pid]["status"].get("status_str") != "success":
        print(name, "FAILED", json.dumps(hist[pid]["status"])[:600]); return
    o = next(v for v in hist[pid]["outputs"].values() if v.get("images") or v.get("videos"))
    f = (o.get("videos") or o.get("images"))[0]
    with urllib.request.urlopen(S.HOST + f"/view?filename={f['filename']}&subfolder={f.get('subfolder','')}&type=output", timeout=600) as rr:
        open(dest, "wb").write(rr.read())
    print(f"{name} -> {dest} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    image_name = upload(FRAME)
    for name, (action, line) in LINES.items():
        run(name, action, line, image_name)
