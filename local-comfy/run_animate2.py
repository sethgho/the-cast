#!/usr/bin/env python3
"""Wan Animate 2 motion transfer via local ComfyUI API.

Reference image (who) + driving video (what motion) -> character performing it.
Settings mirror the official video_wan_animate2 template: lcm/simple, 6 steps,
cfg 1, shift 5. Uses the distill int8_convrot build (no lightx2v LoRA needed);
pass --lora to add it if running the non-distill build.
"""
import argparse, json, sys, time, urllib.request

API = "http://127.0.0.1:8188"

NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
       "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
       "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def build(a):
    g = {}
    g["1"] = {"class_type": "LoadImage", "inputs": {"image": a.ref}}
    g["2"] = {"class_type": "LoadVideo", "inputs": {"file": a.drive}}
    g["3"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["2", 0]}}
    g["4"] = {"class_type": "UNETLoader", "inputs": {
        "unet_name": a.model, "weight_dtype": "default"}}
    model = ["4", 0]
    if a.lora:
        g["24"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": a.lora, "strength_model": 1.0}}
        model = ["24", 0]
    g["5"] = {"class_type": "ModelSamplingSD3", "inputs": {
        "model": model, "shift": 5.0}}
    g["12"] = {"class_type": "WanAnimate2Cache", "inputs": {
        "model": ["5", 0], "device": a.cache, "dtype": "int8"}}
    g["6"] = {"class_type": "CLIPLoader", "inputs": {
        "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "type": "wan", "device": "default"}}
    g["7"] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["6", 0], "text": a.prompt}}
    g["8"] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["6", 0], "text": NEG}}
    g["9"] = {"class_type": "CLIPTextEncode", "inputs": {
        "clip": ["6", 0], "text": a.pose_prompt}}
    g["10"] = {"class_type": "CLIPVisionLoader", "inputs": {
        "clip_name": "clip_vision_h.safetensors"}}
    g["11"] = {"class_type": "CLIPVisionEncode", "inputs": {
        "clip_vision": ["10", 0], "image": ["1", 0], "crop": "none"}}
    g["13"] = {"class_type": "WanAnimate2ToVideo", "inputs": {
        "positive": ["7", 0], "negative": ["8", 0], "vae": ["14", 0],
        "width": a.width, "height": a.height, "length": a.length,
        "batch_size": 1,
        "reference_image": ["1", 0], "pose_video": ["3", 0],
        "clip_vision_output": ["11", 0], "positive_pose": ["9", 0],
        "video_frame_offset": 0,
        "pose_strength": a.pose_strength,
        "pose_start_percent": 0.0, "pose_end_percent": 1.0,
        "reference_image_strength": a.ref_strength}}
    g["14"] = {"class_type": "VAELoader", "inputs": {
        "vae_name": "Wan2_1_VAE_bf16.safetensors"}}
    g["16"] = {"class_type": "BasicScheduler", "inputs": {
        "model": ["12", 0], "scheduler": "simple", "steps": a.steps,
        "denoise": 1.0}}
    g["17"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}}
    g["15"] = {"class_type": "SamplerCustom", "inputs": {
        "model": ["12", 0], "add_noise": True, "noise_seed": a.seed,
        "cfg": 1.0, "positive": ["13", 0], "negative": ["13", 1],
        "sampler": ["17", 0], "sigmas": ["16", 0],
        "latent_image": ["13", 2]}}
    g["18"] = {"class_type": "TrimVideoLatent", "inputs": {
        "samples": ["15", 0], "trim_amount": ["13", 3]}}
    g["19"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["18", 0], "vae": ["14", 0]}}
    g["20"] = {"class_type": "CreateVideo", "inputs": {
        "images": ["19", 0], "fps": a.fps}}
    g["21"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["20", 0], "filename_prefix": a.out,
        "format": "mp4", "codec": "h264"}}
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--drive", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--pose-prompt", required=True)
    p.add_argument("--out", default="animate2/a2")
    p.add_argument("--model",
                   default="wan_animate_2_distill_int8_convrot.safetensors")
    p.add_argument("--lora", default=None)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=832)
    p.add_argument("--length", type=int, default=69)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--cache", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--pose-strength", type=float, default=1.0)
    p.add_argument("--ref-strength", type=float, default=1.0)
    p.add_argument("--timeout", type=int, default=5400)
    a = p.parse_args()

    urllib.request.urlopen(API + "/free",
        json.dumps({"unload_models": True, "free_memory": True}).encode())

    req = urllib.request.Request(API + "/prompt",
        json.dumps({"prompt": build(a)}).encode(),
        {"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        print(e.read().decode(), file=sys.stderr)
        sys.exit(1)
    print("prompt_id", pid, flush=True)

    t0 = time.time()
    while time.time() - t0 < a.timeout:
        time.sleep(15)
        h = json.load(urllib.request.urlopen(API + "/history/" + pid))
        if pid in h:
            st = h[pid]["status"]
            if st.get("completed"):
                for o in h[pid]["outputs"].values():
                    for v in o.get("images", []) + o.get("video", []):
                        print("OUTPUT", v["subfolder"] + "/" + v["filename"])
                print("elapsed %.0fs" % (time.time() - t0))
                return
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        print("ERROR", m[1].get("node_type"),
                              m[1].get("exception_message"), file=sys.stderr)
                sys.exit(1)
    print("TIMEOUT", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
