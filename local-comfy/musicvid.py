#!/usr/bin/env python3
"""The cast music video for "Everything's Fine" (MiniMax Music 3), rendered on gpu-worker.

    python3 musicvid.py still <shot> [--seed N]     Qwen + Follies: the cast member, posed for the shot
    python3 musicvid.py sing <shot> [--seed N]      LTX-2.5 image + audio -> video, lips locked to the song

The August attempt stalled on lip sync (InfiniteTalk at cfg 1 on 12 GB was soft). LTX-2.5 generates
video and audio jointly; here the song's own audio is encoded, frozen (LTXVFreezeLatent) and concatenated,
so the video is sampled against it and the mouth follows the singing. The full mix goes back on in the edit.
Every image step is a ComfyUI node; this file only queues graphs and cuts audio.
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sethmoji as M  # noqa: E402
import sethmoji_recast as R  # noqa: E402

ROOT = os.path.expanduser("~/scratch/musicvid")
SONG = f"{ROOT}/song.mp3"
VOCALS = f"{ROOT}/stems/vocals.wav"
W, H = 640, 352          # LTX first pass; the x2 latent upscaler makes 1280x704
FPS = 24
STAGE = ("on a 1930s vaudeville stage with a red velvet curtain, warm footlights from below and a single "
         "spotlight, drawn in warm sepia ink with soft halftone shading")
SHOTS = {
    # id: (cast member, start s, seconds, still pose, motion prompt)
    "seth-v1a": ("seth", 10.0, 5.0,
                 "head and shoulders, his face large in the frame, singing with his mouth open, one hand raised, "
                 "his moustache swept up clear of his lips",
                 "The cartoon man sings every word to the camera like a vaudeville crooner. His mouth opens wide "
                 "below his moustache on each sung word, showing his teeth and tongue, and closes between words. "
                 "His eyebrows lift with the melody and one hand gestures with the music. Static camera."),
}


def tag(shot):
    return f"musicvid-{shot}"


def still(shot, seed):
    who, _, _, pose, _ = SHOTS[shot]
    prompt = (f"f0llie5, image 1 is the reference drawing of this character. Draw the same character {STAGE}, "
              f"{pose}. Keep his face, hair, moustache and clothes exactly as in image 1. Landscape framing.")
    g = {
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
        "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "follies-final.safetensors", "strength_model": 1.2, "model": ["3", 0]}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "6": {"class_type": "LoadImage", "inputs": {"image": f"cast-{who}-plate.png"}},
        "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W * 2, "height": H * 2, "batch_size": 1}},
        "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0]}},
        "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": M.NEGATIVE, "clip": ["4", 0]}},
        "22": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": M.STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["30", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
        "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
        "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"{tag(shot)}/still{seed}", "images": ["23", 0]}},
    }
    os.makedirs(f"{ROOT}/{shot}", exist_ok=True)
    M.submit(g, "99", "images", f"{ROOT}/{shot}/still{seed}.png")
    print(shot, "still", seed)


def free():
    M.urllib.request.urlopen(M.urllib.request.Request(  # /free has an empty body; S.api parses JSON
        M.S.HOST + "/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
        {"Content-Type": "application/json"}), timeout=60).read()


def graph_file(g, shot):
    path = f"{ROOT}/{shot}/graph.api.json"
    json.dump(g, open(path, "w"), indent=1)
    return path


def up(local, name):
    M.sh("scp", "-q", local, f"{R.GPU}:comfyui/input/{name}")
    return name


def sing(shot, seed, still_seed, audio_src):
    who, start, secs, _, motion = SHOTS[shot]
    n = int(secs * FPS) // 8 * 8 + 1
    d = f"{ROOT}/{shot}"
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", f"{n / FPS:.3f}", "-i", audio_src,
         "-ac", "2", "-ar", "48000", f"{d}/drive.wav")
    img = up(f"{d}/still{still_seed}.png", f"{tag(shot)}-still.png")
    aud = up(f"{d}/drive.wav", f"{tag(shot)}-drive.wav")
    g = json.load(open(f"{HERE}/lab/ltx25_i2v_audio_api.json"))
    g["pos"]["inputs"]["text"] = (f"A 1930s rubber-hose cartoon man drawn in warm sepia ink, {STAGE}. {motion} "
                                  "He sings along to the music.")
    g["latV"]["inputs"].update(width=W, height=H, length=n)
    g["startLoad"]["inputs"]["image"] = img
    g["noise1"]["inputs"]["noise_seed"] = seed
    g["noise2"]["inputs"]["noise_seed"] = seed + 1
    # the song drives the picture: encode it, freeze it, and sample the video against it
    del g["latA"]
    g["aload"] = {"class_type": "LoadAudio", "inputs": {"audio": aud}}
    g["aenc"] = {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["aload", 0], "audio_vae": ["avae", 0]}}
    g["afrz"] = {"class_type": "LTXVFreezeLatent", "inputs": {"latent": ["aenc", 0]}}
    g["cat1"]["inputs"]["audio_latent"] = ["afrz", 0]
    g["cat2"]["inputs"]["audio_latent"] = ["afrz", 0]
    g["vid"]["inputs"]["fps"] = FPS
    g["save"]["inputs"]["filename_prefix"] = f"{tag(shot)}/sing"
    free()
    t0 = time.time()
    M.sh(sys.executable, f"{HERE}/lab/lab_run.py", graph_file(g, shot), f"{d}/sing{seed}.mp4")
    print(shot, "sing", seed, f"{n} frames", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("shot")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--still", type=int, default=7, help="which still seed to animate")
    p.add_argument("--audio", default="vocals", choices=["vocals", "mix"])
    a = p.parse_args()
    if a.stage == "still":
        still(a.shot, a.seed)
    elif a.stage == "sing":
        sing(a.shot, a.seed, a.still, VOCALS if a.audio == "vocals" else SONG)
