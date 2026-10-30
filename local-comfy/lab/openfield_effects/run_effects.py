"""Run OpenField effect graphs on main ComfyUI, patched for 12 GB and ComfyUI 0.43.2 core nodes."""
import json, subprocess, sys, urllib.request
HOST = "http://192.168.0.210:8188"
VIDEO_W, VIDEO_H, VIDEO_N = 768, 448, 73
JOBS = {  # effect: (input file, kind, extra prompt prefix)
    "flood": ("thump.mp4", "video"), "nightfall": ("thump.mp4", "video"), "empty": ("rule.mp4", "video"),
    "cutout": ("rule.mp4", "video"),
    "living-photo": ("thump-ref0.png", "image"), "pop": ("rule-ref0.png", "image"), "inflate": ("thump-ref0.png", "image"),
    "cake": ("thump-ref0.png", "image"), "squish": ("rule-ref0.png", "image"), "unzip": ("rule-ref0.png", "image"),
    "gulp": ("rule-ref0.png", "image"),
}
RETRY = {"cake": ("wilson-wide.png", "image"), "squish": ("wilson-wide.png", "image"), "flood": ("worthy.mp4", "video")}
if "--retry" in sys.argv:
    sys.argv.remove("--retry"); JOBS.update(RETRY); SUFFIX = "-retry"
else:
    SUFFIX = ""
def patch(e):
    g = json.load(open(f"effect-{e}.json"))
    clip, kind = JOBS[e]
    for k, v in list(g.items()):
        c, i = v["class_type"], v["inputs"]
        if c == "LTXICLoRALoaderModelOnly":  # 0.43.2 core equivalent
            v["class_type"] = "LoraLoaderModelOnly"
            g[k + "_p"] = {"class_type": "GetICLoRAParameters", "inputs": {"iclora_model": [k, 0]}}
        if c == "VAEDecodeTiled":
            i.update(tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
        if c == "CLIPTextEncode" and "pc game" in i.get("text", ""):
            i["text"] = i["text"].replace(", cartoon", "")
        if c == "LoadImage":
            i["image"] = clip
        if c == "VHS_LoadVideo":
            i.update(video=clip, custom_width=VIDEO_W, custom_height=VIDEO_H, frame_load_cap=VIDEO_N)
        if kind == "video" and c == "ImageScale":
            i.update(width=VIDEO_W, height=VIDEO_H)
        if kind == "video" and c == "EmptyLTXVLatentVideo":
            i.update(width=VIDEO_W, height=VIDEO_H, length=VIDEO_N)
        if kind == "video" and c == "LTXVEmptyLatentAudio":
            i.update(frames_number=VIDEO_N)
        if e == "living-photo" and c == "EmptyLTXVLatentVideo":
            i.update(width=768, height=448)
    for k, v in list(g.items()):
        if v["class_type"] == "LTXAddVideoICLoRAGuide":
            i = v["inputs"]
            icl = i.pop("latent_downscale_factor")[0]
            v["class_type"] = "LTXVAddGuide"
            v["inputs"] = {"positive": i["positive"], "negative": i["negative"], "vae": i["vae"], "latent": i["latent"],
                           "image": i["image"], "frame_idx": i.get("frame_idx", 0), "strength": i.get("strength", 1),
                           "iclora_parameters": [icl + "_p", 0]}
    return g
for e in sys.argv[1:]:
    g = patch(e)
    json.dump(g, open(f"patched-{e}.json", "w"), indent=1)
    urllib.request.urlopen(urllib.request.Request(HOST + "/free", json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                                  {"Content-Type": "application/json"}), timeout=60)
    r = subprocess.run([sys.executable, "../lab_run.py", f"patched-{e}.json", f"fx-{e}{SUFFIX}.mp4"], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    print(e, out[0][:120] if out else "?", "|", (out[-1][-300:] if r.returncode else "ok"), flush=True)
