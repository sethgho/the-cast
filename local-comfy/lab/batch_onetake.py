"""One-take VFX presets on our recast clips, via lab_run-style queueing (run.py)."""
import json, os, subprocess, sys
JOBS = [  # clip, frames(8n+1), w, h, preset, who
    ("rule.mp4", 97, 768, 448, "hologram", "the cartoon news anchor"),
    ("rule.mp4", 97, 768, 448, "disintegrate", "the cartoon news anchor"),
    ("thump.mp4", 73, 768, 320, "portal", "the cartoon man"),
    ("thump.mp4", 73, 768, 320, "time-freeze", "the cartoon man"),
    ("worthy.mp4", 33, 768, 320, "slow-mo-debris", "the two cartoon characters"),
    ("hangin.mp4", 89, 448, 768, "levitate", "the cartoon man"),
    ("thump.mp4", 73, 768, 320, "rain-stop", "the cartoon man"),
    ("rule.mp4", 97, 768, 448, "aurora-trails", "the cartoon news anchor"),
    ("rule.mp4", 97, 768, 448, "liquid-chrome", "the cartoon news anchor"),
    ("thump.mp4", 73, 768, 320, "neon-wireframe", "the cartoon man"),
    ("thump.mp4", 73, 768, 320, "particle-bloom", "the cartoon man"),
]
import urllib.request
def free_lab():
    """Unload the previous job's models: Canny runs on the GPU before LTX loads and OOMs on leftovers."""
    urllib.request.urlopen(urllib.request.Request(os.environ.get("COMFY_HOST", "http://192.168.0.210:8188") + "/free",
        json.dumps({"unload_models": True, "free_memory": True}).encode(), {"Content-Type": "application/json"}), timeout=60)
only = sys.argv[1:]
for clip, n, w, h, preset, who in JOBS:
    if only and preset not in only:
        continue
    p = json.load(open(f"presets/{preset}.json"))
    pos = p["prompt"].replace("The person's", who[0].upper() + who[1:] + "'s").replace("the person's", who + "'s") \
        .replace("The person", who[0].upper() + who[1:]).replace("the person", who)
    neg = p["negative_prompt"].replace(", cartoon", "")
    s = p["control_strength"]
    args = [f"load_video.video={json.dumps(clip)}", f"load_video.frame_load_cap={n}", f"fit.width={w}", f"fit.height={h}",
            f"control_small.width={w // 2}", f"control_small.height={h // 2}", f"empty_latent.width={w}",
            f"empty_latent.height={h}", f"empty_latent.length={n}", f"audio_latent.frames_number={n}",
            f"ic_lora.strength_model={s}", f"guide.strength={s}",
            f"positive.text={json.dumps(pos)}", f"negative.text={json.dumps(neg)}"]
    gm = p.get("guide_motion")
    if gm:
        args += [f"guide_motion.dy={gm.get('dy', 0)}", f"guide_motion.dx={gm.get('dx', 0)}",
                 f"guide_motion.fade_below={gm.get('fade_below', 0)}",
                 f"guide_motion.region_x0={gm.get('region', [0, 1])[0]}", f"guide_motion.region_x1={gm.get('region', [0, 1])[1]}"]
    if p.get("anchor_strength") is not None:
        args.append(f"anchor.strength={p['anchor_strength']}")
    out = f"fx-{clip.split('.')[0]}-{preset}.mp4"
    free_lab()
    r = subprocess.run([sys.executable, "lab_run.py", "onetake_vfx_core_api.json", out, *args], capture_output=True, text=True)
    print(preset, clip, (r.stdout + r.stderr).strip().splitlines()[0][:160] if (r.stdout + r.stderr).strip() else "?", flush=True)
