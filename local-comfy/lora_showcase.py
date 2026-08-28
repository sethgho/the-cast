"""Three unrelated ways to drive STUDIO 1939, same beat in each.

    python3 lora_showcase.py <a|b|c>

The beat: a moustached, long-haired man pops out of a hole in a Hill Country garden and says
"Whuffie, it's the one with the guarantee!". Nothing about the cast is imposed -- no plate, no
sepia, no rubber-hose lock. Each variant changes ONE axis so the comparison means something:

  a  card-literal   1.0, 16:9, exhaustive art direction, ends with what must not appear
  b  blended        0.5, 16:9, terse action-first prompt -- the card's "under a modern art
                    direction" advice, and a test of whether it needs the long prompt at all
  c  dialogue-first 1.0, 4:3, medium shot, the line is the subject and the scene is one clause

EasyCache is off everywhere: with it on it reused 10 of 20 steps and every character melted.
"""
import json, sys, time, os, subprocess
sys.path.insert(0, '/home/wilson/dev/the-cast/local-comfy')
import smoke_test as S
from build_transition import H3_UNET, H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE

LINE = "Whuffie, it's the one with the guarantee!"
V = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1939

VARIANTS = {
 # (width, height, lora strength, prompt)
 "a": (1280, 720, 1.0,
   "gulliv3r, A golden-age hand-painted animated feature. A lush Central Texas Hill Country "
   "garden at golden hour: silver-green live oaks with twisting limbs, tall spikes of purple "
   "salvia and red autumn sage, pale limestone boulders, prickly pear pads, bluebonnets, and dry "
   "grass the colour of straw. A round patch of dark crumbly soil sits in the middle of the "
   "flowerbed. A cheerful white man with long wavy chestnut hair to his shoulders and a huge "
   "bushy brown handlebar moustache bursts up out of the hole in a spray of dirt clods, blinks, "
   "grins broadly, leans toward the camera and says out loud, clearly and brightly: "
   f"\"{LINE}\" His mouth moves in sync with the words. Rich saturated inks, soft airbrushed cel "
   "shading, deep painted background, warm amber light, a locked-off camera. Audio: his clear "
   "cheerful speaking voice saying the line, a soft puff of earth, and quiet birdsong. "
   "Not photographic, not 3D, not modern digital vector art, no music, no text, no subtitles, "
   "no watermark."),

 "b": (1280, 720, 0.5,
   f"gulliv3r, A man with long hair and a big moustache pops up out of a hole in a Texas garden "
   f"and says \"{LINE}\" Cinematic, shallow depth of field, warm evening light."),

 "c": (1024, 768, 1.0,
   "gulliv3r, Medium shot, a man's head and shoulders fill the frame. A white man with long wavy "
   "chestnut hair and a huge bushy brown moustache rises up out of a hole in the ground, dirt "
   "falling from his shoulders, looks straight into the camera and speaks the line aloud with "
   f"clear, well-articulated lip sync: \"{LINE}\" Behind him, out of focus, a lush Central Texas "
   "Hill Country garden of live oaks, purple salvia and limestone. Golden-age hand-painted "
   "animation, saturated inks, soft cel shading. Audio: only his voice, close and clear, plus "
   "faint birdsong. No music, no narrator, no text."),

 # d/e: variant c's framing and strength, but the line handled the way a and b handled it --
 # dropped in quotes with no lip-sync coaching and no "Audio:" section. c added both and lost the
 # words entirely, so this isolates that. e repeats d at 16:9 to settle whether 4:3 mattered.
 "d": (1024, 768, 1.0,
   "gulliv3r, Medium shot, a man's head and shoulders fill the frame. A white man with long wavy "
   "chestnut hair and a huge bushy brown moustache bursts up out of a hole in the ground in a "
   "spray of dirt, looks straight into the camera, grins and says: "
   f"\"{LINE}\" Behind him, out of focus, a lush Central Texas Hill Country garden of live oaks, "
   "purple salvia and limestone. Golden-age hand-painted animation, saturated inks, soft cel "
   "shading, warm evening light."),

 # f: the failing case (d, at 4:3) rewritten in H3's OWN documented format. The official base
 # prompt guide says dialogue belongs in a <d> tag with a language marker and a speaker id --
 # "Preserve every original word and punctuation mark verbatim" -- not in quotation marks, which
 # is what every render above used. If 4:3 speaks now, the aspect ratio was never the cause.
 "f": (1024, 768, 1.0,
   "gulliv3r, Medium shot, a man's head and shoulders fill the frame. A white man with long wavy "
   "chestnut hair and a huge bushy brown moustache bursts up out of a hole in the ground in a "
   "spray of dirt, looks straight into the camera and grins. The camera is locked off. Behind "
   "him, out of focus, a lush Central Texas Hill Country garden of live oaks, purple salvia and "
   "limestone. Golden-age hand-painted animation, saturated inks, soft cel shading, warm evening "
   f"light. The man (S1) says: <d>[English] {LINE}</d> "
   "Overall soundscape: a soft puff of earth and faint birdsong. Non-diegetic music: none."),
}
VARIANTS["e"] = (1280, 720) + VARIANTS["d"][2:]
VARIANTS["g"] = (1280, 720) + VARIANTS["f"][2:]  # same, at the aspect that already worked
# h: g's prompt, Spectromachina's sampling recipe. Trades pixels for seconds -- 0.74MP buys 7.3s
# where 0.92MP capped us at 5.2s -- and replaces EasyCache with the turbo LoRA. Turbo DISTILS
# steps; EasyCache REUSED them, which is what melted every character earlier today.
VARIANTS["h"] = (1152, 640) + VARIANTS["f"][2:]
TURBO = V == "h"


W, H, STRENGTH, PROMPT = VARIANTS[V]
LENGTH, STEPS, FPS = (175, 12, 24) if V == "h" else (124, 20, 24)

g = {
 "6": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}},
 **({"turbo": {"class_type": "MiniMaxH3TurboLoRA", "inputs": {"model": ["6", 0], "lora_name": "minimax_h3_turbo_4step.safetensors", "strength": 1.0, "low_vram": True}}} if TURBO else {}),
 "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["turbo" if TURBO else "6", 0], "lora_name": "studio1939-strong.safetensors", "strength_model": STRENGTH}},
 "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}},
 "11": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
 "24": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
 "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["13", 0], "vae": ["11", 0], "prompt": PROMPT,
         "width": W, "height": H, "length": LENGTH}},
 "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
 "16": {"class_type": "BasicGuider", "inputs": {"model": ["lora", 0], "conditioning": ["104", 0]}},
 "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
 "9": {"class_type": "BasicScheduler", "inputs": {"model": ["lora", 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}},
 "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0], "sigmas": ["9", 0], "latent_image": ["104", 1]}},
 "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
 "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
 "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": FPS, "bit_depth": 8}},
 "92": {"class_type": "SaveVideo", "inputs": {"video": ["91", 0], "filename_prefix": f"video/whuffie-{V}-{SEED}", "format": "auto", "codec": "auto"}, "_meta": {"title": "RESULT"}},
}
for n, d in g.items(): d.setdefault("_meta", {"title": n})
t0 = time.time()
r = S.api("/prompt", {"prompt": g, "client_id": "whuffie"})
if "prompt_id" not in r:
    print("REJECTED", json.dumps(r)[:1000]); sys.exit(1)
pid = r["prompt_id"]
while True:
    h = S.api(f"/history/{pid}")
    if pid in h: break
    time.sleep(10)
st = h[pid]['status']
if st.get('status_str') != 'success':
    print("FAILED", json.dumps(st)[:1000]); sys.exit(1)
o = h[pid]['outputs']['92']['images'][0]
raw = f"/tmp/whuffie-{V}.mp4"
subprocess.run(["curl", "-s", f"{S.HOST}/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output", "-o", raw], check=True)
print(f"{V}: {time.time()-t0:.0f}s -> {raw}")
