#!/usr/bin/env python3
"""The cast music video for "Everything's Fine" (MiniMax Music 3), rendered on gpu-worker.

    python3 musicvid.py still <shot|all|a,b> [--seed N]     Qwen + Follies: the cast member, posed for the shot
    python3 musicvid.py sing <shot|all> [--seed N]     LTX-2.5 image + audio -> video, lips locked to the song

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
SING = ("sings every word to the camera like a vaudeville crooner. His mouth opens wide below his moustache on each "
        "sung word, showing his teeth and tongue, and closes between words.")
# LTX drifts toward a 3D render on big body moves (Seth's bow, 2026-10-31); every motion prompt restates the medium
LOOK = ("Flat 2D hand-drawn 1930s cartoon animation: black ink outlines, flat sepia colour and halftone shading on "
        "every frame, the drawing stays flat.")
# Coverage: no two cuts in a row share a framing (Seth: a jump cut needs a change of angle, perspective or crop).
# frame = wide (the whole still) | medium | close: cropped around the face before LTX animates it. LTX's first
# pass is 640x352, so a crop of the 1280x704 still loses nothing the render would have kept.
# who, start, end, still pose (with the camera angle), motion, audio, frame. Cuts sit on measured downbeats.
SHOTS = {
    "01-curtain": ("cadbury", 0.0, 4.06, "a wide shot of the whole stage seen from the back of the theatre, the "
                   "curtain closed, the butler stepping into the spotlight at centre stage",
                   "The butler strides into the spotlight and raises one gloved hand to quiet the crowd. Static camera.",
                   "mix", "wide"),
    "02-announce": ("cadbury", 4.06, 8.45, "seen from a low angle at the footlights, the butler with his chin up and "
                    "one gloved hand sweeping toward the curtain, mouth open mid-word",
                    "The butler announces the evening's performance in a grand voice, his mouth shaping each word, "
                    "and sweeps his gloved hand toward the curtain. Slow push in.", "vocals", "medium"),
    "03-friday": ("seth", 8.45, 12.98, "a wide shot, the curtain open, the man at centre stage in the spotlight with "
                  "both arms open, singing with his mouth open, his moustache swept up clear of his lips",
                  f"The cartoon man walks to the front of the stage and {SING} Static camera.", "vocals", "wide"),
    "04-green": ("seth", 12.98, 17.46, "seen in three-quarter view from his left, a row of little round green lights "
                 "glowing behind him, singing with his mouth open, his moustache swept up clear of his lips",
                 f"The cartoon man {SING} The green lights blink. Static camera.", "vocals", "close"),
    "05-small": ("seth", 17.46, 21.97, "seen in three-quarter view from his right, holding up a finger and thumb a "
                 "tiny distance apart beside his face, singing with his mouth open, his moustache swept up clear of "
                 "his lips", f"The cartoon man holds up a finger and thumb a tiny distance apart and {SING} "
                 "Slow push in.", "vocals", "medium"),
    "06-coffee": ("seth", 21.97, 26.42, "holding a steaming cup of coffee up near his face, singing with his mouth "
                  "open, his moustache swept up clear of his lips",
                  f"The cartoon man lifts his steaming coffee cup and {SING} Static camera.", "vocals", "close"),
    "07-proud": ("seth", 26.42, 30.81, "a wide shot seen from high up in the balcony, the man small on the big stage "
                 "with his chest puffed out and one thumb pointing at himself, singing",
                 f"The cartoon man puffs out his chest with pride and {SING} Slow push in.", "vocals", "wide"),
    "08-bow": ("seth", 30.81, 33.85, "seen from a low angle at the footlights, the man bending into a small bow with "
               "one arm across his waist, looking up at the audience and singing with his mouth open",
               "The cartoon man dips into a small bow and comes back up, singing to the crowd. Static camera.",
               "vocals", "medium"),
    "08b-crowd": ("crowd", 33.85, 35.39, "the theatre seats filled with a cheering 1930s audience of cartoon "
                  "theatre-goers in hats and bow ties, clapping and whistling, seen from the stage",
                  "The cartoon audience claps, cheers and throws their hats in the air. Static camera.", "mix", "wide"),
    "09-machine": ("seth", 35.39, 39.85, "facing the camera straight on, one hand on his heart and the other raised "
                   "as if swearing an oath, singing with his mouth open, his moustache swept up clear of his lips",
                   f"The cartoon man swears an oath with his hand raised and {SING} Static camera.", "vocals", "close"),
    "10-fine": ("seth", 39.85, 44.28, "a wide shot, the man at centre stage with arms spread wide, a tall wall of "
                "round green light bulbs in neat rows glowing behind him, singing with his mouth open",
                f"The cartoon man spreads his arms wide as the green lights blink behind him and {SING} "
                "Slow pull back.", "vocals", "wide"),
    "11-prettiest": ("seth", 44.28, 48.74, "seen in profile from the side, the man pointing proudly at the glowing "
                     "green lights beside him, singing with his mouth open",
                     f"The cartoon man points proudly at the green lights and {SING} Static camera.", "vocals",
                     "medium"),
    "12-curtain": ("seth", 48.74, 53.17, "standing in front of the closed curtain, wagging a finger beside his face "
                   "at the audience, singing with his mouth open, his moustache swept up clear of his lips",
                   f"The cartoon man wags a finger at the audience and {SING} Static camera.", "vocals", "close"),
    "13-smell": ("seth", 53.17, 57.61, "seen from a low angle, the man sniffing the air with his nose lifted and a "
                 "nervous grin, a thin grey wisp of smoke rising from the curtain behind his shoulder, singing",
                 f"Grey smoke curls up from the curtain behind the cartoon man as he sniffs the air nervously and "
                 f"{SING} Slow push in.", "vocals", "medium"),
    "14-finale": ("seth", 57.61, 62.04, "a wide shot of the whole stage seen from high up in the balcony, the man in "
                  "the spotlight with both arms thrown up in a big finish, singing with his mouth wide open",
                  f"The cartoon man throws both arms up for a big finish and {SING} Slow pull back.", "vocals",
                  "wide"),
}
WSING = ("sings along to the music: his hat and his eyes bob with every sung word and his white-gloved hands gesture "
         "with the melody.")
REACT = "He does not sing; his mouth stays shut."
SHOTS.update({
    "15-fella": ("wilson", 62.04, 66.456, "seen in three-quarter view, the fence leaning his elbows over a little "
                 "garden rail at the side of the stage, eyes wide and friendly",
                 f"The fence leans over the rail and {WSING} Static camera.", "vocals", "medium"),
    "16-son": ("wilson", 66.456, 70.891, "the fence leaning in toward the camera, wagging one gloved finger, his white "
               "eyes narrowed knowingly under the brim of his hat",
               f"The fence wags one gloved finger knowingly and {WSING} Slow push in.", "vocals", "close"),
    "17-1909": ("wilson", 70.891, 75.279, "a wide shot of the stage seen from the side, the fence at centre stage "
                "holding up a dusty yellowed old theatre playbill",
                f"The fence holds up the old playbill and taps it, and {WSING} Static camera.", "vocals", "wide"),
    "18-disk": ("wilson", 75.279, 79.691, "seen from a low angle, the fence counting off on his gloved fingers, "
                "eyes looking up under the brim of his hat",
                f"The fence counts off problems on his gloved fingers one by one and {WSING} Static camera.",
                "vocals", "medium"),
    "19-gulp": ("seth", 79.691, 84.079, "a nervous sweating grin, beads of sweat on his forehead, eyes darting "
                "sideways, his moustache swept up clear of his lips",
                f"The cartoon man listens nervously, sweat beading on his forehead, and gulps. {REACT} Static camera.",
                "mix", "close"),
    "20-stand": ("wilson", 84.079, 88.468, "the fence shrugging with both gloved hands open, palms up, his white eyes "
                 "kind under the brim of his hat",
                 f"The fence shrugs with both hands open and {WSING} Slow pull back.", "vocals", "medium"),
    "21-fine2": ("seth", 88.468, 92.88, "a wide shot seen from high up in the balcony, the man at centre stage with "
                 "arms spread wide in front of a tall wall of round green light bulbs, singing with his mouth open",
                 f"The cartoon man spreads his arms wide as the green lights blink and {SING} Slow push in.",
                 "vocals", "wide"),
    "22-deploy": ("seth", 92.88, 97.292, "seen in three-quarter view, singing with his mouth wide open and eyes "
                  "shut in bliss, his moustache swept up clear of his lips, a thin wisp of smoke behind him",
                  f"The cartoon man {SING} Static camera.", "vocals", "close"),
    "23-crowd2": ("crowd", 97.292, 101.68, "the theatre seats filled with a 1930s audience of cartoon theatre-goers "
                  "in hats waving handkerchiefs and swaying along, seen from the stage",
                  "The cartoon audience sways and waves their handkerchiefs to the music. Static camera.", "mix",
                  "wide"),
    "24-ake": ("ake", 101.68, 106.046, "the goldfish in his round bowl rolling onto the stage on his little wheeled "
               "cart, looking worried, bubbles rising",
               "The goldfish's bowl rolls slowly onto the stage on its little cart; he looks around worried and "
               "bubbles rise. Static camera.", "mix", "medium"),
    "25-ledger": ("cadbury", 106.046, 110.434, "the butler opening a huge leather-bound ledger, peering into it "
                  "through his monocle",
                  "The butler opens the heavy ledger and peers into it through his monocle. Slow push in.", "mix",
                  "close"),
    "26-butler": ("cadbury", 110.434, 114.823, "the butler stepping forward holding the open ledger in one gloved "
                  "hand, chin up, mouth open mid-word",
                  "The butler steps forward with the open ledger and sings the words gravely, his mouth shaping each "
                  "word. Static camera.", "vocals", "medium"),
    "27-column": ("cadbury", 114.823, 119.188, "the butler reading from the ledger, one eyebrow raised above his "
                  "monocle, mouth open mid-word",
                  "The butler reads from the ledger, raises an eyebrow and sings the words gravely, his mouth shaping "
                  "each word. Static camera.", "vocals", "close"),
    "28-red": ("cadbury", 119.188, 123.6, "a wide shot seen from a low angle at the footlights, the butler at centre "
               "stage holding the open ledger out toward the audience",
               "The butler holds the ledger out to the audience and sings the words gravely. Slow pull back.",
               "vocals", "wide"),
    "29-items": ("cadbury", 123.6, 127.942, "the butler ticking off lines in the ledger with a fountain pen, mouth "
                 "open mid-word", "The butler ticks off each line with his pen and sings the words gravely, his "
                 "mouth shaping each word. Static camera.", "vocals", "close"),
    "30-backup": ("seth", 127.942, 132.307, "seen from a low angle, the man frozen in horror with both hands on his "
                  "cheeks, eyes huge, mouth hanging open",
                  f"The cartoon man freezes in horror, both hands on his cheeks. {REACT} Slow push in.", "mix",
                  "medium"),
    "31-chalk": ("ake", 132.307, 136.719, "the goldfish's face pressed close to the glass of his bowl, eyes huge with "
                 "fright, his orange scales turning pale white",
                 "The goldfish's orange scales drain to chalk white as he stares in fright, trembling. Static camera.",
                 "mix", "close"),
    "32-squeak": ("ake", 136.719, 141.084, "a wide shot of the big empty stage, the tiny goldfish bowl on its cart "
                  "alone in the spotlight, the pale fish trembling",
                  "The tiny pale goldfish trembles alone in the spotlight, a single bubble rising. Slow pull back.",
                  "mix", "wide"),
    "33-told": ("wilson", 141.084, 145.473, "the fence holding up a lit red signal flare in one gloved hand, sparks "
                "flying, his white eyes wide under the brim",
                f"The fence waves the sparking red flare overhead and {WSING} Static camera.", "vocals", "medium"),
    "34-fire": ("seth", 145.473, 149.815, "a wide shot, the stage curtains behind him on fire with tall cartoon "
                "flames, the man at centre stage singing with arms wide",
                f"Cartoon flames leap up the curtains behind the cartoon man as he {SING} Static camera.", "vocals",
                "wide"),
    "35-calm": ("seth", 149.815, 154.18, "sipping calmly from a cup of coffee with cartoon flames all around him, "
                "singing with his mouth open, his moustache swept up clear of his lips",
                f"Cartoon flames dance all around the cartoon man as he calmly lifts his coffee and {SING} "
                "Static camera.", "vocals", "close"),
    "36-fine3": ("seth", 154.18, 158.569, "seen in three-quarter view, singing with his mouth open, smoke and embers "
                 "drifting past him, a burning curtain behind",
                 f"Smoke and embers drift past the cartoon man as he {SING} Slow push in.", "vocals", "medium"),
    "37-green3": ("seth", 158.569, 162.934, "a wide shot seen from high up in the balcony, the whole stage ablaze, a "
                  "wall of round green light bulbs glowing, the man in the spotlight with arms thrown wide",
                  f"The whole stage burns as the green lights blink and the cartoon man {SING} Slow pull back.",
                  "vocals", "wide"),
    "38-end": ("seth", 162.934, 165.0, "a big grin straight at the camera giving a thumbs up, a little soot on his "
               "cheek, his moustache swept up clear of his lips",
               "The cartoon man grins and gives a big thumbs up to the camera. Static camera.", "mix", "close"),
})
PLATES = {"seth": "cast-seth-plate.png", "cadbury": "cast-cadbury-plate.png", "wilson": "cast-wilson-plate.png",
          "ake": "cast-ake-plate.png",
          "crowd": "cast-seth-scene-empty-auditorium.png"}
WHO = {"seth": "A 1930s rubber-hose cartoon man", "cadbury": "A 1930s rubber-hose cartoon butler",
       "crowd": "A 1930s cartoon theatre audience",
       "wilson": "A 1930s cartoon living picket fence in a bucket hat, white oval eyes in the shadow under the brim, "
                 "white-gloved hands,",
       "ake": "A 1930s cartoon goldfish with a bushy moustache and buck teeth, in a round glass fishbowl on a little "
              "wheeled cart,"}
FIND = {"wilson": "hat", "ake": "fish"}  # what SAM looks for when a character has no human face
CROP = {"medium": 0.62, "close": 0.42}  # crop height as a share of the still


def tag(shot):
    return f"musicvid-{shot}"


def still(shot, seed):
    who, _, _, pose, *_ = SHOTS[shot]
    if who == "crowd":
        prompt = (f"f0llie5, image 1 is an empty theatre. Redraw it as {pose}, drawn in warm sepia ink with soft "
                  "halftone shading like a 1930s cartoon. Landscape framing.")
    else:
        keep = ("Keep his face, hair, moustache and clothes exactly as in image 1." if who == "seth"
                else "Keep the character exactly as drawn in image 1.")
        prompt = (f"f0llie5, image 1 is the reference drawing of this character. Draw the same character {STAGE}, "
                  f"{pose}. {keep} Landscape framing.")
    g = {
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
        "30": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "follies-final.safetensors", "strength_model": 1.2, "model": ["3", 0]}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "6": {"class_type": "LoadImage", "inputs": {"image": PLATES[who]}},
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


def assemble(upto):
    """Trim each shot to its downbeat length, concatenate, and lay the full song back under it."""
    names = [k for k in SHOTS if SHOTS[k][2] <= upto]
    lst = f"{ROOT}/cut.txt"
    with open(lst, "w") as f:
        for k in names:
            _, start, end, *_ = SHOTS[k]
            clip = f"{ROOT}/{k}/cut.mp4"
            # frame counts come from one absolute timeline: cutting each shot by seconds rounded every cut up a
            # frame or two, and the drift reached 8 frames (0.33 s) by shot 14, visibly off the lips
            frames = round(end * FPS) - round(start * FPS)
            M.sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{ROOT}/{k}/sing.mp4", "-frames:v", str(frames), "-an",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS), clip)
            f.write(f"file '{clip}'\n")
    end = SHOTS[names[-1]][2]
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-t", f"{end:.3f}", "-i", SONG,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
         "-movflags", "+faststart", f"{ROOT}/everything-is-fine-{int(end)}s.mp4")
    print("assembled", len(names), "shots", f"{end:.2f}s")


def frame(shot, still_seed):
    """Crop the still to the shot's framing around the face (SAM 3.1 finds it); wide keeps the whole still."""
    *_, framing = SHOTS[shot]
    d = f"{ROOT}/{shot}"
    if framing == "wide":
        shutil.copy(f"{d}/still{still_seed}.png", f"{d}/framed.png")
        print(shot, "frame wide")
        return
    img = up(f"{d}/still{still_seed}.png", f"{tag(shot)}-still.png")
    who = SHOTS[shot][0]
    for text in ("face", FIND.get(who, "person")):
        g, out = R.detect_graph(img, f"{tag(shot)}/face", text=text)
        M.submit(g, out, "images", f"{d}/face.png")
        m = R.np.asarray(R.Image.open(f"{d}/face.png").convert("L")) > 127
        if m.any():
            break
    Hs, Ws = m.shape
    if not m.any():  # nothing found: frame the centre
        m[Hs // 3:Hs // 2, Ws * 2 // 5:Ws * 3 // 5] = True
    ys, xs = R.np.nonzero(m)
    ch = int(Hs * CROP[framing])
    cw = int(ch * Ws / Hs)
    cx, top = (xs.min() + xs.max()) // 2, ys.min()
    x = int(min(max(cx - cw // 2, 0), Ws - cw))
    y = int(min(max(top - ch * 0.15, 0), Hs - ch))  # headroom above the hair
    g = R.Graph()
    crop = g.add("ImageCrop", image=[g.add("LoadImage", image=img), 0], width=cw, height=ch, x=x, y=y)
    fit = g.add("ImageScale", image=[crop, 0], upscale_method="lanczos", width=Ws, height=Hs, crop="disabled")
    M.submit({**g, "out": {"class_type": "SaveImage", "inputs": {"images": [fit, 0],
              "filename_prefix": f"{tag(shot)}/framed"}}}, "out", "images", f"{d}/framed.png")
    print(shot, "frame", framing, (x, y, cw, ch))


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
    who, start, end, _, motion, *_ = SHOTS[shot]
    n = int((end - start) * FPS + 7) // 8 * 8 + 1  # round UP to the 8n+1 grid; the edit trims to the downbeat
    d = f"{ROOT}/{shot}"
    M.sh("ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", f"{n / FPS:.3f}", "-i", audio_src,
         "-ac", "2", "-ar", "48000", f"{d}/drive.wav")
    img = up(f"{d}/framed.png", f"{tag(shot)}-framed.png")
    aud = up(f"{d}/drive.wav", f"{tag(shot)}-drive.wav")
    g = json.load(open(f"{HERE}/lab/ltx25_i2v_audio_api.json"))
    g["pos"]["inputs"]["text"] = f"{WHO[who]} {STAGE}. {motion.replace('The cartoon man', 'He')} {LOOK}"
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
    shutil.copy(f"{d}/sing{seed}.mp4", f"{d}/sing.mp4")
    print(shot, "sing", seed, f"{n} frames", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("shot")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--still", type=int, default=7, help="which still seed to animate")
    p.add_argument("--upto", type=float, default=62.1, help="assemble: last shot end time")
    a = p.parse_args()
    shots = list(SHOTS) if a.shot == "all" else a.shot.split(",")
    for sh in shots:
        if a.stage == "still":
            still(sh, a.seed)
        elif a.stage == "frame":
            frame(sh, a.still)
        elif a.stage == "sing":
            sing(sh, a.seed, a.still, VOCALS if SHOTS[sh][5] == "vocals" else SONG)
    if a.stage == "assemble":
        assemble(a.upto)
