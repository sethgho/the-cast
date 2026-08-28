#!/usr/bin/env python3
"""Three ways to put a webcam frame into the Follies cast, compared on the same inputs.

    python3 selfie_methods.py a|b|c <webcam.jpg> [label]

The action-selfie job currently ships the whole photograph to a hosted model and asks for a
restyle. That lands too close to canonical Seth and throws most of the webcam away. These are the
three distinct ways to spend the webcam signal, and they differ in WHAT the edit target is --
which, from the sprite work, is the single decision that determines what survives:

  a  PHOTO AS CANVAS    edit target = the webcam frame. Pose, gesture, framing and lighting are
                        preserved exactly, because they are already in the pixels being edited.
                        Risk: the man who comes back is whoever was in the photo, not cast Seth.
  b  PLATE AS CANVAS    edit target = Seth's canonical plate; the webcam is cropped to the face
                        and consulted for ONE property, the expression. Maximum character
                        fidelity, and everything else about the moment is discarded.
  c  WORDS AS BRIDGE    no webcam pixels at all. The frame is read into a written description of
                        pose, gesture, wardrobe and mood, and that text drives a render off the
                        canonical plate. The bridge is semantic, so it carries intent rather than
                        geometry -- and it composes with a scene prompt for free.

a and c are new graphs; b is the existing seth-expression app with its face source repointed.
"""
import json, os, sys, time, urllib.request, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S
from build_workflows import STYLE_LOCK, SETH_LOOK, NEGATIVE, STEPS

SEED = 7
PLATE = "cast-seth-headshot-neutral.png"
W, H = 1152, 648

# The vision step, done by hand for this bake-off. In production this is the same claude-cli call
# the job already makes for the scene text, so it costs nothing extra.
DESCRIPTIONS = {
 "f1": "He sits square to the camera at his desk wearing large over-ear headphones, long hair "
       "loose on his shoulders, in a plain dark t-shirt. His mouth is closed under the moustache "
       "and his eyes are steady and a little tired. Daylight from a window behind him.",
 "f2": "He leans on one elbow with his hand propping up his cheek, headphones on, long hair "
       "loose, gazing sideways and down at a screen with a weary, patient expression.",
 "f3": "He leans in close to the camera in a dark room lit only by the blue glow of a monitor, "
       "mouth open mid-sentence, eyes wide and animated, in a sleeveless dark top.",
}


def upload(path):
    b = uuid.uuid4().hex
    body = b"".join([
        f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{os.path.basename(path)}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode(),
        open(path, "rb").read(),
        f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n".encode()])
    req = urllib.request.Request(S.HOST + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["name"]


def engine(g):
    """The loaders every method shares, straight off the expression app."""
    g.update({
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE, "clip": ["4", 0]}},
     "22": {"class_type": "KSampler", "inputs": {"seed": SEED, "steps": STEPS, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "model": ["3", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
    })
    return g


def graph_a(name):
    """Photo as canvas: the frame itself is the thing being redrawn."""
    p = ("Image 1 is a photograph. Redraw the whole photograph as an original cartoon drawing. "
         "This is a repaint: the man's pose, the tilt of his head, where his hands are, what he "
         "is wearing, where he sits in the frame and the direction the light comes from are "
         f"already exactly right and must not change at all. Draw the man as {SETH_LOOK}. "
         f"{STYLE_LOCK} Above all, keep his pose, his gesture and his expression exactly as they "
         "are in the photograph, and change only how it is drawn.")
    g = {"16": {"class_type": "LoadImage", "inputs": {"image": name}},
         "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
         "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": p, "clip": ["4", 0], "vae": ["5", 0], "image1": ["16", 0]}}}
    return engine(g)


def graph_b(name):
    """Plate as canvas: the shipped expression app, its face source repointed at the webcam."""
    g = json.load(open(os.path.join(HERE, "api", "seth-expression.api.json")))
    g["6"]["inputs"]["image"] = name          # face read for expression
    g["16"]["inputs"]["image"] = PLATE        # edit target stays canonical Seth
    g["22"]["inputs"]["seed"] = SEED
    del g["24"], g["25"]
    g["23"] = {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}}
    return g


def graph_c(desc):
    """Words as bridge: canonical plate plus a written account of the moment. No webcam pixels."""
    p = (f"Image 1 is the drawing to edit, and the man in image 1 is the only character in the "
         f"finished picture. Redraw him exactly as he is: {SETH_LOOK}. Now draw him in this "
         f"moment: {desc} {STYLE_LOCK} Above all, the man in the finished drawing is the man "
         f"from image 1 — the same long wavy hair, the same bushy handlebar moustache, the same "
         f"dark v-neck t-shirt — and he is doing exactly what the description says.")
    g = {"16": {"class_type": "LoadImage", "inputs": {"image": PLATE}},
         "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
         "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": p, "clip": ["4", 0], "vae": ["5", 0], "image1": ["16", 0]}}}
    return engine(g)


method, path = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else "x"
name = upload(path)
g = {"a": lambda: graph_a(name), "b": lambda: graph_b(name), "c": lambda: graph_c(DESCRIPTIONS[label])}[method]()
out_id = "23"
g["99"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"cast/selfie-{method}-{label}", "images": [out_id, 0]}}
for n, d in g.items(): d.setdefault("_meta", {"title": n})

t0 = time.time()
r = S.api("/prompt", {"prompt": g, "client_id": "selfie"})
if "prompt_id" not in r:
    print("REJECTED", json.dumps(r)[:1000]); sys.exit(1)
pid = r["prompt_id"]
while True:
    h = S.api(f"/history/{pid}")
    if pid in h: break
    time.sleep(5)
st = h[pid]["status"]
if st.get("status_str") != "success":
    print("FAILED", json.dumps(st)[:1000]); sys.exit(1)
o = h[pid]["outputs"]["99"]["images"][0]
dest = f"/home/wilson/scratch/selfie-out/{method}-{label}.png"
os.makedirs(os.path.dirname(dest), exist_ok=True)
with urllib.request.urlopen(S.HOST + f"/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output", timeout=300) as rr:
    open(dest, "wb").write(rr.read())
print(f"{method}/{label}: {time.time()-t0:.0f}s -> {dest}")
