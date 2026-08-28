#!/usr/bin/env python3
"""Mode 2 probe: a webcam frame's SHAPE, rendered as abstract pattern.

    python3 abstract_methods.py <map> <style> <frame>

Vaudeville mode spends the webcam signal on a written description and throws the pixels away.
This mode does the opposite: it keeps only the geometry and throws the SUBJECT away. The frame is
reduced to a structure map on the CPU first (see selfie-src/map-*.png), and that map is the Qwen
edit target -- the same trick the sprite repaint uses, where editing an image in place preserves
its geometry and changes only how it is drawn.

Three encodings, because they afford different pictures:

  edges     thin contour lines where the picture changes. Sparse: lots of empty
            space for a pattern to fill, but little sense of volume.
  depth     heavily blurred luminance. No lines at all, just near/far masses --
            the one that can carry volumetric light.
  contours  posterised iso-luminance bands. Reads as a topographic map, which is
            already halfway to an interference pattern.

No ControlNet is involved: none is installed, and this needs no download. If the shape does not
hold tightly enough, a Qwen Blockwise ControlNet (depth or canny) is the next step up.
"""
import json, os, sys, time, urllib.request, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S
from build_workflows import STEPS

W, H, SEED = 1152, 648, 7

KEEP = ("Follow the light and dark shapes of image 1 exactly: every edge, mass and boundary in the "
        "finished picture sits precisely where it does in image 1, at the same scale and in the "
        "same place in the frame. Nothing is drawn outside those shapes and nothing is moved.")
DROP = ("Do not draw a person, a face, a room, furniture or any recognisable object. The finished "
        "picture is pure abstract pattern.")

STYLES = {
 "particles": ("Redraw image 1 as a psychedelic particle field: millions of tiny luminous points "
               "swarming in dense streams, hot magenta, cyan and acid green against deep black, "
               "with glowing trails and soft bloom where the streams crowd together. " + KEEP + " " + DROP),
 "waves":     ("Redraw image 1 as spatial sound waves: concentric interference rings spreading and "
               "colliding, thin bright wavefronts on a dark field, iridescent blue-violet through "
               "gold where the ripples overlap, like a cymatics plate photographed in the dark. "
               + KEEP + " " + DROP),
 "plasma":    ("Redraw image 1 as a neon flow field: long smooth streamlines of liquid light "
               "bending around invisible obstacles, electric teal, ultraviolet and hot orange, "
               "deep black between the strands, with a soft volumetric haze. " + KEEP + " " + DROP),

 # The first three were all neon-on-black with rainbow hues -- one aesthetic in three hats. These
 # six are chosen to be materially different from each other: each names a real physical medium
 # and a CLOSED palette, because "abstract" with an open palette collapses to tie-dye every time.
 "chladni":  ("Redraw image 1 as a Chladni figure: fine pale sand scattered on a black steel plate, "
              "vibrated into sharp nodal ridges and swept bare in between. Photographed from "
              "directly above in raking light, so the sand grains cast tiny shadows and the bare "
              "steel is dark and slightly scuffed. Monochrome: bone-white sand, gunmetal plate, "
              "nothing else. " + KEEP + " " + DROP),
 "ferro":    ("Redraw image 1 as ferrofluid on glass: glossy black magnetic liquid pulled into "
              "dense fields of sharp spikes and smooth beading pools, wet and mirror-bright, lit "
              "by one hard studio light from the left. Almost entirely black on black, readable "
              "only by specular highlights and reflection. No colour at all. " + KEEP + " " + DROP),
 "agate":    ("Redraw image 1 as a cut and polished agate slab: concentric mineral banding in "
              "rust, ochre, cream and smoky grey, with crystalline druzy pockets where the bands "
              "close, veined in white quartz. Photographed lit from behind so the thin bands glow. "
              "Earth tones only, no bright or saturated colour. " + KEEP + " " + DROP),
 "schlieren":("Redraw image 1 as a schlieren photograph of moving air: shockwaves and thermal "
              "plumes rendered as smooth grey density gradients with knife-edge dark and light "
              "fringes, on a flat grey field. A laboratory image, greyscale only, no colour "
              "whatsoever. " + KEEP + " " + DROP),
 "kirlian":  ("Redraw image 1 as a Kirlian corona-discharge photograph: fine electric filaments "
              "branching off every boundary into a black photographic ground, violet-white at the "
              "core fading to deep indigo, with film grain and a slight halation bloom. Two "
              "colours only, violet and white on black. " + KEEP + " " + DROP),
 "sumi":     ("Redraw image 1 as suminagashi marbling: black sumi ink floated on still water and "
              "drawn into fine concentric feathered rings, printed onto damp cream washi paper "
              "with visible fibre and deckle. Ink black and one muted indigo on undyed paper, "
              "nothing more. " + KEEP + " " + DROP),
}

mapname, style, frame = sys.argv[1], sys.argv[2], sys.argv[3]
src = f"/home/wilson/scratch/selfie-src/map-{mapname}-{frame}.png"


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


name = upload(src)
g = {
 "2":  {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
 "3":  {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
 "4":  {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
 "5":  {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
 "6":  {"class_type": "LoadImage", "inputs": {"image": name}},
 "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
 "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": STYLES[style], "clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0]}},
 "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": "a person, a face, a room, furniture, text, watermark, blurry, muddy, rainbow, tie-dye, neon, psychedelic, oversaturated, garish", "clip": ["4", 0]}},
 "22": {"class_type": "KSampler", "inputs": {"seed": SEED, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["3", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
 "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
 "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"cast/abs-{mapname}-{style}-{frame}", "images": ["23", 0]}},
}
for n, d in g.items(): d.setdefault("_meta", {"title": n})

t0 = time.time()
r = S.api("/prompt", {"prompt": g, "client_id": "abstract"})
if "prompt_id" not in r:
    print("REJECTED", json.dumps(r)[:900]); sys.exit(1)
pid = r["prompt_id"]
while True:
    h = S.api(f"/history/{pid}")
    if pid in h: break
    time.sleep(5)
st = h[pid]["status"]
if st.get("status_str") != "success":
    print("FAILED", json.dumps(st)[:900]); sys.exit(1)
o = h[pid]["outputs"]["99"]["images"][0]
dest = f"/home/wilson/scratch/abstract-out/{mapname}-{style}-{frame}.png"
os.makedirs(os.path.dirname(dest), exist_ok=True)
with urllib.request.urlopen(S.HOST + f"/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output", timeout=300) as rr:
    open(dest, "wb").write(rr.read())
print(f"{mapname}/{style}/{frame}: {time.time()-t0:.0f}s -> {dest}")
