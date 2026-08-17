#!/usr/bin/env python3
"""Build the local-ComfyUI cast workflows.

One character spec in, two files out:

  workflows/<id>-pose-and-scene.json   the graph you load in the ComfyUI builder
  api/<id>-pose-and-scene.api.json     the same graph in API format, for smoke tests

The graph is Qwen-Image-Edit-2511 (GGUF) + a Lightning LoRA, driven from a character
reference plate. Two boolean switches carry the whole control surface: SCENE on/off and
TRANSPARENT PNG on/off.

Usage:  python3 build_workflows.py [character_id ...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- engine defaults

UNET = "qwen-image-edit-2511-Q4_K_S.gguf"
LORA = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
BGREMOVAL = "birefnet.safetensors"
STEPS = 8

STYLE_LOCK = (
    "Draw it in exactly the same style as image 1: a 1933 Fleischer-era rubber-hose cartoon "
    "posed with real weight and balance, confident smooth ink outlines of even weight, shaded "
    "only with soft halftone dots, warm sepia duotone on aged grey-brown newsprint, muted "
    "desaturated ink and no bright modern colour. The art fills the frame with no border and no "
    "margin. There are no speech balloons, no caption boxes, no lettering, no signature and no "
    "watermark."
)

PLATE_CLAUSE = (
    "He is centred in frame, whole body inside the frame, on a single flat mid-grey backdrop of "
    "one even tone — no scenery, no horizon, no floor line, no cast shadow on the ground and no "
    "props except the ones named above."
)

SCENE_LEAD_IN = (
    "He is standing inside the scene shown in image 2, at believable scale, lit by that scene's "
    "light and casting its shadows. The scene is:"
)

NEGATIVE = (
    "blurry, deformed, extra limbs, photorealistic, 3d render, bright saturated colour, "
    "speech balloon, caption box, lettering, watermark"
)

HOW_TO_USE = """# {name} — pose & scene

**Type in the two green boxes. Flip the two switches. Press Run.**

## The controls

| Control | What it does |
|---|---|
| **▶ YOUR SHOT** | The one box you always edit. Say what he is *doing*, in one sentence. |
| **▶ YOUR SCENE** | Where he is. Only used when `SCENE` is ON. |
| **SCENE** switch | OFF = flat mid-grey plate (pose-library style). ON = puts him in the scene image + your scene text. |
| **TRANSPARENT PNG** switch | ON = cuts the background out and saves an RGBA PNG. Works with either scene setting, but is meant for plate shots. |
| **OUTPUT SIZE** | 1024×1024 by default. 1024×1536 for a tall figure, 1536×1024 for a wide gag. |

## What NOT to edit

The blue boxes are the character and house style. They are what makes the output look like
{name} and not a stock cartoon. Leave them alone unless you are deliberately re-canonising him.

## Speed on gpu-worker (RTX 3080 Ti, 12GB)

- **8 steps ≈ 48s** at 1024×1024 — the shipped default, best hands and halftone.
- **4 steps ≈ 25s** — swap the LoRA to `...Lightning-4steps...` and set `steps` to 4 for drafting.

## Two rules learned the hard way

1. **Never wire a second character plate into `image2`.** Qwen treats every reference as a
   *subject*, not a style — two character plates get merged into one creature.
2. Style lives in **words**, not in a reference. If the palette drifts modern or the newsprint
   grain vanishes, the style box got weakened.
"""

# ---------------------------------------------------------------- characters

CHARACTERS = {
    "seth": {
        "name": "Seth",
        "plate": "cast-seth-plate.png",
        "scene_plate": "cast-seth-scene-empty-auditorium.png",
        "character_lock": (
            "Redraw the character in image 1 as a single full-body cartoon figure, keeping his "
            "identity exactly: a lanky cartoon man with long wavy shoulder-length brown hair and a "
            "full bushy handlebar moustache, wearing a plain v-neck t-shirt of very dark olive "
            "charcoal — near-black, the same flat dark tone as the t-shirt in image 1 — with warm "
            "grey trousers and black-and-white sneakers. Keep his face, hair, moustache, build and "
            "clothing identical to image 1, and keep every colour as muted and desaturated as image "
            "1. The new drawing shows him as follows:"
        ),
        "shot": (
            "he stands with both hands on his hips, chest out, one foot forward, looking off to "
            "one side, pleased with himself."
        ),
        "scene": "the empty auditorium seen from the stage, house lights low, rows of seats receding.",
    },
}

# ---------------------------------------------------------------- graph builder

# widget order per node type, straight out of /object_info
WIDGETS = {
    "UnetLoaderGGUF": ["unet_name"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "LoadBackgroundRemovalModel": ["bg_removal_name"],
    "LoadImage": ["image", "upload"],
    "PrimitiveStringMultiline": ["value"],
    "PrimitiveBoolean": ["value"],
    "StringConcatenate": ["string_a", "string_b", "delimiter"],
    "ComfySwitchNode": ["switch"],
    "TextEncodeQwenImageEditPlus": ["prompt"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "VAEDecode": [],
    "RemoveBackground": [],
    "InvertMask": [],
    "JoinImageWithAlpha": [],
    "SaveImage": ["filename_prefix"],
    "MarkdownNote": ["text"],
}

# widgets that are NOT sent to the API (frontend-only)
UI_ONLY_WIDGETS = {("LoadImage", "upload"), ("KSampler", "control_after_generate")}


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.groups = []
        self._nid = 0
        self._lid = 0

    def add(self, ntype, title, pos, size, widgets=None, links=None, outputs=None, color=None):
        """links: {input_name: (node, slot, TYPE, is_widget_input)}"""
        self._nid += 1
        node = {
            "id": self._nid,
            "type": ntype,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": self._nid,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "title": title,
            "properties": {"Node name for S&R": ntype},
            "widgets_values": [],
        }
        if color:
            node["color"] = color[0]
            node["bgcolor"] = color[1]

        wnames = WIDGETS[ntype]
        widgets = widgets or {}
        node["_widgets"] = widgets
        node["widgets_values"] = [widgets.get(w, "") for w in wnames]

        for name, (src, slot, wtype, is_widget) in (links or {}).items():
            self._lid += 1
            entry = {"name": name, "type": wtype, "link": self._lid}
            if is_widget:
                entry["widget"] = {"name": name}
            node["inputs"].append(entry)
            src["outputs"][slot]["links"].append(self._lid)
            self.links.append([self._lid, src["id"], slot, self._nid, len(node["inputs"]) - 1, wtype])
        node["_link_names"] = list((links or {}).keys())

        for oname, otype in (outputs or []):
            node["outputs"].append({"name": oname, "type": otype, "links": []})

        self.nodes.append(node)
        return node

    def group(self, title, bounding, color):
        self.groups.append({
            "id": len(self.groups) + 1,
            "title": title,
            "bounding": list(bounding),
            "color": color,
            "font_size": 24,
            "flags": {},
        })

    # -------------------------------------------------- serialisation

    def to_ui(self):
        nodes = []
        for n in self.nodes:
            c = {k: v for k, v in n.items() if not k.startswith("_")}
            c["outputs"] = [{"name": o["name"], "type": o["type"], "links": o["links"] or None}
                            for o in n["outputs"]]
            nodes.append(c)
        return {
            "id": "cast-local-qwen-edit",
            "revision": 0,
            "last_node_id": self._nid,
            "last_link_id": self._lid,
            "nodes": nodes,
            "links": self.links,
            "groups": self.groups,
            "definitions": {"subgraphs": []},
            "config": {},
            "extra": {"ds": {"scale": 0.55, "offset": [1600, 700]}},
            "version": 0.4,
        }

    def to_api(self):
        by_link = {l[0]: (l[1], l[2]) for l in self.links}
        out = {}
        for n in self.nodes:
            if n["type"] == "MarkdownNote":
                continue
            inputs = {}
            for w in WIDGETS[n["type"]]:
                if (n["type"], w) in UI_ONLY_WIDGETS or w not in n["_widgets"]:
                    continue
                inputs[w] = n["_widgets"][w]
            for entry in n["inputs"]:
                src_id, src_slot = by_link[entry["link"]]
                inputs[entry["name"]] = [str(src_id), src_slot]
            out[str(n["id"])] = {"class_type": n["type"], "inputs": inputs,
                                 "_meta": {"title": n["title"]}}
        return out


# ---------------------------------------------------------------- the workflow

BLUE = ("#223", "#335")        # locked / canon
GREEN = ("#232", "#353")       # type here
GREY = ("#323", "#535")        # switches
ENGINE = ("#222", "#000")

GROUP_LOCK = "#3f789e"
GROUP_SHOT = "#4a7c3f"
GROUP_OUT = "#8a6d3b"
GROUP_ENGINE = "#444"


def build(cid, spec):
    g = Graph()
    S = lambda t: ("STRING", 0)  # noqa: E731 (readability)

    # --- notes -----------------------------------------------------------
    g.add("MarkdownNote", f"HOW TO USE — {spec['name']}", (-1560, -560), (480, 900),
          {"text": HOW_TO_USE.format(name=spec["name"])})

    # --- 1. character + style lock ---------------------------------------
    char_lock = g.add("PrimitiveStringMultiline", f"CHARACTER LOCK — who {spec['name']} is (leave alone)",
                      (-1020, -560), (430, 300), {"value": spec["character_lock"]},
                      outputs=[("STRING", "STRING")], color=BLUE)
    style_lock = g.add("PrimitiveStringMultiline", "STYLE LOCK — house style (leave alone)",
                       (-1020, -230), (430, 300), {"value": STYLE_LOCK},
                       outputs=[("STRING", "STRING")], color=BLUE)
    plate_clause = g.add("PrimitiveStringMultiline", "PLATE BACKGROUND — used when SCENE is OFF (leave alone)",
                         (-1020, 100), (430, 190), {"value": PLATE_CLAUSE},
                         outputs=[("STRING", "STRING")], color=BLUE)
    scene_lead = g.add("PrimitiveStringMultiline", "SCENE LEAD-IN — used when SCENE is ON (leave alone)",
                       (-1020, 320), (430, 190), {"value": SCENE_LEAD_IN},
                       outputs=[("STRING", "STRING")], color=BLUE)
    g.group(f"① {spec['name'].upper()} — CHARACTER & STYLE LOCK · leave alone",
            (-1050, -630, 490, 1180), GROUP_LOCK)

    # --- 2. your shot ----------------------------------------------------
    shot = g.add("PrimitiveStringMultiline", "▶ YOUR SHOT — what he is doing (TYPE HERE)",
                 (-520, -560), (440, 330), {"value": spec["shot"]},
                 outputs=[("STRING", "STRING")], color=GREEN)
    scene_text = g.add("PrimitiveStringMultiline", "▶ YOUR SCENE — where he is (TYPE HERE, needs SCENE = true)",
                       (-520, -200), (440, 250), {"value": spec["scene"]},
                       outputs=[("STRING", "STRING")], color=GREEN)
    scene_on = g.add("PrimitiveBoolean", "SCENE  ·  false = flat grey plate,  true = in a scene",
                     (-520, 80), (440, 90), {"value": False},
                     outputs=[("BOOLEAN", "BOOLEAN")], color=GREY)
    transparent_on = g.add("PrimitiveBoolean", "TRANSPARENT PNG  ·  true = cut the background out",
                           (-520, 200), (440, 90), {"value": False},
                           outputs=[("BOOLEAN", "BOOLEAN")], color=GREY)
    plate = g.add("LoadImage", f"CHARACTER PLATE — {spec['name']}'s reference (image 1)",
                  (-520, 320), (440, 420), {"image": spec["plate"], "upload": "image"},
                  outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    scene_plate = g.add("LoadImage", "SCENE PLATE (image 2) — only read when SCENE = true",
                        (-520, 780), (440, 420), {"image": spec["scene_plate"], "upload": "image"},
                        outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    g.group("② YOUR SHOT · type in the green boxes, flip the switches",
            (-550, -630, 500, 1860), GROUP_SHOT)

    # --- 3. prompt assembly ---------------------------------------------
    scene_clause = g.add("StringConcatenate", "scene lead-in + your scene",
                         (-20, -200), (330, 150), {"delimiter": " "},
                         links={"string_a": (scene_lead, 0, "STRING", True),
                                "string_b": (scene_text, 0, "STRING", True)},
                         outputs=[("STRING", "STRING")])
    bg_clause = g.add("ComfySwitchNode", "BACKGROUND text switch",
                      (-20, 0), (330, 130), {"switch": False},
                      links={"on_false": (plate_clause, 0, "STRING", False),
                             "on_true": (scene_clause, 0, "STRING", False),
                             "switch": (scene_on, 0, "BOOLEAN", True)},
                      outputs=[("output", "STRING")])
    cat1 = g.add("StringConcatenate", "character lock + your shot",
                 (-20, -560), (330, 150), {"delimiter": " "},
                 links={"string_a": (char_lock, 0, "STRING", True),
                        "string_b": (shot, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")])
    cat2 = g.add("StringConcatenate", "+ background",
                 (-20, -390), (330, 150), {"delimiter": " "},
                 links={"string_a": (cat1, 0, "STRING", True),
                        "string_b": (bg_clause, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")])
    prompt = g.add("StringConcatenate", "FINAL PROMPT (= what the model reads)",
                   (-20, 160), (330, 150), {"delimiter": " "},
                   links={"string_a": (cat2, 0, "STRING", True),
                          "string_b": (style_lock, 0, "STRING", True)},
                   outputs=[("STRING", "STRING")])

    # --- 4. engine -------------------------------------------------------
    unet = g.add("UnetLoaderGGUF", "diffusion model", (-20, 700), (400, 80),
                 {"unet_name": UNET}, outputs=[("MODEL", "MODEL")], color=ENGINE)
    lora = g.add("LoraLoaderModelOnly", "Lightning LoRA (8 steps)", (-20, 820), (400, 110),
                 {"lora_name": LORA, "strength_model": 1.0},
                 links={"model": (unet, 0, "MODEL", False)},
                 outputs=[("MODEL", "MODEL")], color=ENGINE)
    clip = g.add("CLIPLoader", "text encoder", (-20, 970), (400, 130),
                 {"clip_name": CLIP, "type": "qwen_image", "device": "default"},
                 outputs=[("CLIP", "CLIP")], color=ENGINE)
    vae = g.add("VAELoader", "VAE", (-20, 1140), (400, 80), {"vae_name": VAE},
                outputs=[("VAE", "VAE")], color=ENGINE)
    bgmodel = g.add("LoadBackgroundRemovalModel", "BiRefNet (for TRANSPARENT PNG)",
                    (-20, 1260), (400, 80), {"bg_removal_name": BGREMOVAL},
                    outputs=[("BACKGROUND_REMOVAL", "BACKGROUND_REMOVAL")], color=ENGINE)
    g.group("engine · don't touch", (-50, 630, 460, 760), GROUP_ENGINE)

    # --- 5. conditioning + sampling --------------------------------------
    enc_plate = g.add("TextEncodeQwenImageEditPlus", "encode — plate (image 1 only)",
                      (420, -560), (400, 130), {"prompt": ""},
                      links={"clip": (clip, 0, "CLIP", False),
                             "vae": (vae, 0, "VAE", False),
                             "image1": (plate, 0, "IMAGE", False),
                             "prompt": (prompt, 0, "STRING", True)},
                      outputs=[("CONDITIONING", "CONDITIONING")])
    enc_scene = g.add("TextEncodeQwenImageEditPlus", "encode — scene (image 1 + image 2)",
                      (420, -380), (400, 130), {"prompt": ""},
                      links={"clip": (clip, 0, "CLIP", False),
                             "vae": (vae, 0, "VAE", False),
                             "image1": (plate, 0, "IMAGE", False),
                             "image2": (scene_plate, 0, "IMAGE", False),
                             "prompt": (prompt, 0, "STRING", True)},
                      outputs=[("CONDITIONING", "CONDITIONING")])
    positive = g.add("ComfySwitchNode", "POSITIVE switch (same SCENE toggle)",
                     (420, -190), (400, 130), {"switch": False},
                     links={"on_false": (enc_plate, 0, "CONDITIONING", False),
                            "on_true": (enc_scene, 0, "CONDITIONING", False),
                            "switch": (scene_on, 0, "BOOLEAN", True)},
                     outputs=[("output", "CONDITIONING")])
    negative = g.add("TextEncodeQwenImageEditPlus", "negative (only bites when cfg > 1)",
                     (420, 0), (400, 130), {"prompt": NEGATIVE},
                     links={"clip": (clip, 0, "CLIP", False)},
                     outputs=[("CONDITIONING", "CONDITIONING")], color=ENGINE)
    latent = g.add("EmptySD3LatentImage", "OUTPUT SIZE", (420, 180), (400, 130),
                   {"width": 1024, "height": 1024, "batch_size": 1},
                   outputs=[("LATENT", "LATENT")], color=GREY)
    sampler = g.add("KSampler", "sampler · seed + steps live here", (420, 360), (400, 270),
                    {"seed": 1, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                     "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
                    links={"model": (lora, 0, "MODEL", False),
                           "positive": (positive, 0, "CONDITIONING", False),
                           "negative": (negative, 0, "CONDITIONING", False),
                           "latent_image": (latent, 0, "LATENT", False)},
                    outputs=[("LATENT", "LATENT")])
    decode = g.add("VAEDecode", "decode", (420, 680), (400, 80),
                   links={"samples": (sampler, 0, "LATENT", False),
                          "vae": (vae, 0, "VAE", False)},
                   outputs=[("IMAGE", "IMAGE")])

    # --- 6. output -------------------------------------------------------
    cutout = g.add("RemoveBackground", "cut the figure out", (900, -560), (380, 90),
                   links={"bg_removal_model": (bgmodel, 0, "BACKGROUND_REMOVAL", False),
                          "image": (decode, 0, "IMAGE", False)},
                   outputs=[("MASK", "MASK")])
    # JoinImageWithAlpha inverts the mask internally (ComfyUI's mask convention is
    # 1 = masked OUT), so the foreground mask has to be flipped first or the figure
    # is what goes transparent.
    inverted = g.add("InvertMask", "flip to ComfyUI mask convention", (900, -430), (380, 80),
                     links={"mask": (cutout, 0, "MASK", False)},
                     outputs=[("MASK", "MASK")])
    rgba = g.add("JoinImageWithAlpha", "make it RGBA", (900, -330), (380, 90),
                 links={"image": (decode, 0, "IMAGE", False),
                        "alpha": (inverted, 0, "MASK", False)},
                 outputs=[("IMAGE", "IMAGE")])
    final = g.add("ComfySwitchNode", "TRANSPARENT switch", (900, -200), (380, 130),
                  {"switch": False},
                  links={"on_false": (decode, 0, "IMAGE", False),
                         "on_true": (rgba, 0, "IMAGE", False),
                         "switch": (transparent_on, 0, "BOOLEAN", True)},
                  outputs=[("output", "IMAGE")])
    g.add("SaveImage", "SAVE", (900, -100), (600, 700),
          {"filename_prefix": f"cast/{cid}"},
          links={"images": (final, 0, "IMAGE", False)})
    g.group("③ OUTPUT", (870, -630, 660, 1250), GROUP_OUT)

    return g


def main(ids):
    os.makedirs(os.path.join(HERE, "workflows"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "api"), exist_ok=True)
    for cid in ids:
        g = build(cid, CHARACTERS[cid])
        ui = os.path.join(HERE, "workflows", f"{cid}-pose-and-scene.json")
        api = os.path.join(HERE, "api", f"{cid}-pose-and-scene.api.json")
        json.dump(g.to_ui(), open(ui, "w"), indent=1)
        json.dump(g.to_api(), open(api, "w"), indent=1)
        print(f"wrote {ui}\nwrote {api}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(CHARACTERS))
