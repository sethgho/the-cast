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

Everything you touch is the numbered column on the left, top to bottom.

| Control | What it does |
|---|---|
| **1 · YOUR SHOT** | The one box you always edit. Say what he is *doing*, in one sentence. |
| **2 · SCENE** | OFF = flat mid-grey plate (pose-library style). ON = puts him in the scene plate + your scene text. |
| **3 · YOUR SCENE** | Where he is. Only read when `SCENE` is ON. |
| **4 · SCENE PLATE** | The image 2 backdrop. Only read when `SCENE` is ON. |
| **5 · TRANSPARENT PNG** | ON = cuts the background out and saves an RGBA PNG. Meant for plate shots. |
| **6 · OUTPUT SIZE** | 1024×1024 by default. 1024×1536 for a tall figure, 1536×1024 for a wide gag. |
| **7 · SEED + STEPS** | Re-roll here. 8 steps ships; drop to 4 with the 4-step LoRA for drafts. |

## What NOT to edit

Everything else is collapsed on purpose. The blue group is the character and house style —
what makes the output look like {name} and not a stock cartoon. Expand it only to re-canonise
him. Expand **FINAL PROMPT** in that group any time you want to read exactly what the model got.

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

# slot types for the widgets promoted into the App Mode form
WIDGET_TYPES = {
    ("PrimitiveStringMultiline", "value"): "STRING",
    ("PrimitiveBoolean", "value"): "BOOLEAN",
    ("LoadImage", "image"): "COMBO",
    ("LoadImage", "upload"): "IMAGEUPLOAD",
    ("EmptySD3LatentImage", "width"): "INT",
    ("EmptySD3LatentImage", "height"): "INT",
    ("EmptySD3LatentImage", "batch_size"): "INT",
    ("KSampler", "seed"): "INT",
    ("KSampler", "steps"): "INT",
    ("KSampler", "cfg"): "FLOAT",
    ("KSampler", "sampler_name"): "COMBO",
    ("KSampler", "scheduler"): "COMBO",
    ("KSampler", "denoise"): "FLOAT",
}

# widgets that are NOT sent to the API (frontend-only)
UI_ONLY_WIDGETS = {("LoadImage", "upload"), ("KSampler", "control_after_generate")}


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.groups = []
        self.app_inputs = []   # [[node_id, widget_name], ...] — the App Mode form, in order
        self.app_outputs = []  # [node_id, ...]
        self._nid = 0
        self._lid = 0

    def app_input(self, node, *widget_names, **labels):
        """Promote widgets into the App Mode form.

        App Mode labels a field `widget.label || widget.name`, so a bare PrimitiveBoolean
        shows up as "value". The label rides on the node's input slot for that widget —
        the same field the UI's right-click → Rename writes.
        """
        for w in widget_names:
            self.app_inputs.append([str(node["id"]), w])
        # The label only survives a reload when the widget's input slot is written the
        # way the frontend writes it — name + localized_name + real widget type.
        existing = {i["name"] for i in node["inputs"]}
        for w in WIDGETS[node["type"]]:
            if w in existing or (node["type"], w) == ("KSampler", "control_after_generate"):
                continue
            slot = {
                "name": w,
                "localized_name": "choose file to upload" if w == "upload" else w,
                "type": WIDGET_TYPES[(node["type"], w)],
                "widget": {"name": w},
                "link": None,
            }
            if labels.get(w):
                slot["label"] = labels[w]
            node["inputs"].append(slot)
        return node

    def app_output(self, node):
        self.app_outputs.append(str(node["id"]))
        return node

    def add(self, ntype, title, pos, size, widgets=None, links=None, outputs=None, color=None,
            collapsed=False):
        """links: {input_name: (node, slot, TYPE, is_widget_input)}"""
        self._nid += 1
        node = {
            "id": self._nid,
            "type": ntype,
            "pos": list(pos),
            "size": list(size),
            "flags": {"collapsed": True} if collapsed else {},
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
            # NB: no "definitions" key. An empty {"subgraphs": []} sends the loader down
            # the subgraph path, which rebuilds every widget input slot and throws away
            # the App Mode labels on them.
            "config": {},
            "extra": {
                "ds": {"scale": 0.55, "offset": [1600, 700]},
                # App Mode ("linear mode"): the form the workflow opens as.
                "linearMode": True,
                "linearData": {"inputs": self.app_inputs, "outputs": self.app_outputs},
            },
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
                if entry.get("link") is None:   # label-only slot, no connection
                    continue
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

    # --- 1. the column you actually touch --------------------------------
    shot = g.add("PrimitiveStringMultiline", "▶ 1 · YOUR SHOT — what he is doing (TYPE HERE)",
                 (-40, -620), (460, 300), {"value": spec["shot"]},
                 outputs=[("STRING", "STRING")], color=GREEN)
    scene_on = g.add("PrimitiveBoolean", "▶ 2 · SCENE   false = flat grey plate · true = in a scene",
                     (-40, -290), (460, 80), {"value": False},
                     outputs=[("BOOLEAN", "BOOLEAN")], color=GREY)
    scene_text = g.add("PrimitiveStringMultiline", "▶ 3 · YOUR SCENE — where he is (only read when SCENE = true)",
                       (-40, -180), (460, 210), {"value": spec["scene"]},
                       outputs=[("STRING", "STRING")], color=GREEN)
    scene_plate = g.add("LoadImage", "▶ 4 · SCENE PLATE (image 2) — only read when SCENE = true",
                        (-40, 60), (460, 400), {"image": spec["scene_plate"], "upload": "image"},
                        outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    transparent_on = g.add("PrimitiveBoolean", "▶ 5 · TRANSPARENT PNG   true = cut the background out",
                           (-40, 490), (460, 80), {"value": False},
                           outputs=[("BOOLEAN", "BOOLEAN")], color=GREY)
    latent = g.add("EmptySD3LatentImage", "▶ 6 · OUTPUT SIZE", (-40, 600), (460, 130),
                   {"width": 1024, "height": 1024, "batch_size": 1},
                   outputs=[("LATENT", "LATENT")], color=GREY)
    g.group("② YOUR SHOT · everything you touch is in this column, top to bottom",
            (-70, -690, 520, 1470), GROUP_SHOT)

    # App Mode form, in the order it is asked for
    g.app_input(shot, "value", value="1 · YOUR SHOT — what he is doing")
    g.app_input(scene_on, "value", value="2 · SCENE — off = flat grey plate, on = in a scene")
    g.app_input(scene_text, "value", value="3 · YOUR SCENE — where he is (only read when SCENE is on)")
    g.app_input(scene_plate, "image", image="4 · SCENE PLATE (only read when SCENE is on)")
    g.app_input(transparent_on, "value", value="5 · TRANSPARENT PNG — cut the background out")
    g.app_input(latent, "width", "height", width="6 · WIDTH", height="6 · HEIGHT")

    # --- 2. output --------------------------------------------------------
    # (sampler sits here because seed + steps are the only engine dials worth a poke)
    sampler_pos = (520, 640)

    # --- 3. character + style lock (collapsed; expand to re-canonise) ------
    LX, LY, DY = 1240, -620, 46
    char_lock = g.add("PrimitiveStringMultiline", f"CHARACTER LOCK — who {spec['name']} is",
                      (LX, LY), (430, 300), {"value": spec["character_lock"]},
                      outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    style_lock = g.add("PrimitiveStringMultiline", "STYLE LOCK — house style",
                       (LX, LY + DY), (430, 300), {"value": STYLE_LOCK},
                       outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    plate_clause = g.add("PrimitiveStringMultiline", "PLATE BACKGROUND — used when SCENE is off",
                         (LX, LY + 2 * DY), (430, 190), {"value": PLATE_CLAUSE},
                         outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    scene_lead = g.add("PrimitiveStringMultiline", "SCENE LEAD-IN — used when SCENE is on",
                       (LX, LY + 3 * DY), (430, 190), {"value": SCENE_LEAD_IN},
                       outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    plate = g.add("LoadImage", f"CHARACTER PLATE — {spec['name']}'s reference (image 1)",
                  (LX, LY + 4 * DY), (430, 400), {"image": spec["plate"], "upload": "image"},
                  outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], collapsed=True)

    # prompt assembly, collapsed — FINAL PROMPT is the one worth expanding to read
    cat1 = g.add("StringConcatenate", "character lock + your shot",
                 (LX, LY + 5 * DY), (330, 150), {"delimiter": " "},
                 links={"string_a": (char_lock, 0, "STRING", True),
                        "string_b": (shot, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    scene_clause = g.add("StringConcatenate", "scene lead-in + your scene",
                         (LX, LY + 6 * DY), (330, 150), {"delimiter": " "},
                         links={"string_a": (scene_lead, 0, "STRING", True),
                                "string_b": (scene_text, 0, "STRING", True)},
                         outputs=[("STRING", "STRING")], collapsed=True)
    bg_clause = g.add("ComfySwitchNode", "background text switch",
                      (LX, LY + 7 * DY), (330, 130), {"switch": False},
                      links={"on_false": (plate_clause, 0, "STRING", False),
                             "on_true": (scene_clause, 0, "STRING", False),
                             "switch": (scene_on, 0, "BOOLEAN", True)},
                      outputs=[("output", "STRING")], collapsed=True)
    cat2 = g.add("StringConcatenate", "+ background",
                 (LX, LY + 8 * DY), (330, 150), {"delimiter": " "},
                 links={"string_a": (cat1, 0, "STRING", True),
                        "string_b": (bg_clause, 0, "STRING", True)},
                 outputs=[("STRING", "STRING")], collapsed=True)
    prompt = g.add("StringConcatenate", "FINAL PROMPT — expand to read what the model got",
                   (LX, LY + 9 * DY), (330, 150), {"delimiter": " "},
                   links={"string_a": (cat2, 0, "STRING", True),
                          "string_b": (style_lock, 0, "STRING", True)},
                   outputs=[("STRING", "STRING")], collapsed=True)
    g.group(f"① {spec['name'].upper()} — CHARACTER & STYLE LOCK · expand only to re-canonise him",
            (LX - 30, LY - 70, 480, 10 * DY + 90), GROUP_LOCK)

    # --- 4. engine (all collapsed) ---------------------------------------
    EX, EY = LX, LY + 10 * DY + 60
    unet = g.add("UnetLoaderGGUF", "diffusion model", (EX, EY), (400, 80),
                 {"unet_name": UNET}, outputs=[("MODEL", "MODEL")], color=ENGINE, collapsed=True)
    lora = g.add("LoraLoaderModelOnly", "Lightning LoRA (8 steps)", (EX, EY + DY), (400, 110),
                 {"lora_name": LORA, "strength_model": 1.0},
                 links={"model": (unet, 0, "MODEL", False)},
                 outputs=[("MODEL", "MODEL")], color=ENGINE, collapsed=True)
    clip = g.add("CLIPLoader", "text encoder", (EX, EY + 2 * DY), (400, 130),
                 {"clip_name": CLIP, "type": "qwen_image", "device": "default"},
                 outputs=[("CLIP", "CLIP")], color=ENGINE, collapsed=True)
    vae = g.add("VAELoader", "VAE", (EX, EY + 3 * DY), (400, 80), {"vae_name": VAE},
                outputs=[("VAE", "VAE")], color=ENGINE, collapsed=True)
    bgmodel = g.add("LoadBackgroundRemovalModel", "BiRefNet (for TRANSPARENT PNG)",
                    (EX, EY + 4 * DY), (400, 80), {"bg_removal_name": BGREMOVAL},
                    outputs=[("BACKGROUND_REMOVAL", "BACKGROUND_REMOVAL")], color=ENGINE,
                    collapsed=True)
    enc_plate = g.add("TextEncodeQwenImageEditPlus", "encode — plate (image 1 only)",
                      (EX, EY + 5 * DY), (400, 130), {"prompt": ""},
                      links={"clip": (clip, 0, "CLIP", False),
                             "vae": (vae, 0, "VAE", False),
                             "image1": (plate, 0, "IMAGE", False),
                             "prompt": (prompt, 0, "STRING", True)},
                      outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    enc_scene = g.add("TextEncodeQwenImageEditPlus", "encode — scene (image 1 + image 2)",
                      (EX, EY + 6 * DY), (400, 130), {"prompt": ""},
                      links={"clip": (clip, 0, "CLIP", False),
                             "vae": (vae, 0, "VAE", False),
                             "image1": (plate, 0, "IMAGE", False),
                             "image2": (scene_plate, 0, "IMAGE", False),
                             "prompt": (prompt, 0, "STRING", True)},
                      outputs=[("CONDITIONING", "CONDITIONING")], collapsed=True)
    positive = g.add("ComfySwitchNode", "positive switch (same SCENE toggle)",
                     (EX, EY + 7 * DY), (400, 130), {"switch": False},
                     links={"on_false": (enc_plate, 0, "CONDITIONING", False),
                            "on_true": (enc_scene, 0, "CONDITIONING", False),
                            "switch": (scene_on, 0, "BOOLEAN", True)},
                     outputs=[("output", "CONDITIONING")], collapsed=True)
    negative = g.add("TextEncodeQwenImageEditPlus", "negative (only bites when cfg > 1)",
                     (EX, EY + 8 * DY), (400, 130), {"prompt": NEGATIVE},
                     links={"clip": (clip, 0, "CLIP", False)},
                     outputs=[("CONDITIONING", "CONDITIONING")], color=ENGINE, collapsed=True)
    g.group("engine · don't touch", (EX - 30, EY - 70, 480, 9 * DY + 90), GROUP_ENGINE)

    # --- 5. sampling + output --------------------------------------------
    sampler = g.add("KSampler", "▶ 7 · SEED + STEPS", sampler_pos, (460, 270),
                    {"seed": 1, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                     "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
                    links={"model": (lora, 0, "MODEL", False),
                           "positive": (positive, 0, "CONDITIONING", False),
                           "negative": (negative, 0, "CONDITIONING", False),
                           "latent_image": (latent, 0, "LATENT", False)},
                    outputs=[("LATENT", "LATENT")])
    decode = g.add("VAEDecode", "decode", (520, 950), (400, 80),
                   links={"samples": (sampler, 0, "LATENT", False),
                          "vae": (vae, 0, "VAE", False)},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)
    cutout = g.add("RemoveBackground", "cut the figure out", (520, 1000), (380, 90),
                   links={"bg_removal_model": (bgmodel, 0, "BACKGROUND_REMOVAL", False),
                          "image": (decode, 0, "IMAGE", False)},
                   outputs=[("MASK", "MASK")], collapsed=True)
    # JoinImageWithAlpha inverts the mask internally (ComfyUI's mask convention is
    # 1 = masked OUT), so the foreground mask has to be flipped first or the figure
    # is what goes transparent.
    inverted = g.add("InvertMask", "flip to ComfyUI mask convention", (520, 1050), (380, 80),
                     links={"mask": (cutout, 0, "MASK", False)},
                     outputs=[("MASK", "MASK")], collapsed=True)
    rgba = g.add("JoinImageWithAlpha", "make it RGBA", (520, 1100), (380, 90),
                 links={"image": (decode, 0, "IMAGE", False),
                        "alpha": (inverted, 0, "MASK", False)},
                 outputs=[("IMAGE", "IMAGE")], collapsed=True)
    final = g.add("ComfySwitchNode", "transparent switch", (520, 1150), (380, 130),
                  {"switch": False},
                  links={"on_false": (decode, 0, "IMAGE", False),
                         "on_true": (rgba, 0, "IMAGE", False),
                         "switch": (transparent_on, 0, "BOOLEAN", True)},
                  outputs=[("output", "IMAGE")], collapsed=True)
    g.app_input(sampler, "seed", "steps", seed="7 · SEED", steps="7 · STEPS (8 ships, 4 drafts)")
    g.app_output(g.add("SaveImage", "RESULT", (520, -620), (620, 1200),
                       {"filename_prefix": f"cast/{cid}"},
                       links={"images": (final, 0, "IMAGE", False)}))
    g.group("③ RESULT · seed, steps and the finished image",
            (490, -690, 680, 1930), GROUP_OUT)

    return g


def main(ids):
    os.makedirs(os.path.join(HERE, "workflows"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "api"), exist_ok=True)
    for cid in ids:
        g = build(cid, CHARACTERS[cid])
        ui = os.path.join(HERE, "workflows", f"{cid}-pose.json")
        api = os.path.join(HERE, "api", f"{cid}-pose.api.json")
        json.dump(g.to_ui(), open(ui, "w"), indent=1)
        json.dump(g.to_api(), open(api, "w"), indent=1)
        print(f"wrote {ui}\nwrote {api}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(CHARACTERS))
