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
    "ImageCrop": ["width", "height", "x", "y"],
    "ImageScale": ["upscale_method", "width", "height", "crop"],
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


HOW_TO_USE = """# {name} — {kind_title}

**Fill in the numbered fields, flip the switches, press Run.**

## The controls

{control_table}

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

# ---------------------------------------------------------------- kinds

# A kind is one workflow shape: which plate it edits, which locked prompt it uses, and the
# ordered list of free-text fields the operator fills in. Everything downstream — layout,
# App Mode form, numbering — is derived from this.
KINDS = {
    "pose": {
        "title": "pose & scene",
        "plate_key": "plate",
        "lock_key": "pose_lock",
        "fields": [
            {"key": "shot", "label": "YOUR SHOT — what he is doing", "height": 300,
             "hint": "The one box you always edit. Say what he is *doing*, in one sentence."},
        ],
        "plate_clause": (
            "He is centred in frame, whole body inside the frame, on a single flat mid-grey "
            "backdrop of one even tone — no scenery, no horizon, no floor line, no cast shadow on "
            "the ground and no props except the ones named above."
        ),
        "scene_lead": (
            "He is standing inside the scene shown in image 2, at believable scale, lit by that "
            "scene's light and casting its shadows. The scene is:"
        ),
        "restate": "Above all, his body clearly shows this action:",
    },
    "headshot": {
        "title": "headshot & expression",
        "plate_key": "headshot_plate",
        "lock_key": "headshot_lock",
        "fields": [
            {"key": "expression", "label": "HIS EXPRESSION — what his face is doing", "height": 260,
             "hint": "Brows, eyes, mouth. One sentence, e.g. weary with heavy lids and a flat mouth."},
            {"key": "framing", "label": "FRAMING — crop and head angle", "height": 200,
             "hint": "How close, which way he faces, where he looks. In SCENE mode give him room — a tight crop leaves the scene nowhere to go."},
        ],
        "plate_clause": (
            "Behind him is a single flat mid-grey backdrop of one even tone — no scenery, no "
            "horizon and no props."
        ),
        "scene_lead": (
            "The background behind him is the scene shown in image 2, drawn in the same ink and "
            "halftone style, simplified and a little soft so his face stays the subject, but still "
            "clearly readable as that place. The scene is:"
        ),
        "restate": (
            "Above all, redraw his brows, eyes and mouth so his face clearly shows this "
            "expression, even though everything else about him is unchanged:"
        ),
    },
}

# ---------------------------------------------------------------- characters

SETH_LOOK = (
    "a lanky cartoon man with long wavy shoulder-length brown hair and a full bushy handlebar "
    "moustache, wearing a plain v-neck t-shirt of very dark olive charcoal — near-black, the same "
    "flat dark tone as the t-shirt in image 1"
)

CHARACTERS = {
    "seth": {
        "name": "Seth",
        "plate": "cast-seth-plate.png",
        "headshot_plate": "cast-seth-headshot.png",
        "scene_plate": "cast-seth-scene-empty-auditorium.png",
        "pose_lock": (
            "Redraw the character in image 1 as a single full-body cartoon figure, keeping his "
            f"identity exactly: {SETH_LOOK} — with warm grey trousers and black-and-white sneakers. "
            "Keep his face, hair, moustache, build and clothing identical to image 1, and keep "
            "every colour as muted and desaturated as image 1. The new drawing shows him as "
            "follows:"
        ),
        "headshot_lock": (
            "Redraw the character in image 1 as a head-and-shoulders portrait, keeping his "
            f"identity exactly: {SETH_LOOK}. Keep the shape of his face, his hair, his moustache "
            "and his colouring identical to image 1, keep both eyes and the whole moustache inside "
            "the frame, and keep every colour as muted and desaturated as image 1. The portrait "
            "shows him as follows:"
        ),
        "shot": (
            "he stands with both hands on his hips, chest out, one foot forward, looking off to "
            "one side, pleased with himself."
        ),
        "expression": "delighted — brows up, eyes wide, a big open grin under the moustache.",
        "framing": "head and shoulders, facing the viewer, chin level, filling most of the frame.",
        "scene": "the empty auditorium seen from the stage, house lights low, rows of seats receding.",
    },
}


CHARACTERS["wilson"] = {
    "name": "Wilson",
    "plate": "cast-wilson-plate.png",
    "headshot_plate": "cast-wilson-headshot.png",
    "scene_plate": "cast-scene-empty-stage.png",
    "pose_lock": (
        "Redraw the character in image 1 as a single full-body figure, keeping his identity exactly: "
        "a living picket fence: one wide wooden fence panel of vertical slats with pointed tops IS his whole body, with two segmented riveted metal arms attached at its side edges, two short stubby metal legs with rolled cuffs and heavy black boots beneath its bottom rail, and a battered wide-brimmed floppy canvas bucket hat resting on the picket points; in the deep flat shadow between hat brim and pickets float two simple cartoon eyes, white ovals with round dark pupils. He has no head, no neck, no torso and no waist — the panel is the body. Keep the picket panel, the bucket hat, the two cartoon eyes, the segmented metal "
        "arms and the black boots identical to image 1. Keep every colour as muted and desaturated as image 1. The new drawing shows him as follows:"
    ),
    "headshot_lock": (
        "Redraw the character in image 1 as a close portrait of the top of him, keeping his identity "
        "exactly: a living picket fence: one wide wooden fence panel of vertical slats with pointed tops IS his whole body, with two segmented riveted metal arms attached at its side edges, two short stubby metal legs with rolled cuffs and heavy black boots beneath its bottom rail, and a battered wide-brimmed floppy canvas bucket hat resting on the picket points; in the deep flat shadow between hat brim and pickets float two simple cartoon eyes, white ovals with round dark pupils. He has no head, no neck, no torso and no waist — the panel is the body. Crop in tight on the bucket hat, the pointed picket tops and the two cartoon eyes so they fill the whole frame — the arms, the legs and the boots are cropped away entirely. Keep the eyes "
        "floating in the deep flat shadow under the brim. He has no face beyond those two eyes — no "
        "nose, no mouth, no eyebrows. Keep every colour as muted and desaturated as image 1. The portrait shows him as follows:"
    ),
    "shot": "he leans his side against a tall fence post, one boot crossed over the other, relaxed.",
    "expression": "attentive — both eyes wide and level, the hat brim tipped a little forward.",
    "framing": "seen straight on, hat level, the picket tops square to the viewer.",
    # The render is cropped to this box (x, y, w, h at 1024x1024) and scaled back up.
    "headshot_crop": [230, 40, 570, 570],
    "restate_headshot": (
        "Above all, redraw the two cartoon eyes and the tilt of the hat so he clearly reads as:"
    ),
    "scene": "the empty vaudeville stage, footlights low, the curtain half drawn behind him.",
}

CHARACTERS["cadbury"] = {
    "name": "Cadbury",
    "plate": "cast-cadbury-plate.png",
    "headshot_plate": "cast-cadbury-headshot.png",
    "scene_plate": "cast-scene-empty-stage.png",
    "pose_lock": (
        "Redraw the character in image 1 as a single full-body cartoon figure, keeping his identity "
        "exactly: a thin, impeccably composed butler in a black tailcoat, waistcoat and bow tie, with slicked-back black hair and a neat pencil moustache, a white folded towel draped over one forearm. His left eye is a brass clockwork monocle — a mechanical camera-iris lens in an ornate rim, its fine chain running down to his waistcoat pocket — and it sits on the eye at the VIEWER'S RIGHT when he faces the camera. Keep the tailcoat, waistcoat, bow tie, hair, pencil moustache, the "
        "white towel over his forearm and the monocle identical to image 1. Keep every colour as muted and desaturated as image 1. The new drawing "
        "shows him as follows:"
    ),
    "headshot_lock": (
        "Redraw the character in image 1 as a head-and-shoulders portrait, keeping his identity "
        "exactly: a thin, impeccably composed butler in a black tailcoat, waistcoat and bow tie, with slicked-back black hair and a neat pencil moustache, a white folded towel draped over one forearm. His left eye is a brass clockwork monocle — a mechanical camera-iris lens in an ornate rim, its fine chain running down to his waistcoat pocket — and it sits on the eye at the VIEWER'S RIGHT when he faces the camera. Keep the shape of his face, his slicked hair, his pencil moustache and "
        "the brass monocle on the eye at the viewer's right identical to image 1. Keep every colour as muted and desaturated as image 1. The "
        "portrait shows him as follows:"
    ),
    "shot": "he presents a silver tray flat on one upturned palm, the other hand behind his back, mid-stride.",
    "expression": "disapproving — the brow above the monocle raised, mouth pressed to a thin line.",
    "framing": "head and shoulders, facing the viewer, chin level.",
    "scene": "the empty vaudeville stage, footlights low, the curtain half drawn behind him.",
}

CHARACTERS["ake"] = {
    "name": "Ake",
    "plate": "cast-ake-plate.png",
    "headshot_plate": "cast-ake-headshot.png",
    "scene_plate": "cast-scene-empty-stage.png",
    "pose_lock": (
        "Redraw the character in image 1 keeping his identity exactly: a plump goldfish inside a spherical glass fishbowl mounted on a small wheeled wooden dolly, with huge googly eyes, a drooping walrus moustache and prominent front teeth, wearing a small dark cap lettered FEW perched on top of the bowl. Keep the glass bowl, "
        "the wheeled dolly, the googly eyes, the walrus moustache, the buck teeth and the FEW cap "
        "identical to image 1. Keep every colour as muted and desaturated as image 1. The new drawing shows him as follows:"
    ),
    "headshot_lock": (
        "Redraw the character in image 1 as a close portrait of the bowl and the fish inside it, "
        "keeping his identity exactly: a plump goldfish inside a spherical glass fishbowl mounted on a small wheeled wooden dolly, with huge googly eyes, a drooping walrus moustache and prominent front teeth, wearing a small dark cap lettered FEW perched on top of the bowl. Keep the googly eyes, the drooping walrus moustache, "
        "the buck teeth and the FEW cap identical to image 1. Keep every colour as muted and desaturated as image 1. The portrait shows him as "
        "follows:"
    ),
    "shot": "the dolly rolls fast to the left, the bowl tipped back and the water sloshing up one side.",
    "expression": "alarmed — googly eyes bulging, the walrus moustache flared, buck teeth showing.",
    "framing": "close on the bowl, the fish filling it, facing the viewer.",
    "restate_headshot": (
        "Above all, redraw the fish's eyes, moustache and teeth so his face clearly shows this:"
    ),
    "scene": "the empty vaudeville stage, footlights low, the curtain half drawn behind him.",
}

# ---------------------------------------------------------------- the workflow

BLUE = ("#223", "#335")        # locked / canon
GREEN = ("#232", "#353")       # type here
GREY = ("#323", "#535")        # switches
ENGINE = ("#222", "#000")

GROUP_LOCK = "#3f789e"
GROUP_SHOT = "#4a7c3f"
GROUP_OUT = "#8a6d3b"
GROUP_ENGINE = "#444"


def control_table(kind, fields):
    rows = [f"| **{i} · {f['label'].split(' — ')[0]}** | {f['hint']} |"
            for i, f in enumerate(fields, start=1)]
    n = len(fields)
    rows += [
        f"| **{n+1} · SCENE** | OFF = flat mid-grey plate. ON = the scene plate plus your scene text. |",
        f"| **{n+2} · YOUR SCENE** | Where he is. Only read when SCENE is ON. |",
        f"| **{n+3} · SCENE PLATE** | The image 2 backdrop. Only read when SCENE is ON. |",
        f"| **{n+4} · TRANSPARENT PNG** | ON = cuts him out and saves an RGBA PNG. |",
        f"| **{n+5} · WIDTH / HEIGHT** | 1024×1024 by default. |",
        f"| **{n+6} · SEED + STEPS** | Re-roll here. 8 steps ships; 4 drafts with the 4-step LoRA. |",
    ]
    return "| Control | What it does |\n|---|---|\n" + "\n".join(rows)


def build(cid, spec, kind_id):
    kind = KINDS[kind_id]
    g = Graph()

    fields = kind["fields"]
    g.add("MarkdownNote", f"HOW TO USE — {spec['name']}", (-560, -620), (480, 900),
          {"text": HOW_TO_USE.format(name=spec["name"], kind_title=kind["title"],
                                     control_table=control_table(kind_id, fields))})

    # --- 1. the column you actually touch --------------------------------
    X, y = -40, -620
    field_nodes = []
    for i, f in enumerate(fields, start=1):
        node = g.add("PrimitiveStringMultiline", f"▶ {i} · {f['label']} (TYPE HERE)",
                     (X, y), (460, f["height"]), {"value": spec[f["key"]]},
                     outputs=[("STRING", "STRING")], color=GREEN)
        g.app_input(node, "value", value=f"{i} · {f['label']}")
        field_nodes.append(node)
        y += f["height"] + 30

    n = len(fields)
    scene_on = g.add("PrimitiveBoolean", f"▶ {n+1} · SCENE   off = flat grey plate · on = in a scene",
                     (X, y), (460, 80), {"value": False},
                     outputs=[("BOOLEAN", "BOOLEAN")], color=GREY)
    g.app_input(scene_on, "value", value=f"{n+1} · SCENE — off = flat grey plate, on = in a scene")
    y += 110
    scene_text = g.add("PrimitiveStringMultiline", f"▶ {n+2} · YOUR SCENE — where he is (only read when SCENE is on)",
                       (X, y), (460, 210), {"value": spec["scene"]},
                       outputs=[("STRING", "STRING")], color=GREEN)
    g.app_input(scene_text, "value", value=f"{n+2} · YOUR SCENE — only read when SCENE is on")
    y += 240
    scene_plate = g.add("LoadImage", f"▶ {n+3} · SCENE PLATE (image 2) — only read when SCENE is on",
                        (X, y), (460, 400), {"image": spec["scene_plate"], "upload": "image"},
                        outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    g.app_input(scene_plate, "image", image=f"{n+3} · SCENE PLATE — only read when SCENE is on")
    y += 430
    transparent_on = g.add("PrimitiveBoolean", f"▶ {n+4} · TRANSPARENT PNG   on = cut the background out",
                           (X, y), (460, 80), {"value": False},
                           outputs=[("BOOLEAN", "BOOLEAN")], color=GREY)
    g.app_input(transparent_on, "value", value=f"{n+4} · TRANSPARENT PNG — cut the background out")
    y += 110
    latent = g.add("EmptySD3LatentImage", f"▶ {n+5} · OUTPUT SIZE", (X, y), (460, 130),
                   {"width": 1024, "height": 1024, "batch_size": 1},
                   outputs=[("LATENT", "LATENT")], color=GREY)
    g.app_input(latent, "width", "height", width=f"{n+5} · WIDTH", height=f"{n+5} · HEIGHT")
    g.group("① YOUR SHOT · everything you touch is in this column, top to bottom",
            (X - 30, -690, 520, y + 830), GROUP_SHOT)

    # --- 2. character + style lock (collapsed; expand to re-canonise) ------
    LX, LY, DY = 1240, -620, 46
    char_lock = g.add("PrimitiveStringMultiline", f"CHARACTER LOCK — who {spec['name']} is",
                      (LX, LY), (430, 300), {"value": spec[kind["lock_key"]]},
                      outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    style_lock = g.add("PrimitiveStringMultiline", "STYLE LOCK — house style",
                       (LX, LY + DY), (430, 300), {"value": STYLE_LOCK},
                       outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    plate_clause = g.add("PrimitiveStringMultiline", "PLATE BACKGROUND — used when SCENE is off",
                         (LX, LY + 2 * DY), (430, 190), {"value": kind["plate_clause"]},
                         outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    scene_lead = g.add("PrimitiveStringMultiline", "SCENE LEAD-IN — used when SCENE is on",
                       (LX, LY + 3 * DY), (430, 190), {"value": kind["scene_lead"]},
                       outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    plate = g.add("LoadImage", f"CHARACTER PLATE — {spec['name']}'s reference (image 1)",
                  (LX, LY + 4 * DY), (430, 400),
                  {"image": spec[kind["plate_key"]], "upload": "image"},
                  outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], collapsed=True)

    # prompt assembly, collapsed — FINAL PROMPT is the one worth expanding to read
    row = 5
    acc = char_lock
    for f, node in zip(fields, field_nodes):
        acc = g.add("StringConcatenate", f"+ {f['key']}", (LX, LY + row * DY), (330, 150),
                    {"delimiter": " "},
                    links={"string_a": (acc, 0, "STRING", True),
                           "string_b": (node, 0, "STRING", True)},
                    outputs=[("STRING", "STRING")], collapsed=True)
        row += 1
    scene_clause = g.add("StringConcatenate", "scene lead-in + your scene",
                         (LX, LY + row * DY), (330, 150), {"delimiter": " "},
                         links={"string_a": (scene_lead, 0, "STRING", True),
                                "string_b": (scene_text, 0, "STRING", True)},
                         outputs=[("STRING", "STRING")], collapsed=True)
    row += 1
    bg_clause = g.add("ComfySwitchNode", "background text switch",
                      (LX, LY + row * DY), (330, 130), {"switch": False},
                      links={"on_false": (plate_clause, 0, "STRING", False),
                             "on_true": (scene_clause, 0, "STRING", False),
                             "switch": (scene_on, 0, "BOOLEAN", True)},
                      outputs=[("output", "STRING")], collapsed=True)
    row += 1
    acc = g.add("StringConcatenate", "+ background", (LX, LY + row * DY), (330, 150),
                {"delimiter": " "},
                links={"string_a": (acc, 0, "STRING", True),
                       "string_b": (bg_clause, 0, "STRING", True)},
                outputs=[("STRING", "STRING")], collapsed=True)
    row += 1
    # A second reference image pulls the model back toward image 1's face, so the primary
    # field is restated at the end of the prompt. Without this, SCENE mode ignored the
    # expression entirely and returned the reference grin.
    restate_lead = g.add("PrimitiveStringMultiline", "RESTATEMENT — keeps the first field from being ignored",
                         (LX, LY + row * DY), (430, 190),
                         {"value": spec.get(f"restate_{kind_id}", kind["restate"])},
                         outputs=[("STRING", "STRING")], color=BLUE, collapsed=True)
    row += 1
    restate = g.add("StringConcatenate", "restatement + first field",
                    (LX, LY + row * DY), (330, 150), {"delimiter": " "},
                    links={"string_a": (restate_lead, 0, "STRING", True),
                           "string_b": (field_nodes[0], 0, "STRING", True)},
                    outputs=[("STRING", "STRING")], collapsed=True)
    row += 1
    acc = g.add("StringConcatenate", "+ restatement", (LX, LY + row * DY), (330, 150),
                {"delimiter": " "},
                links={"string_a": (acc, 0, "STRING", True),
                       "string_b": (restate, 0, "STRING", True)},
                outputs=[("STRING", "STRING")], collapsed=True)
    row += 1
    prompt = g.add("StringConcatenate", "FINAL PROMPT — expand to read what the model got",
                   (LX, LY + row * DY), (330, 150), {"delimiter": " "},
                   links={"string_a": (acc, 0, "STRING", True),
                          "string_b": (style_lock, 0, "STRING", True)},
                   outputs=[("STRING", "STRING")], collapsed=True)
    row += 1
    g.group(f"② {spec['name'].upper()} — CHARACTER & STYLE LOCK · expand only to re-canonise him",
            (LX - 30, LY - 70, 480, row * DY + 90), GROUP_LOCK)

    # --- 3. engine (all collapsed) ---------------------------------------
    EX, EY = LX, LY + row * DY + 60
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

    # --- 4. sampling + output --------------------------------------------
    sampler = g.add("KSampler", f"▶ {n+6} · SEED + STEPS", (520, 640), (460, 270),
                    {"seed": 1, "control_after_generate": "randomize", "steps": STEPS, "cfg": 1.0,
                     "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0},
                    links={"model": (lora, 0, "MODEL", False),
                           "positive": (positive, 0, "CONDITIONING", False),
                           "negative": (negative, 0, "CONDITIONING", False),
                           "latent_image": (latent, 0, "LATENT", False)},
                    outputs=[("LATENT", "LATENT")])
    g.app_input(sampler, "seed", "steps",
                seed=f"{n+6} · SEED", steps=f"{n+6} · STEPS (8 ships, 4 drafts)")
    decode = g.add("VAEDecode", "decode", (520, 950), (400, 80),
                   links={"samples": (sampler, 0, "LATENT", False),
                          "vae": (vae, 0, "VAE", False)},
                   outputs=[("IMAGE", "IMAGE")], collapsed=True)
    # Some characters can't be framed by prompting alone — Wilson has no head, so the model
    # re-completes the whole figure however tightly the crop is described. For those, the
    # framing is done deterministically after the render.
    crop = spec.get(f"{kind_id}_crop")
    if crop:
        x, y, cw, ch = crop
        cropped = g.add("ImageCrop", "crop to the portrait", (520, 900), (380, 150),
                        {"width": cw, "height": ch, "x": x, "y": y},
                        links={"image": (decode, 0, "IMAGE", False)},
                        outputs=[("IMAGE", "IMAGE")], collapsed=True)
        decode = g.add("ImageScale", "back up to output size", (520, 925), (380, 150),
                       {"upscale_method": "lanczos", "width": 1024, "height": 1024,
                        "crop": "disabled"},
                       links={"image": (cropped, 0, "IMAGE", False)},
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
    g.app_output(g.add("SaveImage", "RESULT", (520, -620), (620, 1200),
                       {"filename_prefix": f"cast/{cid}-{kind_id}"},
                       links={"images": (final, 0, "IMAGE", False)}))
    g.group("③ RESULT · seed, steps and the finished image",
            (490, -690, 680, 1930), GROUP_OUT)

    return g


def main(argv):
    os.makedirs(os.path.join(HERE, "workflows"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "api"), exist_ok=True)
    ids = argv or list(CHARACTERS)
    for cid in ids:
        for kind_id in KINDS:
            g = build(cid, CHARACTERS[cid], kind_id)
            stem = f"{cid}-{kind_id}"
            ui = os.path.join(HERE, "workflows", f"{stem}.json")
            app = os.path.join(HERE, "workflows", f"{stem}.app.json")
            api = os.path.join(HERE, "api", f"{stem}.api.json")
            json.dump(g.to_ui(), open(ui, "w"), indent=1)
            json.dump(g.to_ui(), open(app, "w"), indent=1)
            json.dump(g.to_api(), open(api, "w"), indent=1)
            print(f"wrote {ui}\nwrote {app}\nwrote {api}")


if __name__ == "__main__":
    main(sys.argv[1:])
