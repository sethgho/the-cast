#!/usr/bin/env python3
"""Repaint the chosen sprite cells as drawings, instead of shipping video frames.

    python3 repaint_cells.py seth       # every move; cached cells are free

## Why

H3 gives motion; it does not give a clean drawing. Its frames carry codec speckle, soft coloured
edges and a background it repaints slightly differently every frame, and no amount of post
recovers ink that was never crisp. So the clip is demoted to a POSE SOURCE: the packer still picks
the cells off it, but each picked frame then goes back through Qwen-Image-Edit as an EDIT of
itself — same pose, same framing, redrawn.

The distinction that makes it work: the H3 frame is the *edit target*, not a reference image. A
reference image is an identity reference and Qwen will happily re-pose the character to suit the
prompt. Editing the frame in place keeps the pose and changes only the drawing. An earlier attempt
did it the other way round and came back crisp but re-posed, which is why this looked impossible.

Measured on the punch: edge sharpness 8.8 -> 14.9, and the magenta comes back flat, so the key
gets a hard boundary and cel_clean is not needed at all.

## The one thing to be careful about

Every cell is an independent generation, so the character comes back a few percent bigger or
smaller each time. One locked seed keeps the drawing consistent; `unify_height` in the packer
rescales each cell to the set median for moves whose height should not change (walk, idle). Moves
where the height IS the animation — jump, crouch — must not use it.

Cost: ~48s a cell. The seven-move set is 54 cells, about 45 minutes.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/wilson/scratch/local-cast")

import smoke_test as S  # noqa: E402
import sprite_sheet as SS  # noqa: E402
from build_workflows import UNET, LORA, CLIP, VAE  # noqa: E402

OUT = "/home/wilson/artifacts/cast-fighter/sprites"
REPAINT_DIR = "/tmp/repaint"
SEED = 77          # one seed for every cell, so the drawing itself does not wander
STEPS = 8
CELL = 512

# move -> (clip recipe in build_sprite.MOVES, cells, fps, loop, hold_key, unify)
#
# NOT seed values that stop mattering once a manifest exists. clip_path(), pick_frames() and
# sprite_editor.state() all read MOVES at runtime, keyed by move name, for the clip recipe and the
# fields bootstrap_manifest() would give a move that has never had a manifest entry. Renaming a
# tag here therefore still breaks clip_path() for that move, even though the tag itself carries
# its own `clip` field once written — the two are independent copies of the same fact.
#
# The recipe name is not always the move name: a sprite "walk" comes from the `walk-cycle` clip,
# which for a character with no legs reads as a roll instead. The clip path is derived from the
# character id, so nothing here is Seth-specific.
MOVES = {
    # unify="head" normalises the few percent of scale drift between independent repaints without
    # touching the stride's real rise and fall. unify=True (total height) is only for moves that
    # should not change height at all; jump and crouch must not unify, the height IS the animation.
    "walk":   ("walk-cycle",   10, 14, True,  False, "head"),
    # Idle is a breath. More cells does NOT buy more subtlety: asked for 16, the repaint returned
    # only 8 distinct drawings, because a locked seed maps two near-identical source frames onto
    # the same output. The extra cells became uneven holds, which reads as a stutter.
    "idle":   ("idle-breathe",  8,  8, True,  False, True),
    "punch":  ("punch",         8, 16, False, False, False),
    "kick":   ("kick",          8, 16, False, False, False),
    "jump":   ("jump",          8, 16, False, False, False),
    "block":  ("block",         6, 16, False, True,  False),
    "crouch": ("crouch",        6, 16, False, True,  False),
}
SIZE, CLIP_STEPS = 832, 20


def clip_path(cid, move):
    return f"/tmp/{cid}-{MOVES[move][0]}-{SIZE}-{CLIP_STEPS}.mp4"


# The repaint instruction. It says "repaint" three ways on purpose: the failure mode is Qwen
# treating the frame as inspiration and drawing a fresh pose, and every clause here exists to
# close that door.
REPAINT = (
    "Redraw the picture as clean crisp 1933 rubber-hose cartoon line art: confident smooth black "
    "ink outlines of even weight, sharp clear edges, soft halftone dot shading, muted sepia inks of "
    "aged newsprint, strong contrast, never pale or washed out. This is a repaint, not a new "
    "drawing: the man's pose, the angle of every limb, his size in the frame and his position in "
    "the frame are already exactly right and must not change at all. Keep the flat magenta "
    "background exactly as it is, one single solid magenta colour with nothing drawn on it. Only "
    "the drawing quality changes — remove all blur, smearing and video compression noise, and "
    "restate every edge as a crisp ink line."
)
NEG = ("blurry, smeared, soft focus, video artifacts, photographic, different pose, moved, "
       "resized, cropped")


def graph(image_name, seed=SEED):
    return {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": LORA, "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "pos": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": REPAINT,
                           "image1": ["img", 0]}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": NEG,
                           "image1": ["img", 0]}},
        "lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["img", 0], "vae": ["vae", 0]}},
        "k": {"class_type": "KSampler",
              "inputs": {"model": ["lora", 0], "positive": ["pos", 0], "negative": ["neg", 0],
                         "latent_image": ["lat", 0], "seed": seed, "steps": STEPS, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["vae", 0]}},
        "RESULT": {"class_type": "SaveImage",
                   "inputs": {"images": ["dec", 0], "filename_prefix": "cast/repaint"},
                   "_meta": {"title": "RESULT"}},
    }


PAD = 0.86     # how much of the frame the character is shrunk into before repainting


def pad_for_repaint(src, dst):
    """Shrink the frame into itself before handing it to the repaint.

    H3 draws the character filling the 832 frame, and the repaint reliably draws him a few percent
    LARGER than he arrived -- so his hair ran off the top edge. Eight of ten walk frames came back
    with a flat-topped head, which then poisoned everything downstream: a clipped silhouette has no
    true top, so the head-size measure and the alignment were both reading a truncated shape.

    Padding first costs a little resolution and buys a guaranteed margin. The key colour is
    measured from the frame rather than assumed, so the padding is the same magenta the model
    painted and the keyer cannot see a seam.
    """
    import numpy as np
    from PIL import Image
    im = Image.open(src).convert("RGB")
    key = tuple(int(v) for v in SS.measure_key(np.asarray(im).astype(np.float32)))
    w, h = im.size
    small = im.resize((int(w * PAD), int(h * PAD)), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), key)
    # centred horizontally, sitting low: the headroom is what we are buying
    canvas.paste(small, ((w - small.width) // 2, h - small.height - int(h * 0.02)))
    canvas.save(dst)
    return dst


def repaint(src, dst, seed=SEED):
    import run as R
    padded = pad_for_repaint(src, dst.replace(".png", "-padded.png"))
    g = graph(R.upload(padded), seed)
    for node in g.values():
        node.setdefault("_meta", {"title": "node"})
    pid = S.api("/prompt", {"prompt": g, "client_id": "repaint"})["prompt_id"]
    while True:
        hist = S.api(f"/history/{pid}")
        if pid in hist:
            break
        time.sleep(3)
    st = hist[pid]["status"]
    if st.get("status_str") != "success":
        raise SystemExit(f"{src}: FAILED {st}")
    im = hist[pid]["outputs"]["RESULT"]["images"][0]
    subprocess.run(["curl", "-s",
                    f"{S.HOST}/view?filename={im['filename']}&subfolder={im.get('subfolder','')}"
                    "&type=output", "-o", dst], check=True)
    return dst


def pick_frames(cid, move, clip, n_cells):
    """Let the packer choose the cells off the CLIP, then hand back their file paths.

    Cell choice needs the motion, so it stays on the video: gait period, pose extremes and the
    sharpest-frame nudge all read the sequence. Only the chosen frames get repainted.
    """
    SS.CEL_CLEAN = True                        # picking reads video frames, so clean them
    name = f"{cid}-{move}-pick"
    prep = SS._prepare(clip, name, n_cells, skip=6, cycle=MOVES[move][3], anchor="feet",
                       smooth=True, stop=MOVES[move][4])
    paths = sorted(os.listdir(f"/tmp/sprite-{name}"))
    paths = [os.path.join(f"/tmp/sprite-{name}", p) for p in paths if p.endswith(".png")][6:]
    return [paths[i] for i in prep["picks"]]


def match_palette(prep):
    """Pull every cell in a move onto the same colours.

    Each cell is an independent generation, so the ink and the flats drift a few percent between
    them — one jump cell came back with grey trousers among seven sepia ones, which flashes badly
    at 16fps. Inside the silhouette, each cell's per-channel mean is scaled to the set median.
    That corrects a global cast without touching the drawing.
    """
    import numpy as np
    from PIL import Image
    picks = prep["picks"]
    means = []
    for i in picks:
        a = np.asarray(prep["rgba"][i]).astype(np.float32)
        m = a[:, :, 3] > 128
        means.append(a[:, :, :3][m].mean(axis=0) if m.sum() > 100 else np.array([1.0, 1.0, 1.0]))
    target = np.median(np.stack(means), axis=0)
    for i, mean in zip(picks, means):
        gain = np.clip(target / np.maximum(mean, 1.0), 0.85, 1.18)
        a = np.asarray(prep["rgba"][i]).astype(np.float32)
        rgb = np.clip(a[:, :, :3] * gain, 0, 255)
        prep["rgba"][i] = Image.fromarray(
            np.dstack([rgb.astype(np.uint8), a[:, :, 3].astype(np.uint8)]), "RGBA")


MANIFEST_DIR = "/home/wilson/dev/the-cast/local-comfy/sheets"


def manifest_path(cid):
    return os.path.join(MANIFEST_DIR, f"{cid}.json")


def sheet_pivot():
    """The packer's feet line, seeded into the manifest as the sheet-level pivot."""
    return list(SS.cell_pivot(CELL, "feet"))


def _frame(src, seed, png=None, hold=1, pivot_nudge=(0, 0)):
    """One cell record, always the same shape and the same key order.

    `hold` is a DURATION multiplier over its tag's fps -- 2 means this drawing is on screen for
    two beats, which is how an animator sits on an extreme. It is not the tag's `hold_key`, which
    is the game's freeze-on-the-last-cell behaviour. The two were one word for a week and the
    words are now different on purpose.
    """
    f = {"src": src, "seed": seed}
    if png:
        f["png"] = png
    f["hold"] = int(hold)
    f["pivot_nudge"] = list(pivot_nudge)
    return f


def _migrated(cid):
    """Fold whatever `sheets/<cid>-<move>.json` files exist into one character manifest.

    A move is not a file: it is a NAMED RANGE of frames, which every sprite tool has called a tag
    for twenty years. The seven-way split also split the things that MUST agree across moves --
    one cell size, one pivot, one scale -- across seven files free to contradict each other.

    Frame order follows MOVES order, which is the order the packer already packed in, so nothing
    a cell was judged by moves under it.

    Returns (manifest_or_None, folded) where `folded` lists the moves actually folded in. A move
    whose file fails to parse is skipped, not fatal to the rest -- and it is NOT in `folded`, so
    the caller knows not to delete a file it never actually got the picks out of.
    """
    frames, tags, folded = [], [], []
    for move in MOVES:
        old = os.path.join(MANIFEST_DIR, f"{cid}-{move}.json")
        if not os.path.exists(old):
            continue
        try:
            man = json.load(open(old))
            new_frames = [_frame(c["src"], c["seed"], c.get("png")) for c in man["cells"]]
            start = len(frames)
            tag = {"name": move, "from": start, "to": start + len(new_frames) - 1,
                   "fps": man["fps"], "direction": "forward", "loop": man["loop"],
                   # The old per-move `hold` was the game semantic; it becomes `hold_key`.
                   "hold_key": man["hold"], "unify": man["unify"], "clip": man["clip"]}
        except (KeyError, ValueError) as e:
            print(f"  {cid}-{move}: {old} did not fold cleanly ({e}) -- leaving it on disk")
            continue
        frames.extend(new_frames)
        tags.append(tag)
        folded.append(move)
    man = {"character": cid, "cell": CELL, "pivot": sheet_pivot(), "tags": tags,
           "frames": frames} if tags else None
    return man, folded


def _backfill_missing_moves(cid, man):
    """Give every move in MOVES a tag, even one a partial migration or an older MOVES never wrote.

    `bootstrap_manifest` only runs when there is NO manifest at all, so a manifest that migrated
    from 3 of 7 per-move files -- or was written before an eighth move existed -- had no path back
    to the other four: they were gone the moment migration deleted the originals, and every run
    after that packed a permanently amputated character. This tops up whatever's missing the same
    way bootstrap would have built it from scratch. Returns True if it changed anything, so the
    caller only re-saves the manifest when there was something to add.
    """
    changed = False
    for move, (_, n, fps, loop, hold_key, unify) in MOVES.items():
        if _tag_index(man, move) is not None:
            continue
        start = len(man["frames"])
        for p in pick_frames(cid, move, clip_path(cid, move), n):
            man["frames"].append(_frame(p, SEED))
        man["tags"].append({"name": move, "from": start, "to": len(man["frames"]) - 1, "fps": fps,
                            "direction": "forward", "loop": loop, "hold_key": hold_key,
                            "unify": unify, "clip": clip_path(cid, move)})
        changed = True
    return changed


def load_character_manifest(cid):
    """The whole character in one file: `sheets/<cid>.json`.

    The cell list, once chosen, is an ARTEFACT -- not something recomputed every run. Automatic
    choice gets a move to about 90%: the right gait period, the right extremes, the sharpest frame
    of each. The last 10% is always per-cell and always a judgement -- this repaint garbled a hand,
    that one drifted in scale, this pose reads better two frames later. Recomputing the picks on
    every run threw those judgements away, so one bad cell cost a full re-pick and re-repaint.

    So the picks are written down, and the file wins over the picker whenever it exists. Every
    return path still runs through `_backfill_missing_moves` -- a manifest is never allowed to be
    permanently short a move that MOVES says should exist.
    """
    p = manifest_path(cid)
    if os.path.exists(p):
        man = json.load(open(p))
    else:
        man, folded = _migrated(cid)
        if not man:
            return None
        save_character_manifest(cid, man)
        # Only after the new file is on disk, and only the files that actually made it in: until
        # then, and for anything that didn't fold, the old file is the only copy of those picks.
        for move in folded:
            old = os.path.join(MANIFEST_DIR, f"{cid}-{move}.json")
            if os.path.exists(old):
                os.remove(old)
    if _backfill_missing_moves(cid, man):
        save_character_manifest(cid, man)
    return man


def save_character_manifest(cid, man):
    """Write `sheets/<cid>.json` atomically.

    This is the one file of record for a character: migration deletes the per-move originals once
    it lands, so a kill mid-write here truncates the only copy and every later run dies in
    json.load on the character for good. Serialise to a temp file in the same directory, so a
    partial write or a mid-serialise exception never touches the real path, then os.replace() it
    in -- a rename on the same filesystem is atomic, a partially-written file never is.
    """
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    p = manifest_path(cid)
    tmp = f"{p}.tmp{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(man, f, indent=1)
        os.replace(tmp, p)
    except Exception:
        # A mid-serialise failure (e.g. a bad value slipped into `man`) must not leave a stray
        # half-written file next to the real manifest for the next run to trip over.
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _tag_index(man, move):
    for i, t in enumerate(man["tags"]):
        if t["name"] == move:
            return i
    return None


def load_manifest(cid, move):
    """One tag, in the per-move shape sprite_editor.py still asks for.

    A VIEW of the character manifest, never a second store: there is one file on disk. The editor
    is rewritten against tags in a later stage, and it must not stop working in the meantime.
    """
    man = load_character_manifest(cid)
    i = _tag_index(man, move) if man else None
    if i is None:
        return None
    tag = man["tags"][i]
    return {"character": cid, "move": move,
            "cells": [dict(f) for f in man["frames"][tag["from"]:tag["to"] + 1]],
            "fps": tag["fps"], "loop": tag["loop"], "hold": tag["hold_key"],
            "unify": tag["unify"], "clip": tag["clip"]}


def save_manifest(cid, move, cells, meta):
    """Splice an edited move back in. Dropping or adding a cell moves every later tag, so the
    ranges after it are renumbered here -- a tag range left stale points at another move's cells.
    """
    man = load_character_manifest(cid)
    i = _tag_index(man, move) if man else None
    if i is None:
        raise SystemExit(f"{cid}: no {move} tag to write")
    tag = man["tags"][i]
    new = [_frame(c["src"], c["seed"], c.get("png"), c.get("hold", 1),
                  c.get("pivot_nudge", (0, 0))) for c in cells]
    shift = len(new) - (tag["to"] + 1 - tag["from"])
    man["frames"][tag["from"]:tag["to"] + 1] = new
    tag["to"] += shift
    for later in man["tags"][i + 1:]:
        later["from"] += shift
        later["to"] += shift
    for k in ("fps", "loop", "unify", "clip"):
        if k in meta:
            tag[k] = meta[k]
    if "hold" in meta:
        tag["hold_key"] = meta["hold"]
    save_character_manifest(cid, man)


def repaint_path(cid, move, src, seed):
    """One file per (source frame, seed), so a re-roll never overwrites the cell it replaces."""
    stem = os.path.basename(src).replace(".png", "")
    tag = "" if seed == SEED else f"-s{seed}"
    return os.path.join(REPAINT_DIR, f"{cid}-{move}-{stem}{tag}.png")


def bootstrap_manifest(cid):
    """A character with no manifest yet: choose every move's cells off its clip, in MOVES order.

    This is the only path that reads MOVES for anything but a clip path. Once the file exists the
    manifest is the truth and MOVES cannot silently overrule an edit.
    """
    frames, tags = [], []
    for move, (_, n, fps, loop, hold_key, unify) in MOVES.items():
        start = len(frames)
        for p in pick_frames(cid, move, clip_path(cid, move), n):
            frames.append(_frame(p, SEED))
        tags.append({"name": move, "from": start, "to": len(frames) - 1, "fps": fps,
                     "direction": "forward", "loop": loop, "hold_key": hold_key,
                     "unify": unify, "clip": clip_path(cid, move)})
    return {"character": cid, "cell": CELL, "pivot": sheet_pivot(), "tags": tags, "frames": frames}


def main():
    cid = sys.argv[1] if len(sys.argv) > 1 else "seth"
    os.makedirs(REPAINT_DIR, exist_ok=True)
    # Always every tag, never a subset. The cell scale is shared across the whole set -- that is
    # what stops the character resizing when the state machine switches move -- so packing one tag
    # on its own would silently rescale it against itself. Repaints are cached, so the moves you
    # did not touch cost nothing.
    man = load_character_manifest(cid) or bootstrap_manifest(cid)

    prepared = []
    for tag in man["tags"]:
        t0 = time.time()
        span = list(range(tag["from"], tag["to"] + 1))
        print(f"  {cid}-{tag['name']}: {len(span)} cells from the manifest", flush=True)
        for n, idx in enumerate(span, start=1):
            f = man["frames"][idx]
            dst = repaint_path(cid, tag["name"], f["src"], f["seed"])
            if not os.path.exists(dst):
                repaint(f["src"], dst, seed=f["seed"])
            man["frames"][idx] = _frame(f["src"], f["seed"], dst, f.get("hold", 1),
                                        f.get("pivot_nudge", (0, 0)))
            print(f"  {cid}-{tag['name']} cell {n}/{len(span)}  {time.time()-t0:.0f}s", flush=True)

        frames = man["frames"][tag["from"]:tag["to"] + 1]
        SS.CEL_CLEAN = False                   # repaints are drawn, not decoded
        prep = SS._prepare_stills([f["png"] for f in frames], smooth=True, unify=tag["unify"])
        match_palette(prep)
        prepared.append((tag, prep, frames))

    # Written before anything is packed. A pack can crash or be killed; if it does, the picks and
    # the repaint paths are still recorded and the next run costs nothing.
    save_character_manifest(cid, man)

    # One scale across the whole set, or he changes size when the state machine switches move.
    tallest = max(p["natural"] for _, p, _ in prepared)
    highest = max(p["up"] for _, p, _ in prepared)
    scale = min((CELL * 0.92) / tallest, (CELL * 0.88) / highest)

    SS.CEL_CLEAN = False
    # The per-move atlases are a bridge: cast-fighter.html and sprite_editor.py still load them.
    # They come off the same prep and the same scale as the tagged sheet, so the two agree cell
    # for cell until the consumers are moved over.
    meta = {}
    for tag, prep, frames in prepared:
        name = f"{cid}-{tag['name']}"
        SS._emit(name, prep, CELL, OUT, "feet", scale, unify_height=tag["unify"])
        meta[tag["name"]] = {"file": name, "frames": len(prep["picks"]), "fps": tag["fps"],
                             "loop": tag["loop"], "hold": tag["hold_key"]}

    path = os.path.join(OUT, f"{cid}-moves.json")
    json.dump(meta, open(path, "w"), indent=1)
    print("wrote", path)

    SS._emit_sheet(cid, prepared, CELL, OUT, scale, tuple(man["pivot"]))


if __name__ == "__main__":
    main()
