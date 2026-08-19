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
smaller each time. One locked seed keeps the drawing consistent; a tag's `unify`
field rescales each cell to the set median for tags whose height should not change (walk, idle).
Tags where the height IS the animation — jump, crouch — must not use it.

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
from canon import digest  # noqa: E402
import sprite_sheet as SS  # noqa: E402
import build_sprite as BS  # noqa: E402
from build_transition import TRAITS  # noqa: E402
from build_workflows import UNET, LORA, CLIP, VAE  # noqa: E402

OUT = "/home/wilson/artifacts/cast-fighter/sprites"
REPAINT_DIR = "/tmp/repaint"
SEED = 77          # one seed for every cell, so the drawing itself does not wander
STEPS = 8
CELL = 512

# The whole set is packed at ONE scale, which is what stops the character resizing when the state
# machine switches move: the tallest cell fills this much of the cell, and the highest reach fits
# in that much above the pivot. Named rather than inline because the pack step's cache key carries
# them -- changing either one really does invalidate the atlas.
SHEET_HEIGHT_CAP, SHEET_UP_CAP = 0.92, 0.88

# move -> (clip recipe in build_sprite.MOVES, cells, fps, loop, hold_key, unify)
#
# BOOTSTRAP DEFAULTS ONLY. Nothing at runtime reads this table for a move that already has a tag:
# a move is manifest data now, and its tag carries its own recipe, recipe text, cell count, fps,
# loop, hold_key, unify, cyclic flag and clip path. clip_path() returns the tag's `clip`, and
# pick_frames() is told the cycle and stop flags by its caller.
#
# This comment used to say the recipe existed here and in the tag as two independent copies free
# to disagree, so renaming a move broke its clip path while the tag looked fine. That is fixed:
# the table now only supplies the FIRST values a move that has never been built gets, and the
# manifest is the truth from the moment it is written.
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


def _pipeline():
    """The step graph, imported on demand.

    `pipeline.py` reads this module's paths, constants and prompt text, so importing it at module
    scope would be a cycle. Every caller in here goes through this one function rather than
    scattering local imports, so there is one place that says why.
    """
    import pipeline
    return pipeline


def bootstrap_clip_path(cid, recipe):
    """Where hires_sprite.py writes a clip. Only a move with no tag yet has to guess at it."""
    return f"/tmp/{cid}-{recipe}-{SIZE}-{CLIP_STEPS}.mp4"


def clip_path(cid, move, man=None):
    """This move's clip, from the manifest's own `clip` field wherever there is one.

    The MOVES fallback is reached only for a move that has never had a tag. Deriving the path from
    the recipe table instead was the bug the table's comment complained about: the tag recorded a
    `clip` of its own, so renaming a move made this function point at a file that did not exist
    while the tag still named the right one.
    """
    if man is not None:
        i = _tag_index(man, move)
        if i is not None:
            return man["tags"][i]["clip"]
    return bootstrap_clip_path(cid, MOVES[move][0])


def recipe_of_clip(cid, clip):
    """The recipe name a clip path encodes — the tag's own record of which brief rendered it."""
    stem = os.path.basename(clip)[:-len(".mp4")]
    return stem[len(f"{cid}-"):-len(f"-{SIZE}-{CLIP_STEPS}")]


def is_cyclic(recipe):
    """Does this brief end where it began? Mirrors hires_sprite.py, which pins one-shots only.

    A cycle's clip must be left unpinned (see the cast-sprites skill: pinning Cadbury's walk took
    self-similarity from 0.0061 to 0.1223 against a 0.139 random baseline), and its cells must be
    picked by gait period rather than spaced by motion. Both readings are this one fact, which is
    why it is a tag field and not two.
    """
    return recipe.endswith("-cycle") or recipe == "idle-breathe"


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
        # RuntimeError, not SystemExit: this is a library function called from a worker thread
        # (sprite_agent.runner()). SystemExit is a BaseException, which used to slip past a
        # bare `except Exception` in the worker and silently kill the thread — the job stayed
        # "running" forever and every job queued after it never ran.
        raise RuntimeError(f"{src}: FAILED {st}")
    im = hist[pid]["outputs"]["RESULT"]["images"][0]
    subprocess.run(["curl", "-s",
                    f"{S.HOST}/view?filename={im['filename']}&subfolder={im.get('subfolder','')}"
                    "&type=output", "-o", dst], check=True)
    return dst


PICK_SKIP, PICK_ANCHOR, PICK_SMOOTH = 6, "feet", True


def pick_frames(cid, move, clip, n_cells, cycle, stop):
    """Let the packer choose the cells off the CLIP, then hand back their file paths.

    Cell choice needs the motion, so it stays on the video: gait period, pose extremes and the
    sharpest-frame nudge all read the sequence. Only the chosen frames get repainted.

    `cycle` and `stop` are facts about the MOTION -- is there a gait to detect, and does the move
    end on a held pose -- and they arrive from the caller rather than from MOVES. They used to be
    read as the tag's `loop` and `hold_key`, which are what the GAME does at playback; the two
    pairs agree today, and conflating them meant re-tagging a move for playback changed which
    frames got picked.
    """
    SS.CEL_CLEAN = True                        # picking reads video frames, so clean them
    name = f"{cid}-{move}-pick"
    prep = SS._prepare(clip, name, n_cells, skip=PICK_SKIP, cycle=cycle, anchor=PICK_ANCHOR,
                       smooth=PICK_SMOOTH, stop=stop)
    paths = sorted(os.listdir(f"/tmp/sprite-{name}"))
    paths = [os.path.join(f"/tmp/sprite-{name}", p)
             for p in paths if p.endswith(".png")][PICK_SKIP:]
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


# The order sheets/<cid>.json is written in. Fixed in one place because both this module and the
# Durable Object write this file: a manifest that came back from the DO with `trait` appended
# after `frames` would otherwise re-order the whole file on the next local run, for no change in
# meaning, and every diff of a real edit would be buried in it.
# `name` and `plate` are only ever present on a character CREATED from an uploaded image
# (DESIGN-pipeline.md, "a new subject"). Both are absent from seth's and cadbury's manifests,
# whose cutouts were hand-made, and `_ordered` only emits the keys a manifest actually has -- so
# naming them here re-orders nothing that already exists on disk.
MANIFEST_FIELDS = ("character", "name", "cell", "pivot", "trait", "plate", "build", "tags",
                   "frames", "steps")


def _ordered(man):
    known = {k: man[k] for k in MANIFEST_FIELDS if k in man}
    # Anything this module has not heard of is kept, at the end. Dropping it would silently
    # discard a field a newer Worker had added.
    known.update({k: v for k, v in man.items() if k not in known})
    return known


def _tag(name, start, end, recipe, fps, loop, hold_key, unify, clip, cells=None,
         recipe_text=None):
    """One move, as manifest data, always the same shape and the same key order.

    A move is `{name, recipe_text, cells, fps, loop, hold_key, unify, cyclic}` plus the range it
    owns and the clip it came from. The recipe and its text live HERE, not in MOVES: that table is
    the bootstrap default a brand-new move gets, and from this record on the manifest is the truth.

    The new fields are appended after the old ones on purpose. This record is serialised straight
    into sheets/<cid>.json and mirrored by the Durable Object, so inserting a key in the middle
    would rewrite every tag of every manifest for no change in meaning.
    """
    return {"name": name, "from": start, "to": end, "fps": fps, "direction": "forward",
            "loop": loop, "hold_key": hold_key, "unify": unify, "clip": clip,
            "recipe": recipe, "recipe_text": recipe_text or BS.MOVES[recipe],
            # How many cells the next picks run should choose, which is NOT the range's width once
            # an empty tag exists: a move whose clip has never been rendered owns no frames at all,
            # and reading its cell count off `end - start + 1` would ask the picker for zero.
            "cells": cells if cells is not None else end - start + 1,
            "cyclic": is_cyclic(recipe)}


def build_block():
    """The locked constants every step's cache key is computed against.

    Carried IN the manifest rather than read from this module by both halves, because the Durable
    Object recomputes the same keys and has no way to import Python. One copy, pushed to the DO
    with the manifest, is what stops the two sides drifting into disagreeing about which steps are
    stale. Every number here is measured; the cast-sprites skill says why each one is what it is.
    """
    return {
        # One digest over every prompt TEMPLATE, so re-wording a lock invalidates the clips and
        # the repaints that were briefed with the old words -- and nothing else. It is in the key;
        # a move's own recipe text is a param, because that is data, not template.
        "template_version": digest([BS.SPRITE_LOCK, BS.WHO_LEAD, BS.STAGE_RESTATE,
                                    BS.SOUND_LOCK, REPAINT, NEG]),
        "clip": {"size": SIZE, "steps": CLIP_STEPS, "length": BS.LENGTH, "seed": BS.CLIP_SEED},
        "picks": {"skip": PICK_SKIP, "anchor": PICK_ANCHOR, "smooth": PICK_SMOOTH},
        "repaint": {"pad": PAD, "steps": STEPS, "cfg": 1.0, "sampler": "euler",
                    "scheduler": "simple", "denoise": 1.0},
        "pack": {"columns": SS.SHEET_COLUMNS, "height_cap": SHEET_HEIGHT_CAP,
                 "up_cap": SHEET_UP_CAP},
        # Prices, not key material. Carried here because the Durable Object has to price a step it
        # has just invented -- a re-roll makes a repaint instance that has never existed -- and it
        # cannot import Python to ask.
        "cost_s": dict(_pipeline().COST_S),
    }


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
            # The old per-move `hold` was the game semantic; it becomes `hold_key`.
            tag = _tag(move, start, start + len(new_frames) - 1,
                       recipe_of_clip(cid, man["clip"]), man["fps"], man["loop"], man["hold"],
                       man["unify"], man["clip"])
        except (KeyError, ValueError) as e:
            print(f"  {cid}-{move}: {old} did not fold cleanly ({e}) -- leaving it on disk")
            continue
        frames.extend(new_frames)
        tags.append(tag)
        folded.append(move)
    return (_character(cid, tags, frames) if tags else None), folded


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
    for move, (recipe, n, fps, loop, hold_key, unify) in MOVES.items():
        if _tag_index(man, move) is not None:
            continue
        start = len(man["frames"])
        clip = clip_path(cid, move, man)
        # A character created from an uploaded plate has no clips at all yet, so there is nothing
        # to pick cells off. That move gets an EMPTY tag -- `from` one past `to`, which slices to
        # no frames -- and the rail then prices it as a to-do rather than this dying on a missing
        # mp4. Picking against a clip that is not there was the only behaviour before, which is
        # why a new subject could not exist without someone rendering seven clips by hand first.
        picks = pick_frames(cid, move, clip, n, cycle=is_cyclic(recipe),
                            stop=hold_key) if os.path.isfile(clip) else []
        for p in picks:
            man["frames"].append(_frame(p, SEED))
        man["tags"].append(_tag(move, start, len(man["frames"]) - 1, recipe, fps, loop, hold_key,
                                unify, clip, cells=n))
        changed = True
    return changed


def _backfill_move_data(cid, man):
    """Top a manifest written before a move was data up to the current shape.

    A tag used to record only how it PLAYS -- fps, loop, hold_key, unify -- and left the brief that
    rendered it in a Python table. It now carries the recipe and its text as well, and the
    character carries his own trait line, so a rename or a re-word cannot leave the two disagreeing.
    The recipe name is recovered from the tag's own `clip` path, which has always encoded it, so no
    move has to be matched back to MOVES by name.

    The locked build constants are REFRESHED, not defaulted: they are still owned by this module,
    and the manifest carries a copy only so the Durable Object can compute the same cache keys.
    Returns True when it changed something, so a manifest that is already current is not rewritten.
    """
    before = json.dumps(man, sort_keys=True)
    for tag in man["tags"]:
        if "recipe" not in tag:
            recipe = recipe_of_clip(cid, tag["clip"])
            tag.update({"recipe": recipe, "recipe_text": BS.MOVES[recipe],
                        "cells": tag["to"] - tag["from"] + 1, "cyclic": is_cyclic(recipe)})
    man.setdefault("trait", TRAITS.get(cid, ""))
    man["build"] = build_block()
    return json.dumps(man, sort_keys=True) != before


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
    topped_up = _backfill_move_data(cid, man)
    if _backfill_missing_moves(cid, man) or topped_up:
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
            json.dump(_ordered(man), f, indent=1)
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


def repaint_path(cid, move, src, seed):
    """One file per (source frame, seed), so a re-roll never overwrites the cell it replaces."""
    stem = os.path.basename(src).replace(".png", "")
    tag = "" if seed == SEED else f"-s{seed}"
    return os.path.join(REPAINT_DIR, f"{cid}-{move}-{stem}{tag}.png")


def _character(cid, tags, frames):
    """The whole-character record, always the same shape and the same key order.

    `trait` is the character's own one-sentence description of how he moves, and it is manifest
    data for the same reason a recipe is: Ake has no legs, so "walk cycle" has to read as rolling
    for him, and that fact belongs to the character rather than to a table in a Python file. A
    character with no line of his own yet starts from build_transition.TRAITS.
    """
    return _ordered({"character": cid, "cell": CELL, "pivot": sheet_pivot(),
                     "trait": TRAITS.get(cid, ""), "build": build_block(),
                     "tags": tags, "frames": frames})


def bootstrap_manifest(cid):
    """A character with no manifest yet: choose every move's cells off its clip, in MOVES order.

    This is the only path that reads MOVES at all. Once the file exists the manifest is the truth
    and MOVES cannot silently overrule an edit.
    """
    frames, tags = [], []
    for move, (recipe, n, fps, loop, hold_key, unify) in MOVES.items():
        start = len(frames)
        clip = bootstrap_clip_path(cid, recipe)
        picks = pick_frames(cid, move, clip, n, cycle=is_cyclic(recipe),
                            stop=hold_key) if os.path.isfile(clip) else []
        for p in picks:
            frames.append(_frame(p, SEED))
        tags.append(_tag(move, start, len(frames) - 1, recipe, fps, loop, hold_key, unify, clip,
                         cells=n))
    return _character(cid, tags, frames)


def print_steps(cid, rows):
    """One line per step kind: how many instances, how many stale and what re-running them costs."""
    total = sum(r["cost_s"] for r in rows.values())
    print(f"STEPS {cid}: " +
          "  ".join(f"{k} {r['n'] - r['stale']}/{r['n']} fresh" for k, r in rows.items()) +
          (f"  -- rebuilding the stale ones costs about {total}s" if total else ""))


def main():
    cid = sys.argv[1] if len(sys.argv) > 1 else "seth"
    os.makedirs(REPAINT_DIR, exist_ok=True)
    PL = _pipeline()
    # Always every tag, never a subset. The cell scale is shared across the whole set -- that is
    # what stops the character resizing when the state machine switches move -- so packing one tag
    # on its own would silently rescale it against itself. Repaints are cached, so the moves you
    # did not touch cost nothing.
    man = load_character_manifest(cid) or bootstrap_manifest(cid)

    # Which steps this run really produced an artifact for, so their `built_key` is stamped at the
    # key they were built at. Everything else carries its old built_key forward -- see
    # pipeline.build_steps: a run that packs without re-rendering must not clear the staleness an
    # edit created.
    built = {"pack"}
    prepared = []
    for tag in man["tags"]:
        t0 = time.time()
        span = list(range(tag["from"], tag["to"] + 1))
        # An EMPTY tag: a move that exists as manifest data and whose clip has never been
        # rendered. It is skipped rather than packed, so it contributes no cell to the atlas and
        # no row to the sheet table -- which is what keeps a half-built character's atlas the
        # honest packing of the moves that DO have drawings.
        if not span:
            print(f"  {cid}-{tag['name']}: no cells yet — nothing to pack", flush=True)
            continue
        print(f"  {cid}-{tag['name']}: {len(span)} cells from the manifest", flush=True)
        for n, idx in enumerate(span, start=1):
            f = man["frames"][idx]
            dst = repaint_path(cid, tag["name"], f["src"], f["seed"])
            if not os.path.exists(dst):
                repaint(f["src"], dst, seed=f["seed"])
                built.add(PL.repaint_id(dst))
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

    # A brand-new character, whose every move is still an empty tag. There is no atlas to emit and
    # `pack` must NOT be stamped built, or the rail would call a character with no drawings fresh.
    # The step records are still written, because the whole point of the rail on a new character
    # is that it prices what has not been made yet.
    if not prepared:
        built.discard("pack")
        man["steps"] = PL.build_steps(cid, man, built)
        save_character_manifest(cid, man)
        print(f"  {cid}: no move has any cells yet — nothing packed", flush=True)
        print_steps(cid, PL.summary(man))
        return

    # One scale across the whole set, or he changes size when the state machine switches move.
    tallest = max(p["natural"] for _, p, _ in prepared)
    highest = max(p["up"] for _, p, _ in prepared)
    scale = min((CELL * SHEET_HEIGHT_CAP) / tallest, (CELL * SHEET_UP_CAP) / highest)

    SS.CEL_CLEAN = False
    # One atlas, one JSON, and nothing else. The per-move atlases and the `<cid>-moves.json` that
    # stitched them back together are gone: both consumers -- cast-fighter.html and
    # sprite_files.py -- read the tagged sheet now, and a second copy of the same cells is only
    # somewhere for the two to disagree.
    SS._emit_sheet(cid, prepared, CELL, OUT, scale, tuple(man["pivot"]))

    # Measured off the atlas that was just written, never fed back into it. Numbers, not eyes:
    # every one of these has caught a real defect here after it shipped.
    SS.print_qc(SS.sheet_qc(cid, prepared, CELL, OUT, tuple(man["pivot"])))

    # Last, because every step's record carries the content hash of an artifact, and the atlas is
    # only final once the emit above has written it. Nothing reads these yet and nothing runs off
    # them -- they make the pipeline's state explicit, no more (DESIGN-pipeline.md, stage 1).
    man["steps"] = PL.build_steps(cid, man, built)
    save_character_manifest(cid, man)
    print_steps(cid, PL.summary(man))


if __name__ == "__main__":
    main()
