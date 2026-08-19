"""The seven-step pipeline as data: one step instance per thing that can be built or re-built.

    plate -+-> clip(move) -> frames(move) -> picks(move) -> repaint(cell) -+
           +-> (traits and recipes are inputs, not steps)      pack(char) <-+ -> export(format)

Seven step KINDS, fixed wiring, instances per (character, move) and per cell. It is a typed DAG
with a hardcoded shape, not a node canvas — see DESIGN-pipeline.md.

## What a step record is for

Each instance records what it was asked to make (`params`), what it was made FROM (`inputs`, the
content hash of each upstream artifact it actually consumed), the cache `key` those two compute
to, and the `built_key` the artifact on disk was really built at. `stale = key != built_key`,
propagated down every edge.

**Staleness never runs anything.** Nothing in this module builds, queues or prices work; it only
makes the state explicit so a person can look at it and decide.

## What is in the key, and what is deliberately not

IN: the step kind, its params, the content hashes of its inputs, and the prompt TEMPLATE version.
OUT: fps, hold, loop, direction, pivot, pivot_nudge, tag names, and every QC threshold.

That second list is playback metadata. The pack reads it; a clip has never heard of it. Putting
fps in a clip's key would mean re-tagging a move to 12fps invalidates a 170-second render, which
is the single most expensive wrong answer this file can give. The `pack` step is where all of it
belongs, and it is the only step whose key those fields enter.

## The two hash implementations

`celld-editor/worker.js` computes the same keys in the Durable Object, because the DO owns every
mutation and has to know what a mutation invalidated without asking a machine that may be asleep.
Both sides serialise through the canonical form in `canon.py` and hash it with SHA-256; the JS
copy carries the same rules and the same warnings. `test_pipeline_keys.py` runs the two against
the real manifests and fails if one digit differs.
"""
import hashlib
import os

import exports as EX
import repaint_cells as RC
from canon import DIGEST_CHARS, digest

# A rough price per run of one step, in seconds, so a rail can show a cost before it spends one.
# Measured, not guessed: a clip is the ~170s in the cast-sprites skill, a repaint the ~45s a cell.
# `frames` and `picks` are the extraction and the choice, both a few seconds of CPU. `export` is a
# pure function of the atlas that is computed on request, so it is free. Copied into the manifest's
# `build` block, because the Durable Object prices a step it has just invented and cannot import
# Python. Never in a cache key: a price is not an input.
COST_S = {"plate": 20, "clip": 170, "frames": 10, "picks": 5, "repaint": 45, "pack": 5,
          "export": 0}

# The key order every step record is written in, on BOTH sides. Fixed because this record is
# serialised straight into sheets/<cid>.json: a reordered key would rewrite every manifest on
# disk for no change in meaning, exactly as the frame record's order already guards against.
STEP_FIELDS = ("id", "kind", "tag", "params", "inputs", "key", "built_key", "artifact",
               "artifact_hash", "cost_s")


def edge(from_id, path, hash_):
    """One incoming edge: which step it came from, which file was read, and that file's hash.

    Three facts and only the third is key material. The `from` id and the `path` both carry the
    TAG NAME, and a tag name must never enter a cache key -- renaming `walk` to `stroll` would
    otherwise re-key its picks, all ten of its repaints and the pack, which is 8 minutes of GPU to
    answer a rename. `step_key` takes the hashes alone; the rest is here so a person, and the
    Durable Object, can see what fed what.
    """
    return {"from": from_id, "path": path, "hash": hash_}


def step_key(version, kind, params, inputs):
    """The cache key. The step's ID and its tag NAME are absent on purpose.

    A key addresses CONTENT, so two moves briefed with the same recipe on the same plate compute
    the same key and that is correct — renaming a tag must never invalidate anything.
    """
    return digest([version, kind, params, [e["hash"] for e in inputs]])


# --- artifact hashes ---------------------------------------------------------------------------

def hash_file(path):
    """The content hash of one artifact, or None when it does not exist yet.

    None is not an error: a step whose artifact has never been built has no hash, its key cannot
    match any built_key, and it reads stale. That is the correct answer for "not built".
    """
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:DIGEST_CHARS]


def hash_dir(path):
    """The content hash of a whole directory of frames, name by name.

    Genuinely reads every byte — the pick directories are ~800MB across both characters and it
    measures 0.7s warm, which is cheaper than the risk of a size-and-name fingerprint calling a
    re-extracted clip identical.
    """
    if not path or not os.path.isdir(path):
        return None
    names = sorted(n for n in os.listdir(path) if not n.startswith("."))
    return digest([[n, hash_file(os.path.join(path, n))] for n in names])


# --- where the artifacts live ------------------------------------------------------------------

def plate_path(cid):
    """The transparent cutout every clip of this character is composited from."""
    return os.path.join(RC.HERE, "plates", f"cast-cutout-{cid}.png")


def frames_dir(cid, move):
    """Where `pick_frames` extracted this move's clip to. Mirrors sprite_files.source_frames.

    This path is named after the TAG, which is the one place a tag name still reaches an artifact:
    renaming a move orphans its extracted frames, so `frames` and `picks` read stale until the
    clip is re-extracted (about 15 seconds), and the repaints under them inherit that. No clip and
    no repaint is re-keyed by it -- test_pipeline_keys.py checks exactly that -- so the 170-second
    render a rename must never touch is safe. Recording the directory on the tag the way `clip` is
    recorded would close it completely; nothing renames a tag yet, so it is written down here
    rather than fixed speculatively.
    """
    return f"/tmp/sprite-{cid}-{move}-pick"


def atlas_paths(cid):
    return os.path.join(RC.OUT, f"{cid}.png"), os.path.join(RC.OUT, f"{cid}.json")


# --- the step graph ------------------------------------------------------------------------------

def _record(sid, kind, tag, params, inputs, build, artifact, artifact_hash, built_key):
    return dict(zip(STEP_FIELDS, (
        sid, kind, tag, params, inputs,
        step_key(build["template_version"], kind, params, inputs), built_key,
        artifact, artifact_hash, build["cost_s"][kind])))


def repaint_id(png):
    """A repaint step is named by the file it makes, which is one per (source frame, seed).

    Not by the cell's position in its tag: a reorder or a drop moves every position along, and a
    step whose identity moved would lose the built_key that says its drawing is current.
    """
    return f"repaint:{os.path.basename(png)[:-len('.png')]}"


def build_steps(cid, man, built=()):
    """Every step instance for one character, in pipeline order.

    `built` names the step ids this run actually produced an artifact for; those get
    `built_key = key`. Every other step CARRIES ITS PREVIOUS built_key forward, and that carry is
    the whole point: the packer runs on every editor edit without re-rendering a clip, so
    stamping "built" from the mere existence of a file would quietly erase the staleness an edit
    had just created.

    A step with no previous record and an artifact already on disk is backfilled as built — the
    two characters that predate this file have current artifacts, and calling them stale would be
    a lie that costs 20 minutes of GPU to answer.
    """
    prior = {s["id"]: s for s in man.get("steps", [])}
    build = man["build"]
    steps = []

    def add(sid, kind, tag, params, inputs, artifact, artifact_hash, current=None):
        """One step record. `current` forces "this artifact is current" for the steps that are
        not built by a job at all — an uploaded plate, an export computed on request."""
        rec = _record(sid, kind, tag, params, inputs, build, artifact, artifact_hash, None)
        was = prior.get(sid)
        if current is not None:
            rec["built_key"] = rec["key"] if current else None
        elif sid in built:
            rec["built_key"] = rec["key"]
        elif was and was.get("built_key") is not None:
            rec["built_key"] = was["built_key"]
        elif artifact_hash is not None:
            # The backfill: an artifact on disk with no built_key anywhere is current.
            #
            # It reaches PAST an existing record on purpose, and only when that record says
            # "never built". A re-pick lands a cell on a source frame something already drew --
            # the file is named by (source frame, seed), so a cell that has been there before
            # comes back to its own drawing -- and the Durable Object, which cannot open a file,
            # necessarily invented that step with `built_key: null`. Carrying that null forward
            # read "never built" forever, because the packer paints only what is MISSING: running
            # repaint could not answer it, so the rail showed a permanent stale chip that no
            # amount of GPU would clear. A record that DOES carry a built_key is never touched
            # here, so a genuinely stale drawing stays stale.
            rec["built_key"] = rec["key"]
        steps.append(rec)
        return rec

    # --- plate. Uploaded, never computed, so the file IS the build: there is no job to run and
    # reporting it stale would be an instruction nobody can follow. Its content is its only param,
    # so replacing the plate changes its hash, which changes every clip's input and marks the
    # clips stale — which is the signal that actually matters.
    plate = plate_path(cid)
    phash = hash_file(plate)
    add("plate", "plate", None, {}, [edge(None, plate, phash)], plate, phash,
        current=phash is not None)

    for tag in man["tags"]:
        move = tag["name"]
        clip_id, frames_id, picks_id = f"clip:{move}", f"frames:{move}", f"picks:{move}"

        clip = tag["clip"]
        chash = hash_file(clip)
        add(clip_id, "clip", move,
            {"recipe": tag["recipe"], "recipe_text": tag["recipe_text"], "trait": man["trait"],
             "cyclic": tag["cyclic"], **build["clip"]},
            [edge("plate", plate, phash)], clip, chash)

        fdir = frames_dir(cid, move)
        fhash = hash_dir(fdir)
        add(frames_id, "frames", move, {"skip": build["picks"]["skip"]},
            [edge(clip_id, clip, chash)], fdir, fhash)

        # `cycle` and `stop` are pick-time facts about the MOTION -- is there a gait to detect,
        # and does the move end on a held pose -- and they are not the tag's `loop` and
        # `hold_key`, which are what the game does at playback. The two pairs happen to agree
        # today and were one field for a week; keeping them apart is what lets a re-tag leave the
        # picks alone.
        srcs = [f["src"] for f in man["frames"][tag["from"]:tag["to"] + 1]]
        src_hashes = {s: hash_file(s) for s in srcs}
        add(picks_id, "picks", move,
            {"cells": tag["cells"], "cycle": tag["cyclic"], "stop": tag["hold_key"],
             "anchor": build["picks"]["anchor"], "smooth": build["picks"]["smooth"]},
            [edge(frames_id, fdir, fhash)],
            # The chosen cells are written down IN the manifest, not in a file of their own; the
            # hash is over the chosen set, unordered, because the order within a tag is playback.
            RC.manifest_path(cid), digest(sorted(set(srcs))))

        for f in man["frames"][tag["from"]:tag["to"] + 1]:
            png = f.get("png") or RC.repaint_path(cid, move, f["src"], f["seed"])
            add(repaint_id(png), "repaint", move,
                {"seed": f["seed"], **build["repaint"]},
                # The edge carries the hash of what this step actually CONSUMED from its upstream
                # -- one source frame -- not the upstream's whole artifact. A repaint is cached per
                # (source frame, seed), so re-picking a different cell of the same tag must not
                # invalidate the drawings that did not move.
                [edge(picks_id, f["src"], src_hashes[f["src"]])],
                png, hash_file(png))

    # --- pack. Every field the key deliberately excludes upstream lands here, because this is the
    # step that actually reads them.
    png, js = atlas_paths(cid)
    atlas, table = hash_file(png), hash_file(js)
    repaints = [s for s in steps if s["kind"] == "repaint"]
    pack = add("pack", "pack", None,
        {"cell": man["cell"], "pivot": man["pivot"], **build["pack"],
         "tags": [{"name": t["name"], "fps": t["fps"], "direction": t["direction"],
                   "loop": t["loop"], "hold_key": t["hold_key"], "unify": t["unify"],
                   "cells": [{"hold": f.get("hold", 1),
                              "pivot_nudge": list(f.get("pivot_nudge", (0, 0)))}
                             for f in man["frames"][t["from"]:t["to"] + 1]]}
                  for t in man["tags"]]},
        [edge(s["id"], s["artifact"], s["artifact_hash"]) for s in repaints],
        # Both halves of the deliverable, in one hash: the atlas is a PNG and a JSON table, and a
        # run that wrote one without the other has not packed anything a consumer can read.
        png, None if atlas is None or table is None else digest([atlas, table]))

    for fmt in sorted(EX.FORMATS):
        # An export is computed on request straight off the atlas and streamed; there is no file
        # to go stale. So it is built exactly when the atlas it reads exists, and it goes stale
        # only by inheriting a stale pack.
        add(f"export:{fmt}", "export", None, {"format": fmt},
            [edge("pack", pack["artifact"], pack["artifact_hash"])], None, None,
            current=pack["artifact_hash"] is not None)
    return steps


# --- staleness ------------------------------------------------------------------------------------

def staleness(steps):
    """`stale` and why, for every step, with the answer propagated down every edge.

    Two independent reasons, and both are needed. A step goes stale on its OWN key when its params
    or an input's content changed. It goes stale by INHERITANCE when an upstream step is stale but
    its artifact has not been rebuilt yet — a re-worded recipe leaves the old mp4 byte-identical on
    disk, so no hash downstream of it moves, and only the propagation carries the news.

    Derived, never stored: a stale flag written into sheets/<cid>.json would be a second answer to
    a question the keys already answer, free to disagree with them.
    """
    out, order = {}, {s["id"]: s for s in steps}
    for s in steps:
        upstream = [e["from"] for e in s["inputs"] if e["from"] in order]
        if s["built_key"] is None:
            state = (True, "never built")
        elif s["key"] != s["built_key"]:
            state = (True, "params or inputs changed")
        else:
            behind = [u for u in upstream if out[u]["stale"]]
            state = (True, f"upstream is stale: {behind[0]}") if behind else (False, None)
        out[s["id"]] = {"stale": state[0], "reason": state[1]}
    return out


def summary(man):
    """One line per step kind: how many instances, how many stale. For humans and for tests."""
    steps = man.get("steps", [])
    stale = staleness(steps)
    rows = {}
    for s in steps:
        row = rows.setdefault(s["kind"], {"n": 0, "stale": 0, "cost_s": 0})
        row["n"] += 1
        if stale[s["id"]]["stale"]:
            row["stale"] += 1
            row["cost_s"] += s["cost_s"]
    return rows
