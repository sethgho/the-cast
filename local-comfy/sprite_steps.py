#!/usr/bin/env python3
"""The three upstream steps, each as one killable child process.

    python3 sprite_steps.py clip    <cid> <move> <result.json>
    python3 sprite_steps.py extract <cid> <move> <result.json>
    python3 sprite_steps.py picks   <cid> <move> <result.json>

## Why this is a separate process and not a function in the agent

`sprite_agent.py` already runs the packer as a child, and every reason applies twice as hard
here. A clip is ~170 seconds inside one blocking wait on gpu-worker; there is no flag any of
this could check often enough, and Python cannot interrupt a thread. So a cancel is a signal to
a process, and everything cancellable has to BE a process.

## Why the result comes back in a file

stdout is the job's log and the agent streams it straight through to the journal, which is the
only way to watch a 170-second render. So the machine-readable answer goes to a file the caller
names, written last, atomically — a partial result file read as a whole one would tell the
Durable Object that a killed job had picked cells it never picked.

## What each one is allowed to decide

Nothing about the manifest. `clip` and `extract` write FILES and report their hashes; `picks`
reports the source frames the picker chose and nothing else. Which cell keeps which hand edit is
the Durable Object's answer alone (`CharacterDO.reconcile`), because the edits are its data —
see DESIGN-pipeline.md, "hand edits under a re-run".
"""
import glob
import json
import os
import signal
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import repaint_cells as RC  # noqa: E402
import sprite_sheet as SS  # noqa: E402
import pipeline as PL  # noqa: E402
import smoke_test as S  # noqa: E402

# Where hires_sprite.py leaves the id of the prompt it queued, so that a cancel can reach the
# card rather than only the process waiting on it. Without it a cancelled clip stopped this
# process and left gpu-worker rendering 170 seconds of video for nobody — on a box whose whole
# job is shared with other tenants.
PID_FILE_ENV = "SPRITE_CLIP_PID_FILE"


def tag_of(cid, move):
    """This move's tag, read straight off the manifest the agent just wrote.

    Straight off disk rather than through `load_character_manifest`, which tops a manifest up with
    any move it is missing by PICKING CELLS off a clip. Two of the three steps here exist because
    that clip is about to change, so a backfill firing inside them would choose cells from the
    artifact they are replacing.
    """
    path = RC.manifest_path(cid)
    if not os.path.exists(path):
        raise SystemExit(f"no manifest for {cid}; the agent writes it before every job")
    for tag in json.load(open(path))["tags"]:
        if tag["name"] == move:
            return tag
    raise SystemExit(f"no {move} tag for {cid}")


def write_result(path, obj):
    """Atomically, for the same reason `save_character_manifest` is atomic: the caller reads this
    file to decide what happened, and a half-written one is a lie it cannot detect."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# --- clip -----------------------------------------------------------------------------------

def interrupt_comfy(pid_file):
    """Stop the render this process queued, and only that one.

    gpu-worker's ComfyUI is shared, so a bare /interrupt would kill whatever happened to be
    executing — which on that box is routinely somebody else's job. The prompt id is checked
    against the running one first, and a prompt still QUEUED is deleted rather than interrupted,
    because interrupting only ever touches the one that is executing.
    """
    try:
        pid = open(pid_file).read().strip()
    except OSError:
        return
    if not pid:
        return
    try:
        q = S.api("/queue")
        running = [r[1] for r in q.get("queue_running", []) if len(r) > 1]
        pending = [r[1] for r in q.get("queue_pending", []) if len(r) > 1]
        if pid in pending:
            S.api("/queue", {"delete": [pid]})
            print(f"  cancelled: removed queued prompt {pid}", flush=True)
        if pid in running:
            req = urllib.request.Request(S.HOST + "/interrupt", data=b"{}",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
            print(f"  cancelled: interrupted running prompt {pid}", flush=True)
    except Exception as e:                      # noqa: BLE001 - a cancel must never itself fail
        print(f"  could not reach ComfyUI to cancel: {type(e).__name__}: {e}", flush=True)


def run_clip(cid, move, result):
    """Render this move's clip, briefed from the MANIFEST.

    `hires_sprite.brief()` already prefers the tag's own `recipe_text`, `trait` and `cyclic` over
    the Python tables; it is called here with the tag's RECIPE, which is what names both the brief
    and the file, and the file it writes is checked against the path the tag records. That check
    is the whole reason this is not a bare subprocess call: a recipe rename that made the two
    disagree would leave the tag pointing at the previous render and the step reading fresh.
    """
    tag = tag_of(cid, move)
    want = tag["clip"]
    was = PL.hash_file(want)
    # Removed before the render, never after it. A cancel arriving in the same instant the render
    # completes would otherwise read a file this process had just deleted and give up quietly; a
    # stale id left behind is harmless, because `interrupt_comfy` checks it against what is
    # actually on gpu-worker's queue before it touches anything.
    pid_file = f"/tmp/sprite-clip-{cid}-{move}.pid"
    if os.path.exists(pid_file):
        os.remove(pid_file)
    os.environ[PID_FILE_ENV] = pid_file
    signal.signal(signal.SIGTERM, lambda *_: (interrupt_comfy(pid_file), sys.exit(143)))

    argv = [sys.argv[0], cid, tag["recipe"], str(RC.SIZE), str(RC.CLIP_STEPS)]
    old = sys.argv
    sys.argv = argv
    try:
        import runpy
        # In-process, not a grandchild: the agent cancels by signalling THIS pid, and a render
        # sitting behind one more fork would have carried on polling gpu-worker unreachably.
        runpy.run_path(os.path.join(HERE, "hires_sprite.py"), run_name="__main__")
    finally:
        sys.argv = old
    if not os.path.isfile(want):
        raise SystemExit(f"the render finished but {want} is not there — the tag's clip path and "
                         f"the recipe {tag['recipe']!r} disagree")
    now = PL.hash_file(want)
    write_result(result, {"clip": want, "hash": now, "changed": now != was})


# --- extract --------------------------------------------------------------------------------

def run_extract(cid, move, result):
    """Re-extract every frame of this move's clip into its pick directory.

    The directory is EMPTIED first. ffmpeg numbers from `f_0001` and overwrites, so a clip that
    got shorter would otherwise leave the tail of the previous render behind — frames a re-pick
    can still choose, and which the directory hash then counts as part of the current extraction.

    A changed extraction also invalidates every DRAWING of this move, and the packer caches
    repaints by file NAME: `f_0011.png` of the new clip repaints to the same path as `f_0011.png`
    of the old one, so without deleting them the next pack would silently reuse pictures of a
    performance that no longer exists. They are deleted only when the bytes actually moved, so
    re-extracting an unchanged clip stays free.
    """
    tag = tag_of(cid, move)
    clip = tag["clip"]
    if not os.path.isfile(clip):
        raise SystemExit(f"no clip at {clip} — render it before extracting from it")
    out = PL.frames_dir(cid, move)
    was = PL.hash_dir(out)
    os.makedirs(out, exist_ok=True)
    for old in glob.glob(os.path.join(out, "f_*.png")):
        os.remove(old)
    n = len(SS.extract_frames(clip, out))
    now = PL.hash_dir(out)
    cleared = []
    if now != was:
        cleared = sorted(glob.glob(os.path.join(RC.REPAINT_DIR, f"{cid}-{move}-*.png")))
        for png in cleared:
            os.remove(png)
    print(f"  {cid}-{move}: {n} frames, {'new' if now != was else 'identical'} extraction, "
          f"{len(cleared)} repaint(s) cleared", flush=True)
    write_result(result, {"dir": out, "frames": n, "hash": now, "changed": now != was,
                          "repaints_cleared": len(cleared)})


# --- picks ----------------------------------------------------------------------------------

def run_picks(cid, move, result):
    """Choose this move's cells off its clip, and report only the frames chosen.

    `pick_frames` re-extracts as part of measuring the motion, so the directory hash reported here
    is taken AFTER the pick and is the extraction the choice was really made against. That hash is
    what the Durable Object compares to decide whether a hand edit can follow its frame: equal
    means these are the same bytes the edits were made on, and anything else means a new
    performance and an honest orphaning.
    """
    tag = tag_of(cid, move)
    clip = tag["clip"]
    if not os.path.isfile(clip):
        raise SystemExit(f"no clip at {clip} — render it before picking cells from it")
    srcs = RC.pick_frames(cid, move, clip, tag["cells"], cycle=tag["cyclic"],
                          stop=tag["hold_key"])
    out = PL.frames_dir(cid, move)
    write_result(result, {"srcs": srcs, "frames_hash": PL.hash_dir(out),
                          "cells": len(srcs)})


STEPS = {"clip": run_clip, "extract": run_extract, "picks": run_picks}

if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in STEPS:
        raise SystemExit(f"usage: sprite_steps.py {{{'|'.join(STEPS)}}} <cid> <move> <result.json>")
    STEPS[sys.argv[1]](sys.argv[2], sys.argv[3], sys.argv[4])
