#!/usr/bin/env python3
"""The wilson half of the sprite editor: the GPU, the packer, and the files.

    python3 sprite_agent.py        # long-polls celld, serves images on :8812

## Why there are two halves

The editor used to be one Python process that owned everything — the page, the manifest, the
queue and the pixels. Stage 6 of DESIGN-editor.md split it along the only line that matters:
what can run in V8 and what cannot. The manifest and the job queue are pure state, so they moved
into a Durable Object on the celld `sprites` fleet, where two browsers can no longer splice the
same frame list or queue the same 45-second GPU job, and the page moved with them. The repaint
drives ComfyUI and the packer is numpy and PIL, so both stay here, forever.

There is no second editor any more, and that is not tidiness. Both halves wrote
`sheets/<cid>.json`, this process overwrites that file with the DO's copy at the start of every
job, and the two job queues could not see each other — so an edit made in the old page was
destroyed by the next celld edit, and a re-roll in each could put two jobs on one 12GB card.

This process therefore does exactly two things:

1. Long-polls each character's Durable Object for work. When it gets a job it writes the DO's
   manifest to `sheets/<cid>.json`, runs `repaint_cells.py` as a child process, and reports the
   result. It never decides what the manifest should say — the DO owns that, and the only thing
   reported back is the `png` path the packer filled in for each frame. The DO's queue can hold
   several jobs; it only ever offers the head of one, so this process is never asked to run two.
2. Serves the files the page needs to look at: the packed cells, the source frames and the
   atlas, read-only and cross-origin, because the page is served from another host. It also
   ACCEPTS one thing -- the image a new character is cut out of -- because a plate has to land on
   the machine with the GPU and the disk, and there is nowhere else for it to go. That is the
   only write route, and `sprite_files.stage_upload` is what makes it a bounded one.

Both halves are answers about FILES, so the answers themselves live in `sprite_files.py` — the
path allowlist, the atlas crop, the metrics and the variant listing. Keeping them out of this
file keeps the HTTP surface from growing a second, looser copy of the allowlist.
"""
import io
import json
import os
import secrets
import socketserver
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import http.server

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exports as EX  # noqa: E402
import pipeline as PL  # noqa: E402
import repaint_cells as RC  # noqa: E402
import sprite_files as SF  # noqa: E402

PORT = int(os.environ.get("SPRITE_AGENT_PORT", "8812"))
CELLD = os.environ.get("SPRITE_CELLD_BASE", "http://192.168.0.19:8087").rstrip("/")
TOKEN = os.environ.get("SPRITE_AGENT_TOKEN", "")

# The DO holds a claim open for up to 20s. Asking for slightly less keeps the answer inside the
# HTTP timeout below, so a poll that finds no work returns rather than raising every cycle.
CLAIM_WAIT_MS = 18000
HTTP_TIMEOUT = 30

# The heartbeat is two things at once, and the second is why it is seconds rather than a minute:
# it renews the running lease, AND its response is the only channel a cancel has to reach a job
# that is already on the GPU. The DO gives a running job a 60s lease, so this is twelve beats of
# headroom, and a cancel is felt within one beat instead of within a paint.
HEARTBEAT_S = 5

# This process's identity, new on every start. A claim carries it so the DO can tell "the agent
# is holding this job" from "the agent that held this job is gone": celld drops a long poll when
# its client disconnects without telling the object, so a restart used to leave the head of the
# queue claimed by nobody for the full ten-minute lease. Only correct while exactly one agent
# process runs per fleet, which is what the systemd unit guarantees.
RUN_ID = secrets.token_hex(8)


def celld(path, body=None, timeout=HTTP_TIMEOUT):
    """One call to the fleet. Every agent route is token-authenticated and POSTs its `cid`."""
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{CELLD}/api/agent/{path}", data=data, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# --- what the page cannot ask the Durable Object -------------------------------------------
# All three of these are answers about FILES: where the packer put a cell, what that cell
# measures, and which repaints already exist. The DO has none of them and should not.


def catalogue(cid, man):
    """The choices a mutation is allowed to make, as the filesystem currently offers them.

    The DO validates a re-pick against the tag's pick directory and a variant switch against the
    repaints on disk, exactly as the Python editor did — but it cannot read a directory, so the
    listing is pushed to it with every seed and every finished job. Sending it rather than
    trusting the page is the point: the page could otherwise name any path it liked.
    """
    sources, variants = {}, {}
    for tag in man["tags"]:
        sources[tag["name"]] = SF.source_frames(cid, tag["name"])
        for f in man["frames"][tag["from"]:tag["to"] + 1]:
            variants[f["src"]] = SF.variants(cid, tag["name"], f["src"], f.get("prompt"))
    return {"sources": sources, "variants": variants}


def derived(cid, man):
    """Where each of the manifest's cells actually landed, and what it measures there.

    A cell is a crop out of one atlas, not a file, so its position is build OUTPUT and belongs
    with the build. `atlas_base` refuses to guess when the packer emitted fewer cells than the
    tag has frames, and that refusal is carried through as a null rather than an off-by-one:
    showing the wrong drawing beside the wrong source frame defeats the whole editor.
    """
    sh = SF.sheet(cid)
    tags = {}
    for tag in man["tags"]:
        span = list(range(tag["from"], tag["to"] + 1))
        base = SF.atlas_base(sh, tag["name"], len(span)) if sh else None
        cells = []
        for i, fi in enumerate(span):
            f = man["frames"][fi]
            ai = None if base is None else base + i
            cells.append({
                "atlas": ai,
                "xy": None if ai is None else [sh["frames"][ai]["x"], sh["frames"][ai]["y"]],
                "metrics": None if ai is None else SF.cell_metrics(cid, ai),
                "variants": SF.variants(cid, tag["name"], f["src"], f.get("prompt"))})
        tags[tag["name"]] = {"cells": cells}
    return {"atlas": SF.sheet_paths(cid)[0], "outdir": RC.OUT, "tags": tags, "locks": LOCKS}


def defaults():
    """What a character who does not exist yet is made of.

    A new subject is created by the PAGE against the Durable Object (DESIGN-pipeline.md, "a new
    subject"), and the DO cannot import Python — so the bootstrap defaults have to travel. What
    travels is exactly what `repaint_cells` already owns: the cell size, the sheet pivot, the
    locked build block every cache key is computed against, and the seven moves from `MOVES` with
    their recipe text from `build_sprite.MOVES`.

    The DO validates every field of this before it seeds anything, so nothing here is trusted for
    being ours. The build block is the one part that is taken as given, and it is self-healing:
    `_backfill_move_data` refreshes it from this module on every load, and every finished job
    reports it back, so a wrong one survives exactly until the character's first job.

    `recipes` and `traits` are for the page's own forms — the recipe library a new move is written
    against, and the existing trait lines shown as examples of the one sentence a person has to
    write. Neither is manifest data.
    """
    moves = {}
    for move, (recipe, n, fps, loop, hold_key, unify) in RC.MOVES.items():
        moves[move] = {"recipe": recipe, "recipe_text": RC.BS.MOVES[recipe], "cells": n,
                       "fps": fps, "loop": loop, "hold_key": hold_key, "unify": unify,
                       "cyclic": RC.is_cyclic(recipe)}
    return {"cell": RC.CELL, "pivot": RC.sheet_pivot(), "build": RC.build_block(),
            "moves": moves, "recipes": dict(RC.BS.MOVES), "traits": dict(RC.TRAITS)}


# --- the gait curve -------------------------------------------------------------------------
# The picks step chooses cells INSIDE one gait period, and until now the only evidence of that
# period was a line in a log. The workbench draws it under the picks filmstrip, so the curve and
# the detected period have to be askable. Both are read-only measurements of files that already
# exist: no GPU, no manifest, nothing written.

# 48x48 is what `sprite_sheet.silhouettes` compares poses at, so the numbers here are the same
# numbers `find_cycle` was tuned against.
GAIT_SIZE = 48
# The matte is measured on a THUMBNAIL rather than through `sprite_sheet.key_out`, which costs
# ~140ms a frame because it also despills and cel-cleans for the packer. None of that changes a
# silhouette, and 67 frames of it is a nine-second page hang. The threshold is the same
# `SOFT_OUT`: a pixel further than that from the measured screen colour is solid subject.
GAIT_THUMB = 96

# One curve per (character, move), invalidated by the pick directory's own mtime. The frames only
# change when an extraction rewrites them, and that rewrites the directory.
_GAIT = {}


def _silhouette(path):
    """One frame's binary silhouette at GAIT_SIZE, keyed against its own measured screen colour."""
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((GAIT_THUMB, GAIT_THUMB), Image.BILINEAR)
    a = np.asarray(im).astype(np.float32)
    dist = np.sqrt(((a - RC.SS.measure_key(a)) ** 2).sum(axis=2))
    solid = Image.fromarray(((dist > RC.SS.SOFT_OUT) * 255).astype("uint8"))
    return np.asarray(solid.resize((GAIT_SIZE, GAIT_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0


def gait(cid, tag_name):
    """Per-frame motion energy across a move's extracted frames, and the gait period if there is one.

    `energy[i]` is how much of the silhouette moved between frame i-1 and frame i, normalised so
    the busiest frame reads 1. That is the same quantity `pose_extremes` integrates when a move is
    NOT periodic, and the period below is the same one it picks inside when a move is — so a flag
    sitting on a trough is a cell drawn on a blurred in-between, and it now says so on the page.
    """
    import numpy as np
    paths = SF.source_frames(cid, tag_name)
    d = f"/tmp/sprite-{cid}-{tag_name}-pick"
    stamp = (os.path.getmtime(d), len(paths)) if paths else None
    hit = _GAIT.get((cid, tag_name))
    if hit and hit[0] == stamp:
        return hit[1]
    if not paths:
        out = {"frames": [], "energy": [], "period": None, "score": None}
    else:
        sils = [_silhouette(p) for p in paths]
        energy = [0.0]
        for a, b in zip(sils, sils[1:]):
            energy.append(float(np.abs(b - a).mean()))
        top = max(energy) or 1.0
        # find_cycle needs half a clip to score a period against; below that it would be scoring
        # one shift against two samples, which is noise with a number on it.
        period, score = RC.SS.find_cycle(sils) if len(sils) >= 16 else (None, None)
        if score is None or score >= RC.SS.CYCLE_SCORE_MAX:
            period = None
        out = {"frames": paths, "energy": [round(e / top, 4) for e in energy],
               "period": period, "score": None if score is None else round(score, 4)}
    _GAIT[(cid, tag_name)] = (stamp, out)
    return out


# The prompt TEMPLATES, read-only, so the editor can show a person the words their clips and their
# repaints were actually briefed with. They are not in the manifest and must not be: the manifest
# carries `template_version`, one digest over exactly these strings, and a second copy of the text
# beside the digest would be free to disagree with it. They live on this side because this is the
# side that imports the Python they are declared in.
LOCKS = {
    "sprite_lock": RC.BS.SPRITE_LOCK,
    "who_lead": RC.BS.WHO_LEAD,
    "stage_restate": RC.BS.STAGE_RESTATE,
    "sound_lock": RC.BS.SOUND_LOCK,
    "repaint": RC.REPAINT,
    "negative": RC.NEG,
}


# --- the work ------------------------------------------------------------------------------


def fetch_manifest(cid):
    """The DO's manifest, plus the per-frame bookkeeping that travels BESIDE it.

    Beside, and not inside, because this manifest is written verbatim to `sheets/<cid>.json`: a
    frame id or a revision carried in the frame records would land in every manifest on disk. The
    list is handed straight back with the finished report, and the DO uses it to decide which
    painted `png` paths still belong to the frames it now holds.
    """
    got = celld("manifest", {"cid": cid})
    return got["manifest"], got["frames"]


def supervise(argv, beat):
    """Run one child to completion, heartbeating while it works. Returns (exit code, killed).

    The heartbeat thread is the only thing that can hear a cancel — the agent is behind a long
    poll and the Worker cannot call it — so the beat continues at a fixed cadence whatever the
    child is doing, and a clip's 170 seconds is no different from a repack's two.
    """
    proc = subprocess.Popen(argv)
    stop = threading.Event()
    killed = threading.Event()

    def heartbeat():
        while not stop.wait(HEARTBEAT_S):
            if beat() and not killed.is_set():
                killed.set()
                # SIGTERM, not SIGKILL: sprite_steps.py catches it to take the render off
                # gpu-worker's queue as well. A kill would stop this process and leave the card
                # rendering a clip nobody will ever collect.
                proc.terminate()

    heart = threading.Thread(target=heartbeat, daemon=True)
    heart.start()
    try:
        code = proc.wait()
    finally:
        stop.set()
        heart.join(timeout=1)
    return code, killed.is_set()


def report_built(cid, job, built_id, cleared_tag=None):
    """Report one upstream step's result: the manifest as the DO now holds it, re-keyed.

    The manifest is re-fetched rather than reused, because the DO owns it and this job may have
    taken minutes — a re-roll typed during a clip render is already spliced in there, and the
    reconcile above has just rewritten a whole tag. Only `built_id` is stamped as built: every
    other step carries its previous built_key forward, or a run that packed nothing would erase
    the staleness an edit had just created.
    """
    man, frames = fetch_manifest(cid)
    RC.save_character_manifest(cid, man)
    man["steps"] = RC._pipeline().build_steps(cid, man, {built_id})
    RC.save_character_manifest(cid, man)
    celld("done", {"cid": cid, "id": job["id"], "message": f"{job['label']} — done",
                   "manifest": man, "frames": frames, "catalogue": catalogue(cid, man),
                   # A changed extraction deleted every drawing of this move, so the tray entries
                   # that still name one have to be told. They keep their facts and lose their
                   # picture; without this the page asks the agent for a deleted file every time
                   # somebody opens the Picks panel.
                   "cleared_tag": cleared_tag})


# What each job type is actually doing, in the words the queue shows while it does it. Saying
# "repainting" for a reorder made a two-second repack look like a 45-second generation, which is
# how this tool earned a reputation for doing things nobody asked for; the same honesty is what
# the clip's 170 seconds needs most, because it is the longest wait this queue can produce.
NOTES = {
    "pack": "repacking, a few seconds",
    "plate": "cutting the subject out on the card, ~20s",
    "clip": "rendering the clip on the card, ~170s",
    "extract": "re-extracting the frames, ~10s",
    "picks": "choosing the cells, ~5s",
}


def run_job(cid, job):
    """One claimed job: adopt the DO's manifest, do the one thing it names, report the result.

    The manifest is written to disk BEFORE anything runs, for the same reason the Python editor
    did it: work can crash or be killed, and the edit must survive that. It is also what makes the
    upstream steps manifest-driven — `hires_sprite.brief()` and `sprite_steps.tag_of()` both read
    this file, so a clip is briefed with the words the tag carries and never with the Python
    tables (DESIGN-pipeline.md, "a new move").

    Every kind runs as a CHILD PROCESS rather than in this thread, and that is what makes a cancel
    real. A queued job can simply be dropped, but a job already on the card is 45 seconds of
    ComfyUI — 170 for a clip — inside one blocking call; there is no flag the work could check
    often enough, and Python cannot interrupt a thread. So the abort is a signal to a process. It
    also puts the packer's numpy and PIL arenas in a process that exits, which this long-lived
    server does not.
    """
    man, frames = fetch_manifest(cid)
    RC.save_character_manifest(cid, man)

    kind = job.get("kind") or "pack"
    if kind in ("pack", "repaint"):
        # Counted off the disk, not guessed from the job's `generates` flag. That flag says
        # whether the EDIT was a generating one; the packer paints whatever drawing is missing,
        # whoever made it missing. A restore landing on a tag whose cells were re-picked reported
        # "repacking, a few seconds" and then held the queue for three minutes of painting.
        missing = sum(1 for f in man["frames"] if not f.get("png") or not os.path.exists(f["png"]))
        note = (f"painting {missing} cell{'' if missing == 1 else 's'}, ~{45 * missing}s"
                if missing else NOTES["pack"])
    else:
        note = NOTES[kind]
    message = f"{job['label']} — {note}"

    def beat():
        """One heartbeat. Returns True when the DO has asked for this job to stop.

        This is also what renews the job's RUNNING lease, and it is why a 170-second clip does not
        need a longer one: the beat is every 5 seconds whatever the child is doing, so the DO's
        60-second running lease keeps twelve beats of headroom at any job length. The lease bounds
        the AGENT's silence, not the work's duration, which is the property that has to hold.
        """
        try:
            return celld("progress", {"cid": cid, "id": job["id"], "message": message}) \
                .get("cancel", False)
        except Exception as e:
            print(f"[{cid}] progress report failed: {type(e).__name__}: {e}", flush=True)
            return False

    def cancelled():
        celld("failed", {"cid": cid, "id": job["id"], "message": f"{job['label']} — cancelled"})
        print(f"[{cid}] {job['id']} cancelled", flush=True)

    def fail(why):
        celld("failed", {"cid": cid, "id": job["id"], "message": f"{job['label']} — {why}"})

    # Beat once before starting anything. A job can be claimed and then cancelled while it waits
    # behind the other character's job on the one card, and starting a paint nobody wants any more
    # is exactly the cost this queue exists to avoid.
    if beat():
        cancelled()
        return

    if kind in ("pack", "repaint"):
        argv = [sys.executable, os.path.join(HERE, "repaint_cells.py"), cid]
        result_path = None
    else:
        result_path = f"/tmp/sprite-step-{job['id']}.json"
        if os.path.exists(result_path):
            os.remove(result_path)
        # A plate belongs to the character and not to a move, so its job carries no tag. The
        # placeholder keeps ONE command line for every step kind (sprite_steps.py's docstring says
        # why), and the plate step ignores it.
        argv = [sys.executable, os.path.join(HERE, "sprite_steps.py"), kind, cid,
                job["tag"] or "-", result_path]

    code, killed = supervise(argv, beat)
    if killed:
        cancelled()
        return
    if code != 0:
        # gpu-worker being down is routine, so this path is reachable on any re-roll. In-process
        # this arrived as SystemExit — which slips past `except Exception` and used to kill the
        # worker thread silently; as a child process it is simply an exit code.
        fail(f"{kind} exited {code}")
        return

    if kind in ("pack", "repaint"):
        packed = RC.load_character_manifest(cid)
        # `manifest` carries the step records and the locked build constants the packer just
        # wrote, which is the BUILD REPORT: every artifact hash and every built_key in there is
        # knowledge only this half has, because only this half can open a file. The DO takes those
        # as given and recomputes the cache keys on top of them.
        celld("done", {"cid": cid, "id": job["id"], "message": f"{job['label']} — done",
                       "manifest": packed, "frames": frames,
                       "catalogue": catalogue(cid, packed)})
        return

    result = json.load(open(result_path))
    os.remove(result_path)
    move = job["tag"]
    if kind == "picks":
        # The picker chose frames; it does NOT get to decide what that costs the hand edits on the
        # frames it replaced. That answer is the Durable Object's, because the edits are its data,
        # and it has to land before the step records below are computed — those key off the chosen
        # source frames, and stamping built_key against picks the DO has not accepted yet would
        # report "fresh" about a manifest that never existed.
        r = celld("picked", {"cid": cid, "tag": move, "srcs": result["srcs"],
                             "frames_hash": result["frames_hash"]})
        print(f"[{cid}] {move}: {r['cells']} cells, {r['matched']} kept their edits, "
              f"{r['orphaned']} to the tray ({r['edits_orphaned']} hand-edited), "
              f"{r['suppressed']} still dropped, "
              f"carry-over {'on' if r['carried_over'] else 'off — a new extraction'}", flush=True)
    built = {"plate": "plate", "clip": f"clip:{move}", "extract": f"frames:{move}",
             "picks": f"picks:{move}"}[kind]
    report_built(cid, job, built, move if kind == "extract" and result["changed"] else None)


def fleet_roster():
    """Which characters the fleet says exist. The DURABLE OBJECTS are the record, not `sheets/`.

    Read off the Worker's own health route, which already answers it, rather than through a new
    agent verb: the roster is a public fact about the fleet, and it is the only thing this process
    needs in order to notice a character somebody created in the page a minute ago.
    """
    with urllib.request.urlopen(f"{CELLD}/api/health", timeout=HTTP_TIMEOUT) as r:
        return json.load(r)["characters"]


def supervisor(pending, wake):
    """Start a long-poll thread for every character on the fleet, including ones created later.

    A character created in the page is a Durable Object with a queue of its own, and until
    something claims from it his very first job -- cutting his plate out -- sits there forever.
    This used to require restarting the agent, which meant the new-subject flow could not complete
    without a shell. One thread per character is the design (each character is one DO), so this
    only ever ADDS: a character is never un-polled while this process lives, because a poller that
    stopped could not be restarted without also proving nothing was in flight on it.
    """
    running = set()
    while True:
        try:
            for cid in fleet_roster():
                if cid in running:
                    continue
                running.add(cid)
                threading.Thread(target=poller, args=(cid, pending, wake), daemon=True).start()
                print(f"[{cid}] polling", flush=True)
        except Exception as e:
            print(f"roster read failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(10)


def poller(cid, pending, wake):
    """One long-poll loop per character, because each character is its own Durable Object.

    Claiming and running are separated on purpose. Each DO offers only the head of its own
    queue, but the GPU is one card for both characters, so a claimed job for Cadbury waits in
    this list while Seth's runs rather than starting beside it. The page sees it as "claimed"
    the whole time, which is the truth — and a cancel arriving in that window is felt at the
    first heartbeat, before any paint starts.
    """
    while True:
        try:
            job = celld("claim", {"cid": cid, "waitMs": CLAIM_WAIT_MS, "agent": RUN_ID})["job"]
        except Exception as e:
            print(f"[{cid}] claim failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(5)
            continue
        if not job:
            continue
        print(f"[{cid}] claimed {job['id']}: {job['label']}", flush=True)
        with wake:
            pending.append((cid, job))
            wake.notify()


def runner(pending, wake):
    while True:
        with wake:
            while not pending:
                wake.wait()
            cid, job = pending.pop(0)
        try:
            run_job(cid, job)
        except Exception as e:
            traceback.print_exc()
            try:
                celld("failed", {"cid": cid, "id": job["id"],
                                 "message": f"{job['label']} — {type(e).__name__}: {e}"})
            except Exception:
                traceback.print_exc()


# --- HTTP ----------------------------------------------------------------------------------


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The page is served by celld on another host, so every read here is cross-origin. The
        # server is read-only and LAN-bound, and it holds nothing private that a wildcard could
        # leak — the allowlist below is what keeps it honest, not the origin check.
        self.send_header("Access-Control-Allow-Origin", "*")
        # Every file here is overwritten in place by a repack, so a cached image is the most
        # likely way the editor lies about what is on disk.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, code, obj):
        self.send(code, json.dumps(obj), "application/json")

    def do_OPTIONS(self):
        """The upload's preflight. An `image/png` body is not a CORS-simple content type, so the
        browser asks first — and the page is served by celld on another host, so it always asks."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        """The ONE write route: stage the image a new character will be cut out of.

        The body is read only after Content-Length has been checked, so an over-sized upload is
        refused without this process ever holding it. Everything else the request claims about
        itself — its type, its name — is ignored in favour of what the bytes actually are; see
        `sprite_files.stage_upload`.
        """
        if urllib.parse.urlparse(self.path).path != "/upload":
            self.json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.json(400, {"error": "a bad Content-Length"})
            return
        if length <= 0:
            self.json(411, {"error": "the upload needs a Content-Length"})
            return
        if length > SF.UPLOAD_MAX:
            self.json(413, {"error": f"that image is {length >> 20}MB and the limit is "
                                     f"{SF.UPLOAD_MAX >> 20}MB"})
            return
        data = self.rfile.read(length)
        try:
            path, w, h = SF.stage_upload(data)
        except ValueError as e:
            self.json(400, {"error": str(e)})
            return
        print(f"staged an upload: {path} ({w}x{h}, {len(data) >> 10}KB)", flush=True)
        self.json(200, {"path": path, "width": w, "height": h})

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        one = lambda k, d=None: (q.get(k) or [d])[0]  # noqa: E731
        try:
            if u.path == "/health":
                self.json(200, {"ok": True, "celld": CELLD})
            elif u.path == "/derived":
                # The DURABLE OBJECT decides whether a character exists, not this filesystem. A
                # character created from an upload has a manifest in his cell from the instant he
                # is seeded, and `sheets/<cid>.json` only appears when his first job runs -- so
                # gating on the file made a brand-new character's page open on an error.
                try:
                    man = fetch_manifest(one("cid", "seth"))[0]
                except urllib.error.HTTPError as e:
                    self.json(e.code, {"error": f"unknown character: {one('cid', 'seth')}"})
                    return
                self.json(200, derived(man["character"], man))
            elif u.path == "/defaults":
                self.json(200, defaults())
            elif u.path == "/frames":
                self.json(200, {"frames": SF.source_frames(one("cid", "seth"), one("tag", "walk"))})
            elif u.path == "/gait":
                self.json(200, gait(one("cid", "seth"), one("tag", "walk")))
            elif u.path == "/clip":
                self.serve_clip(one("cid", "seth"), one("tag", "walk"))
            elif u.path == "/cell":
                self.serve_cell(one("cid", ""), one("i"), one("w"))
            elif u.path == "/img":
                self.serve_image(one("path", ""), one("w"), one("dl") == "1")
            elif u.path == "/export":
                self.serve_export(one("cid", "seth"), one("format", ""))
            else:
                self.json(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self.json(500, {"error": f"{type(e).__name__}: {e}"})

    TYPES = {".png": "image/png", ".json": "application/json"}

    def png(self, im, width):
        """PNG bytes for an image, optionally shrunk to fit `width`.

        `thumbnail` mutates in place, which is safe only because every caller hands over an image
        it just created — a crop off the cached atlas, or a freshly opened file. Never pass the
        cached atlas itself.
        """
        from PIL import Image
        if width:
            w = max(16, min(512, int(width)))
            im.thumbnail((w, w), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    def serve_cell(self, cid, index, width):
        if cid not in SF.characters() or index is None or not index.isdigit():
            self.json(403, {"error": "unknown character or cell"})
            return
        im = SF.cell_image(cid, int(index))
        if im is None:
            self.json(404, {"error": "no such cell in the atlas"})
            return
        self.send(200, self.png(im, width), "image/png")

    def serve_image(self, path, width, download=False):
        real = os.path.realpath(path)
        ext = os.path.splitext(real)[1]
        # sprite_files' allowlist, imported rather than restated: source frames, repaints and
        # the packed output are the only things the page needs, and an unrestricted path
        # parameter on any server is a file-read hole.
        if not real.startswith(SF.ALLOWED_ROOTS) or ext not in self.TYPES:
            self.json(403, {"error": "path not allowed"})
            return
        if not os.path.exists(real):
            self.json(404, {"error": "no such file"})
            return
        if width and ext == ".png":
            from PIL import Image
            body = self.png(Image.open(real).convert("RGBA"), width)
        else:
            body = open(real, "rb").read()
        extra = ({"Content-Disposition": f'attachment; filename="{os.path.basename(real)}"'}
                 if download else None)
        self.send(200, body, self.TYPES[ext], extra)

    def serve_clip(self, cid, tag_name):
        """The mp4 one move was rendered from, named by its MOVE and never by its path.

        The clip is the clip step's artifact, so the bench has to be able to play it — and it is
        the one artifact that lives loose in /tmp rather than under a `sprite-` prefix, so `/img`'s
        allowlist cannot reach it and must not be widened to `/tmp`. Taking the character and the
        move instead, and reading the path out of the Durable Object's own manifest, means the
        page names a MOVE and this side names the file: an unrestricted path parameter never
        exists to be abused.
        """
        try:
            man = fetch_manifest(cid)[0]
        except urllib.error.HTTPError:
            self.json(404, {"error": f"unknown character: {cid}"})
            return
        tag = next((t for t in man["tags"] if t["name"] == tag_name), None)
        if tag is None:
            self.json(404, {"error": f"no {tag_name} move for {cid}"})
            return
        path = tag["clip"]
        if not os.path.exists(path):
            self.json(404, {"error": f"no clip rendered for {cid} {tag_name} yet"})
            return
        self.send(200, open(path, "rb").read(), "video/mp4")

    def serve_export(self, cid, fmt):
        if cid not in SF.characters():
            self.json(404, {"error": f"unknown character: {cid}"})
            return
        entry = EX.FORMATS.get(fmt)
        if entry is None:
            self.json(400, {"error": f"unknown format {fmt!r}; choose one of "
                                     f"{', '.join(sorted(EX.FORMATS))}"})
            return
        builder, ctype, name_for = entry
        try:
            atlas = EX.load_atlas(cid)
        except OSError:
            self.json(404, {"error": f"no emitted atlas for {cid} -- run the packer first"})
            return
        self.send(200, builder(atlas, cid), ctype,
                  {"Content-Disposition": f'attachment; filename="{name_for.format(cid=cid)}"'})


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def seed():
    """Push each character's manifest on disk into its Durable Object.

    Only meaningful once, at cutover — after that the DO is the record and this would overwrite
    it with whatever the file happens to say. So it is a flag, not the startup path.
    """
    for cid in SF.characters():
        man = RC.load_character_manifest(cid)
        if not man:
            print(f"  {cid}: no manifest on disk, skipped", flush=True)
            continue
        r = celld("seed", {"cid": cid, "manifest": man, "catalogue": catalogue(cid, man)})
        print(f"  {cid}: seeded {r['frames']} frames", flush=True)


def forget(cid):
    """Delete one character completely: his Durable Object cell, his roster entry, and his files.

    Deliberately NOT a button on the page. Everything a character is lives in exactly two places
    and this is the only thing that empties both, so it is a command somebody has to type -- with
    the cid spelled out -- rather than a click next to the character switcher.

    The order is the record first and the files second. A crash between the two leaves files
    nothing refers to, which the next `--forget` or a `rm` clears; the reverse would leave a
    character on the roster whose every drawing had gone, and the page would then 404 its way
    through a set that still claims to exist.
    """
    import glob
    import shutil
    # Read BEFORE the record is erased, and from the DURABLE OBJECT rather than from disk: the
    # image the plate was cut out of is named in the manifest and nowhere else, and a character
    # created a minute ago has no `sheets/<cid>.json` at all -- that file is written by his first
    # job. Asking the record rather than the copy is what makes this work at any age.
    try:
        source = ((fetch_manifest(cid)[0].get("plate")) or {}).get("source")
    except urllib.error.HTTPError:
        source = None       # already erased, or never seeded; the file sweep below still runs
    got = celld("erase-character", {"cid": cid})
    print(f"  cell: {'cleared' if got.get('existed') else 'was already empty'}; "
          f"roster is now {', '.join(got['characters']) or '(empty)'}")
    gone = 0
    for pattern in (source or "", RC.manifest_path(cid), PL.plate_path(cid),
                    os.path.join(RC.OUT, f"{cid}.png"), os.path.join(RC.OUT, f"{cid}.json"),
                    os.path.join(RC.OUT, f"{cid}-qc.json"),
                    os.path.join(RC.REPAINT_DIR, f"{cid}-*.png"),
                    f"/tmp/{cid}-*.mp4", f"/tmp/sprite-{cid}-*", f"/tmp/sprite-clip-{cid}-*.pid",
                    f"/tmp/sprite-plate-{cid}.pid", f"/tmp/sprite-plate-raw-{cid}.png"):
        for path in glob.glob(pattern):
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            gone += 1
    print(f"  files: {gone} removed")
    # The copies this process cannot reach. ComfyUI loads by bare FILENAME, so both the plate and
    # every padded repaint input were uploaded to its input directory on gpu-worker — another
    # host, one this service has no credential for and should not grow one for. They are harmless
    # (each is overwritten by name on the next run) but they are still this character, so they are
    # named rather than left unsaid.
    print(f"  still on gpu-worker, remove them there: ~/comfyui/input/cast-cutout-{cid}.png "
          f"and ~/comfyui/input/{cid}-*-padded.png")


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("SPRITE_AGENT_TOKEN is not set; the fleet will refuse every call")
    if "--seed" in sys.argv:
        seed()
        sys.exit(0)
    if "--forget" in sys.argv:
        forget(sys.argv[sys.argv.index("--forget") + 1])
        sys.exit(0)
    pending, wake = [], threading.Condition()
    threading.Thread(target=runner, args=(pending, wake), daemon=True).start()
    threading.Thread(target=supervisor, args=(pending, wake), daemon=True).start()
    print(f"sprite agent on http://0.0.0.0:{PORT}/ , polling {CELLD}", flush=True)
    Server(("0.0.0.0", PORT), Handler).serve_forever()
