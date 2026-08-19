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
   atlas, read-only and cross-origin, because the page is served from another host.

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
            variants[f["src"]] = SF.variants(cid, tag["name"], f["src"])
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
                "variants": SF.variants(cid, tag["name"], f["src"])})
        tags[tag["name"]] = {"cells": cells}
    return {"atlas": SF.sheet_paths(cid)[0], "outdir": RC.OUT, "tags": tags, "locks": LOCKS}


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


def run_job(cid, job):
    """One claimed job: adopt the DO's manifest, repaint and repack, report what was packed.

    The manifest is written to disk BEFORE the repack for the same reason the Python editor did
    it: a repack can crash or be killed, and the edit must survive that. Nothing in here decides
    anything about the manifest — `repaint_cells.main()` fills in each frame's `png` as it
    repaints, and that is the only field the DO will accept back.

    The repack runs as a CHILD PROCESS rather than in this thread, and that is what makes a cancel
    real. A queued job can simply be dropped, but a job already on the GPU is 45 seconds of
    ComfyUI inside one blocking call; there is no flag `repaint_cells` could check often enough,
    and Python cannot interrupt a thread. So the abort is a signal to a process. It also puts the
    packer's numpy and PIL arenas in a process that exits, which this long-lived server does not.
    """
    man, frames = fetch_manifest(cid)
    RC.save_character_manifest(cid, man)

    # Saying "repainting" for a reorder made a two-second repack look like a 45-second
    # generation, which is how this tool earned a reputation for doing things nobody asked for.
    note = "painting a new cell, ~45s" if job["generates"] else "repacking, a few seconds"
    message = f"{job['label']} — {note}"

    def beat():
        """One heartbeat. Returns True when the DO has asked for this job to stop."""
        try:
            return celld("progress", {"cid": cid, "id": job["id"], "message": message}) \
                .get("cancel", False)
        except Exception as e:
            print(f"[{cid}] progress report failed: {type(e).__name__}: {e}", flush=True)
            return False

    def cancelled():
        celld("failed", {"cid": cid, "id": job["id"], "message": f"{job['label']} — cancelled"})
        print(f"[{cid}] {job['id']} cancelled", flush=True)

    # Beat once before starting anything. A job can be claimed and then cancelled while it waits
    # behind the other character's job on the one card, and starting a paint nobody wants any more
    # is exactly the cost this queue exists to avoid.
    if beat():
        cancelled()
        return

    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "repaint_cells.py"), cid])
    stop = threading.Event()
    killed = threading.Event()

    def heartbeat():
        while not stop.wait(HEARTBEAT_S):
            if beat() and not killed.is_set():
                killed.set()
                proc.terminate()

    heart = threading.Thread(target=heartbeat, daemon=True)
    heart.start()
    try:
        code = proc.wait()
    finally:
        stop.set()
        heart.join(timeout=1)
    if killed.is_set():
        cancelled()
        return
    if code != 0:
        # gpu-worker being down is routine, so this path is reachable on any re-roll. In-process
        # this arrived as SystemExit — which slips past `except Exception` and used to kill the
        # worker thread silently; as a child process it is simply an exit code.
        celld("failed", {"cid": cid, "id": job["id"],
                         "message": f"{job['label']} — the repack exited {code}"})
        return
    packed = RC.load_character_manifest(cid)
    # `manifest` carries the step records and the locked build constants the packer just wrote,
    # which is the BUILD REPORT: every artifact hash and every built_key in there is knowledge only
    # this half has, because only this half can open a file. The DO takes those as given and
    # recomputes the cache keys on top of them.
    celld("done", {"cid": cid, "id": job["id"], "message": f"{job['label']} — done",
                   "manifest": packed, "frames": frames, "catalogue": catalogue(cid, packed)})


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

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        one = lambda k, d=None: (q.get(k) or [d])[0]  # noqa: E731
        try:
            if u.path == "/health":
                self.json(200, {"ok": True, "celld": CELLD})
            elif u.path == "/derived":
                cid = one("cid", "seth")
                if cid not in SF.characters():
                    self.json(404, {"error": f"unknown character: {cid}"})
                    return
                self.json(200, derived(cid, fetch_manifest(cid)[0]))
            elif u.path == "/frames":
                self.json(200, {"frames": SF.source_frames(one("cid", "seth"), one("tag", "walk"))})
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


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("SPRITE_AGENT_TOKEN is not set; the fleet will refuse every call")
    if "--seed" in sys.argv:
        seed()
        sys.exit(0)
    pending, wake = [], threading.Condition()
    threading.Thread(target=runner, args=(pending, wake), daemon=True).start()
    # Read once, at start: each character needs its own long-poll thread against its own Durable
    # Object. A character seeded later is served by every read route immediately -- those ask
    # SF.characters() per request -- but does not get a poller until this process is restarted.
    for cid in SF.characters():
        threading.Thread(target=poller, args=(cid, pending, wake), daemon=True).start()
    print(f"sprite agent on http://0.0.0.0:{PORT}/ , polling {CELLD}", flush=True)
    Server(("0.0.0.0", PORT), Handler).serve_forever()
