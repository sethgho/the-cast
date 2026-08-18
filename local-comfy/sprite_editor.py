#!/usr/bin/env python3
"""A per-cell editor for the character sprite manifest, so a bad cell costs one repaint instead of
a whole tag.

    python3 sprite_editor.py        # http://<host>:8811/

## Why

`repaint_cells.py` gets a tag to about 90% on its own, and the last 10% is always the same
judgement: this cell's hand came back garbled, that one was picked two frames too early, this one
drifted in scale. The manifest in `sheets/<cid>.json` exists so those judgements survive a rerun.
Editing it by hand works but is blind — you cannot see which cell is cell 4, and you cannot tell a
bad REPAINT from a bad SOURCE FRAME without opening two files in two viewers. That distinction
decides the fix: a bad repaint wants a new seed, a bad source frame wants a different frame.

So the page shows both, side by side, for the selected cell.

Nothing about packing lives here. Every mutation writes the manifest and then calls
`repaint_cells.main()`, which repaints whatever is missing and repacks the WHOLE character at one
shared scale. Packing a single tag on its own would rescale it against itself and the character
would change size when the game switches move.

## The rules the design is built around

1. One ComfyUI job at a time. A repaint is ~45s on a 12GB card and two at once means two failures,
   so every mutating request goes onto a single-worker queue and the caller gets a job id to poll.
2. The manifest is written BEFORE the repack. A repack can crash or be killed; if it does, the edit
   is still recorded and the next run picks it up. The other order loses the edit silently.
3. Every manifest read and write goes through `repaint_cells.load_character_manifest` /
   `save_character_manifest` and nothing else — no `json.load` on a manifest path in this file.
   Stage 6 of DESIGN-editor.md swaps that pair for an HTTP call to a Durable Object; a scattered
   read means writing this server twice.
"""
import hashlib
import http.server
import io
import json
import os
import queue
import random
import socketserver
import sys
import threading
import time
import traceback
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exports as EX  # noqa: E402
import repaint_cells as RC  # noqa: E402
import sprite_sheet as SS  # noqa: E402

PORT = 8811
CHARACTERS = ["seth", "cadbury"]
PICK_SKIP = 6                    # settle-in frames pick_frames drops; not offerable as a re-pick

# /api/export?cid=&format= -- each value is (builder, content-type, filename(cid)). The builder
# is a pure function of the atlas table (`exports.load_atlas`), so this list is the entire
# contract with exports.py; nothing else in this file needs to know the export formats exist.
EXPORT_FORMATS = {
    "json-hash": (EX.export_json_hash, "application/json", lambda cid: f"{cid}-atlas-hash.json"),
    "json-array": (EX.export_json_array, "application/json", lambda cid: f"{cid}-atlas-array.json"),
    "phaser3": (EX.export_phaser3, "application/json", lambda cid: f"{cid}-phaser3.json"),
    "godot": (EX.export_godot, "text/plain; charset=utf-8", lambda cid: f"{cid}-frames.tres"),
    "css": (EX.export_css, "text/css; charset=utf-8", lambda cid: f"{cid}-sprite.css"),
}

# /img serves nothing outside these. The editor only ever needs source frames, repaints and the
# packed output, and an unrestricted path parameter on a LAN-bound server is a file-read hole.
ALLOWED_ROOTS = ("/tmp/sprite-", RC.REPAINT_DIR + "/", os.path.realpath(RC.OUT) + "/")


def sheet_paths(cid):
    return os.path.join(RC.OUT, f"{cid}.png"), os.path.join(RC.OUT, f"{cid}.json")


# One decoded 512px atlas is 10 x 512 wide by six rows: ~63MB of RGBA. Keeping every character's
# would quietly cost more than the packer does, so only the last one asked for is held.
_SHEET = {}


def sheet(cid):
    """The emitted atlas and its frame table, cached against the files' mtimes.

    This is build OUTPUT, not the manifest. The manifest says which drawings a tag is made of; the
    sheet says where the packer actually put them. Both are needed because a cell is no longer a
    file of its own — it is a crop out of the one atlas.
    """
    png, js = sheet_paths(cid)
    try:
        stamp = (os.path.getmtime(png), os.path.getmtime(js))
    except OSError:
        return None
    hit = _SHEET.get(cid)
    if hit and hit[0] == stamp:
        return hit[1]
    from PIL import Image
    meta = json.load(open(js))
    meta["image"] = Image.open(png).convert("RGBA")
    meta["metrics"] = {}
    _SHEET.clear()
    _SHEET[cid] = (stamp, meta)
    return meta


def atlas_base(sh, tag_name, n_frames):
    """Where a manifest tag's cells landed in the atlas, or None if they cannot be located.

    `_prepare_stills` DROPS a frame that keys to nothing, so the atlas can hold fewer cells than
    the manifest tag has frames — and nothing in the emitted schema records WHICH frame went
    missing. When the counts agree the mapping is exact. When they do not, refusing to guess is the
    only correct answer: an off-by-one here shows the wrong drawing beside the wrong source frame,
    which is precisely the judgement this editor exists to make.
    """
    for t in sh["tags"]:
        if t["name"] == tag_name:
            return t["from"] if t["to"] - t["from"] + 1 == n_frames else None
    return None


def cell_image(cid, index):
    """The packed 512px cell, cropped out of the character atlas.

    Shown instead of the raw repaint because it is what actually ships, and because the metrics
    below are only meaningful in packed coordinates.
    """
    sh = sheet(cid)
    if not sh or not 0 <= index < len(sh["frames"]):
        return None
    f, c = sh["frames"][index], sh["cell"]
    return sh["image"].crop((f["x"], f["y"], f["x"] + c, f["y"] + c))


def cell_metrics(cid, index):
    """Silhouette height, top clearance and feet offset for one packed cell.

    These three are the numbers that have caught every bad cell in this project. A repaint that
    came back larger shows as a small clearance; one that came back smaller shows as a short
    silhouette; a keyed-in shadow or a clipped foot shows as a feet offset away from zero, which is
    the character sliding off the ground line mid-animation.

    Measured against THIS FRAME'S OWN emitted pivot (`sh["frames"][index]["pivot"]`), not the
    manifest-level constant `man["pivot"]`. The two agree today because every frame still carries
    the same pivot, but the demo already honours each frame's own pivot — once a later stage makes
    pivots vary per frame, measuring against the manifest constant would silently score a correctly
    placed cell as off the ground line. The cache hangs off the sheet, so a repack invalidates it
    for free.
    """
    sh = sheet(cid)
    if not sh:
        return None
    hit = sh["metrics"].get(index)
    if hit is not None:
        return hit[0]
    im = cell_image(cid, index)
    if im is None:
        return None
    ground = sh["frames"][index]["pivot"][1]
    import numpy as np
    mask = np.asarray(im)[:, :, 3] > 10
    rows = np.where(mask.any(axis=1))[0]
    m = None if len(rows) == 0 else {
        "height": int(rows[-1] - rows[0] + 1),
        "top": int(rows[0]),
        "feet": int(rows[-1] - ground),
        "ground": int(ground),
    }
    sh["metrics"][index] = (m,)
    return m


def source_frames(cid, tag_name):
    d = f"/tmp/sprite-{cid}-{tag_name}-pick"
    if not os.path.isdir(d):
        return []
    names = sorted(n for n in os.listdir(d) if n.endswith(".png"))
    return [os.path.join(d, n) for n in names[PICK_SKIP:]]


def variants(cid, tag_name, src):
    """Every repaint that already exists on disk for this source frame.

    A re-roll used to feel destructive: the previous drawing vanished from the sheet with no way
    back, and getting it back meant remembering the seed. But nothing is ever deleted -- each
    (source frame, seed) pair has its own file. So list them, and let a click swap between them.
    Switching to one that already exists costs no GPU at all.
    """
    stem = os.path.basename(src).replace(".png", "")
    prefix = f"{cid}-{tag_name}-{stem}"
    out = []
    for n in sorted(os.listdir(RC.REPAINT_DIR)):
        if not n.startswith(prefix + "-s") and n != prefix + ".png":
            continue
        if n.endswith("-padded.png") or not n.endswith(".png"):
            continue
        tail = n[len(prefix):-4]
        out.append({"png": os.path.join(RC.REPAINT_DIR, n),
                    "seed": int(tail[2:]) if tail.startswith("-s") else RC.SEED})
    return out


def state(cid):
    """The whole character, as the page needs it: tags in playback order, cells within each tag.

    A cell's `index` is its position WITHIN ITS TAG, because that is what the buttons act on and
    what the user counts. `atlas` is its position in the flat emitted sheet, which is what /cell
    crops by; the two differ by the tag's start and are never the same number.
    """
    man = RC.load_character_manifest(cid)
    if not man:
        raise ValueError(f"no manifest for {cid}")
    sh = sheet(cid)
    pivot = man["pivot"]
    tags = []
    for tag in man["tags"]:
        span = list(range(tag["from"], tag["to"] + 1))
        base = atlas_base(sh, tag["name"], len(span)) if sh else None
        cells = []
        for i, fi in enumerate(span):
            f = man["frames"][fi]
            ai = None if base is None else base + i
            cells.append({
                "index": i, "src": f["src"], "seed": f["seed"], "png": f.get("png"),
                "hold": int(f.get("hold", 1)),
                # The nudge is an OFFSET from the sheet pivot, never an absolute point: the page
                # draws the crosshair at pivot+nudge and posts the offset back, so a later change
                # to the sheet pivot carries every nudged cell with it.
                "pivot_nudge": [int(v) for v in f.get("pivot_nudge", (0, 0))], "atlas": ai,
                "xy": None if ai is None else [sh["frames"][ai]["x"], sh["frames"][ai]["y"]],
                "metrics": None if ai is None else cell_metrics(cid, ai),
                "variants": variants(cid, tag["name"], f["src"])})
        tags.append({"name": tag["name"], "fps": tag["fps"], "loop": tag["loop"],
                     "direction": tag["direction"], "hold_key": tag["hold_key"], "cells": cells})
    return {"character": cid, "characters": CHARACTERS, "tags": tags, "pivot": pivot,
            "cell": man["cell"], "columns": (sh or {}).get("columns", SS.SHEET_COLUMNS),
            "atlas": sheet_paths(cid)[0], "outdir": RC.OUT}


# --- the job queue -------------------------------------------------------------------------
# One worker, one ComfyUI job at a time. Anything that repaints or repacks goes through here.

JOBS = {}
JOBS_LOCK = threading.Lock()
WORK = queue.Queue()


def submit(label, fn, generates=True):
    jid = hashlib.sha1(f"{label}{time.time()}{random.random()}".encode()).hexdigest()[:12]
    with JOBS_LOCK:
        JOBS[jid] = {"state": "queued", "message": f"queued: {label}"}
    WORK.put((jid, label, fn, generates))
    return jid


def set_job(jid, st, message):
    with JOBS_LOCK:
        JOBS[jid] = {"state": st, "message": message}


def worker():
    while True:
        jid, label, fn, generates = WORK.get()
        # Reordering and dropping do not draw anything. Saying "repainting" for them made a
        # two-second repack look like a 45-second generation, which is how the tool earned its
        # reputation for doing things nobody asked for.
        set_job(jid, "running", f"{label} — painting a new cell, ~45s" if generates
                else f"{label} — repacking, a few seconds")
        try:
            fn(lambda msg: set_job(jid, "running", msg))
            set_job(jid, "done", f"{label} — done")
        except BaseException as e:
            # BaseException, not Exception: repaint_cells.repaint() and sprite_sheet's frame-
            # keying checks used to raise SystemExit on a failed ComfyUI job, which is a
            # BaseException and slips straight past `except Exception`. That silently killed
            # this thread — the job stayed "running" forever and every job queued behind it
            # never ran, so the whole editor looked dead until someone restarted the server.
            # gpu-worker being down is routine here, so this path is reachable on any re-roll.
            traceback.print_exc()
            set_job(jid, "error", f"{label} — {type(e).__name__}: {e}")
        WORK.task_done()


def repack(cid):
    """Repaint anything missing and repack the character. Never write a packer here."""
    argv = sys.argv
    sys.argv = ["repaint_cells", cid]
    try:
        RC.main()
    finally:
        sys.argv = argv


def edit_manifest(cid, tag_name, mutate):
    """Apply `mutate` to one tag's frames, splice them back, and save — in that order, always.

    Dropping or adding a frame moves every LATER tag along the flat frame list, so their ranges are
    renumbered here. A range left stale points at the neighbouring tag's cells, and the packer then
    emits another move's drawings under this tag's name.
    """
    man = RC.load_character_manifest(cid)
    i = RC._tag_index(man, tag_name) if man else None
    if i is None:
        raise ValueError(f"no {tag_name} tag for {cid}")
    tag = man["tags"][i]
    frames = mutate([dict(f) for f in man["frames"][tag["from"]:tag["to"] + 1]])
    if not frames:
        raise ValueError("a tag cannot have zero frames")
    new = [RC._frame(f["src"], f["seed"], f.get("png"), f.get("hold", 1),
                     f.get("pivot_nudge", (0, 0))) for f in frames]
    shift = len(new) - (tag["to"] + 1 - tag["from"])
    man["frames"][tag["from"]:tag["to"] + 1] = new
    tag["to"] += shift
    for later in man["tags"][i + 1:]:
        later["from"] += shift
        later["to"] += shift
    RC.save_character_manifest(cid, man)


def edit_character(cid, mutate):
    """Apply `mutate` to the whole manifest and save it.

    The sheet pivot and a tag's fps and direction sit OUTSIDE any frame range, so `edit_manifest`
    -- which exists to splice one tag's frame list and renumber the tags after it -- cannot express
    them. Same order as every other mutation here: read, mutate, write, and only then repack.
    """
    man = RC.load_character_manifest(cid)
    if not man:
        raise ValueError(f"no manifest for {cid}")
    mutate(man)
    RC.save_character_manifest(cid, man)


# --- validation ----------------------------------------------------------------------------
# Every one of these refuses with a sentence rather than writing a value the packer would then
# have to survive. A pivot off the cell pastes the artwork outside its own cell in the atlas, and
# an fps of 0 divides by zero in both consumers -- neither fails where it was caused.


def _whole(value, what):
    """A JSON whole number. `bool` is an `int` in Python, and `true` is not a coordinate."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be a whole number, not {value!r}")
    return value


def _xy(value, what):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{what} must be a pair [x, y]")
    return _whole(value[0], f"{what} x"), _whole(value[1], f"{what} y")


def _in_cell(x, y, cell, what):
    if not (0 <= x < cell and 0 <= y < cell):
        raise ValueError(f"{what} {x},{y} is outside the {cell}px cell")


def _cell_index(man, tag_name, req):
    """A cell's position WITHIN ITS TAG, checked against the tag that is actually on disk."""
    tag = man["tags"][RC._tag_index(man, tag_name)]
    n = tag["to"] - tag["from"] + 1
    i = _whole(req.get("index"), "index")
    if not 0 <= i < n:
        raise ValueError(f"cell {i + 1} is not in the {tag_name} tag — it has {n}")
    return i


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
        # Every file this server serves is overwritten in place by a repack. A cached image is the
        # most likely way the editor lies about what is on disk, so nothing here is cacheable.
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
            if u.path == "/":
                self.send(200, PAGE, "text/html; charset=utf-8")
            elif u.path == "/api/state":
                self.json(200, state(one("cid", "seth")))
            elif u.path == "/api/frames":
                self.json(200, {"frames": source_frames(one("cid", "seth"), one("tag", "walk"))})
            elif u.path == "/cell":
                self.serve_cell(one("cid", ""), one("i"), one("w"))
            elif u.path == "/img":
                self.serve_image(one("path", ""), one("w"), one("dl") == "1")
            elif u.path == "/api/export":
                self.serve_export(one("cid", "seth"), one("format", ""))
            elif u.path.startswith("/api/job/"):
                jid = u.path.rsplit("/", 1)[-1]
                with JOBS_LOCK:
                    job = JOBS.get(jid)
                self.json(200 if job else 404, job or {"state": "error", "message": "no such job"})
            else:
                self.json(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self.json(500, {"error": f"{type(e).__name__}: {e}"})

    # The atlas is only half the deliverable: the frame table ships with it, so it has to be
    # downloadable too, or the sheet reads as something that only exists inside this page.
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
        """One packed cell, cropped live out of the atlas.

        The cells used to be written out as `<cid>-<move>-frames/*.png` purely so this page could
        link to them. One sheet means one file, so the crop happens here instead of on disk — the
        editor never shows a cell the demo would not draw.
        """
        if cid not in CHARACTERS or index is None or not index.isdigit():
            self.json(403, {"error": "unknown character or cell"})
            return
        im = cell_image(cid, int(index))
        if im is None:
            self.json(404, {"error": "no such cell in the atlas"})
            return
        self.send(200, self.png(im, width), "image/png")

    def serve_image(self, path, width, download=False):
        real = os.path.realpath(path)
        ext = os.path.splitext(real)[1]
        if not real.startswith(ALLOWED_ROOTS) or ext not in self.TYPES:
            self.json(403, {"error": "path not allowed"})
            return
        if not os.path.exists(real):
            self.json(404, {"error": "no such file"})
            return
        if width and ext == ".png":
            # The re-pick strip is 60+ full 832px frames. Sending them whole makes the strip take
            # tens of seconds to appear, which is long enough that you stop using the feature.
            from PIL import Image
            body = self.png(Image.open(real).convert("RGBA"), width)
        else:
            body = open(real, "rb").read()
        extra = ({"Content-Disposition": f'attachment; filename="{os.path.basename(real)}"'}
                 if download else None)
        self.send(200, body, self.TYPES[ext], extra)

    def serve_export(self, cid, fmt):
        """One export preset (`exports.py`), generated fresh on every request.

        Never cached, for the same reason `/img` never is: a repack overwrites `<cid>.json` on
        disk in place, and a stale export would describe cells that no longer match the atlas a
        moment later.
        """
        if cid not in CHARACTERS:
            self.json(404, {"error": f"unknown character: {cid}"})
            return
        entry = EXPORT_FORMATS.get(fmt)
        if entry is None:
            self.json(400, {"error": f"unknown format {fmt!r}; choose one of "
                                      f"{', '.join(sorted(EXPORT_FORMATS))}"})
            return
        builder, ctype, name_for = entry
        try:
            atlas = EX.load_atlas(cid)
        except OSError:
            self.json(404, {"error": f"no emitted atlas for {cid} -- run the packer first"})
            return
        body = builder(atlas, cid)
        self.send(200, body, ctype,
                  {"Content-Disposition": f'attachment; filename="{name_for(cid)}"'})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            cid, tag = req.get("cid", "seth"), req.get("tag", "walk")
            if cid not in CHARACTERS:
                raise ValueError("unknown character")
            # The tag list comes from the manifest, never from the MOVES table: MOVES is only the
            # bootstrap recipe, and a tag that exists on disk must stay editable whether or not
            # MOVES still names it.
            man = RC.load_character_manifest(cid)
            if not man or RC._tag_index(man, tag) is None:
                raise ValueError(f"no {tag} tag for {cid}")
            handler = {"/api/reroll": self.reroll, "/api/repick": self.repick,
                       "/api/drop": self.drop, "/api/reorder": self.reorder,
                       "/api/use": self.use, "/api/pivot": self.pivot,
                       "/api/hold": self.hold, "/api/tag": self.tag_settings}.get(u.path)
            if not handler:
                self.json(404, {"error": "not found"})
                return
            self.json(200, {"job": handler(cid, tag, req)})
        except Exception as e:
            traceback.print_exc()
            self.json(400, {"error": f"{type(e).__name__}: {e}"})

    def queue_edit(self, label, cid, tag, mutate, generates=True):
        # The manifest edit itself runs on the worker too, so a queued edit always sees the frame
        # list the edit before it left behind, not the one it was submitted against.
        def run(_progress):
            edit_manifest(cid, tag, mutate)
            repack(cid)
        return submit(label, run, generates)

    def reroll(self, cid, tag, req):
        i, seed = int(req["index"]), req.get("seed")

        def mutate(frames):
            f = frames[i]
            new = int(seed) if seed is not None else random.randint(1, 2 ** 31 - 1)
            if new == f["seed"]:
                raise ValueError("that is the seed the cell already has")
            f["seed"] = new
            f["png"] = RC.repaint_path(cid, tag, f["src"], new)
            return frames
        return self.queue_edit(f"{cid}-{tag} cell {i + 1} re-roll", cid, tag, mutate)

    def repick(self, cid, tag, req):
        i, src = int(req["index"]), req["src"]
        if src not in source_frames(cid, tag):
            raise ValueError("that source frame is not in this tag's pick directory")

        def mutate(frames):
            f = frames[i]
            f["src"] = src
            f["png"] = RC.repaint_path(cid, tag, src, f["seed"])
            return frames
        return self.queue_edit(f"{cid}-{tag} cell {i + 1} re-pick", cid, tag, mutate)

    def use(self, cid, tag, req):
        """Point a cell at a repaint that already exists. Repack only -- nothing to generate."""
        i, png = int(req["index"]), req["png"]

        def mutate(frames):
            f = frames[i]
            if png not in [v["png"] for v in variants(cid, tag, f["src"])]:
                raise ValueError("that is not a variant of this cell's source frame")
            f["png"] = png
            stem = os.path.basename(png)[:-4].split("-s")
            f["seed"] = int(stem[-1]) if len(stem) > 1 else RC.SEED
            return frames
        return self.queue_edit(f"{cid}-{tag} cell {i + 1} use variant", cid, tag, mutate,
                               generates=False)

    def drop(self, cid, tag, req):
        i = int(req["index"])

        def mutate(frames):
            if len(frames) <= 1:
                raise ValueError("refusing to drop the last cell of a tag")
            del frames[i]
            return frames
        return self.queue_edit(f"{cid}-{tag} drop cell {i + 1}", cid, tag, mutate,
                               generates=False)

    def reorder(self, cid, tag, req):
        order = [int(x) for x in req["order"]]

        def mutate(frames):
            if sorted(order) != list(range(len(frames))):
                raise ValueError("order must be a permutation of the current cell indices")
            return [frames[i] for i in order]
        return self.queue_edit(f"{cid}-{tag} reorder", cid, tag, mutate, generates=False)

    def queue_character(self, label, cid, mutate):
        """Queue a whole-manifest edit. Never generates: no cell's drawing changes, only its
        placement or its timing, so the repack costs the pack alone and every repaint is a cache
        hit."""
        def run(_progress):
            edit_character(cid, mutate)
            repack(cid)
        return submit(label, run, generates=False)

    def pivot(self, cid, tag, req):
        """Move an origin. `scope` says WHICH origin, because the two are different operations.

        `sheet` moves the origin of every cell in the character at once — it is the ground line the
        whole set is packed against. `frame` moves one cell's artwork relative to that shared
        origin and touches nothing else. Folding them into one "drag the pivot" action would mean
        a fix for one bad cell silently re-seated all 54.
        """
        man = RC.load_character_manifest(cid)
        cell = man["cell"]
        scope = req.get("scope")
        if scope == "sheet":
            x, y = _xy(req.get("pivot"), "pivot")
            _in_cell(x, y, cell, "pivot")
            return self.queue_character(f"{cid} sheet pivot -> {x},{y}", cid,
                                        lambda m: m.__setitem__("pivot", [x, y]))
        if scope == "frame":
            i = _cell_index(man, tag, req)
            dx, dy = _xy(req.get("pivot_nudge"), "pivot_nudge")
            px, py = man["pivot"]
            # The nudge is an offset, so it is the RESULT that must land inside the cell: the
            # artwork is pasted at pivot+nudge, and a nudge past the edge paints it into a
            # neighbouring cell of the atlas.
            _in_cell(px + dx, py + dy, cell, "nudged pivot")

            def mutate(frames):
                frames[i]["pivot_nudge"] = [dx, dy]
                return frames
            return self.queue_edit(f"{cid}-{tag} cell {i + 1} nudge {dx:+d},{dy:+d}",
                                   cid, tag, mutate, generates=False)
        raise ValueError('scope must be "sheet" or "frame"')

    def hold(self, cid, tag, req):
        """One cell's duration multiplier over its tag's fps — an animator sitting on an extreme.

        Not the tag's `hold_key`, which is the game freezing on the last cell while a key is down.
        """
        i = _cell_index(RC.load_character_manifest(cid), tag, req)
        beats = _whole(req.get("hold"), "hold")
        if beats < 1:
            raise ValueError("hold is a count of beats, so it cannot be less than 1")

        def mutate(frames):
            frames[i]["hold"] = beats
            return frames
        return self.queue_edit(f"{cid}-{tag} cell {i + 1} hold {beats}", cid, tag, mutate,
                               generates=False)

    DIRECTIONS = ("forward", "reverse", "pingpong")

    def tag_settings(self, cid, tag, req):
        fps = _whole(req.get("fps"), "fps")
        if fps <= 0:
            raise ValueError("fps must be greater than zero — both consumers divide by it")
        direction = req.get("direction")
        if direction not in self.DIRECTIONS:
            raise ValueError(f"direction must be one of {', '.join(self.DIRECTIONS)}")

        def mutate(man):
            t = man["tags"][RC._tag_index(man, tag)]
            t["fps"], t["direction"] = fps, direction
        return self.queue_character(f"{cid}-{tag} {fps}fps {direction}", cid, mutate)


PAGE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sprite cell editor</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #14120f; color: #e8e2d8;
         font: 16px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1180px; margin: 0 auto; padding: 1.6rem 1.25rem 4rem; }
  h1 { font-size: 1.5rem; margin: 0 0 .2rem; }
  p.lede { color: #b3a894; margin: 0 0 1.2rem; max-width: 68ch; }
  h2 { font-size: .95rem; letter-spacing: .04em; text-transform: uppercase; color: #c9bfae;
       margin: 1.8rem 0 .5rem; }
  .keys { display: flex; gap: .4rem; flex-wrap: wrap; margin: 0 0 .7rem; align-items: center; }
  button { font: 600 .85rem/1 ui-monospace, monospace; color: #e8e2d8; background: #241f19;
           border: 1px solid #4a4136; border-radius: 6px; padding: .55rem .7rem; cursor: pointer; }
  button.on { background: #d8a657; color: #1a1610; }
  button:disabled { opacity: .38; cursor: not-allowed; }
  .row { display: flex; gap: 1rem; align-items: flex-start; flex-wrap: wrap; }
  .pane { background: #efe9d8; border: 1px solid #3a332a; border-radius: 10px; padding: .6rem; }
  .pane img { display: block; width: 260px; height: 260px; object-fit: contain; }
  .pane canvas { display: block; width: 300px; height: 300px; }
  #bigcell { cursor: grab; touch-action: none; }
  #bigcell:active { cursor: grabbing; }
  .pane .cap { font: 600 .72rem/1.6 ui-monospace, monospace; color: #6b6152; letter-spacing: .06em;
               max-width: 300px; }
  .ctl { display: flex; gap: .8rem; align-items: center; flex-wrap: wrap; margin: 0 0 .7rem;
         font: .82rem/1.7 ui-monospace, monospace; color: #c9bfae; }
  .pane .ctl { color: #6b6152; margin: .5rem 0 0; }
  input, select { font: 600 .85rem/1 ui-monospace, monospace; color: #e8e2d8; background: #241f19;
                  border: 1px solid #4a4136; border-radius: 6px; padding: .45rem .5rem; }
  input[type=number] { width: 4.6rem; }
  input:disabled, select:disabled { opacity: .38; }
  .facts { font: .82rem/1.7 ui-monospace, monospace; color: #c9bfae; }
  .facts b { color: #d8a657; font-weight: 700; }
  .dl { font: 600 .85rem/1 ui-monospace, monospace; color: #e8e2d8; background: #241f19;
        border: 1px solid #4a4136; border-radius: 6px; padding: .55rem .7rem; text-decoration: none; }
  .dl:hover { border-color: #d8a657; color: #d8a657; }

  /* --- filmstrip ------------------------------------------------------------------------- */
  /* Tag bands line up with their frames BY CONSTRUCTION: each tag is a column whose width is
     set by the row of frames above the band. Absolutely positioning the bands over a single
     flat row looked identical until a hold made one frame wider, and then every band was off. */
  .film { background: #efe9d8; border: 1px solid #3a332a; border-radius: 10px; padding: .6rem;
          display: flex; gap: .5rem; overflow-x: auto; align-items: flex-start; }
  .tgroup { display: flex; flex-direction: column; gap: .3rem; }
  .frames { display: flex; gap: .25rem; }
  .band { border-radius: 4px; padding: .18rem .4rem; cursor: pointer; opacity: .5;
          font: 700 .68rem/1.5 ui-monospace, monospace; color: #1a1610; white-space: nowrap;
          overflow: hidden; text-overflow: ellipsis; border: 2px solid transparent; }
  .band.cur { opacity: 1; border-color: #3a332a; }
  .fbox { position: relative; border: 2px solid transparent; border-radius: 6px; padding: 2px 0 0;
          cursor: pointer; background: #efe9d8; box-sizing: border-box; flex: none; }
  .fbox.sel { border-color: #b4462f; }
  .fbox .n { font: 700 .7rem/1.3 ui-monospace, monospace; color: #6b6152; text-align: center; }
  .fbox .art { display: flex; align-items: stretch; }
  .fbox img, .fbox .empty { display: block; width: 72px; height: 72px; flex: none; }
  .fbox .empty { font: .6rem/1.2 ui-monospace, monospace; color: #6b6152; text-align: center;
                 padding-top: 26px; box-sizing: border-box; }
  /* The extra width IS the hold. The stripes make it read as duration rather than as a wider
     drawing, which a plain stretched thumbnail did not. */
  .fbox .held { flex: 1 1 auto; border-radius: 0 3px 3px 0;
                background: repeating-linear-gradient(135deg, #d8c9a4 0 6px, #efe9d8 6px 12px); }
  .fbox .m { font: .62rem/1.3 ui-monospace, monospace; color: #6b6152; text-align: center;
             white-space: pre; }
  .fbox .grip { position: absolute; top: 18px; right: 0; width: 7px; bottom: 20px;
                border-radius: 0 4px 4px 0; cursor: ew-resize; touch-action: none;
                background: linear-gradient(#d6cbb0, #b09a6d); border-left: 1px solid #efe9d8; }
  .fbox .grip:hover { background: #b4462f; }

  /* --- transport ------------------------------------------------------------------------- */
  .transport { display: flex; gap: .3rem; align-items: center; margin-top: .5rem; }
  .transport button { padding: .4rem .5rem; }
  .transport input[type=range] { flex: 1 1 auto; width: 100%; min-width: 0; accent-color: #b4462f; }

  .picker { background: #efe9d8; border: 1px solid #3a332a; border-radius: 10px; padding: .5rem;
            display: flex; gap: .3rem; overflow-x: auto; max-width: 100%; }
  .picker figure { margin: 0; cursor: pointer; border: 2px solid transparent; border-radius: 4px; }
  .picker figure.cur { border-color: #b4462f; }
  .picker img { display: block; width: 76px; height: 76px; }
  .picker figcaption { font: .62rem/1.4 ui-monospace, monospace; color: #6b6152; text-align: center; }
  button.thumb { padding: .25rem; display: flex; flex-direction: column; align-items: center;
                 gap: .2rem; background: #efe9d8; }
  button.thumb img { display: block; width: 74px; height: 74px; }
  button.thumb span { font: .62rem/1.3 ui-monospace, monospace; color: #6b6152; }
  button.thumb.on { background: #d8a657; }
  #status { border-left: 3px solid #d8a657; padding: .35rem 0 .35rem .9rem; margin: 1rem 0;
            color: #c9bfae; font: .85rem/1.5 ui-monospace, monospace; min-height: 1.5em; }
  #status.err { border-color: #b4462f; color: #e6a394; }
  .muted { color: #8d8371; }
  kbd { font: 600 .75rem/1 ui-monospace, monospace; border: 1px solid #4a4136; border-radius: 4px;
        padding: .15rem .3rem; color: #c9bfae; }
</style>
<main>
  <h1>Sprite cell editor</h1>
  <p class="lede">One filmstrip per character: every frame in order, with a tag band under the
  range it covers. Click a frame to select it, drag its right edge to hold it longer, click a band
  to make that tag current. The onion skin ghosts the neighbouring frames on the same canvas as the
  ground line and the pivot crosshair — that is how a scale drift or a slipping foot becomes
  visible. Each edit writes <code>sheets/&lt;cid&gt;.json</code> and repacks the whole character.
  <span class="muted"><kbd>←</kbd><kbd>→</kbd> step frames · <kbd>space</kbd> play/pause</span></p>

  <div class="keys" id="chars"></div>
  <div id="status">ready</div>

  <h2>Filmstrip</h2>
  <div class="film" id="film"></div>

  <div class="row" style="margin-top:1rem">
    <div class="pane">
      <canvas id="bigcell" width="512" height="512"></canvas>
      <div class="cap" id="bigcap"></div>
      <div class="keys" style="margin-top:.5rem" id="pivotmode"></div>
      <div class="cap" id="pivotcap"></div>
      <div class="keys" style="margin-top:.5rem" id="onionkeys"></div>
      <div class="keys" id="onionmode"></div>
      <div class="cap" id="onioncap"></div>
    </div>
    <div class="pane">
      <canvas id="pv" width="512" height="512"></canvas>
      <div class="transport">
        <button id="t-first" title="first frame">|&#9664;</button>
        <button id="t-prev" title="step back">&#9664;</button>
        <button id="t-play" title="play / pause">&#9654;</button>
        <button id="t-next" title="step forward">&#9654;</button>
        <button id="t-last" title="last frame">&#9654;|</button>
      </div>
      <div class="transport"><input id="scrub" type="range" min="0" max="0" step="1" value="0"></div>
      <div class="cap" id="pvcap"></div>
      <div class="ctl">
        <label>fps <input id="fps" type="number" min="1" step="1"></label>
        <label>direction <select id="direction">
          <option value="forward">forward</option>
          <option value="reverse">reverse</option>
          <option value="pingpong">pingpong</option>
        </select></label>
      </div>
    </div>
    <div style="flex: 1 1 300px">
      <div class="facts" id="facts"></div>
      <div class="ctl" style="margin-top:.8rem">
        <label>hold <input id="hold" type="number" min="1" step="1"></label>
        <span class="muted">beats at this tag's fps</span>
      </div>
      <div class="keys" style="margin-top:.8rem">
        <button id="b-reroll">Re-roll</button>
        <button id="b-repick">Re-pick&#8230;</button>
        <button id="b-drop">Drop</button>
        <button id="b-left">&#9664; move left</button>
        <button id="b-right">move right &#9654;</button>
      </div>
    </div>
  </div>

  <div id="sel" hidden>
    <h2>Source frame <span class="muted">&mdash; the drawing above was painted from this</span></h2>
    <div class="row">
      <div class="pane"><img id="bigsrc" alt="source frame"><div class="cap" id="srccap"></div></div>
      <div style="flex:1 1 380px">
        <h2 style="margin-top:0">Versions of this cell <span class="muted">&mdash; every seed already painted for this source frame; clicking one is instant and costs no GPU</span></h2>
        <div class="picker" id="variants"></div>
      </div>
    </div>

    <div id="pickwrap" hidden>
      <h2>Re-pick source frame <span class="muted" id="pickcount"></span></h2>
      <div class="picker" id="picker"></div>
    </div>
  </div>

  <div class="out">
    <h2>Where this ends up</h2>
    <p class="muted">The atlas and its frame table are the deliverable; every preset below is a
    template over that same table, generated fresh on each click.</p>
    <div class="keys" id="downloads"></div>
    <div class="keys" id="exports"></div>
  </div>
</main>
<script>
// The tag list is not hard-coded here: it comes off the character manifest, so a tag added or
// renamed in `sheets/<cid>.json` appears without touching this page.
let cid = "seth", tagName = "walk", ST = null, sel = 0, busy = false, bust = Date.now();
// Dragging the crosshair is TWO operations, and the page makes you say which, because they are not
// interchangeable: the sheet pivot is the ground line the whole character is packed against and
// moving it re-seats all 54 cells, while a frame's nudge moves one drawing relative to that shared
// line. Both are posted as the DELTA the pointer travelled, so the gesture is identical and only
// the field it lands in differs. "this cell" is the default because it is the reversible one.
let pivotMode = "frame", cellImg = null, drag = null;
// Onion skin is a QA overlay here, not a drawing aid: prev on by default, because the drift this
// catches is between a cell and the one before it.
let onion = {prev: true, next: false, mode: "alpha"};
const GHOST_ALPHA = 0.3;
const PREV_INK = "#b4462f", NEXT_INK = "#3a6ea5";
// Preview transport. `seq` is the play order for the current tag with `hold` expanded, so the
// scrubber counts BEATS, not cells -- that is what the game plays and what the fps divides.
let seq = [], head = 0, playing = true, timer = null, rest = 0;
let atlasImg = null, atlasKey = "", atlasLoading = "";
let holdDrag = null, suppressClick = false, scrubSelect = null;
const HOLD_PX = 26;   // one extra beat widens a frame box by this much, and drag reads it back

const img = (p, w) => "/img?path=" + encodeURIComponent(p) + (w ? "&w=" + w : "") + "&v=" + bust;
const cellSrc = (i, w) => "/cell?cid=" + cid + "&i=" + i + (w ? "&w=" + w : "") + "&v=" + bust;
const el = (id) => document.getElementById(id);
const TAG = () => ST.tags.find(t => t.name === tagName) || ST.tags[0];
const CELLS = () => TAG().cells;
const CUR = () => CELLS()[sel];

function status(msg, err) {
  el("status").textContent = msg;
  el("status").className = err ? "err" : "";
}

function buttons(host, items, current, pick, live) {
  host.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    b.textContent = it;
    if (it === current) b.className = "on";
    b.disabled = busy && !live;
    b.onclick = () => pick(it);
    host.appendChild(b);
  }
}

// A tag's colour has to survive a reload and a rename of its neighbours, so it is derived from the
// name rather than handed out by position: adding a tag must not recolour the other six.
function tagInk(name) {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return "hsl(" + h + ", 52%, 62%)";
}

// Global frame numbers run across the whole character, exactly as the manifest and the atlas
// number them. The per-tag index is what every endpoint takes, so both are carried together.
function flat() {
  const out = [];
  for (const t of ST.tags) t.cells.forEach((c, i) => out.push({tag: t, i, c}));
  return out;
}

async function load() {
  const r = await fetch("/api/state?cid=" + cid + "&v=" + Date.now());
  ST = await r.json();
  if (ST.error) { status(ST.error, true); return; }
  tagName = TAG().name;
  if (sel >= CELLS().length) sel = Math.max(0, CELLS().length - 1);
  render();
}

// Full rebuild: refetches every thumbnail, so it runs on a state load and a tag switch, never on
// a plain selection change. Nothing here is cacheable (the atlas is overwritten in place), so a
// rebuild per arrow key would put 54 image requests on every keystroke.
function render() {
  renderDownloads();
  buttons(el("chars"), ST.characters, cid, (c) => {
    cid = c; sel = 0; tagName = ""; el("pickwrap").hidden = true; load();
  });
  renderFilm();
  renderTagControls();
  rebuildSequence();
  renderSelected();
}

function renderFilm() {
  const film = el("film");
  film.innerHTML = "";
  let n = 0;
  for (const tag of ST.tags) {
    const group = document.createElement("div");
    group.className = "tgroup";
    const frames = document.createElement("div");
    frames.className = "frames";
    tag.cells.forEach((c, i) => frames.appendChild(frameBox(tag, i, c, ++n)));
    const band = document.createElement("div");
    band.className = "band" + (tag.name === tagName ? " cur" : "");
    band.style.background = tagInk(tag.name);
    band.textContent = tag.name + " · " + tag.fps + "fps · " + tag.cells.length + "f";
    band.title = tag.name + ", " + tag.cells.length + " frames at " + tag.fps + "fps, " +
                 tag.direction + (tag.loop ? ", loop" : tag.hold_key ? ", hold" : ", once");
    band.onclick = () => { if (tag.name !== tagName) selectFrame(tag.name, 0); };
    group.appendChild(frames);
    group.appendChild(band);
    film.appendChild(group);
  }
  scrollSelectionIntoView();
}

function frameBox(tag, i, c, n) {
  const box = document.createElement("div");
  const chosen = tag.name === tagName && i === sel;
  box.className = "fbox" + (chosen ? " sel" : "");
  box.style.width = (80 + (Math.max(1, c.hold) - 1) * HOLD_PX) + "px";
  box.dataset.tag = tag.name;
  box.dataset.i = i;
  const m = c.metrics;
  box.innerHTML = '<div class="n">' + n + (c.hold > 1 ? " ×" + c.hold : "") + "</div>" +
    '<div class="art">' +
    (c.atlas !== null ? '<img src="' + cellSrc(c.atlas, 72) + '" alt="frame ' + n + '">'
                      : '<div class="empty">not packed</div>') +
    (c.hold > 1 ? '<div class="held"></div>' : "") + "</div>" +
    '<div class="m">' + (m ? "h" + m.height + " f" + (m.feet > 0 ? "+" : "") + m.feet : "empty") +
    (c.pivot_nudge[0] || c.pivot_nudge[1] ? " ◈" : "") + "</div>";
  // A drag that ends off the box never produces a click, so the suppression flag is cleared on
  // the next press as well as on the click it was raised for -- otherwise one duration drag ate
  // the next frame selection.
  box.addEventListener("pointerdown", () => { suppressClick = false; });
  box.onclick = () => {
    if (suppressClick) { suppressClick = false; return; }
    selectFrame(tag.name, i);
  };
  const grip = document.createElement("div");
  grip.className = "grip";
  grip.title = "drag to change how long frame " + n + " is held";
  grip.addEventListener("pointerdown", (ev) => startHoldDrag(ev, tag, i, c, box, n));
  box.appendChild(grip);
  return box;
}

// Duration is edited ON the strip, because the panel version made every timing change a trip
// away from the thing being timed. The drag is applied optimistically to the box width and only
// POSTed on release, so a drag that ends where it started costs nothing.
function startHoldDrag(ev, tag, i, c, box, n) {
  if (busy) return;
  ev.stopPropagation();
  ev.preventDefault();
  ev.target.setPointerCapture(ev.pointerId);
  holdDrag = {tag: tag.name, i, from: ev.clientX, start: Math.max(1, c.hold), hold: Math.max(1, c.hold), box, n};
  const move = (e) => {
    if (!holdDrag) return;
    const v = Math.max(1, holdDrag.start + Math.round((e.clientX - holdDrag.from) / HOLD_PX));
    if (v === holdDrag.hold) return;
    holdDrag.hold = v;
    box.style.width = (80 + (v - 1) * HOLD_PX) + "px";
    box.querySelector(".n").textContent = n + (v > 1 ? " ×" + v : "");
    status("frame " + n + " held for " + v + " beat" + (v === 1 ? "" : "s") + " — release to apply");
  };
  const up = () => {
    ev.target.removeEventListener("pointermove", move);
    ev.target.removeEventListener("pointerup", up);
    ev.target.removeEventListener("pointercancel", up);
    const d = holdDrag;
    holdDrag = null;
    suppressClick = true;
    if (!d || d.hold === d.start) { status("ready"); render(); return; }
    tagName = d.tag; sel = d.i;
    post("/api/hold", {index: d.i, hold: d.hold}, "frame " + d.n + " held for " + d.hold);
  };
  ev.target.addEventListener("pointermove", move);
  ev.target.addEventListener("pointerup", up);
  ev.target.addEventListener("pointercancel", up);
}

function scrollSelectionIntoView() {
  const box = el("film").querySelector(".fbox.sel");
  if (!box) return;
  const film = el("film");
  const left = box.offsetLeft - film.offsetLeft;
  // Scrolls the strip only, never the page: scrollIntoView on a 4000px-wide strip jumped the
  // whole document every time an arrow key moved the selection.
  if (left < film.scrollLeft) film.scrollLeft = left - 12;
  else if (left + box.offsetWidth > film.scrollLeft + film.clientWidth)
    film.scrollLeft = left + box.offsetWidth - film.clientWidth + 12;
}

// A selection change inside the current tag repaints the marks and the panels; a change of tag
// rebuilds the strip's bands and the preview sequence with it.
function selectFrame(name, i) {
  const switching = name !== tagName;
  tagName = name; sel = i;
  el("pickwrap").hidden = true;
  if (switching) {
    render();
  } else {
    for (const b of el("film").querySelectorAll(".fbox"))
      b.classList.toggle("sel", b.dataset.tag === name && +b.dataset.i === i);
    renderSelected();
    scrollSelectionIntoView();
  }
  seekToCell(i);
}

function stepSelection(delta) {
  const all = flat();
  const at = all.findIndex(f => f.tag.name === tagName && f.i === sel);
  const next = all[Math.min(all.length - 1, Math.max(0, at + delta))];
  if (next && (next.tag.name !== tagName || next.i !== sel)) selectFrame(next.tag.name, next.i);
}

function renderTagControls() {
  const tag = TAG();
  el("fps").value = tag.fps;
  el("direction").value = tag.direction;
  el("fps").disabled = el("direction").disabled = busy;
}

// --- preview transport -----------------------------------------------------------------------
// Ping-pong follows the Aseprite convention the demo page uses: play from -> to, then back down
// to (but not including) the first cell. Per-frame `hold` repeats a cell in the step list, which
// is the same thing the game does with it.
function rebuildSequence() {
  const tag = TAG(), order = [];
  if (tag.direction === "reverse") {
    for (let i = tag.cells.length - 1; i >= 0; i--) order.push(i);
  } else if (tag.direction === "pingpong") {
    for (let i = 0; i < tag.cells.length; i++) order.push(i);
    for (let i = tag.cells.length - 2; i > 0; i--) order.push(i);
  } else {
    for (let i = 0; i < tag.cells.length; i++) order.push(i);
  }
  seq = [];
  for (const i of order) {
    const c = tag.cells[i];
    if (!c.xy) continue;
    for (let k = 0; k < Math.max(1, c.hold); k++) seq.push({i, xy: c.xy});
  }
  const scrub = el("scrub");
  scrub.max = String(Math.max(0, seq.length - 1));
  scrub.disabled = seq.length < 2;
  if (head >= seq.length) head = 0;
  rest = 0;
  retime();
  seekToCell(sel);
  drawPreview();
}

function seekToCell(i) {
  // Already showing a beat of that cell? Leave the playhead alone. Snapping to the cell's FIRST
  // beat here made "step back" skip every held beat of the cell it stepped into.
  if (seq[head] && seq[head].i === i) return;
  const at = seq.findIndex(s => s.i === i);
  if (at >= 0) { head = at; el("scrub").value = String(head); drawPreview(); }
}

function retime() {
  if (timer) { clearInterval(timer); timer = null; }
  el("t-play").innerHTML = playing ? "&#10073;&#10073;" : "&#9654;";
  el("t-play").className = playing ? "on" : "";
  if (!playing || seq.length < 2) return;
  timer = setInterval(advance, 1000 / TAG().fps);
}

function advance() {
  const tag = TAG();
  // A once-tag ends on its last cell in the game. Here it rests there and replays, because a
  // preview frozen on the recovery pose tells you nothing about the motion.
  if (!tag.loop && head === seq.length - 1 && rest < Math.ceil(tag.fps * 0.7)) { rest++; return; }
  rest = 0;
  head = (head + 1) % seq.length;
  el("scrub").value = String(head);
  drawPreview();
}

// Keyed on the PATH being fetched, never on `cid`. `cid` changes the instant the character
// button is clicked, while `ST` is still the previous character until its fetch lands -- keying on
// cid cached the outgoing character's atlas under the incoming one's name, and the preview then
// played Seth's drawings at Cadbury's cell coordinates until the next repack.
// One in-flight load at a time as well: the preview redraws every beat, and a 10MB atlas fetched
// per beat starved the thumbnails on the same connection.
function withAtlas(cb) {
  const path = ST.atlas, key = path + ":" + bust;
  if (atlasKey === key && atlasImg) { cb(atlasImg); return; }
  if (atlasLoading === key) return;
  atlasLoading = key;
  const im = new Image();
  im.onload = () => { atlasImg = im; atlasKey = key; atlasLoading = ""; cb(im); };
  im.onerror = () => { atlasLoading = ""; };
  im.src = img(path);
}

function drawPreview() {
  const cv = el("pv"), ctx = cv.getContext("2d");
  const tag = TAG();
  el("pvcap").textContent = tag.name + " — beat " + (seq.length ? head + 1 : 0) + "/" +
    seq.length + " · cell " + (seq.length ? seq[head].i + 1 : 0) + "/" + tag.cells.length +
    " · " + tag.fps + "fps " + tag.direction +
    (tag.loop ? " loop" : tag.hold_key ? " hold" : " once");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!seq.length) return;
  const step = seq[head];
  withAtlas((sheet) => {
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(sheet, step.xy[0], step.xy[1], ST.cell, ST.cell, 0, 0, cv.width, cv.height);
  });
}

function transport(fn) { return () => { if (!ST || !seq.length) return; fn(); }; }
el("t-play").onclick = transport(() => { playing = !playing; retime(); });
el("t-first").onclick = transport(() => { pause(); head = 0; syncScrub(); });
el("t-last").onclick = transport(() => { pause(); head = seq.length - 1; syncScrub(); });
el("t-prev").onclick = transport(() => { pause(); head = (head - 1 + seq.length) % seq.length; syncScrub(); });
el("t-next").onclick = transport(() => { pause(); head = (head + 1) % seq.length; syncScrub(); });
function pause() { if (playing) { playing = false; retime(); } }
function syncScrub() {
  el("scrub").value = String(head);
  drawPreview();
  // Stepping the transport moves the SELECTION with it, so the canvas, the metrics and the
  // onion ghost describe the frame you just stepped onto. Playback deliberately does not:
  // reselecting 12 times a second would refetch the source frame and the variants each beat.
  if (seq[head] && seq[head].i !== sel) selectFrame(tagName, seq[head].i);
}
el("scrub").oninput = () => {
  pause();
  head = parseInt(el("scrub").value, 10) || 0;
  drawPreview();
  // Debounced: dragging the scrubber across 30 beats would otherwise fire 30 selection changes,
  // each refetching a 512px cell, a source frame and the variant strip.
  if (scrubSelect) clearTimeout(scrubSelect);
  scrubSelect = setTimeout(() => {
    if (seq[head] && seq[head].i !== sel) selectFrame(tagName, seq[head].i);
  }, 150);
};

// --- the selected cell -----------------------------------------------------------------------
function renderSelected() {
  const c = CUR(), n = CELLS().length;
  el("sel").hidden = !c;
  if (!c) return;
  const vs = c.variants || [];
  el("variants").innerHTML = "";
  for (const v of vs) {
    const b = document.createElement("button");
    b.className = "thumb" + (v.png === c.png ? " on" : "");
    b.innerHTML = '<img src="' + img(v.png, 150) + '"><span>seed ' +
                  (v.seed === 77 ? "77 (first)" : v.seed) + "</span>";
    b.disabled = busy || v.png === c.png;
    b.onclick = () => post("/api/use", {index: c.index, png: v.png}, "use seed " + v.seed);
    el("variants").appendChild(b);
  }
  cellImg = null;
  paintCell(c);
  if (c.atlas !== null) {
    withCell(c.atlas, (im) => { if (CUR() === c) { cellImg = im; paintCell(c); } });
    for (const g of ghosts(c)) withCell(g.atlas, () => { if (CUR() === c) paintCell(c); });
  }
  buttons(el("pivotmode"), ["drag: this cell", "drag: whole sheet"],
          pivotMode === "sheet" ? "drag: whole sheet" : "drag: this cell",
          (m) => { pivotMode = m === "drag: whole sheet" ? "sheet" : "frame"; renderSelected(); });
  el("pivotcap").textContent = "pivot " + ST.pivot.join(",") + " · nudge " +
    c.pivot_nudge.map(v => (v > 0 ? "+" : "") + v).join(",");
  renderOnionControls(c);
  el("hold").value = c.hold;
  el("hold").disabled = busy;
  el("bigcap").textContent = "cell " + (c.index + 1) + " of " + tagName + " — packed, seed " + c.seed;
  el("bigsrc").src = img(c.src, 512);
  el("srccap").textContent = "source — " + c.src.split("/").pop();
  const m = c.metrics;
  el("facts").innerHTML = m
    ? "silhouette height <b>" + m.height + "</b>px<br>top clearance <b>" + m.top +
      "</b>px<br>feet offset <b>" + (m.feet > 0 ? "+" : "") + m.feet +
      "</b>px from the ground line (y=" + m.ground + ")<br>seed <b>" + c.seed + "</b>"
    : "no packed cell in the atlas yet";
  for (const [id, off] of [["b-reroll", false], ["b-repick", false], ["b-drop", n <= 1],
                           ["b-left", c.index === 0], ["b-right", c.index === n - 1]]) {
    el(id).disabled = busy || off;
  }
}

// --- onion skin ------------------------------------------------------------------------------
// The neighbours are the cells either side WITHIN the tag, not either side in the atlas: the
// frame after the last cell of `walk` is the first of `punch`, and ghosting that compares two
// unrelated poses.
function ghosts(c) {
  const cells = CELLS(), out = [];
  if (onion.prev && sel > 0 && cells[sel - 1].atlas !== null)
    out.push({atlas: cells[sel - 1].atlas, ink: PREV_INK, label: "prev"});
  if (onion.next && sel < cells.length - 1 && cells[sel + 1].atlas !== null)
    out.push({atlas: cells[sel + 1].atlas, ink: NEXT_INK, label: "next"});
  return out;
}

// Cached by URL, so the cache can never disagree with what was fetched. A 512px cell is small,
// but the ghost redraws on every mode toggle and every selection step, and the server sends
// no-store -- without this each toggle refetched three cells.
const CELL_IMGS = new Map();
function withCell(atlasIndex, cb) {
  const url = cellSrc(atlasIndex);
  const hit = CELL_IMGS.get(url);
  if (hit) { if (hit.complete && hit.naturalWidth) cb(hit); return; }
  const im = new Image();
  CELL_IMGS.set(url, im);
  im.onload = () => cb(im);
  im.src = url;
}
function cachedCell(atlasIndex) {
  const im = CELL_IMGS.get(cellSrc(atlasIndex));
  return im && im.complete && im.naturalWidth ? im : null;
}

const OUTLINES = new Map();
// Plain alpha reads badly when both drawings are the same sepia ink on cream: two nearly aligned
// silhouettes just look like one slightly bolder one. The outline is the silhouette's edge only,
// which is what you actually compare when checking a foot or a shoulder for drift.
function outlineOf(im, ink, key) {
  const hit = OUTLINES.get(key);
  if (hit) return hit;
  const cell = ST.cell;
  const cv = document.createElement("canvas");
  cv.width = cell; cv.height = cell;
  const ctx = cv.getContext("2d");
  const R = 3;
  for (const [dx, dy] of [[-R, 0], [R, 0], [0, -R], [0, R]]) ctx.drawImage(im, dx, dy, cell, cell);
  ctx.globalCompositeOperation = "destination-out";
  ctx.drawImage(im, 0, 0, cell, cell);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = ink;
  ctx.fillRect(0, 0, cell, cell);
  OUTLINES.set(key, cv);
  return cv;
}

function renderOnionControls(c) {
  const keys = el("onionkeys");
  keys.innerHTML = "";
  for (const which of ["prev", "next"]) {
    const b = document.createElement("button");
    b.textContent = "ghost " + which;
    b.style.borderColor = which === "prev" ? PREV_INK : NEXT_INK;
    if (onion[which]) b.className = "on";
    b.onclick = () => { onion[which] = !onion[which]; renderSelected(); };
    keys.appendChild(b);
  }
  buttons(el("onionmode"), ["alpha", "outline", "difference"], onion.mode,
          (m) => { onion.mode = m; renderSelected(); }, true);
  const g = ghosts(c);
  el("onioncap").textContent = g.length
    ? "ghosting " + g.map(x => x.label).join(" + ") + " at " + onion.mode +
      (onion.mode === "difference" ? " — black means aligned" : "")
    : "no ghost — " + (onion.prev || onion.next ? "no neighbouring cell in this tag" : "both off");
}

// The crosshair and the ground line are drawn from the manifest, never from a constant in this
// page: the ground line IS the sheet pivot's y, and the crosshair is where this cell's artwork is
// actually anchored, which is pivot + this frame's nudge.
function paintCell(c) {
  const cv = el("bigcell"), cell = ST.cell;
  if (cv.width !== cell) { cv.width = cell; cv.height = cell; }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cell, cell);
  const g = ghosts(c);
  if (onion.mode === "alpha") {
    // Under the selected cell, so the current drawing stays the one you read.
    ctx.globalAlpha = GHOST_ALPHA;
    for (const x of g) { const im = cachedCell(x.atlas); if (im) ctx.drawImage(im, 0, 0, cell, cell); }
    ctx.globalAlpha = 1;
    if (cellImg) ctx.drawImage(cellImg, 0, 0, cell, cell);
  } else if (onion.mode === "difference") {
    if (cellImg) ctx.drawImage(cellImg, 0, 0, cell, cell);
    ctx.globalCompositeOperation = "difference";
    for (const x of g) { const im = cachedCell(x.atlas); if (im) ctx.drawImage(im, 0, 0, cell, cell); }
    ctx.globalCompositeOperation = "source-over";
  } else {
    if (cellImg) ctx.drawImage(cellImg, 0, 0, cell, cell);
    for (const x of g) {
      const im = cachedCell(x.atlas);
      if (im) ctx.drawImage(outlineOf(im, x.ink, cellSrc(x.atlas) + ":" + x.ink), 0, 0);
    }
  }
  const d = drag ? drag.d : [0, 0];
  const sheet = pivotMode === "sheet" ? [ST.pivot[0] + d[0], ST.pivot[1] + d[1]] : ST.pivot;
  const nudge = pivotMode === "frame" ? [c.pivot_nudge[0] + d[0], c.pivot_nudge[1] + d[1]]
                                      : c.pivot_nudge;
  const art = [sheet[0] + nudge[0], sheet[1] + nudge[1]];

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#b4462f";
  ctx.setLineDash([9, 7]);
  ctx.beginPath(); ctx.moveTo(0, sheet[1]); ctx.lineTo(cell, sheet[1]); ctx.stroke();
  ctx.globalAlpha = 0.4;
  ctx.beginPath(); ctx.moveTo(sheet[0], 0); ctx.lineTo(sheet[0], cell); ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);

  ctx.strokeStyle = "#14120f";
  ctx.lineWidth = 5;
  for (let pass = 0; pass < 2; pass++) {
    // Drawn twice, dark then gold: a thin gold crosshair vanishes against the cream cell and the
    // sepia ink both, and this is the one thing on the canvas that has to be seen exactly.
    ctx.beginPath();
    ctx.moveTo(art[0] - 22, art[1]); ctx.lineTo(art[0] + 22, art[1]);
    ctx.moveTo(art[0], art[1] - 22); ctx.lineTo(art[0], art[1] + 22);
    ctx.stroke();
    ctx.beginPath(); ctx.arc(art[0], art[1], 11, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = "#d8a657";
    ctx.lineWidth = 2;
  }
}

function cellPoint(ev) {
  const r = el("bigcell").getBoundingClientRect();
  return [Math.round((ev.clientX - r.left) * ST.cell / r.width),
          Math.round((ev.clientY - r.top) * ST.cell / r.height)];
}

el("bigcell").addEventListener("pointerdown", (ev) => {
  if (busy || !ST || CUR().atlas === null) return;
  el("bigcell").setPointerCapture(ev.pointerId);
  drag = {from: cellPoint(ev), d: [0, 0]};
});
el("bigcell").addEventListener("pointermove", (ev) => {
  if (!drag) return;
  const p = cellPoint(ev);
  drag.d = [p[0] - drag.from[0], p[1] - drag.from[1]];
  paintCell(CUR());
});
el("bigcell").addEventListener("pointercancel", () => { drag = null; paintCell(CUR()); });
el("bigcell").addEventListener("pointerup", () => {
  if (!drag) return;
  const d = drag.d, c = CUR();
  drag = null;
  if (!d[0] && !d[1]) { paintCell(c); return; }
  if (pivotMode === "sheet") {
    post("/api/pivot", {scope: "sheet", pivot: [ST.pivot[0] + d[0], ST.pivot[1] + d[1]]},
         "sheet pivot — every cell of " + cid);
  } else {
    post("/api/pivot", {scope: "frame", index: c.index,
                        pivot_nudge: [c.pivot_nudge[0] + d[0], c.pivot_nudge[1] + d[1]]},
         "nudge cell " + (c.index + 1) + " only");
  }
});

el("hold").onchange = () => {
  const c = CUR(), v = parseInt(el("hold").value, 10);
  if (!Number.isInteger(v) || v < 1) {
    status("hold is a count of beats, so it cannot be less than 1", true);
    el("hold").value = c.hold;
    return;
  }
  if (v === c.hold) return;
  post("/api/hold", {index: c.index, hold: v}, "cell " + (c.index + 1) + " held for " + v);
};

function postTag() {
  const tag = TAG(), fps = parseInt(el("fps").value, 10), direction = el("direction").value;
  if (!Number.isInteger(fps) || fps < 1) {
    status("fps must be a whole number greater than zero", true);
    el("fps").value = tag.fps;
    return;
  }
  if (fps === tag.fps && direction === tag.direction) return;
  post("/api/tag", {fps, direction}, tagName + " at " + fps + "fps " + direction);
}
el("fps").onchange = postTag;
el("direction").onchange = postTag;

// Arrows and space are the two shortcuts every timeline has. They are ignored while a field has
// focus, or space would type into the fps box and an arrow would silently change its value.
addEventListener("keydown", (ev) => {
  if (!ST || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const t = ev.target, tag = t && t.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || (t && t.isContentEditable)) return;
  if (ev.key === "ArrowLeft") { ev.preventDefault(); stepSelection(-1); }
  else if (ev.key === "ArrowRight") { ev.preventDefault(); stepSelection(1); }
  else if (ev.key === " " && tag !== "BUTTON") {
    // A focused button keeps space for itself, which is what a keyboard user expects of it.
    ev.preventDefault();
    if (seq.length) { playing = !playing; retime(); }
  }
});

function renderDownloads() {
  const box = el("downloads");
  const base = ST.outdir + "/" + cid;
  box.innerHTML = '<div class="muted" style="width:100%">' + ST.outdir + "</div>";
  for (const [path, label] of [[base + ".png", "atlas PNG"], [base + ".json", "frame table JSON"]]) {
    const a = document.createElement("a");
    a.className = "dl"; a.href = "/img?path=" + encodeURIComponent(path) + "&dl=1";
    a.download = path.split("/").pop(); a.textContent = "↓ " + label;
    box.appendChild(a);
  }
  const demo = document.createElement("a");
  demo.className = "dl"; demo.href = "http://wilson/cast-fighter.html"; demo.target = "_blank";
  demo.textContent = "↗ play it in the demo";
  box.appendChild(demo);

  // The server names and types each file, so these are plain links: a fetch-and-blob dance would
  // only duplicate the Content-Disposition the endpoint already sends. The EMPTY download
  // attribute keeps the filename the server chose while marking the click as a download rather
  // than a navigation -- without it the browser aborts the navigation once the attachment header
  // arrives, and every export shows up in the network log as a failed request.
  const ex = el("exports");
  ex.innerHTML = "";
  for (const [fmt, label] of [["json-hash", "JSON hash"], ["json-array", "JSON array"],
                              ["phaser3", "Phaser 3"], ["godot", "Godot SpriteFrames"],
                              ["css", "CSS steps()"]]) {
    const a = document.createElement("a");
    a.className = "dl"; a.href = "/api/export?cid=" + cid + "&format=" + fmt;
    a.download = "";
    a.textContent = "↓ " + label;
    ex.appendChild(a);
  }
}

async function post(url, body, label) {
  busy = true; render(); status(label + " — queued…");
  const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"},
                             body: JSON.stringify(Object.assign({cid, tag: tagName}, body))});
  const j = await r.json();
  if (!r.ok || j.error) { busy = false; status(j.error || "request failed", true); render(); return; }
  poll(j.job);
}

function poll(jid) {
  const tick = async () => {
    const r = await fetch("/api/job/" + jid + "?v=" + Date.now());
    const j = await r.json();
    status(j.message, j.state === "error");
    if (j.state === "done" || j.state === "error") {
      busy = false;
      bust = Date.now();          // the atlas was overwritten in place; force every img to refetch
      atlasImg = null; CELL_IMGS.clear(); OUTLINES.clear();
      await load();
      return;
    }
    setTimeout(tick, 1500);
  };
  tick();
}

el("b-reroll").onclick = () => post("/api/reroll", {index: sel}, "re-roll cell " + (sel + 1));
el("b-drop").onclick = () => post("/api/drop", {index: sel}, "drop cell " + (sel + 1));
el("b-left").onclick = () => swap(sel, sel - 1);
el("b-right").onclick = () => swap(sel, sel + 1);

function swap(a, b) {
  const order = CELLS().map((c, i) => i);
  [order[a], order[b]] = [order[b], order[a]];
  sel = b;
  post("/api/reorder", {order}, "reorder");
}

el("b-repick").onclick = async () => {
  const wrap = el("pickwrap");
  if (!wrap.hidden) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const picker = el("picker");
  picker.innerHTML = "loading frames…";
  const r = await fetch("/api/frames?cid=" + cid + "&tag=" + tagName);
  const frames = (await r.json()).frames;
  const cur = CUR().src;
  el("pickcount").textContent = "(" + frames.length + " available)";
  picker.innerHTML = "";
  for (const f of frames) {
    const fig = document.createElement("figure");
    if (f === cur) fig.className = "cur";
    fig.innerHTML = '<img loading="lazy" src="' + img(f, 96) + '"><figcaption>' +
                    f.split("/").pop().replace(".png", "").replace("f_", "") + "</figcaption>";
    fig.onclick = () => { wrap.hidden = true; post("/api/repick", {index: sel, src: f}, "re-pick cell " + (sel + 1)); };
    picker.appendChild(fig);
  }
};

load();
</script>
"""


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    print(f"sprite editor on http://0.0.0.0:{PORT}/", flush=True)
    Server(("0.0.0.0", PORT), Handler).serve_forever()
