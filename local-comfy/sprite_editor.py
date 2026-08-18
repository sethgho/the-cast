#!/usr/bin/env python3
"""A per-cell editor for the sprite manifests, so a bad cell costs one repaint instead of a move.

    python3 sprite_editor.py        # http://<host>:8811/

## Why

`repaint_cells.py` gets a move to about 90% on its own, and the last 10% is always the same
judgement: this cell's hand came back garbled, that one was picked two frames too early, this one
drifted in scale. The manifest in `sheets/` exists so those judgements survive a rerun. Editing it
by hand works but is blind — you cannot see which cell is cell 4, and you cannot tell a bad REPAINT
from a bad SOURCE FRAME without opening two files in two viewers. That distinction decides the fix:
a bad repaint wants a new seed, a bad source frame wants a different frame.

So the page shows both, side by side, for the selected cell.

Nothing about packing lives here. Every mutation writes the manifest and then calls
`repaint_cells.main()`, which repaints whatever is missing and repacks ALL seven moves at one shared
scale. Packing a single move on its own would rescale it against itself and the character would
change size when the game switches move.

## The two rules the design is built around

1. One ComfyUI job at a time. A repaint is ~45s on a 12GB card and two at once means two failures,
   so every mutating request goes onto a single-worker queue and the caller gets a job id to poll.
2. The manifest is written BEFORE the repack. A repack can crash or be killed; if it does, the edit
   is still recorded and the next run picks it up. The other order loses the edit silently.
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

import repaint_cells as RC  # noqa: E402

PORT = 8811
CHARACTERS = ["seth", "cadbury"]
GROUND_Y = int(RC.CELL * 0.94)   # the feet line _emit anchors to, in packed-cell pixels
PICK_SKIP = 6                    # settle-in frames pick_frames drops; not offerable as a re-pick

# /img serves nothing outside these. The editor only ever needs source frames, repaints and the
# packed output, and an unrestricted path parameter on a LAN-bound server is a file-read hole.
ALLOWED_ROOTS = ("/tmp/sprite-", RC.REPAINT_DIR + "/", os.path.realpath(RC.OUT) + "/")


def frames_dir(cid, move):
    return os.path.join(RC.OUT, f"{cid}-{move}-frames")


def cell_png(cid, move, index):
    """The packed 512px cell, transparent-backed. Shown instead of the raw repaint because it is
    what actually ships, and because the metrics below are only meaningful in packed coordinates."""
    return os.path.join(frames_dir(cid, move), f"{cid}-{move}-{index + 1:02d}.png")


def source_frames(cid, move):
    d = f"/tmp/sprite-{cid}-{move}-pick"
    if not os.path.isdir(d):
        return []
    names = sorted(n for n in os.listdir(d) if n.endswith(".png"))
    return [os.path.join(d, n) for n in names[PICK_SKIP:]]


_metrics_cache = {}


def cell_metrics(path):
    """Silhouette height, top clearance and feet offset for one packed cell.

    These three are the numbers that have caught every bad cell in this project. A repaint that
    came back larger shows as a small clearance; one that came back smaller shows as a short
    silhouette; a keyed-in shadow or a clipped foot shows as a feet offset away from zero, which is
    the character sliding off the ground line mid-animation.
    """
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None
    hit = _metrics_cache.get(path)
    if hit and hit[0] == stamp:
        return hit[1]

    import numpy as np
    from PIL import Image
    mask = np.asarray(Image.open(path).convert("RGBA"))[:, :, 3] > 10
    rows = np.where(mask.any(axis=1))[0]
    m = None if len(rows) == 0 else {
        "height": int(rows[-1] - rows[0] + 1),
        "top": int(rows[0]),
        "feet": int(rows[-1] - GROUND_Y),
    }
    _metrics_cache[path] = (stamp, m)
    return m


def state(cid):
    moves = {}
    for move in RC.MOVES:
        man = RC.load_manifest(cid, move)
        _, n_default, fps, loop, hold, _ = RC.MOVES[move]
        cells = []
        for i, c in enumerate((man or {}).get("cells", [])):
            png = cell_png(cid, move, i)
            cells.append({"index": i, "src": c["src"], "seed": c["seed"],
                          "png": c.get("png"), "cell": png if os.path.exists(png) else None,
                          "metrics": cell_metrics(png)})
        moves[move] = {
            "cells": cells,
            "fps": (man or {}).get("fps", fps),
            "loop": (man or {}).get("loop", loop),
            "hold": (man or {}).get("hold", hold),
            "atlas": os.path.join(RC.OUT, f"{cid}-{move}.png"),
            "n_default": n_default,
            "n_source_frames": len(source_frames(cid, move)),
        }
    return {"character": cid, "characters": CHARACTERS, "moves": moves,
            "ground": GROUND_Y, "cell": RC.CELL}


# --- the job queue -------------------------------------------------------------------------
# One worker, one ComfyUI job at a time. Anything that repaints or repacks goes through here.

JOBS = {}
JOBS_LOCK = threading.Lock()
WORK = queue.Queue()


def submit(label, fn):
    jid = hashlib.sha1(f"{label}{time.time()}{random.random()}".encode()).hexdigest()[:12]
    with JOBS_LOCK:
        JOBS[jid] = {"state": "queued", "message": f"queued: {label}"}
    WORK.put((jid, label, fn))
    return jid


def set_job(jid, st, message):
    with JOBS_LOCK:
        JOBS[jid] = {"state": st, "message": message}


def worker():
    while True:
        jid, label, fn = WORK.get()
        set_job(jid, "running", f"{label} — repainting and repacking, ~45s per new cell")
        try:
            fn(lambda msg: set_job(jid, "running", msg))
            set_job(jid, "done", f"{label} — done")
        except Exception as e:
            traceback.print_exc()
            set_job(jid, "error", f"{label} — {type(e).__name__}: {e}")
        WORK.task_done()


def repack(cid):
    """Repaint anything missing and repack every move. Never write a packer here."""
    argv = sys.argv
    sys.argv = ["repaint_cells", cid]
    try:
        RC.main()
    finally:
        sys.argv = argv


def edit_manifest(cid, move, mutate):
    """Apply `mutate` to the cell list, write it, then repack — in that order, always."""
    man = RC.load_manifest(cid, move)
    if not man:
        raise ValueError(f"no manifest for {cid}-{move}")
    cells = [dict(c) for c in man["cells"]]
    cells = mutate(cells)
    if not cells:
        raise ValueError("a move cannot have zero cells")
    meta = {k: man[k] for k in ("fps", "loop", "hold", "unify", "clip") if k in man}
    RC.save_manifest(cid, move, cells, meta)


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
                self.json(200, {"frames": source_frames(one("cid", "seth"), one("move", "walk"))})
            elif u.path == "/img":
                self.serve_image(one("path", ""), one("w"))
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

    def serve_image(self, path, width):
        real = os.path.realpath(path)
        if not real.startswith(ALLOWED_ROOTS) or not real.endswith(".png"):
            self.json(403, {"error": "path not allowed"})
            return
        if not os.path.exists(real):
            self.json(404, {"error": "no such file"})
            return
        if width:
            # The re-pick strip is 60+ full 832px frames. Sending them whole makes the strip take
            # tens of seconds to appear, which is long enough that you stop using the feature.
            from PIL import Image
            im = Image.open(real).convert("RGBA")
            w = max(16, min(512, int(width)))
            im.thumbnail((w, w), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "PNG")
            body = buf.getvalue()
        else:
            body = open(real, "rb").read()
        self.send(200, body, "image/png")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            cid, move = req.get("cid", "seth"), req.get("move", "walk")
            if cid not in CHARACTERS or move not in RC.MOVES:
                raise ValueError("unknown character or move")
            handler = {"/api/reroll": self.reroll, "/api/repick": self.repick,
                       "/api/drop": self.drop, "/api/reorder": self.reorder}.get(u.path)
            if not handler:
                self.json(404, {"error": "not found"})
                return
            self.json(200, {"job": handler(cid, move, req)})
        except Exception as e:
            traceback.print_exc()
            self.json(400, {"error": f"{type(e).__name__}: {e}"})

    def queue_edit(self, label, cid, move, mutate):
        # The manifest edit itself runs on the worker too, so a queued edit always sees the cell
        # list the edit before it left behind, not the one it was submitted against.
        def run(_progress):
            edit_manifest(cid, move, mutate)
            repack(cid)
        return submit(label, run)

    def reroll(self, cid, move, req):
        i, seed = int(req["index"]), req.get("seed")

        def mutate(cells):
            c = cells[i]
            new = int(seed) if seed is not None else random.randint(1, 2 ** 31 - 1)
            if new == c["seed"]:
                raise ValueError("that is the seed the cell already has")
            c["seed"] = new
            c["png"] = RC.repaint_path(cid, move, c["src"], new)
            return cells
        return self.queue_edit(f"{cid}-{move} cell {i + 1} re-roll", cid, move, mutate)

    def repick(self, cid, move, req):
        i, src = int(req["index"]), req["src"]
        if src not in source_frames(cid, move):
            raise ValueError("that source frame is not in this move's pick directory")

        def mutate(cells):
            c = cells[i]
            c["src"] = src
            c["png"] = RC.repaint_path(cid, move, src, c["seed"])
            return cells
        return self.queue_edit(f"{cid}-{move} cell {i + 1} re-pick", cid, move, mutate)

    def drop(self, cid, move, req):
        i = int(req["index"])

        def mutate(cells):
            if len(cells) <= 1:
                raise ValueError("refusing to drop the last cell of a move")
            del cells[i]
            return cells
        return self.queue_edit(f"{cid}-{move} drop cell {i + 1}", cid, move, mutate)

    def reorder(self, cid, move, req):
        order = [int(x) for x in req["order"]]

        def mutate(cells):
            if sorted(order) != list(range(len(cells))):
                raise ValueError("order must be a permutation of the current cell indices")
            return [cells[i] for i in order]
        return self.queue_edit(f"{cid}-{move} reorder", cid, move, mutate)


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
  .keys { display: flex; gap: .4rem; flex-wrap: wrap; margin: 0 0 .7rem; }
  button { font: 600 .85rem/1 ui-monospace, monospace; color: #e8e2d8; background: #241f19;
           border: 1px solid #4a4136; border-radius: 6px; padding: .55rem .7rem; cursor: pointer; }
  button.on { background: #d8a657; color: #1a1610; }
  button:disabled { opacity: .38; cursor: not-allowed; }
  .row { display: flex; gap: 1rem; align-items: flex-start; flex-wrap: wrap; }
  .strip { background: #efe9d8; border: 1px solid #3a332a; border-radius: 10px; padding: .6rem;
           display: flex; gap: .4rem; flex-wrap: wrap; flex: 1 1 620px; }
  .cellbox { border: 2px solid transparent; border-radius: 6px; padding: 2px; cursor: pointer;
             background: #efe9d8; }
  .cellbox.sel { border-color: #b4462f; }
  .cellbox img { display: block; width: 96px; height: 96px; }
  .cellbox .n { font: 700 .72rem/1.4 ui-monospace, monospace; color: #6b6152; text-align: center; }
  .cellbox .m { font: .66rem/1.35 ui-monospace, monospace; color: #6b6152; text-align: center;
                white-space: pre; }
  .preview { background: #efe9d8; border: 1px solid #3a332a; border-radius: 10px; padding: .6rem;
             text-align: center; }
  .preview canvas { display: block; width: 240px; height: 240px; }
  .preview .cap { font: 600 .72rem/1.6 ui-monospace, monospace; color: #6b6152; letter-spacing: .06em; }
  .pair { display: flex; gap: 1rem; flex-wrap: wrap; }
  .pane { background: #efe9d8; border: 1px solid #3a332a; border-radius: 10px; padding: .6rem; }
  .pane img { display: block; width: 260px; height: 260px; object-fit: contain; }
  .pane .cap { font: 600 .72rem/1.6 ui-monospace, monospace; color: #6b6152; letter-spacing: .06em; }
  .facts { font: .82rem/1.7 ui-monospace, monospace; color: #c9bfae; }
  .facts b { color: #d8a657; font-weight: 700; }
  .picker { background: #efe9d8; border: 1px solid #3a332a; border-radius: 10px; padding: .5rem;
            display: flex; gap: .3rem; overflow-x: auto; max-width: 100%; }
  .picker figure { margin: 0; cursor: pointer; border: 2px solid transparent; border-radius: 4px; }
  .picker figure.cur { border-color: #b4462f; }
  .picker img { display: block; width: 76px; height: 76px; }
  .picker figcaption { font: .62rem/1.4 ui-monospace, monospace; color: #6b6152; text-align: center; }
  #status { border-left: 3px solid #d8a657; padding: .35rem 0 .35rem .9rem; margin: 1rem 0;
            color: #c9bfae; font: .85rem/1.5 ui-monospace, monospace; min-height: 1.5em; }
  #status.err { border-color: #b4462f; color: #e6a394; }
  .muted { color: #8d8371; }
</style>
<main>
  <h1>Sprite cell editor</h1>
  <p class="lede">Every cell of every move, as it ships. Pick a cell to see it next to the source
  frame it was painted from — a bad drawing wants a new seed, a bad pose wants a different frame.
  Each edit writes <code>sheets/&lt;cid&gt;-&lt;move&gt;.json</code> and repacks all seven moves.</p>

  <div class="keys" id="chars"></div>
  <div class="keys" id="moves"></div>
  <div id="status">ready</div>

  <div class="row">
    <div class="strip" id="strip"></div>
    <div class="preview">
      <canvas id="pv" width="512" height="512"></canvas>
      <div class="cap" id="pvcap"></div>
    </div>
  </div>

  <div id="sel" hidden>
    <h2>Selected cell</h2>
    <div class="pair">
      <div class="pane"><img id="bigcell" alt="packed cell"><div class="cap" id="bigcap"></div></div>
      <div class="pane"><img id="bigsrc" alt="source frame"><div class="cap" id="srccap"></div></div>
      <div>
        <div class="facts" id="facts"></div>
        <div class="keys" style="margin-top:.8rem">
          <button id="b-reroll">Re-roll</button>
          <button id="b-repick">Re-pick…</button>
          <button id="b-drop">Drop</button>
          <button id="b-left">◀ move left</button>
          <button id="b-right">move right ▶</button>
        </div>
      </div>
    </div>
    <div id="pickwrap" hidden>
      <h2>Re-pick source frame <span class="muted" id="pickcount"></span></h2>
      <div class="picker" id="picker"></div>
    </div>
  </div>
</main>
<script>
const MOVES = ["walk","idle","punch","kick","jump","block","crouch"];
let cid = "seth", move = "walk", ST = null, sel = 0, busy = false, bust = Date.now(), anim = null;

const img = (p, w) => "/img?path=" + encodeURIComponent(p) + (w ? "&w=" + w : "") + "&v=" + bust;
const el = (id) => document.getElementById(id);

function status(msg, err) {
  el("status").textContent = msg;
  el("status").className = err ? "err" : "";
}

function buttons(host, items, current, pick) {
  host.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    b.textContent = it;
    if (it === current) b.className = "on";
    b.disabled = busy;
    b.onclick = () => pick(it);
    host.appendChild(b);
  }
}

async function load() {
  const r = await fetch("/api/state?cid=" + cid + "&v=" + Date.now());
  ST = await r.json();
  const cells = ST.moves[move].cells;
  if (sel >= cells.length) sel = Math.max(0, cells.length - 1);
  render();
}

function render() {
  buttons(el("chars"), ST.characters, cid, (c) => { cid = c; sel = 0; el("pickwrap").hidden = true; load(); });
  buttons(el("moves"), MOVES, move, (m) => { move = m; sel = 0; el("pickwrap").hidden = true; load(); });
  const mv = ST.moves[move], cells = mv.cells;

  const strip = el("strip");
  strip.innerHTML = "";
  cells.forEach((c, i) => {
    const box = document.createElement("div");
    box.className = "cellbox" + (i === sel ? " sel" : "");
    const m = c.metrics;
    box.innerHTML = '<div class="n">' + (i + 1) + "</div>" +
      (c.cell ? '<img src="' + img(c.cell) + '" alt="cell ' + (i + 1) + '">'
              : '<div class="m" style="width:96px;height:96px">not packed</div>') +
      '<div class="m">' + (m ? "h " + m.height + "\ntop " + m.top + "\nfeet " +
        (m.feet > 0 ? "+" : "") + m.feet : "empty") + "</div>";
    box.onclick = () => { sel = i; el("pickwrap").hidden = true; render(); };
    strip.appendChild(box);
  });

  el("pvcap").textContent = mv.cells.length + " cells @ " + mv.fps + "fps" +
    (mv.loop ? " loop" : mv.hold ? " hold" : " once");
  playPreview(mv);
  renderSelected(cells[sel], cells.length);
}

function playPreview(mv) {
  if (anim) { clearInterval(anim); anim = null; }
  const cv = el("pv"), ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, 512, 512);
  if (!mv.cells.length) return;
  const sheet = new Image();
  sheet.onload = () => {
    const n = Math.round(sheet.width / ST.cell), cell = ST.cell;
    // A once-move ends on its last cell in the game. Here it replays after a short rest instead,
    // because a preview frozen on the recovery pose tells you nothing about the motion.
    const total = mv.loop ? n : n + Math.ceil(mv.fps * 0.7);
    let f = 0;
    const step = () => {
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.drawImage(sheet, Math.min(f, n - 1) * cell, 0, cell, cell, 0, 0, cv.width, cv.height);
      f = (f + 1) % total;
    };
    step();
    anim = setInterval(step, 1000 / mv.fps);
  };
  sheet.src = img(mv.atlas);
}

function renderSelected(c, n) {
  el("sel").hidden = !c;
  if (!c) return;
  el("bigcell").src = c.cell ? img(c.cell) : "";
  el("bigcap").textContent = "cell " + (c.index + 1) + " — packed, seed " + c.seed;
  el("bigsrc").src = img(c.src);
  el("srccap").textContent = "source — " + c.src.split("/").pop();
  const m = c.metrics;
  el("facts").innerHTML = m
    ? "silhouette height <b>" + m.height + "</b>px<br>top clearance <b>" + m.top +
      "</b>px<br>feet offset <b>" + (m.feet > 0 ? "+" : "") + m.feet +
      "</b>px from the ground line (y=" + ST.ground + ")<br>seed <b>" + c.seed + "</b>"
    : "no packed cell on disk yet";
  for (const [id, off] of [["b-reroll", false], ["b-repick", false], ["b-drop", n <= 1],
                           ["b-left", c.index === 0], ["b-right", c.index === n - 1]]) {
    el(id).disabled = busy || off;
  }
}

async function post(url, body, label) {
  busy = true; render(); status(label + " — queued…");
  const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"},
                             body: JSON.stringify(Object.assign({cid, move}, body))});
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
      bust = Date.now();          // the files were overwritten in place; force every img to refetch
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
  const order = ST.moves[move].cells.map((c, i) => i);
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
  const r = await fetch("/api/frames?cid=" + cid + "&move=" + move);
  const frames = (await r.json()).frames;
  const cur = ST.moves[move].cells[sel].src;
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
