#!/usr/bin/env python3
"""Prove a re-run cannot eat a hand edit, against the real Durable Object code.

    python3 test_orphan_tray.py

DESIGN-pipeline.md calls this risk 1: "the orphan tray silently eats a hand edit on a half-matching
repick". Seven claims, all run against `celld-editor/worker.js` itself under node with a Map for
storage, so nothing here is a re-implementation of the thing under test:

1. **A re-pick of the SAME extraction carries edits over.** New picks within ±3 frames of an old
   one inherit its id, seed, hold and nudge; a pick that did not move keeps its drawing too.
2. **Anything unmatched is orphaned, not deleted.** It lands in the tag's tray with the frame it
   came from and the reason, and the tray is never emptied by anything but a person.
3. **A re-pick of a DIFFERENT extraction matches nothing.** A re-rendered clip is a new
   performance, so the whole tag goes to the tray and is rebuilt from the fresh pick.
4. **A drop stays dropped.** A cell removed by hand is not resurrected by the next re-pick that
   likes that pose again.
5. **Restore lands where it was aimed**, and moves the tray entry onto the record rather than
   deleting it.
6. **A changed extraction takes the tray's PICTURES and not its facts.** The repaints of a
   re-rendered move are deleted, and a tray entry that still named one made the page ask for a
   file that is not there on every open.
7. **The invariant fires.** A manifest that has lost an id is refused on the next write, loudly,
   with the id named — rather than being written with an edit missing.

No GPU, no network, no writes to anything on disk.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = "node"

# The Durable Object, driven through its own HTTP surface with a Map for `ctx.storage`. Driving it
# through `fetch` rather than by calling methods is deliberate: the routes are what the agent and
# the page really use, and a test that reached past them could pass while the object was
# unreachable.
HARNESS = r"""
import { CharacterDO } from "%s";

const store = new Map();
const ctx = { storage: {
  async get(k) { return store.has(k) ? JSON.parse(JSON.stringify(store.get(k))) : undefined; },
  async put(k, v) { store.set(k, JSON.parse(JSON.stringify(v))); },
  async delete(k) { store.delete(k); },
} };
const DO = new CharacterDO(ctx, {});

const post = (path, body) => DO.fetch(new Request("http://c" + path, {
  method: "POST", headers: { "content-type": "application/json" },
  body: JSON.stringify(body ?? {}) }));
const get = (path) => DO.fetch(new Request("http://c" + path));
const said = async (r) => [r.status, await r.json()];

const src = (n) => `/tmp/sprite-cad-block-pick/f_${String(n).padStart(4, "0")}.png`;
const frame = (n, extra) => ({ src: src(n), seed: 77, hold: 1, pivot_nudge: [0, 0], ...extra });

// A whole character, in the shape `repaint_cells._character()` writes. Small on purpose: one tag,
// six cells, chosen so every branch of the matching rule has a case in it.
const manifest = (picks) => ({
  character: "cad", cell: 512, pivot: [256, 481], trait: "a test butler",
  build: {
    template_version: "v1",
    cost_s: { plate: 20, clip: 170, frames: 10, picks: 5, repaint: 45, pack: 5, export: 0 },
    clip: { size: 832, steps: 20, length: 61, seed: 7 },
    picks: { skip: 6, anchor: "feet", smooth: true },
    repaint: { pad: 0.86, steps: 8, cfg: 1.0, sampler: "euler", scheduler: "simple", denoise: 1 },
    pack: { height_cap: 0.9, up_cap: 0.75 },
  },
  tags: [{ name: "block", from: 0, to: picks.length - 1, recipe: "block",
           recipe_text: "he blocks", cells: picks.length, cyclic: false, fps: 12,
           loop: false, hold_key: true, unify: "head",
           clip: "/tmp/cad-block-832-20.mp4" }],
  frames: picks.map((n) => frame(n)),
});

// The build report the agent sends after a pack. Only the `frames:block` step's hash matters
// here: it is what teaches the object which extraction the live cells were picked out of.
const report = (man, framesHash) => ({
  ...man,
  steps: [{ id: "frames:block", kind: "frames", tag: "block", params: {}, inputs: [],
            key: "k", built_key: "k", artifact: "/tmp/sprite-cad-block-pick",
            artifact_hash: framesHash, cost_s: 10 }],
});

const out = {};
const cells = async () => (await (await get("/state")).json()).tags[0].cells;
const state = async () => (await get("/state")).json();

// --- setup: six cells, three of them hand-edited ---------------------------------------------
await post("/seed", { manifest: manifest([10, 20, 30, 40, 50, 60]) });
const seeded = await cells();
out.seeded = seeded.map((c) => c.fid);
// cell 2 (f_0020) gets a hold and a nudge; cell 4 (f_0040) a nudge; cell 6 (f_0060) a hold.
await post("/hold", { tag: "block", index: 1, hold: 4 });
await post("/pivot", { tag: "block", scope: "frame", index: 1, pivot_nudge: [9, -14] });
await post("/pivot", { tag: "block", scope: "frame", index: 3, pivot_nudge: [-5, 3] });
await post("/hold", { tag: "block", index: 5, hold: 3 });
// A drop, which must survive a re-pick as a drop.
await post("/drop", { tag: "block", index: 4 });          // f_0050
out.dropped_fid = seeded[4].fid;

// The agent's build report, which is the only thing that can say which bytes these picks were
// made from. Everything before this line is the state an existing character is really in.
const man0 = JSON.parse(JSON.stringify(store.get("manifest")));
await post("/done", { id: "none", manifest: report(man0, "HASH-A"),
                      frames: man0.frames.map((f) => ({ fid: f._fid, rev: f._rev })) });
out.before = (await cells()).map((c) => ({ fid: c.fid, src: c.src, seed: c.seed, hold: c.hold,
                                          nudge: c.pivot_nudge, png: c.png }));

// --- 1 to 4: a re-pick of the SAME extraction, at a different cell count ----------------------
// f_0010 unmoved; f_0022 is 2 from f_0020 so it inherits; f_0031 inherits f_0030; f_0044 is 4
// from f_0040 so it does NOT, and f_0040's nudge is orphaned; f_0051 matches the DROPPED f_0050
// and is suppressed; f_0090 is new; f_0060's hold is orphaned because nothing came near it.
out.same = await (await post("/picked", {
  tag: "block", frames_hash: "HASH-A",
  srcs: [10, 22, 31, 44, 51, 90].map(src) })).json();
out.after_same = (await cells()).map((c) => ({ fid: c.fid, src: c.src, seed: c.seed,
                                              hold: c.hold, nudge: c.pivot_nudge, png: c.png }));
out.tray_same = (await state()).tray.map((e) => ({ fid: e.fid, src: e.src, hold: e.hold,
                                                  nudge: e.pivot_nudge, why: e.why }));

// --- 5: restore one orphan onto a live cell ---------------------------------------------------
const orphanHold = out.tray_same.find((e) => e.hold === 3);
out.restore = await said(await post("/restore",
  { tag: "block", fid: orphanHold.fid, index: 2 }));
out.after_restore = (await cells()).map((c) => ({ fid: c.fid, hold: c.hold,
                                                 nudge: c.pivot_nudge }));
const st1 = await state();
out.tray_after_restore = st1.tray.map((e) => e.fid);
out.dropped_after_restore = st1.dropped.map((e) => ({ fid: e.fid, why: e.why }));

// --- 3: a re-pick of a DIFFERENT extraction ---------------------------------------------------
out.live_before_clip = (await cells()).map((c) => c.fid);
out.newclip = await (await post("/picked", {
  tag: "block", frames_hash: "HASH-B",
  srcs: [11, 21, 32, 43].map(src) })).json();
const st2 = await state();
out.after_newclip = (await cells()).map((c) => ({ fid: c.fid, src: c.src, seed: c.seed,
                                                 hold: c.hold, nudge: c.pivot_nudge }));
out.tray_after_newclip = st2.tray.map((e) => e.fid);
out.tray_after_newclip_facts = st2.tray.map((e) => ({ hold: e.hold, nudge: e.pivot_nudge }));

// --- a changed extraction takes the tray's PICTURES with it, never its facts ------------------
out.tray_png_before = (await state()).tray.map((e) => e.png);
const man2 = JSON.parse(JSON.stringify(store.get("manifest")));
await post("/done", { id: "none", cleared_tag: "block", manifest: report(man2, "HASH-B"),
                      frames: man2.frames.map((f) => ({ fid: f._fid, rev: f._rev })) });
const st3 = await state();
out.tray_png_after = st3.tray.map((e) => e.png);
out.tray_facts_after = st3.tray.map((e) => ({ fid: e.fid, src: e.src, hold: e.hold,
                                             nudge: e.pivot_nudge }));

// --- 6: the invariant ---------------------------------------------------------------------------
// A mutation that spliced a frame out of the tag and never recorded where it went. Done by hand
// on the stored manifest because no code path in the object does it -- which is the point: the
// assertion is what guarantees none ever will, including one written next year.
const tampered = JSON.parse(JSON.stringify(store.get("manifest")));
const eaten = tampered.frames[1]._fid;
tampered.frames.splice(1, 1);
tampered.tags[0].to -= 1;
store.set("manifest", tampered);
out.eaten = eaten;
out.assertion = await said(await post("/hold", { tag: "block", index: 0, hold: 2 }));

console.log(JSON.stringify(out));
"""


def run():
    mjs = "/tmp/worker-orphan-test.mjs"
    with open(os.path.join(HERE, "celld-editor", "worker.js")) as src, open(mjs, "w") as dst:
        dst.write(src.read())
    harness = "/tmp/orphan-harness.mjs"
    with open(harness, "w") as f:
        f.write(HARNESS % mjs)
    r = subprocess.run([NODE, harness], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"node failed:\n{r.stderr}")
    return json.loads(r.stdout)


FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILED.append(label)


def main():
    o = run()
    was = {f["src"].split("/")[-1]: f for f in o["before"]}
    now = {f["src"].split("/")[-1]: f for f in o["after_same"]}

    print("1. a re-pick of the same extraction carries hand edits over")
    check("f_0010 did not move, so it keeps its id AND its drawing",
          now["f_0010.png"]["fid"] == was["f_0010.png"]["fid"] and
          now["f_0010.png"]["png"] == was["f_0010.png"]["png"])
    check("f_0022 is 2 frames from f_0020, so it inherits its id, hold and nudge",
          now["f_0022.png"]["fid"] == was["f_0020.png"]["fid"] and
          now["f_0022.png"]["hold"] == 4 and now["f_0022.png"]["nudge"] == [9, -14],
          json.dumps(now["f_0022.png"]))
    check("a frame that MOVED names a drawing that does not exist yet",
          now["f_0022.png"]["png"] != was["f_0020.png"]["png"])
    check("f_0031 inherits f_0030's id", now["f_0031.png"]["fid"] == was["f_0030.png"]["fid"])
    check("f_0090 is brand new, on the base seed with no edits",
          now["f_0090.png"]["fid"] not in [f["fid"] for f in o["before"]] and
          now["f_0090.png"]["seed"] == 77 and now["f_0090.png"]["hold"] == 1)

    print("2. everything unmatched is orphaned, never deleted")
    tray = {e["src"].split("/")[-1]: e for e in o["tray_same"]}
    check("f_0044 is 4 frames away, past the +-3 threshold, so f_0040 is orphaned",
          "f_0040.png" in tray and tray["f_0040.png"]["nudge"] == [-5, 3])
    check("f_0060's hold is orphaned, not lost", "f_0060.png" in tray and
          tray["f_0060.png"]["hold"] == 3)
    check("the tray says which frame each edit came from and why",
          all(e["src"] and e["why"] for e in o["tray_same"]))
    check("the reconcile reports what it did",
          o["same"]["matched"] == 3 and o["same"]["orphaned"] == 2 and
          o["same"]["suppressed"] == 1 and o["same"]["carried_over"] is True,
          json.dumps(o["same"]))

    print("3. a re-pick of a different extraction matches nothing")
    check("carry-over is off", o["newclip"]["carried_over"] is False)
    check("every live frame went to the tray",
          o["newclip"]["orphaned"] == len(o["live_before_clip"]) and
          all(f in o["tray_after_newclip"] for f in o["live_before_clip"]),
          json.dumps(o["newclip"]))
    check("the tag is rebuilt from the fresh pick, all new ids on the base seed",
          len(o["after_newclip"]) == 4 and
          all(f["seed"] == 77 and f["hold"] == 1 and f["nudge"] == [0, 0]
              for f in o["after_newclip"]) and
          not (set(f["fid"] for f in o["after_newclip"]) & set(o["live_before_clip"])))

    print("4. a cell dropped by hand stays dropped")
    check("f_0051 matched the dropped f_0050 and was suppressed",
          "f_0051.png" not in now and o["same"]["suppressed"] == 1)

    print("5. restore lands on the cell it was aimed at")
    check("the request was accepted", o["restore"][0] == 202)
    check("cell 3 took the orphan's hold of 3", o["after_restore"][2]["hold"] == 3)
    check("no other cell moved",
          [c["hold"] for c in o["after_restore"]] ==
          [c["hold"] if i != 2 else 3 for i, c in enumerate(o["after_same"])])
    check("the tray entry is not deleted — it moves to the record with where it went",
          any(d["why"].startswith("restored onto cell 3") for d in o["dropped_after_restore"]))

    print("6. a changed extraction clears the tray's pictures and keeps its facts")
    check("the entries had drawings to begin with",
          any(x is not None for x in o["tray_png_before"]))
    check("every one is forgotten once the repaints are deleted",
          all(x is None for x in o["tray_png_after"]))
    check("the edits themselves are untouched",
          [(e["hold"], e["nudge"]) for e in o["tray_facts_after"]] ==
          [(e["hold"], e["nudge"]) for e in o["tray_after_newclip_facts"]])

    print("7. the invariant refuses a write that would lose a frame")
    status, body = o["assertion"]
    check("the write is refused with 500", status == 500)
    check("the lost id is named in the error", o["eaten"] in body.get("error", ""),
          body.get("error", ""))
    check("the error says what was about to happen",
          "hand edit was about to be lost" in body.get("error", ""))

    print()
    if FAILED:
        sys.exit(f"{len(FAILED)} failed: {', '.join(FAILED)}")
    print("all good")


if __name__ == "__main__":
    main()
