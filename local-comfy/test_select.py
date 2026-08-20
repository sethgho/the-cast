#!/usr/bin/env python3
"""Prove the three things DESIGN-playback.md asks for, against the real Durable Object code.

    python3 test_select.py

Nine claims, all run against `celld-editor/worker.js` itself under node with a Map for storage and
driven through its own HTTP surface, exactly as `test_orphan_tray.py` is -- so nothing here is a
re-implementation of the thing under test:

1. **A frame that stays selected keeps everything.** Its id, seed, drawing, hold and nudge follow
   it, because the edit belongs to the frame and the frame is still chosen.
2. **A frame that is deactivated is orphaned, not deleted.** It lands in the tag's tray with its
   edits and the reason, exactly as an unmatched re-pick does.
3. **A newly activated frame is a cell with no drawing.** Base seed, no png, nothing queued for it
   beyond the repack every edit owes.
4. **The invariant holds across a selection that removes and adds at once**, and every id this
   object has ever written is still live, in the tray, or dropped.
5. **A selection is validated hard** -- range, order, repeats, emptiness, the ceiling, and a
   selection that changes nothing.
6. **A cancelled selection puts the tag back exactly**, including the cells it had orphaned, and
   parks the cell it had created rather than losing it.
7. **A per-frame prompt marks exactly that cell stale** -- one repaint and the pack that reads it,
   and not one other drawing.
8. **Creating a character queues the plate and then the idle clip**, in that order, chained; and
   cancelling the plate cancels the clip rather than leaving it to render against nothing.
9. **A new move queues its own clip**, behind the edit that created it and cancelled with it.

No GPU, no network, no writes to anything on disk.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = "node"

HARNESS = r"""
import { CharacterDO } from "%s";

const make = () => {
  const store = new Map();
  const ctx = { storage: {
    async get(k) { return store.has(k) ? JSON.parse(JSON.stringify(store.get(k))) : undefined; },
    async put(k, v) { store.set(k, JSON.parse(JSON.stringify(v))); },
    async delete(k) { store.delete(k); },
    async deleteAll() { store.clear(); },
  } };
  const DO = new CharacterDO(ctx, {});
  return { store, DO,
    post: (path, body) => DO.fetch(new Request("http://c" + path, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body ?? {}) })),
    get: (path) => DO.fetch(new Request("http://c" + path)) };
};

const said = async (r) => [r.status, await r.json()];
const src = (n) => `/tmp/sprite-cad-block-pick/f_${String(n).padStart(4, "0")}.png`;
const png = (n) => `/tmp/repaint/cad-block-f_${String(n).padStart(4, "0")}.png`;
const frame = (n, extra) => ({ src: src(n), seed: 77, hold: 1, pivot_nudge: [0, 0], ...extra });

const build = {
  template_version: "v1",
  cost_s: { plate: 20, clip: 170, frames: 10, picks: 5, repaint: 45, pack: 5, export: 0 },
  clip: { size: 832, steps: 20, length: 61, seed: 7 },
  picks: { skip: 6, anchor: "feet", smooth: true },
  repaint: { pad: 0.86, steps: 8, cfg: 1.0, sampler: "euler", scheduler: "simple", denoise: 1 },
  pack: { height_cap: 0.9, up_cap: 0.75 },
};

const manifest = (picks) => ({
  character: "cad", cell: 512, pivot: [256, 481], trait: "a test butler", build,
  tags: [{ name: "block", from: 0, to: picks.length - 1, recipe: "block",
           recipe_text: "he blocks", cells: picks.length, cyclic: false, fps: 12,
           loop: false, hold_key: true, unify: "head",
           clip: "/tmp/cad-block-832-20.mp4" }],
  frames: picks.map((n) => frame(n)),
});

// The extraction the timeline is drawn from: 100 frames, of which six are cells. `/api/select`
// indexes into exactly this list, so f_0010 is index 9.
const SOURCES = Array.from({ length: 100 }, (_, i) => src(i + 1));
const out = {};

// The agent's build report for a character every one of whose cells is drawn: it is reported
// against the job the edits queued, so the queue is empty afterwards and the next mutation is a
// job of its own rather than a fourth edit coalesced into the same repack.
//
// The step records are iterated to a fixed point first. An artifact hash is an INPUT to the key of
// everything downstream of it, so stamping built_key against keys computed before those hashes
// existed would report a character every one of whose steps was stale -- which would be a fact
// about this harness and not about the object under test.
const paint = async (c) => {
  const man = JSON.parse(JSON.stringify(c.store.get("manifest")));
  const painted = { ...man, frames: man.frames.map((f) => ({ ...f, png: png(Number(
    /f_(\d+)\.png$/.exec(f.src)[1]) ) })) };
  let steps = (await buildSteps(painted, [])).map((s) => ({
    ...s, artifact_hash: s.artifact_hash ?? `h-${s.id}` }));
  for (let i = 0; i < 3; i++) {
    steps = (await buildSteps(painted, steps))
      .map((s) => ({ ...s, artifact_hash: s.artifact_hash ?? `h-${s.id}`, built_key: s.key }));
  }
  const job = (await (await c.get("/jobs")).json()).jobs.find((j) => j.state === "queued");
  await c.post("/claim", { agent: "test" });
  await c.post("/done", { id: job.id, manifest: { ...painted, steps },
                          frames: man.frames.map((f) => ({ fid: f._fid, rev: f._rev })) });
};

// =============================================================================================
// 8 + 9 first, in cells of their own: `create` refuses a character who already exists.
// =============================================================================================
{
  const c = make();
  const [status, made] = await said(await c.post("/create", {
    cid: "gnu", name: "Gnu", cell: 512, pivot: [256, 481], trait: "he ambles", build,
    source: "/tmp/sprite-uploads/gnu.png",
    moves: [
      { name: "walk", recipe: "walk-cycle", recipe_text: "he walks", cells: 10, fps: 14,
        loop: true, hold_key: false, unify: "head", cyclic: true },
      { name: "idle", recipe: "idle-breathe", recipe_text: "he breathes", cells: 8, fps: 8,
        loop: true, hold_key: false, unify: true, cyclic: true },
    ],
  }));
  out.create = [status, made];
  out.create_jobs = (await (await c.get("/jobs")).json()).jobs;
  // Cancel the plate. The clip behind it must not survive: it would be composited from a cut-out
  // that does not exist.
  out.create_cancel = await said(await c.post("/cancel", { id: out.create_jobs[0].id }));
  out.create_after_cancel = (await (await c.get("/jobs")).json()).jobs;
}
{
  // The other half of the same rule: the plate FINISHING must leave the clip alone.
  const c = make();
  await c.post("/create", {
    cid: "gnu", name: "Gnu", cell: 512, pivot: [256, 481], trait: "he ambles", build,
    source: "/tmp/sprite-uploads/gnu.png",
    moves: [{ name: "idle", recipe: "idle-breathe", recipe_text: "he breathes", cells: 8, fps: 8,
              loop: true, hold_key: false, unify: true, cyclic: true }],
  });
  const jobs = (await (await c.get("/jobs")).json()).jobs;
  await c.post("/claim", { agent: "test" });
  await c.post("/progress", { id: jobs[0].id, message: "cutting out" });
  await c.post("/failed", { id: jobs[0].id, message: "gpu-worker is down" });
  out.plate_failed = (await (await c.get("/jobs")).json()).jobs;
}

// =============================================================================================
// A character in the state the select screen really meets one in: six cells, three hand edits,
// every drawing on disk and nothing stale. Each claim below gets its OWN cell, because a cheap
// mutation coalesces into a repack that is still queued -- so two claims sharing a cell would
// share a job, and cancelling one would unwind the other.
// =============================================================================================
const { buildSteps } = await import("%s");

const setup = async () => {
  const c = make();
  await c.post("/seed", { manifest: manifest([10, 20, 30, 40, 50, 60]),
                          catalogue: { sources: { block: SOURCES }, variants: {} } });
  // cell 2 (f_0020) gets a hold and a nudge; cell 4 (f_0040) a nudge.
  await c.post("/hold", { tag: "block", index: 1, hold: 4 });
  await c.post("/pivot", { tag: "block", scope: "frame", index: 1, pivot_nudge: [9, -14] });
  await c.post("/pivot", { tag: "block", scope: "frame", index: 3, pivot_nudge: [-5, 3] });
  await paint(c);
  return c;
};

// --- the select's cell ------------------------------------------------------------------------
const c = await setup();
const cells = async () => (await (await c.get("/state")).json()).tags[0].cells;
const state = async () => (await c.get("/state")).json();
const stale = (st) => Object.fromEntries(st.steps.map((s) => [s.id, s.stale]));
out.queue_empty_before_select = (await (await c.get("/jobs")).json())
  .jobs.filter((j) => j.state === "queued").length;
out.before = (await cells()).map((c) => ({ fid: c.fid, src: c.src, seed: c.seed, hold: c.hold,
                                          nudge: c.pivot_nudge, png: c.png, prompt: c.prompt,
                                          at: c.source_index }));
out.sources_len = (await state()).tags[0].sources.length;

// =============================================================================================
// 5: validation, before anything is allowed to change ------------------------------------------
// =============================================================================================
out.bad = {};
for (const [why, frames] of [
  ["empty", []],
  ["descending", [19, 9]],
  ["repeated", [9, 9, 19]],
  ["past the end", [9, 100]],
  ["negative", [-1, 9]],
  ["not whole", [9, 19.5]],
  ["too many", Array.from({ length: 33 }, (_, i) => i)],
  ["unchanged", [9, 19, 29, 39, 49, 59]],
]) out.bad[why] = await said(await c.post("/select", { tag: "block", frames }));
out.bad_left_it_alone = (await cells()).map((x) => x.src);

// =============================================================================================
// 1, 2, 3, 4: keep two, drop four, add one, in one write ----------------------------------------
// =============================================================================================
out.select = await said(await c.post("/select", { tag: "block", frames: [9, 19, 74] }));
const st1 = await state();
out.after = st1.tags[0].cells.map((x) => ({ fid: x.fid, src: x.src, seed: x.seed, hold: x.hold,
                                           nudge: x.pivot_nudge, png: x.png, at: x.source_index }));
out.tray = st1.tray.map((e) => ({ fid: e.fid, src: e.src, hold: e.hold, nudge: e.pivot_nudge,
                                  why: e.why }));
// Every id this object has ever written, straight out of storage: the assertion's own ledger.
out.fids_ledger = c.store.get("fids");
out.select_job = [out.select[1].job];

// =============================================================================================
// 6: cancel it ---------------------------------------------------------------------------------
// =============================================================================================
out.cancel = await said(await c.post("/cancel", { id: out.select_job[0].id }));
const st2 = await state();
out.after_cancel = st2.tags[0].cells.map((x) => ({ fid: x.fid, src: x.src, hold: x.hold,
                                                   nudge: x.pivot_nudge, png: x.png }));
out.tray_after_cancel = st2.tray.map((e) => ({ fid: e.fid, src: e.src, why: e.why }));

// =============================================================================================
// 7: one cell's own prompt, on a cell of its own so that it is the prompt being measured --------
// =============================================================================================
{
  const p = await setup();
  const st0 = await (await p.get("/state")).json();
  out.stale_before = stale(st0);
  out.prompt = await said(await p.post("/prompt",
    { tag: "block", index: 1, prompt: "  his left hand is open, not a fist  " }));
  const st3 = await (await p.get("/state")).json();
  out.after_prompt = st3.tags[0].cells.map((x) => ({ src: x.src, prompt: x.prompt, png: x.png }));
  out.stale_after_prompt = stale(st3);
  // Setting the same words twice is not an edit, and clearing one that is not there is not either.
  out.prompt_again = await said(await p.post("/prompt",
    { tag: "block", index: 1, prompt: "his left hand is open, not a fist" }));
  out.prompt_clear_absent = await said(await p.post("/prompt",
    { tag: "block", index: 0, prompt: "" }));
  // Clearing it puts the cell back on the character's words AND back on its own drawing.
  out.prompt_clear = await said(await p.post("/prompt", { tag: "block", index: 1, prompt: "" }));
  const st4 = await (await p.get("/state")).json();
  out.after_clear = st4.tags[0].cells.map((x) => ({ src: x.src, prompt: x.prompt, png: x.png }));
  out.stale_after_clear = stale(st4);
}

// =============================================================================================
// 9: a new move queues its clip -----------------------------------------------------------------
// =============================================================================================
{
  const m = await setup();
  out.newmove = await said(await m.post("/newmove", {
    tag: "wave", from: "block", height: "fixed", fps: 12, loop: true, hold_key: false,
    recipe_text: "he waves", cells: 6, cyclic: true }));
  out.newmove_jobs = (await (await m.get("/jobs")).json()).jobs
    .filter((j) => j.state === "queued");
  out.newmove_cancel = await said(await m.post("/cancel", { id: out.newmove[1].job.id }));
  out.newmove_after = {
    jobs: (await (await m.get("/jobs")).json()).jobs.filter((j) => j.kind !== "pack" ||
      j.id === out.newmove[1].job.id),
    tags: (await (await m.get("/state")).json()).tags.map((t) => t.name),
  };
}

console.log(JSON.stringify(out));
"""


def run():
    mjs = "/tmp/worker-select-test.mjs"
    with open(os.path.join(HERE, "celld-editor", "worker.js")) as src, open(mjs, "w") as dst:
        dst.write(src.read())
    harness = "/tmp/select-harness.mjs"
    with open(harness, "w") as f:
        f.write(HARNESS % (mjs, mjs))
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
    now = {f["src"].split("/")[-1]: f for f in o["after"]}

    print("0. the page can map a cell onto the timeline it is drawn on")
    check("the extraction travels with the state", o["sources_len"] == 100,
          f"{o['sources_len']} frames")
    check("every cell says which extracted frame it is",
          [f["at"] for f in o["before"]] == [9, 19, 29, 39, 49, 59],
          str([f["at"] for f in o["before"]]))

    print("5. a selection is validated before one byte of the manifest moves")
    for why in ("empty", "descending", "repeated", "past the end", "negative", "not whole",
                "too many", "unchanged"):
        st, body = o["bad"][why]
        check(f"{why} is refused", st == 400, body.get("error", "")[:78])
    check("and none of them changed the tag",
          o["bad_left_it_alone"] == [f["src"] for f in o["before"]])

    st, body = o["select"]
    print("1. a frame that stays selected keeps its identity and its edits")
    check("the select was accepted", st == 202, json.dumps(body.get("report", body))[:90])
    check("it reports what it did",
          (body.get("cells"), body.get("kept"), body.get("added"), body.get("orphaned"),
           body.get("edits_orphaned")) == (3, 2, 1, 4, 1),
          f"cells={body.get('cells')} kept={body.get('kept')} added={body.get('added')} "
          f"orphaned={body.get('orphaned')} hand-edited={body.get('edits_orphaned')}")
    check("f_0020 kept its id", now["f_0020.png"]["fid"] == was["f_0020.png"]["fid"])
    check("f_0020 kept the hold and the nudge typed onto it",
          (now["f_0020.png"]["hold"], now["f_0020.png"]["nudge"]) == (4, [9, -14]))
    check("f_0020 kept its drawing, byte for byte",
          now["f_0020.png"]["png"] == was["f_0020.png"]["png"], now["f_0020.png"]["png"] or "")
    check("f_0010 kept its id and its drawing",
          now["f_0010.png"]["fid"] == was["f_0010.png"]["fid"] and
          now["f_0010.png"]["png"] == was["f_0010.png"]["png"])

    print("3. a newly activated frame is a cell with no drawing yet")
    check("f_0075 is in the tag", "f_0075.png" in now)
    check("at the base seed and with no drawing",
          now["f_0075.png"]["seed"] == 77 and now["f_0075.png"]["png"] is None)
    check("with an identity of its own",
          now["f_0075.png"]["fid"] not in {f["fid"] for f in o["before"]})
    check("and the tag is in timeline order", [f["at"] for f in o["after"]] == [9, 19, 74])

    print("2. a deactivated frame is orphaned, never deleted")
    tray = {e["src"].split("/")[-1]: e for e in o["tray"]}
    check("all four deselected frames are in the tray",
          set(tray) == {"f_0030.png", "f_0040.png", "f_0050.png", "f_0060.png"}, str(sorted(tray)))
    check("f_0040's nudge went with it", tray["f_0040.png"]["nudge"] == [-5, 3])
    check("and the tray says a person did it",
          all(e["why"] == "deselected by hand" for e in o["tray"]))

    print("4. the invariant holds across a write that removes and adds at once")
    live = {f["fid"] for f in o["after"]}
    held = {e["fid"] for e in o["tray"]}
    check("every id ever written is live or in the tray",
          set(o["fids_ledger"]) == live | held,
          f"{len(o['fids_ledger'])} known, {len(live)} live, {len(held)} in the tray")
    check("and none of them is in two places at once", not (live & held))

    print("6. a cancelled selection puts the tag back exactly")
    st, body = o["cancel"]
    check("the cancel reverted", st == 200 and body.get("reverted") is True, json.dumps(body))
    check("and it was the select's own job, not a repack four edits had joined",
          o["queue_empty_before_select"] == 0)
    back = {f["src"].split("/")[-1]: f for f in o["after_cancel"]}
    check("all six cells are back, with their ids",
          [f["fid"] for f in o["after_cancel"]] == [f["fid"] for f in o["before"]])
    check("with their edits", back["f_0020.png"]["hold"] == 4 and
          back["f_0040.png"]["nudge"] == [-5, 3])
    check("the tray gave the four orphans back",
          not {e["src"].split("/")[-1] for e in o["tray_after_cancel"]} &
              {"f_0030.png", "f_0040.png", "f_0050.png", "f_0060.png"})
    check("and the cell the selection had created is parked, not lost",
          [e["why"] for e in o["tray_after_cancel"]] == ["left behind by a cancelled selection"],
          json.dumps(o["tray_after_cancel"]))

    print("7. a per-frame prompt marks exactly that cell stale")
    st, body = o["prompt"]
    check("the prompt was accepted", st == 202, json.dumps(body)[:80])
    cell = {f["src"].split("/")[-1]: f for f in o["after_prompt"]}
    check("the words are on that cell, trimmed",
          cell["f_0020.png"]["prompt"] == "his left hand is open, not a fist")
    check("and on no other cell",
          [f["prompt"] for f in o["after_prompt"] if f["src"].endswith("f_0020.png") is False]
          == [None] * 5)
    check("that cell now names a different drawing",
          cell["f_0020.png"]["png"] != was["f_0020.png"]["png"] and
          "-p" in cell["f_0020.png"]["png"], cell["f_0020.png"]["png"])
    before, after = o["stale_before"], o["stale_after_prompt"]
    check("nothing was stale before it", not [k for k, v in before.items() if v],
          str(sorted(k for k, v in before.items() if v)))
    gone = [k for k in before if k not in after]
    fresh_now = [k for k, v in after.items() if not v]
    stale_now = [k for k, v in after.items() if v]
    check("the old drawing's step is replaced, not re-keyed",
          gone == ["repaint:cad-block-f_0020"], str(gone))
    check("exactly the new drawing and the pack that reads it are stale",
          sorted(stale_now) == sorted(["pack"] +
                                      [k for k in after if k.startswith("repaint:") and
                                       k not in before]), str(sorted(stale_now)))
    check("every other drawing is still fresh",
          len([k for k in fresh_now if k.startswith("repaint:")]) == 5)
    check("the clip, the frames and the picks were not touched",
          all(not after[k] for k in ("clip:block", "frames:block", "picks:block")))
    st, body = o["prompt_again"]
    check("setting the same words again is refused", st == 400, body.get("error", ""))
    st, body = o["prompt_clear_absent"]
    check("clearing a prompt that is not there is refused", st == 400, body.get("error", ""))
    st, body = o["prompt_clear"]
    check("clearing it is accepted", st == 202)
    cleared = {f["src"].split("/")[-1]: f for f in o["after_clear"]}
    check("the cell is back on the character's words and its own drawing",
          cleared["f_0020.png"]["prompt"] is None and
          cleared["f_0020.png"]["png"] == was["f_0020.png"]["png"])
    # The cleared cell reads NEVER BUILT, and that is the honest answer from this side: the object
    # cannot open a file, so it cannot know the default prompt's drawing is still on disk. It is
    # the same answer a re-roll back to a seed already drawn gives, and `pipeline.build_steps`
    # backfills it to fresh on the next pack without painting anything.
    check("clearing it leaves only that cell and the pack asking to be looked at",
          sorted(k for k, v in o["stale_after_clear"].items() if v)
          == ["pack", "repaint:cad-block-f_0020"],
          str(sorted(k for k, v in o["stale_after_clear"].items() if v)))

    print("8. creating a character queues the plate and then the idle clip")
    st, body = o["create"]
    check("the character exists", st == 200 and body.get("ok") is True, json.dumps(body)[:80])
    jobs = o["create_jobs"]
    check("two jobs, and only two", len(jobs) == 2, json.dumps([j["label"] for j in jobs]))
    check("the plate is first", jobs[0]["kind"] == "plate" and jobs[0]["queuePos"] == 0)
    check("the idle clip is second, and it is idle",
          jobs[1]["kind"] == "clip" and jobs[1]["tag"] == "idle" and jobs[1]["queuePos"] == 1)
    check("the clip waits on the plate", jobs[1]["after"] == jobs[0]["id"])
    check("neither has started", all(j["state"] == "queued" for j in jobs))
    after = o["create_after_cancel"]
    check("cancelling the plate cancels the clip with it",
          [j["state"] for j in after] == ["cancelled", "cancelled"],
          json.dumps([[j["kind"], j["state"], j["error"]] for j in after]))
    check("and it says why",
          after[1]["error"] == "the job it was waiting for did not finish")
    failed = o["plate_failed"]
    check("a plate that FAILS also takes the clip",
          [j["state"] for j in failed] == ["failed", "cancelled"],
          json.dumps([[j["kind"], j["state"]] for j in failed]))

    print("9. a new move queues its own clip, and nothing else")
    st, body = o["newmove"]
    check("the move was made", st == 202, json.dumps(body)[:70])
    jobs = o["newmove_jobs"]
    check("a repack for the tag, and the clip behind it",
          [j["kind"] for j in jobs] == ["pack", "clip"] and jobs[1]["tag"] == "wave",
          json.dumps([j["label"] for j in jobs]))
    check("the clip waits on the edit that created the move", jobs[1]["after"] == jobs[0]["id"])
    check("cancelling the edit takes the clip and the move with it",
          [j["state"] for j in o["newmove_after"]["jobs"]] == ["cancelled", "cancelled"] and
          o["newmove_after"]["tags"] == ["block"],
          json.dumps(o["newmove_after"]["tags"]))

    print("\n" + ("all good" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
