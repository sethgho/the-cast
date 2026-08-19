#!/usr/bin/env python3
"""Prove the cache key does what DESIGN-pipeline.md says, and that both halves agree on it.

    python3 test_pipeline_keys.py

Four claims, all checked against the REAL manifests in `sheets/`, because a key that is right on a
toy is worth nothing:

1. **The two implementations agree.** `pipeline.py` and `celld-editor/worker.js` compute the same
   key for every step of every character. They must: the Durable Object decides what an edit
   invalidated and the packer decides what it built, and a single digit of disagreement would show
   a healthy character as entirely stale — or, far worse, a stale one as fresh.
2. **Playback metadata is OUT.** Re-tag a move's fps and lengthen a cell's hold: the clip, frames,
   picks and repaint steps stay fresh and only the pack goes stale. This is the expensive one to
   get wrong — a 170-second render must not be invalidated by a number the clip has never heard of.
3. **The recipe is IN.** Re-word a move's brief: that move's clip goes stale and everything
   downstream of it follows, while the other six moves are untouched.
4. **A tag NAME re-keys nothing expensive.** Rename a move and not one clip or repaint key moves.
   Two things below it DO go stale and both are honest and cheap: the pack, because tag names are
   written into the atlas JSON, and the frames and picks, because the extracted-frames DIRECTORY
   is named after the tag and a rename orphans it — which the repaints under it then inherit. The
   170-second render, which is the thing this rule exists to protect, is untouched.

No GPU, no packing, no writes: every check runs on a copy of the manifest in memory.
"""
import copy
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NODE = "node"

import canon as C  # noqa: E402
import pipeline as PL  # noqa: E402
import repaint_cells as RC  # noqa: E402
import sprite_files as SF  # noqa: E402

# worker.js is a Worker module, and node will only import it as one from a `.mjs` path. Copying it
# rather than adding a package.json keeps the deployed directory to exactly the files celld needs.
HARNESS = """
import { canon, digest, buildSteps, staleness } from "%s";
const man = JSON.parse(process.argv[2]);
const steps = await buildSteps(man, man.steps);
const stale = staleness(steps);
console.log(JSON.stringify({
  probe: { canon: canon({ b: 1.0, a: [true, null, 0.86, "\\u2014"] }),
           digest: await digest("celld") },
  keys: Object.fromEntries(steps.map((s) => [s.id, s.key])),
  stale: Object.fromEntries(Object.entries(stale).map(([k, v]) => [k, v.stale])),
}));
"""


def in_node(man):
    """`buildSteps` and `staleness` as the Durable Object runs them, over this exact manifest."""
    mjs = "/tmp/worker-under-test.mjs"
    with open(os.path.join(HERE, "celld-editor", "worker.js")) as src, open(mjs, "w") as dst:
        dst.write(src.read())
    script = "/tmp/pipeline-key-harness.mjs"
    with open(script, "w") as f:
        f.write(HARNESS % mjs)
    out = subprocess.run([NODE, script, json.dumps(man)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def stale_ids(man):
    steps = PL.build_steps(man["character"], man)
    return {k for k, v in PL.staleness(steps).items() if v["stale"]}, steps


def check(label, got, want):
    print(f"  {'ok  ' if got == want else 'FAIL'} {label}")
    if got != want:
        print(f"       got  {sorted(got) if isinstance(got, set) else got}")
        print(f"       want {sorted(want) if isinstance(want, set) else want}")
    return got == want


def main():
    ok = True
    for cid in SF.characters():
        man = json.load(open(RC.manifest_path(cid)))
        print(f"{cid}: {len(man['steps'])} steps")

        # 1 -- the two implementations, over the manifest exactly as it sits on disk.
        js = in_node(man)
        mine = {s["id"]: s["key"] for s in PL.build_steps(cid, man)}
        ok &= check("worker.js computes the same key for every step", js["keys"], mine)
        ok &= check("worker.js agrees on the canonical form",
                    js["probe"]["canon"], C.canon({"b": 1.0, "a": [True, None, 0.86, "—"]}))
        ok &= check("worker.js agrees on the digest", js["probe"]["digest"], C.digest("celld"))
        base, _ = stale_ids(man)
        ok &= check("nothing is stale to begin with", base, set())
        ok &= check("worker.js agrees nothing is stale",
                    {k for k, v in js["stale"].items() if v}, set())

        # 2 -- playback metadata is out of the key. Both edits are the ones the editor makes most.
        played = copy.deepcopy(man)
        played["tags"][0]["fps"] = 30
        played["frames"][played["tags"][0]["from"]]["hold"] = 4
        after, _ = stale_ids(played)
        ok &= check("re-tagging fps and lengthening a hold makes ONLY the pack and its exports "
                    "stale", after, {"pack"} | {s["id"] for s in man["steps"]
                                                if s["kind"] == "export"})

        # 3 -- the recipe is in the key, and the news travels downstream.
        move = man["tags"][0]["name"]
        reworded = copy.deepcopy(man)
        reworded["tags"][0]["recipe_text"] += " He wears a hat."
        after, steps = stale_ids(reworded)
        want = {f"clip:{move}", f"frames:{move}", f"picks:{move}", "pack"}
        want |= {s["id"] for s in steps if s["kind"] == "repaint" and s["tag"] == move}
        want |= {s["id"] for s in steps if s["kind"] == "export"}
        ok &= check(f"re-wording {move}'s recipe makes its clip and everything under it stale",
                    after, want)

        # 4 -- and a tag NAME buys nothing expensive.
        renamed = copy.deepcopy(man)
        renamed["tags"][0]["name"] = "stroll"
        steps = PL.build_steps(cid, renamed)
        moved = {s["id"] for s in steps
                 if s["kind"] in ("clip", "repaint") and s["key"] != s["built_key"]}
        ok &= check("renaming a tag re-keys no clip and no repaint", moved, set())
        before = {s["id"]: s["key"] for s in PL.build_steps(cid, man) if s["kind"] == "clip"}
        now = {s["id"]: s["key"] for s in steps if s["kind"] == "clip"}
        ok &= check("and computes the same clip keys under the new name",
                    sorted(now.values()), sorted(before.values()))

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
