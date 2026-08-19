# From a cell editor to a pipeline tool

The editor we built edits the last two steps of a seven-step pipeline. Everything
upstream — the plate, the clip prompt, the render settings, how cells are chosen — lives
in Python constants and is only changeable by editing code. This is the plan to make the
whole chain inspectable, tweakable and re-runnable, with downstream steps invalidated
when something they depend on changes.

It supersedes `DESIGN-editor.md`'s stage 6 framing, which said "nothing here changes how a
cell is generated; the pipeline stays exactly as it is". That is no longer true.

## The chain, as it actually runs

```
plate ─┬─► clip(move) ─► frames(move) ─► picks(move) ─► repaint(cell) ─┐
       └─► (traits and recipes are inputs, not steps)      pack(char) ◄─┘ ─► export(format)
```

Seven step kinds, fixed wiring, instances per (character, move). So this is a **typed DAG
with a hardcoded shape**, not a node canvas — a free graph editor would be a worse tool and
ten times the work.

## Staleness by content address

Each step instance records its params, its input artifacts' hashes, a cache `key` and the
`built_key` it was actually built at. `stale = key != built_key`, propagated downstream.

**In the key:** step kind, params, upstream artifact hashes, the prompt *template version*.
**Not in the key:** fps, hold, loop, direction, pivot, tag names, QC thresholds — playback
metadata that the pack reads and a clip has never heard of. This is what stops re-tagging a
move from invalidating a 170-second render.

**Staleness never runs anything.** A stale step is drawn stale; running it is an explicit,
priced click. A clip can also be **pinned** — the artifact stays valid though its key has
drifted — which is the escape hatch for fixing a typo you know did not matter.

## Hand edits under a re-run

The hardest question in the design, and a naive cascade throws away every manual fix.

**Hand edits are overlays addressed by stable frame identity, never by artifact hash, and a
re-run never deletes one — it orphans it visibly.**

- **Repaint re-run** — same source frame, new drawing. Identity survives; seed, pivot and
  hold survive by construction. Nothing is lost.
- **Picks re-run** — match new picks to old frames by source-frame *time* within the clip
  (nearest index, ±3 frames). Matched frames inherit their edits and their id; unmatched old
  frames go to a per-tag **orphan tray** — visible, restorable, collected only by hand. A
  drop is recorded against a frame id, so a re-matched frame stays dropped.
- **Clip re-run** — frame times no longer mean anything, so nothing auto-matches. The tag's
  frames go to the tray and the UI offers an adoption view. This is honest: a new clip is a
  new performance, and pretending edits carry over is how a pivot nudge lands on the wrong
  pose.
- **Pack and export** — pure functions of upstream. Always safe.

## What each step exposes

Editable, versus visible-but-locked, versus hidden. The locked column is the opinionated
part: several constants were set by measurement and cost days, so they are shown greyed with
the one-line reason from the skill inline, and there is **no slider UI for them at all**.
Overriding one is an explicit act stored as a named deviation, so QC can report forever after
that this character runs with `PAD=0.90`.

| Step | Editable | Locked, with its reason shown | Hidden |
|---|---|---|---|
| Plate | upload/replace, composite preview | the key colour | the compositing graph |
| Clip | recipe text, trait line, seed, cyclic | `SPRITE_LOCK`, `STAGE_RESTATE`, `SOUND_LOCK` assembled read-only; 832/20/61; the idle-pin rule | node graph, cache params |
| Frames | nothing | frame count, `skip=6` | — |
| Picks | cell count, per-cell repick, cycle on/off | `CYCLE_SCORE_MAX` beside this move's measured score | motion-energy internals |
| Repaint | seed per cell, the repaint prompt per character | `PAD 0.86`, steps, the negative | graph |
| Pack | fps, hold, loop, direction, pivot, tag names, unify | the ±6% clamp, `ANCHOR_PULL`, the feet line, the scale caps | numpy internals |
| Export | format | the naming rule | — |

## What this implies that does not exist yet

**A new subject**: upload an image → cut it out on gpu-worker → write the one-sentence trait
line → the character exists. That means `CHARACTERS` stops being a constant and the DO
namespace is listed from the bucket.

**A new move**: a move becomes manifest data — name, recipe, cell count, fps, loop, hold_key,
unify, cyclic — created from an existing move as a template. `MOVES` in two Python files
becomes bootstrap defaults only. `repaint_cells.py` already complains in a comment that the
recipe exists in two independent copies; this is the fix.

Creating a move forces one question, because getting it wrong is a known silent killer:
**does this move's height change?** Unify on a jump is the bug we shipped and had to measure
our way back out of.

## The UI

A **horizontal pipeline rail** above the editor that already exists: seven chips, each with a
status dot (fresh / stale / pinned / running) and its price (`~170s`, `~45s x n`, `free`).
The clip, frames, picks and repaint chips are scoped by the selected tag, so the tag bar
becomes the scoping control. Clicking a chip opens that step in the right-hand panel.

Running a stale step shows a **cost receipt** first: "re-run Clip(walk): 170s clip + 10
repaints + repack, about 11 minutes, and walk's 3 hand edits move to the tray." There is never
a one-click cascade — the receipt is the confirmation, and each downstream step enqueues as
its own cancellable job.

The whole current centre survives: filmstrip, canvas, preview, variants, scrubber. Re-roll
and re-pick move into the Repaint and Picks chips.

## Mobile

The phone is a **review-and-approve surface plus the cheap edits**, not a pivot-dragging
surface. One column, three screens behind a segmented control: **Preview** (the atlas canvas
playing a tag — this is already a canvas stepping a PNG and it is good on a phone), **Rail**
(the chips vertically with their prices and Run, so approving a queued re-run from the sofa
works), **Cells** (filmstrip as a grid, tap for variants, one-tap re-roll and drop). Pivot
drag and duration handles are hidden below 760px, not shrunk.

The clunky video is mp4 seeking: iOS forces controls and fullscreen and seeking is slow.
Replace it with `playsinline muted preload=metadata` for playback, and scrub against a lazy
**frame strip** instead — every extracted frame is already on disk in the pick directory, and
scrubbing images beats scrubbing an mp4 on every phone.

## Stages

1. **Manifest grows step records.** DO and agent write `key`/`built_key`/`params`; nothing
   reads them yet. Verify playback fields round-trip byte-identical. No GPU.
2. **Moves and characters become data.** Verify `repaint_cells.py seth` reproduces today's
   atlas bit-for-bit from manifest-carried recipes. No GPU.
3. **The rail, staleness and receipts.** Verify a recipe edit marks the clip stale and nothing
   auto-runs. No GPU.
4. **Agent job types `clip` and `extract`, and repick with edit carry-over.** The risky one.
   ~15 min GPU on one throwaway move.
5. **New-character flow.** ~10 min GPU per move tried.
6. **Mobile layout and the frame-strip scrubber.** Verify on a real phone. No GPU.

## The three risks

1. **The orphan tray silently eats a hand edit** on a half-matching repick. Caught by a DO
   invariant asserted on every mutation: every frame id ever created is live, dropped, or in
   the tray — never simply absent.
2. **An unlocked constant quietly regresses a sheet.** Caught by comparing the QC report
   against the last green pack automatically after every pack, with the diff on the Pack chip,
   and by overrides being named so a regression names its own cause.
3. **A 170-second clip job stalls the shared queue**, whose running lease assumes paint-scale
   work. Caught by soak-testing stage 4 with a deliberate mid-clip kill and a cancel.
