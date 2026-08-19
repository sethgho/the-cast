# The workbench — the UI, rethought from the ground up

Two designs were drawn from first principles without sight of the existing page, then
judged adversarially. The workbench won; the ladder lost for a reason worth keeping:
**the pipeline is a tree, and a lineage-first layout draws chains.** Its own author had
already folded the 61 frames into a scrubber and buried the seeds in an inspector — two
admissions that the shape of the UI was fighting the shape of the data. The previous
attempt was rejected for exactly that kind of incoherence.

## The one rule

**The left panel of any step is the right panel of the previous step.**

Press the previous chip and the output you were judging slides left and becomes an input.
Walking chips leftward is walking upstream. No graph is ever drawn, and "where did this
come from" is answered by a sentence a user can hold in their head — which is what the
last attempt lacked.

The ladder contributes exactly two things, as content rather than as a second layout: the
61-tick filmstrip with pick flags, and the seven-dot pipeline strip on an animation row.

## Screens

Two. No modals except the re-run confirm.

- **Home** — a character grid, 160px plate circles, name, animation count, plus a
  new-character card (image + one sentence on how they move; creates nothing else).
- **Bench** — everything else, at `/c/:char/a/:anim/s/:step`, so every state is linkable
  and the back button walks upstream.

## Bench layout, desktop (designed at 1440, minimum 1280)

- **Top bar 48px** — character › animation switchers, the queue line, the tray badge.
- **Left rail 64px** — this character's animations as 48px looping thumbnails, each under
  a seven-dot pipeline strip (green done, grey empty, amber stale, pulsing running).
  Hover widens to 240px for names; click selects and it snaps back.
- **Result strip 224px** — the packed animation looping at game size on a checkerboard
  with 1x/2x/4x zoom, beside a 96px filmstrip of the packed cells. Selecting a cell here
  drives trace and per-cell editing. Never packed yet: a silhouette and "you are N steps
  away".
- **Chip row 44px** — plate · clip · frames · picks · repaint · pack · export, each with a
  status dot and its price. Arrow keys walk them.
- **Judgment area** — two 600px panels, 24px gutter: left is the input, right is the
  output. Changing step slides the content horizontally, 200ms, in the direction of
  travel. The step's single re-run button sits under the right panel; its editable inputs
  under the left.

## What each step shows

| step | left (input) | right (output) |
|---|---|---|
| plate | source photo + the movement sentence, editable | the cut-out plate at 600px |
| clip | plate + recipe textarea + playback settings | the mp4, looping |
| frames | the 61-tick filmstrip, drag to scrub | a 6x11 grid of 88px stills |
| picks | the same filmstrip with the gait curve drawn under it and 6-10 draggable flags | the picks as 140px thumbnails, labelled with their frame number |
| repaint | **breaks the pair, deliberately** — see below | |
| pack | the chosen repaints with a draggable foot-pivot crosshair and a hold stepper | the atlas at 600px with cell outlines |
| export | the file cards | download atlas, data, or both |

**Repaint is a grid, not a pair**, because one-in-one-out cannot express eight picks times
every seed ever painted. A column per pick, minimum 140px: the pick still on top, the
current repaint under it, then a seed shelf of every drawing ever painted for that cell at
72px, newest first, the current one ringed. Clicking a seed swaps it instantly and free.
Per-column re-run, plus one "repaint all N stale".

**Trace** is the cross-step view, pinned rather than hidden: selecting a cell in the result
strip pins a 72px strip above the chip row showing that one cell at every step — source
frame, pick, repaint, packed cell — each clickable to jump to that step with the cell
already selected.

## Intervention

- **Recipe and movement sentence** — textareas in the plate and clip left panels. Editing
  marks the step it feeds stale and runs nothing.
- **Picks** — drag a flag to move one, double-click the strip to add, click a flag's x to
  remove.
- **Seed** — one free click on the shelf. Never confirms, because nothing is spent.
- **Pivot and hold** — on the pack left panel; both free, both mark pack stale.

## Re-runs, staleness, queue, tray

One re-run button per step, always priced in plain time ("Re-run clip · ~2m 50s"). Click
opens an inline confirm stating the price, what goes stale, and — for a clip, in bold —
how many cell edits cannot be carried and will move to the tray. **Nothing auto-runs, ever,
including downstream steps after an upstream finishes. There is no regenerate-everything.**

**Staleness is exactly one visual code**: a solid amber dot on the step whose input
changed, a hollow amber dot on everything downstream, and a 2px amber top border with one
line of explanation on a stale output. No washes, no dimming — the ladder's three
simultaneous codes are what a user who already rejected one incoherent UI cannot afford.

**Queue** is one line in the top bar, expanding to a 280px list with per-job cancel. Serial,
FIFO, no reordering.

**Tray** is a slide-up drawer behind a counted badge. Each entry shows the old frame, the
edits it carries, an "apply to nearest new frame" with the suggestion pre-computed, and a
discard. Entries persist until acted on; the badge is what stops it becoming a graveyard.

## Phone, at 390px — review and approve only

The result loop full width, the chip row scrolling, and **one panel at a time** with a
"◂ input" button to flip. Two things are tappable: the seed shelves (free and safe) and
approve-or-not on a re-run proposed elsewhere. No textareas, no flag dragging, no pivot
nudging. Home gains a "needs review" list.

## Deliberately not built

A graph or node canvas. Batch re-runs across animations. Queue reordering. An undo history
— the seed shelf *is* the history. Editing on the phone. Auto-run chains. Any cost
accounting beyond the price on each button.

## The three risks, and how we would know early

1. **The repaint grid breaks the pair metaphor on purpose.** If it is wrong, users get lost
   exactly there. Tell: at the repaint step, can a user say what is upstream by pointing?
2. **The slide has to actually teach the invariant.** Tell: asked where a pick came from,
   does the user press the previous chip, or reach for trace? If they reach for trace, the
   slide is not landing and should be slowed or labelled before anything is added.
3. **Trace as the only cross-step tool.** Tell: if it is pinned constantly, the single-pair
   altitude is wrong and the result strip needs a per-cell mini-pipeline instead.
