# Playback-centred workbench

Seth's shape, from his mockup. It supersedes the seven-chip bench in
`DESIGN-workbench.md`: the chips were the pipeline's structure showing through, and
the user does not think in seven steps. He thinks: pick the frames, then fix the
frames. Two screens.

## The tree

Left rail is a tree: **character → animations**. Selecting a character shows its
header — avatar (the plate), name, description. The description is the character's
movement sentence and it is used when composing every prompt.

## The flow, and where work is queued

1. **New character** — upload an image and write the description. The image is both
   the avatar and the seed image. Creating the character **immediately queues the
   plate and then an idle clip**, because a character with no animation is not
   useful and idle is the default first animation.
2. **New animation** — name it, write its prompt, press save. **Saving queues the
   video.** Creating a thing is the intent to render it, which is why this is the one
   place work starts without a separate priced confirm. Everything *after* creation
   still costs a confirm.
3. Clicking an animation shows a processing status while its video renders, and the
   Select screen once there is one.

## Screen 1 — Select

One view replacing what were three steps (clip, frames, picks).

```
 ┌─────────────┬──────────────────────────────────────────────┐
 │ tree        │  (o) character name                          │
 │ character   │      description                             │
 │  · idle     │                                              │
 │  · walk     │    ┌──────────┐        ┌──────────┐          │
 │             │    │  video   │        │ selected │          │
 │             │    │ looping  │        │  frames  │          │
 │             │    │          │        │ looping  │          │
 │             │    └──────────┘        └──────────┘          │
 │             │                                              │
 │             │  ──────────── timeline scrubber ──────────── │
 │             │  ───────────  playback controls  ─────────── │
 └─────────────┴──────────────────────────────────────────────┘
```

- **Left: the whole clip**, looping. **Right: only the selected frames**, looping —
  so the thing being built plays beside the thing it came from, and the effect of a
  selection is immediate.
- **The timeline is every frame of the clip.** Scrub it. Each frame is
  **activated or deactivated** as selected — that is the pick step, done by hand,
  with the automatic choice as the starting point rather than the only option.
- Playback controls below: play/pause, step, speed.

## Screen 2 — Enhance

The same shape, and the review-and-repaint phase.

- The timeline and playback controls sit where they did, so the two screens are one
  muscle memory.
- **Pausing on a frame makes that frame editable**: its prompt, its seed, its hold.
  Editing offers a re-render of that frame alone, priced.
- This is the final preview: what plays here is what ships.

## What this needs that does not exist

1. **A `select` mutation.** Picks are currently chosen by algorithm and can only be
   dropped. Activating an arbitrary frame needs a mutation that sets the whole
   selection for a tag, reusing the existing carry-over so edits follow their frames.
2. **A per-frame prompt.** Repaint currently takes one prompt per character. A frame
   the user wants to argue with needs its own override, and that override is part of
   that cell's cache key.
3. **Auto-queue on create**, for the character's plate and idle clip and for a new
   animation's clip. Nothing else auto-runs, ever.

## What goes away

The seven-chip row, the before/after judgment pair, the per-step URLs. Pack and
export stop being screens: pack is what Enhance previews, and export is a menu.
