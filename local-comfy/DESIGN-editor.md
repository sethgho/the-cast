# Sprite editor: what it should become

The editor works, and it is the wrong shape. This is what to build instead, and why —
based on how Aseprite, Piskel, TexturePacker, Unity and Godot actually solve the same
problems.

The finding worth leading with: **three of the hardest problems in this project are
already solved conventions we did not adopt.** We invented worse versions of them in
Python. And there is exactly one axis where no existing tool helps us, because their
frames are drawn by hand and ours are generated.

## 1. One sheet with tags, not seven atlases

Every serious tool treats a character as **one sheet** whose frames are grouped into
named ranges: Aseprite calls them *tags*, Godot calls them *animations*, and both carry
a name, an fps and a loop direction. That is precisely our seven moves.

We ship seven separate atlases and a `<cid>-moves.json` that stitches them back
together. The cost of that choice is already in the code: the packer must compute one
shared scale across all seven, because a per-move scale would make the character resize
when the state machine switches move, and `repaint_cells.main()` therefore has to pack
every move on every run. One sheet cannot have that bug — it has one scale by
construction.

**Adopt:** one atlas per character, tags for the moves, `tags: [{name, from, to, fps,
direction}]` in the JSON.

## 2. Per-frame duration, not one fps per move

Aseprite gives every frame its own duration in milliseconds; that is how an animator
holds an extreme. Godot 4 does the simpler version — a per-frame multiplier over the
animation's fps — and that is the one we want.

We have exactly one lever per move, and it is the wrong one. The punch wants to sit on
full extension for two beats; raising the whole move's fps makes it snappier everywhere
and fixes nothing. `hold_end()` exists because we could not express "hold this frame".

**Adopt:** `hold` as a per-frame integer multiplier, defaulting to 1.

## 3. Pivot is a value you drag, not a constant I tune

Every tool has an editable origin per sprite; TexturePacker has it per *sprite* in a
sheet, which matters more for us than for pixel artists because our frames genuinely
drift.

Ours is hard-coded at 94% of the cell height. Read back the week: `ANCHOR_PULL`,
`torso_cx`, the feet-line arguments, the head-clipping fix — **every one of those was
me editing Python to move a pivot.** That is a slider.

**Adopt:** a sheet-level pivot (the default), a per-frame pivot nudge, both dragged on
the canvas against a ground line. Keep the automatic anchor as the *initial* value, not
the only one.

## 4. Loop direction, trim, and the words themselves

- **Loop direction** — forward / reverse / **ping-pong**, per tag. Free for us, and
  ping-pong is what idle actually wants.
- **Trim** — strip the transparent margin and record the offset so runtime placement is
  unaffected. Our packer already does this internally; expose it under its real name.
- **Terminology**: tag, pivot, ping-pong, loop, trim, atlas, frame, fps, JSON hash vs
  JSON array. Anyone who has used Aseprite or Unity has muscle memory for these. Do not
  invent synonyms.

**Do not adopt** — pixel-art and legacy baggage that would be noise at 512px: extrude,
power-of-two constraints, 9-slice borders, physics shapes, isometric slicing, layers.

## 5. The axis that is ours alone: provenance

No sprite editor has this, because no sprite editor's frames cost 45 seconds of GPU and
carry a source frame, a seed and a prompt behind them.

- **Variants** — version history per frame. Every (source frame, seed) pair is a file
  that already exists; switching between them is instant. Built, keep.
- **Source scrubber** — the H3 clip behind the move, scrubbable, click a frame to repick.
  This is our equivalent of "draw the in-between", and it is free.
- **Cost-aware queue** — the only editor where an action has a price. Say which actions
  generate and which do not, and never let two ComfyUI jobs run at once.

## 6. Onion skinning, repurposed

The received wisdom is that onion skinning is a drawing aid and therefore useless when
frames come from a pipeline. Half right: useless for *drawing*, but ghosting the
neighbouring cell at 30% opacity is exactly how you SEE a scale drift or a foot
slipping — which is what we spent the week measuring numerically instead.

**Adopt it as a QA overlay**, with the ground line and the pivot crosshair on the same
canvas.

## 7. Export presets

Emit what engines actually load: **JSON hash** and **JSON array** (Pixi and Phaser
loaders branch on exactly this), **Phaser 3**, **Godot SpriteFrames**, and our own CSS
`steps()`. The atlas plus one JSON is the deliverable; everything else is a template
over the same data.

## Layout

```
 tags  [ idle ][ walk ][ punch ][ kick ][ jump ][ block ][ crouch ]     <- drag to retag
 film  [1][2][3][4][5][6][7][8][9][10] ...                             <- drag edge = duration
 ─────────────────────────────────────────────────────────────────────
 canvas: selected frame, pivot crosshair, ground line, onion ghost
 preview: always running · fps · loop direction · scrub
 ─────────────────────────────────────────────────────────────────────
 frame: source frame + scrubber · variants · re-roll / re-pick / drop
 export: JSON hash · JSON array · Phaser 3 · Godot · CSS steps()
```

## Order of work

The data model comes first, because everything else hangs off it.

0. **Constraint on everything below:** manifest reads and writes go behind a narrow
   interface — `load_character_manifest` / `save_character_manifest` and nothing else. No
   `json.load` on a manifest path anywhere else in the codebase. Stage 6 replaces that
   interface with an HTTP call to a Durable Object; if the calls are scattered, the
   editor gets rewritten twice.

1. **Tags and per-frame duration** — one manifest per character, tags replacing the
   seven-way split, `hold` per frame. Migrate the existing `sheets/*.json`.
2. **Pivot as data** — sheet pivot plus per-frame nudge, dragged on canvas, with the
   current automatic anchor as the seed value.
3. **Timeline UI** — filmstrip with tag bands, duration handles, transport, onion skin.
4. **Export presets** — the five formats above.
5. **The provenance panel** — fold the existing variants and source scrubber into the
   new layout.

Nothing here changes how a cell is generated. The pipeline stays exactly as it is; this
is all downstream of the atlas.


## Stage 6 — deploy as a private service on celld

The editor should run as a service on the cluster, not as a script someone remembers to
start. celld is the right host for **half** of it, and being precise about which half is
the whole design.

### What cannot move

The packer is numpy and PIL. The repaint drives ComfyUI on gpu-worker. Both read files
under `/tmp` and `/home/wilson/artifacts`. None of that runs in V8, so porting
`sprite_editor.py` to a Worker is not an option and never will be.

### What celld is genuinely right for

**The manifest, as one Durable Object per character.** Single-threaded, consistent, and
its state lives in the bucket, so it survives a node restart and a container reboot —
both measured. Every edit stages 2 through 4 add (retag, per-frame hold, pivot, reorder)
is pure state manipulation with no filesystem, which is exactly the shape celld's docs
describe as its sweet spot: many small, independent, persistent state machines.

The DO also owns the **job queue**, which fixes something the Python server cannot: two
browsers can currently both submit the same 45-second GPU job. A DO is the natural lock.

### The split

```
  browser ──► celld Worker (assets: the editor page)
                 └─► CharacterDO  — manifest, job queue, cost accounting
                        ▲
                        │ long-poll for work, post results
                 wilson agent  — repaint (ComfyUI on gpu-worker), pack (numpy/PIL)
                        │
                        └─► images stay on wilson's artifact server, served over the LAN
```

Images stay where they are. celld's assets are per-deploy and not writable at runtime,
so pushing 5120x3072 atlases through the bucket would mean S3 signing inside the Worker
for no benefit.

### Private means no ingress

Skip the tunnel and the DNS record — those steps exist to publish a fleet. This one is
LAN only: a bucket, a scoped MinIO user, a systemd unit on the next free port (8087,
internal 18087) on CT 113, reachable at `192.168.0.19:8087` or a split-horizon
`sprites.gholson.lan`. Budget ~42 MB idle for the fleet.

### Two things that will bite

- **A deploy takes the DO route down briefly** (~10s on v0.2.1, ~25s on v0.1.0) while the
  asset path keeps answering 200. So health-check a DO-backed route, never `/`, and pick
  one that returns 5xx rather than 404 when routing is broken — `celld-release` counts a
  404 as healthy.
- **workerd extensions are missing from celld.** `Response.redirect()` and
  `crypto.subtle.timingSafeEqual` do not exist; both have broken a live fleet here.
  A test suite running on workerd cannot catch either.
