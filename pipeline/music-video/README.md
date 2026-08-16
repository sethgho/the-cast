# The cast music-video pipeline

Turn a locally-generated song into a beat-locked vaudeville music video, entirely
on gpu-worker's 12GB card, without adopting a second AI stack.

Written 2026-08-15 after looking at [Maestro](https://github.com/Blizaine/Maestro)
v1.8.5, whose "Director mode" does exactly this. Maestro wants 24GB and 150-500GB
of its own models on a WanGP pipeline that shares nothing with our ComfyUI. The
*idea* is worth stealing; the software is not a fit. This is the idea, sized for
what we have.

## What already existed

Almost all of it. This directory adds one thing — the **planner** — on top of the
`minimax-h3` skill's existing stack:

| piece | where | status |
|---|---|---|
| song generation | MiniMax Music 3, `references/music-3.md` | working |
| keyframes from character sheets | nano-banana-pro chain, `references/keyframes-from-character-sheets.md` | working |
| character consistency | reference sheets + tokens, `references/character-consistency-stack.md` | working |
| keyframe-to-keyframe video | `scripts/scene_seq.py`, both-ends conditioning | working |
| the period look | `fps=12` + unsharp + CRF 30, `references/cast-delivery-recipe.md` | working |
| **beat-locked shot planning** | **`plan.py`, here** | **new** |

## The one hard problem: cuts must land on downbeats

A music video is judged on whether the cuts feel intentional, and that is entirely
a question of whether they land on the beat. Three facts make that non-trivial:

1. **H3 shots are quantised.** Frame counts must sit on a `17k+5` grid — 107, 124,
   141, 158 — and off-grid values snap *up*, which can push a marginal job into
   OOM. So you cannot simply ask for 5.94 seconds.
2. **The music is not quantised to that grid.** A bar is whatever the tempo says
   it is.
3. **The tempo is not what you asked for.** The caption for this song requested
   132 BPM. The model delivered **161.5 BPM measured**. A shot list built on the
   requested tempo drifts a full bar within thirty seconds and every cut after
   that is wrong.

The resolution is three rules, all encoded in `plan.py`:

- **Measure the tempo off the rendered audio**, never off the prompt.
- **Shots declare BARS, not seconds.** The planner snaps each shot's end to the
  nearest measured downbeat.
- **Generate long, trim to the beat.** Round frames *up* to the grid and record
  the overshoot per shot. Trimming is free and exact; a short shot can only be
  fixed with a freeze frame. Overshoot is the cheap direction to be wrong in.

## Usage

```bash
./plan.py song.mp3 shots.yaml -o plan.json     # review this before rendering
```

`plan.py` generates nothing. Keyframes cost Comfy Cloud credits and each shot
costs several minutes of exclusive GPU, so the plan is a reviewable artifact and
the render is a separate, deliberate step.

Its warnings are the point — it flags shots that waste GPU on overshoot, shots
that exceed the measured VRAM envelope, and any part of the song left unscored.

## Renderers are data, not code

The planner does not know about H3. It knows about **`renderers.json`**, and H3
is one entry in it. Adding a model is a data change.

That split exists because the first version hardcoded H3's frame grid, fps,
canvas ceiling and cost as module constants. Correct while one model could do
every shot; wrong the instant a shot needed something H3 cannot do.

**A shot declares what it NEEDS; the router picks something that provides it.**

```yaml
- id: kf03_seth_boast
  sings: true        # -> needs an audio_driven renderer
  bars: 4
```

| capability | means |
|---|---|
| `both_ends` | conditions on a first AND last keyframe (bounded drift) |
| `first_frame` | conditions on a starting image only |
| `native_audio` | invents its own audio from the prompt (dialogue, SFX) |
| `audio_driven` | takes EXISTING audio as the driver — lips, head, body |
| `silent` | can be asked to produce no audio |

`audio_driven` is the one that matters for music video and the one H3 does not
have. H3 only ever generates its own audio from text, which is exactly why it
cannot sing along to a song that already exists — no amount of prompting closes
that gap, because it is a different model class.

Three things fall out of the registry that were previously impossible to state:

- **Frames quantise on the ROUTED renderer's grid, at its fps.** H3 is 24fps on
  a 17k+5 grid; Wan-family models are 16fps on 4n+1. The same 6-second shot is
  158 frames on one and 97 on the other. Hardcoding either is a silent bug.
- **Audio-driven shots get an `audio_slice`** in the plan — the exact span of
  song that shot sits over. A beat-locked plan already knows it, so emitting it
  beats making the render step re-derive it and drift a frame.
- **Budget is reported per renderer**, because "three hours of GPU" is a
  different sentence when half of it is on a model nobody here has run.

### Verified renderers

| | H3 | InfiniteTalk |
|---|---|---|
| drives from | keyframes + text | **existing audio** |
| canvas | 1088x608 | 640x384 (832x480 OOMs) |
| fps / grid | 24 / 17k+5 | 25 / 4n+1 |
| max frames | 158 | 81 per window |
| cost | 0.040 min/frame | 0.033 min/frame |

Both measured on the 12GB card, 2026-08-15. The cost per frame is close; the
**resolution is not**. Singing shots come out at 2.7x fewer pixels than the rest
of the cut, so plan for a visible softness on them and grade it deliberately
rather than being surprised by it.

InfiniteTalk needs `blocks_to_swap=40` and `load_device=offload_device` — 11GB of
Q4 weights stream from the box's 31GB of system RAM. Its wav2vec model goes in
`models/wav2vec2`, not `models/wav2vec`; the wrong folder is silently invalid.

**The aesthetic risk did not bite.** Wan 2.1 is trained on photoreal footage, so
the open question was whether an audio-driven performance would drag a sepia
rubber-hose drawing toward live action. Conditioned on our own keyframe it kept
the ink line, the halftone, the proscenium, the aged paper and Seth's design.

### `status` is load-bearing

`measured` means someone ran it on this hardware and wrote the number down.
`unverified` means it came out of a README. The planner warns on every shot
routed to an unverified renderer, and refuses to quote a cost for it.

This is the 124-frame ceiling lesson encoded: that number sat in our docs as
fact for weeks, was never measured, and turned out to be 158. A number with no
experiment behind it is a guess wearing a lab coat.

### The fallback is loud on purpose

A shot that asks for a capability nothing provides still gets planned — on the
default renderer — and raises a warning saying it *will not do what the shot
list says*. Silently downgrading "this character sings" to "silent
interpolation" is how you end up with a finished video nobody can explain the
badness of.

## Rules a shot list must follow

**One action per shot.** From `references/multi-character-scenes.md`: an action
that spans a shot boundary gets double the screen time and reads as slow motion.
Wilson took ten seconds to walk into a room that way. If a character needs to
arrive, they are *already arrived and settled* in the shot's `last` keyframe.

**Describe a performance, not a destination.** A prompt that names an end state
leaves roughly 43% of the clip as a frozen hold. Say what continues throughout.

**Read `cast/{id}/character.yaml` before staging anyone.** The `notes` field
carries staging warnings you would otherwise get wrong: Wilson is a living picket
fence with **no mouth** — he cannot lip-sync, so his verse plays over gesture and
the rocking of the panel. Ake's sheet shows no waterline, so sloshing water must
be stated explicitly. Ake's ochre is the only warm colour in the strip and must be
named in every prompt or the accent spreads.

## Where this departs from the cast delivery recipe

**Generate silent.** The delivery recipe uses native H3 audio, correctly — for
dialogue, external TTS can never sync to lip movement H3 already rendered. A music
video is the opposite case: the song *is* the audio track. Every shot generates
with the vocal-negative list and the song is laid under the stitched cut.

This also means lip-sync to the vocal is **not solved here**. Wilson has no mouth
and Ake is a fish, which covers half the cast; Seth and Cadbury singing in sync
with a pre-existing vocal is an open problem, not a shipped feature.

## Step 3 (optional): review the storyboard before spending GPU

`storyboard.html` is a reusable, JSON-driven reviewer. Point it at a data file
and it plays the song while showing each keyframe at its own timestamp, so you
review the cut at performance speed rather than as a wall of stills.

```
storyboard.html?data=my-song.storyboard.json
```

```json
{ "title": "...", "audio": "song.mp3", "imageBase": "frames/", "tempo": 161.5,
  "shots": [ { "id": "kf02_seth_enters", "start": 11.49, "end": 17.46,
               "frames": 158, "first": "B1_prep.png", "last": "B2_prep.png",
               "prompt": "…the motion prompt…" } ] }
```

Pause anywhere and type; notes autosave to `localStorage` keyed by the data file,
so a refresh or a closed tab does not cost you the pass. **Copy all notes** emits
one markdown block to paste straight back as a revision prompt. `Space` plays,
`←`/`→` step shots, `F` flips between a shot's first and last keyframe.

Two behaviours that are deliberate, not incidental:

- **The song is the clock while it plays**, but an explicit pick wins while it is
  paused. Without that second half, clicking a thumbnail before the audio had
  loaded snapped straight back to shot 1 — `currentTime` never moved, so the
  clock kept re-deciding.
- **A pause timestamp is only reported when it falls inside the shot it is
  attached to.** Jump straight to a thumbnail and the clock reading is
  meaningless; a wrong timestamp in the handoff is worse than none.

The point of doing this before rendering: a keyframe costs one API call to
change, and the shot it becomes costs six minutes of exclusive GPU.

## Keyframes: setups, not one long chain

Chained edits hold framing beautifully *within* a locked-off setup — but a chain
is the wrong tool across a cut, because a cut is precisely where the staging is
supposed to change. So keyframes are generated per SETUP: one base, then pose
edits from it.

The pilot's eleven shots are four setups — marquee (2 frames), the stage (6),
the troupe line (3), Wilson at the rail (4) — which is **15 keyframes, not 12**.
Shots inside a setup share a frame; shots either side of a cut do not.

Evidence it worked: the border-crop boxes came back identical within each setup
(`85,23,1359,728` for all six stage frames), which is the framing holding still
to the pixel.

`prep_keyframes.py` crops the aged-paper margin nano-banana-pro draws no matter
how firmly the prompt forbids one, then cover-fits to 1088x608. It finds the
margin by row/column standard deviation rather than a fixed inset, because the
margin width is not constant — the Wilson setup came back with none at all.

## Budget, honestly

For the 165s of "Everything's Fine":

| | pilot (60s, 11 shots) | full song (~30 shots) |
|---|---|---|
| keyframes (Comfy Cloud credits) | 12 calls | ~31 calls |
| GPU (local, free, exclusive) | ~65 min | ~3 hours |

**Do the pilot first.** 65 minutes is the exact budget already proven in the
delivery recipe. If the look holds across eleven shots and two characters, the
rest is the same thing repeated — and if it does not, it cost an hour instead of
three.

The GPU is exclusive for the whole run: ComfyUI evicts texture-api, whisper-api
and kokoro-tts, so a full-song render takes those three offline for three hours.
Announce it before starting.

## Status

- `plan.py` — working, run against the real song.
- `everything-is-fine.pilot.yaml` — 11-shot pilot covering 0:00-1:00.
- `everything-is-fine.pilot.plan.json` — the generated plan.
- `storyboard.html` + `everything-is-fine.storyboard.json` — the reviewer, live at
  `http://wilson/everything-is-fine-storyboard/`.
- **15 keyframes generated** (2026-08-15) and reviewed — every character on-model,
  framing held within each setup.
- **Video not yet rendered.** ~64 minutes of exclusive GPU, pending the
  storyboard pass.
