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
- **Not yet rendered.** Needs section-boundary verification by ear (marked TODO
  in the YAML) and Seth's go-ahead on the credit spend.
