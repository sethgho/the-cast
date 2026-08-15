#!/usr/bin/env python3
"""Turn a song plus a shot list into a beat-locked H3 render plan.

This is the planner half of the music-video pipeline. It does the one thing a
human should not do by hand — align every cut to a real downbeat in the actual
audio — and emits a beats JSON that `scene_seq.py` can render.

It deliberately does NOT generate anything. Keyframes cost Comfy Cloud credits
and each shot costs ~6 minutes of GPU, so the plan is reviewed before it runs.

    ./plan.py song.mp3 shots.yaml -o plan.json

Read README.md in this directory for the design and the constraints it encodes.
"""
import argparse
import json
import sys

import librosa
import numpy as np
import yaml

FPS = 24.0

# H3 accepts frame counts on a 17k+5 grid; anything else snaps UP, which can
# push a marginal job into OOM. Generate on the grid, trim to the beat.
GRID = [17 * k + 5 for k in range(0, 22)]

# Measured 2026-08-15 on gpu-worker, both-ends conditioning at 1088x608:
# 158 frames (6.58s) completed in 380s and peaked around 9.4GB of 12GB. That is
# the longest verified shot at this canvas — 4 bars at this song's tempo — and it
# left real headroom, so 175 would probably go too. Probably is not measured.
MEASURED_MAX_FRAMES = 158
MINUTES_PER_FRAME = (380 / 60) / 158


def grid_ceil(frames: float) -> int:
    """Smallest legal frame count that is at least `frames`.

    Always rounds UP, never down. A shot generated short cannot be padded — the
    only honest fix would be a freeze frame — whereas a shot generated long is
    trimmed to the downbeat for free. Overshoot is the cheap direction.
    """
    for g in GRID:
        if g >= frames:
            return g
    return GRID[-1]


def downbeats(path):
    """Measured downbeats, in seconds.

    Everything here comes off the audio. The BPM you asked for in the caption is
    a request, not a contract — MiniMax Music 3 answered a 132 BPM prompt with a
    161.5 BPM track, and a shot list built on the prompt would have drifted a
    full bar inside thirty seconds.
    """
    y, sr = librosa.load(path, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_t = librosa.frames_to_time(beats, sr=sr)

    # 4/4 assumed. Pick the downbeat phase with the most onset energy on beat 1.
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    at_beat = onset[np.clip(beats, 0, len(onset) - 1)]
    phase = int(np.argmax([at_beat[p::4].mean() for p in range(4)]))
    return tempo, beat_t[phase::4], librosa.get_duration(y=y, sr=sr)


def snap(t, grid):
    """Nearest downbeat to time t."""
    return float(grid[int(np.argmin(np.abs(grid - t)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("shots", help="YAML shot list")
    ap.add_argument("-o", "--out", default="plan.json")
    args = ap.parse_args()

    tempo, bars, dur = downbeats(args.song)
    shots = yaml.safe_load(open(args.shots))["shots"]

    plan, cursor, warnings = [], 0.0, []
    for i, s in enumerate(shots):
        # Each shot declares how many BARS it wants. Bars, not seconds — a cut
        # that lands mid-bar reads as a mistake no matter how good the art is.
        want_end = cursor + s["bars"] * (4 * 60.0 / tempo)
        end = snap(want_end, bars) if want_end < dur else dur
        length = end - cursor
        if length <= 0:
            warnings.append(f"shot {i} ({s['id']}) has no time left; dropped")
            continue

        frames = grid_ceil(length * FPS)
        generated = frames / FPS
        # Overshoot is trimmed in the stitch. Recorded per shot so the ffmpeg
        # step is deterministic rather than re-derived.
        plan.append({
            "id": s["id"],
            "start": round(cursor, 3),
            "end": round(end, 3),
            "seconds": round(length, 3),
            "frames": frames,
            "generated_seconds": round(generated, 3),
            "trim_seconds": round(generated - length, 3),
            "first": s["first"],
            "last": s["last"],
            "prompt": s["prompt"],
        })
        if generated - length > 1.0:
            warnings.append(
                f"shot {i} ({s['id']}) wastes {generated - length:.1f}s of GPU; "
                f"consider a different bar count")
        if frames > MEASURED_MAX_FRAMES:
            warnings.append(
                f"shot {i} ({s['id']}) is {frames} frames — above the "
                f"{MEASURED_MAX_FRAMES} measured at this canvas with both-ends "
                f"conditioning. Render this one FIRST to prove it fits.")
        cursor = end

    if cursor < dur - 0.5:
        warnings.append(
            f"shot list covers {cursor:.1f}s of a {dur:.1f}s song — "
            f"{dur - cursor:.1f}s unscored")

    out = {
        "song": args.song,
        "measured_tempo_bpm": round(tempo, 2),
        "bar_seconds": round(4 * 60.0 / tempo, 4),
        "song_seconds": round(dur, 2),
        "shot_count": len(plan),
        # Scaled by frames, not per-shot: a 158-frame shot is not a 124-frame
        # shot, and a flat multiplier understated the pilot by a third.
        "gpu_minutes_estimate": round(
            sum(p["frames"] for p in plan) * MINUTES_PER_FRAME, 1),
        "keyframe_calls": len(plan) + 1,
        "shots": plan,
        "warnings": warnings,
    }
    json.dump(out, open(args.out, "w"), indent=1)

    print(f"{len(plan)} shots · {out['gpu_minutes_estimate']} GPU-min · "
          f"{out['keyframe_calls']} keyframes · tempo {out['measured_tempo_bpm']}")
    for w in warnings:
        print("  WARN:", w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
