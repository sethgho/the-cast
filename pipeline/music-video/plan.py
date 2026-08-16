#!/usr/bin/env python3
"""Turn a song plus a shot list into a beat-locked, per-renderer render plan.

This is the planner half of the music-video pipeline. It does the two things a
human should not do by hand:

  1. align every cut to a real downbeat measured in the actual audio, and
  2. route each shot to a renderer that can actually do what the shot needs,
     quantised to THAT renderer's frame grid and fps.

It deliberately does NOT generate anything. Keyframes cost credits and every
shot costs minutes of exclusive GPU, so the plan is a reviewable artifact and
the render is a separate decision.

    ./plan.py song.mp3 shots.yaml -o plan.json

Read README.md in this directory for the design and the constraints it encodes.
"""
import argparse
import json
import os
import sys

import librosa
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


# --- renderers ---------------------------------------------------------------
# Everything model-specific lives in renderers.json. It used to live in module
# constants here, which was fine while one model could do every shot and wrong
# the moment one could not.

def load_renderers(path):
    reg = json.load(open(path))
    return reg["renderers"], reg["routing"]


def grid_ceil(frames, grid):
    """Smallest legal frame count for this renderer that is at least `frames`.

    Always rounds UP. A shot generated short cannot be padded — the only honest
    fix is a freeze frame — whereas a shot generated long is trimmed to the
    downbeat for free. Overshoot is the cheap direction to be wrong in, and it
    is cheap in the same way for every renderer, which is why this rule is
    shared even though the grids are not.
    """
    a, b = grid["a"], grid["b"]
    k = max(0, int(np.ceil((frames - b) / a)))
    return min(a * k + b, a * grid["max_k"] + b)


def route(shot, renderers, routing):
    """Pick a renderer for one shot, and say why.

    Returns (name, capability, reason). A shot that asks for something nothing
    provides still gets planned — on the fallback — and gets a warning, because
    silently downgrading "this character sings" to "silent interpolation" is
    exactly the kind of quiet substitution that produces a finished video nobody
    can explain the badness of.
    """
    for rule in routing:
        cond = rule["when"]
        if cond != "default" and not shot.get(cond):
            continue
        need = rule["needs"]
        for name in rule["prefer"]:
            r = renderers.get(name)
            if r and all(c in r["provides"] for c in need):
                return name, cond, r["status"]
    return routing[-1]["prefer"][0], "default", "fallback"


# --- audio -------------------------------------------------------------------

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

    onset = librosa.onset.onset_strength(y=y, sr=sr)
    at_beat = onset[np.clip(beats, 0, len(onset) - 1)]
    phase = int(np.argmax([at_beat[p::4].mean() for p in range(4)]))
    return tempo, beat_t[phase::4], librosa.get_duration(y=y, sr=sr)


def snap(t, grid):
    return float(grid[int(np.argmin(np.abs(grid - t)))])


# --- planning ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("shots", help="YAML shot list")
    ap.add_argument("-o", "--out", default="plan.json")
    ap.add_argument("--renderers", default=os.path.join(HERE, "renderers.json"))
    args = ap.parse_args()

    renderers, routing = load_renderers(args.renderers)
    tempo, bars, dur = downbeats(args.song)
    doc = yaml.safe_load(open(args.shots))
    shots = doc["shots"]
    bar_s = 4 * 60.0 / tempo

    plan, cursor, warnings = [], 0.0, []
    for i, s in enumerate(shots):
        want_end = cursor + s["bars"] * bar_s
        end = snap(want_end, bars) if want_end < dur else dur
        length = end - cursor
        if length <= 0:
            warnings.append(f"shot {i} ({s['id']}) has no time left; dropped")
            continue

        rname, why, status = route(s, renderers, routing)
        r = renderers[rname]
        frames = grid_ceil(length * r["fps"], r["frame_grid"])
        generated = frames / r["fps"]

        entry = {
            "id": s["id"],
            "renderer": rname,
            "routed_by": why,
            "renderer_status": status,
            "start": round(cursor, 3),
            "end": round(end, 3),
            "seconds": round(length, 3),
            "fps": r["fps"],
            "frames": frames,
            "generated_seconds": round(generated, 3),
            "trim_seconds": round(generated - length, 3),
            "first": s["first"],
            "last": s.get("last"),
            "prompt": s["prompt"],
        }
        # An audio-driven renderer needs the slice of song this shot sits over.
        # A beat-locked plan already knows it exactly, so emit it rather than
        # making the render step re-derive it and drift by a frame.
        if "audio_driven" in r["provides"]:
            entry["audio_slice"] = {"start": round(cursor, 3),
                                    "end": round(end, 3),
                                    "source": os.path.basename(args.song)}
        plan.append(entry)

        if status == "unverified":
            warnings.append(
                f"shot {i} ({s['id']}) routes to {rname}, which is UNVERIFIED on "
                f"this hardware — spike one shot before planning a whole video on it")
        if status == "fallback":
            warnings.append(
                f"shot {i} ({s['id']}) asked for '{why}' and no renderer provides "
                f"it — fell back to {rname}, so this shot will NOT do what the "
                f"shot list says")
        if frames > r["max_frames"]:
            warnings.append(
                f"shot {i} ({s['id']}) is {frames} frames, above {rname}'s "
                f"{r['max_frames']} — render this one FIRST to prove it fits")
        if generated - length > 1.0:
            warnings.append(
                f"shot {i} ({s['id']}) wastes {generated - length:.1f}s of GPU; "
                f"consider a different bar count")
        cursor = end

    if cursor < dur - 0.5:
        warnings.append(
            f"shot list covers {cursor:.1f}s of a {dur:.1f}s song — "
            f"{dur - cursor:.1f}s unscored")

    # Budget per renderer, because "3 hours of GPU" is a different sentence when
    # half of it is on a model we have never run.
    budget = {}
    for p in plan:
        r = renderers[p["renderer"]]
        b = budget.setdefault(p["renderer"], {"shots": 0, "frames": 0,
                                              "minutes": 0.0, "status": r["status"]})
        b["shots"] += 1
        b["frames"] += p["frames"]
        if r["minutes_per_frame"]:
            b["minutes"] = round(b["minutes"] + p["frames"] * r["minutes_per_frame"], 1)
        else:
            b["minutes"] = None

    out = {
        "song": args.song,
        "measured_tempo_bpm": round(tempo, 2),
        "bar_seconds": round(bar_s, 4),
        "song_seconds": round(dur, 2),
        "shot_count": len(plan),
        "keyframe_calls": doc.get("meta", {}).get("keyframes", len(plan) + 1),
        "budget_by_renderer": budget,
        "shots": plan,
        "warnings": warnings,
    }
    json.dump(out, open(args.out, "w"), indent=1)

    print(f"{len(plan)} shots · tempo {out['measured_tempo_bpm']}")
    for name, b in budget.items():
        mins = f"{b['minutes']} min" if b["minutes"] is not None else "cost UNKNOWN"
        print(f"  {name:14} {b['shots']:2} shots · {b['frames']:4} frames · "
              f"{mins}  [{b['status']}]")
    for w in warnings:
        print("  WARN:", w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
