#!/usr/bin/env python3
"""Timed voiceover from an SRT, on our own box.

    python3 build_voiceover.py       # writes workflows/voiceover.json + api/voiceover.api.json

ComfyUI-native, per the standing rule: TTS-Audio-Suite's `UnifiedTTSSRTNode` takes cue timings
and fits the speech to them, so the windows in a script are honoured by the graph rather than by
a pile of ffmpeg `adelay` calls. The only thing left outside is muxing the finished track onto an
existing mp4, which is a container operation, not a pipeline stage.

Four variants, each a different delivery, selected by the VARIANT widget block:

  bright    ChatterBox, exaggeration high. A punchy cartoon read, no reference voice.
  barker    ChatterBox cloning the suite's `broom_salesman` sample -- a carnival pitch.
  f5        F5-TTS instead of ChatterBox: a different model entirely, same cues.
  radio     `bright`, band-limited to a 1930s receiver's passband.

**AudioEqualizer3Band SEGFAULTS the server.** Every run of the `radio` graph killed the ComfyUI
process outright -- six submissions, six `Connection refused`, three systemd restarts -- while the
same three cues through the other variants ran fine. The node is left wired here so the bug stays
reproducible, but do NOT queue this variant on the shared box. The shipped radio track was
band-limited with ffmpeg afterwards instead, which is post-processing on a finished file rather
than a pipeline stage.

`stretch_to_fit` is the timing mode that actually honours a window; `smart_natural` will run a
cue long and push the rest. max_stretch_ratio stays near 1.0 so a line that does not fit is
reported in the timing report rather than being sped up into a chipmunk.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_workflows import Graph, WIDGETS, WIDGET_TYPES, BLUE, GREEN, GREY  # noqa: E402
from build_extras import note  # noqa: E402

WIDGETS.setdefault("ChatterBoxEngineNode",
                   ["language", "device", "exaggeration", "temperature", "cfg_weight",
                    "crash_protection_template"])
WIDGETS.setdefault("F5TTSEngineNode",
                   ["language", "device", "temperature", "speed", "target_rms",
                    "cross_fade_duration", "nfe_step", "cfg_strength"])
WIDGETS.setdefault("UnifiedTTSSRTNode",
                   ["srt_content", "narrator_voice", "seed", "timing_mode", "enable_audio_cache",
                    "fade_for_StretchToFit", "max_stretch_ratio", "min_stretch_ratio",
                    "timing_tolerance", "batch_size", "use_native_duration_targeting"])
# Widget ORDER is the API contract, and it is not the order a human would guess: the node
# interleaves gain and frequency per band. Read it off /object_info, never assume.
WIDGETS.setdefault("AudioEqualizer3Band",
                   ["low_gain_dB", "low_freq", "mid_gain_dB", "mid_freq", "mid_q",
                    "high_gain_dB", "high_freq"])
WIDGETS.setdefault("NormalizeAudioLoudness", ["target_lufs"])
WIDGETS.setdefault("SaveAudioMP3", ["filename_prefix", "quality"])
# app_input() writes a slot for EVERY widget on the node, not just the promoted ones, so each
# node needs its full widget->type map registered or it raises on the first unlisted one.
_TYPES = {
    ("ChatterBoxEngineNode", "language"): "COMBO",
    ("ChatterBoxEngineNode", "device"): "COMBO",
    ("ChatterBoxEngineNode", "crash_protection_template"): "STRING",
    ("UnifiedTTSSRTNode", "enable_audio_cache"): "BOOLEAN",
    ("UnifiedTTSSRTNode", "fade_for_StretchToFit"): "FLOAT",
    ("UnifiedTTSSRTNode", "batch_size"): "INT",
    ("UnifiedTTSSRTNode", "use_native_duration_targeting"): "BOOLEAN",
}
WIDGET_TYPES.update(_TYPES)
for _n, _ws in (("ChatterBoxEngineNode", ("exaggeration", "temperature", "cfg_weight")),
                ("UnifiedTTSSRTNode", ("seed", "max_stretch_ratio", "min_stretch_ratio",
                                       "timing_tolerance")),
                ("AudioEqualizer3Band", ("low_gain_dB", "low_freq", "mid_gain_dB", "mid_freq",
                                         "mid_q", "high_gain_dB", "high_freq"))):
    for _w in _ws:
        WIDGET_TYPES.setdefault((_n, _w), "INT" if _w == "seed" else "FLOAT")
WIDGET_TYPES.setdefault(("UnifiedTTSSRTNode", "srt_content"), "STRING")
WIDGET_TYPES.setdefault(("UnifiedTTSSRTNode", "timing_mode"), "COMBO")
WIDGET_TYPES.setdefault(("UnifiedTTSSRTNode", "narrator_voice"), "COMBO")

# The cues, condensed to fit. The draft ran ~480 words a minute inside its own windows -- three
# times a human read -- so every line here is cut to roughly 2.7 words a second and the technical
# nouns (WHUF, ETH, Ethos) are kept while the connective tissue goes.
SRT = """1
00:00:00,200 --> 00:00:06,000
After your token sale delivers, there's a one-month window where WHUF can only be vouched.

2
00:00:06,200 --> 00:00:10,000
First: you now type the exact WHUF amount you want.

3
00:00:10,300 --> 00:00:16,700
Watch the amount field. Confirm dims while it re-prices, then unlocks the moment the quote agrees.

4
00:00:16,900 --> 00:00:20,700
Second: when two wallets are involved, the app labels each one.

5
00:00:20,900 --> 00:00:25,200
Left, the refund returns to your Ethos wallet. Right, your connected wallet funds the vouch.

6
00:00:25,400 --> 00:00:29,200
Third: the exit fee. Before Unlock, there's no WHUF exchange rate yet.

7
00:00:29,400 --> 00:00:33,800
So hover the breakdown, and the fee now reads in the ETH you actually hold.

8
00:00:34,000 --> 00:00:40,600
Two more guards: a balance change mid-conversion vouches less instead of failing, and the safety switch is checked throughout.
"""

VARIANTS = {
    # name: (engine, engine widgets, narrator_voice, band-limit?)
    "bright": ("chatterbox", {"exaggeration": 1.35, "temperature": 0.9, "cfg_weight": 0.4}, "none", False),
    "barker": ("chatterbox", {"exaggeration": 1.6, "temperature": 1.0, "cfg_weight": 0.3},
               "voices_examples/higgs_audio/broom_salesman.wav", False),
    # F5-TTS is wired and loads, but returns ~1s of audio here with and without a reference
    # voice -- its checkpoint never came down. Left in the builder so the wiring is not lost.
    "f5":     ("f5", {}, "none", False),
    # The sober control: same cues, no scenery-chewing. A product demo usually wants this one.
    "straight": ("chatterbox", {"exaggeration": 0.5, "temperature": 0.7, "cfg_weight": 0.6},
                 "voices_examples/David_Attenborough CC3.wav", False),
    "radio":  ("chatterbox", {"exaggeration": 1.35, "temperature": 0.9, "cfg_weight": 0.4}, "none", True),
}

HOW_TO = """# Timed voiceover

**An SRT in, one audio track fitted to those cues out.**

| Control | What it does |
|---|---|
| **1 · SCRIPT (SRT)** | Cue numbers, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, then the line. Timings are honoured. |
| **2 · DELIVERY** | ChatterBox `exaggeration` is the cartoon dial: 0.5 is flat, 1.3 is animated, 1.6+ starts chewing scenery. |
| **3 · VOICE** | `narrator_voice` clones one of the suite's samples. `none` uses the engine's own voice. |
| **4 · TIMING** | `stretch_to_fit` honours the window. `smart_natural` lets a cue run long and pushes the rest. |
| **5 · RADIO EQ** | Bypass this to keep the clean read. It cuts below ~300Hz and above ~3kHz so the voice sits in a period receiver's passband. |

**Read `timing_report`.** It says which cues did not fit. A line that needs more than
`max_stretch_ratio` gets sped up, and a sped-up read is the first thing that sounds wrong.

Rule of thumb: **2.7 words a second.** A 4-second card holds about eleven words. Scripts written
without that in mind run three times too long and no amount of stretching saves them.
"""


def build(variant):
    engine_kind, ew, voice, band = VARIANTS[variant]
    g = Graph()
    note(g, f"HOW TO USE — voiceover ({variant})", HOW_TO)

    if engine_kind == "chatterbox":
        eng = g.add("ChatterBoxEngineNode", "▶ 2 · DELIVERY", (-60, 60), (420, 260),
                    {"language": "English", "device": "auto", "exaggeration": ew["exaggeration"],
                     "temperature": ew["temperature"], "cfg_weight": ew["cfg_weight"],
                     # The default template is "hmm ,, {seg} hmm ,," and ChatterBox SPEAKS it: the first
                     # pass came back with an "Eh", a "Call it!" and a "YOOO!" wedged between cues.
                     # It exists to stop the model crashing on very short segments; ours are whole
                     # sentences, so bare {seg} is safe and silent.
                     "crash_protection_template": "{seg}"},
                    outputs=[("TTS_ENGINE", "TTS_ENGINE")], color=GREEN)
        g.app_input(eng, "exaggeration", "temperature", "cfg_weight")
    else:
        eng = g.add("F5TTSEngineNode", "▶ 2 · DELIVERY (F5)", (-60, 60), (420, 300),
                    {"language": "F5TTS_v1_Base", "device": "auto", "temperature": 0.9,
                     "speed": 1.0, "target_rms": 0.1, "cross_fade_duration": 0.15,
                     "nfe_step": 32, "cfg_strength": 2.0},
                    outputs=[("TTS_ENGINE", "TTS_ENGINE")], color=GREEN)

    srt = g.add("UnifiedTTSSRTNode", "▶ 1 · SCRIPT (SRT)", (420, 60), (600, 620),
                {"srt_content": SRT, "narrator_voice": voice, "seed": 11,
                 "timing_mode": "stretch_to_fit", # Cached on TEXT+voice+seed, NOT on engine settings: after fixing the spoken filler template the
                 # re-render returned the identical file. Off here so a settings change is always heard.
                 "enable_audio_cache": False,
                 "fade_for_StretchToFit": 0.02, "max_stretch_ratio": 1.15,
                 "min_stretch_ratio": 0.85, "timing_tolerance": 2.0, "batch_size": 0,
                 "use_native_duration_targeting": False},
                links={"TTS_engine": (eng, 0, "TTS_ENGINE", False)},
                outputs=[("audio", "AUDIO"), ("generation_info", "STRING"),
                         ("timing_report", "STRING"), ("Adjusted_SRT", "STRING")], color=BLUE)
    g.app_input(srt, "srt_content", "narrator_voice", "seed", "timing_mode")

    tail = srt
    if band:
        tail = g.add("AudioEqualizer3Band", "▶ 5 · RADIO EQ", (1080, 60), (400, 240),
                     {"low_gain_dB": -18.0, "low_freq": 320, "mid_gain_dB": 6.0,
                      "mid_freq": 1600, "mid_q": 0.9, "high_gain_dB": -16.0,
                      "high_freq": 3000},
                     links={"audio": (srt, 0, "AUDIO", False)},
                     outputs=[("AUDIO", "AUDIO")], color=GREEN)
        g.app_input(tail, "low_gain_dB", "mid_gain_dB", "high_gain_dB", "low_freq", "high_freq")

    out = g.add("SaveAudioMP3", "RESULT", (1080, 340), (480, 200),
                {"filename_prefix": f"vo/{variant}", "quality": "V0"},
                links={"audio": (tail, 0, "AUDIO", False)}, color=GREY)
    g.app_output(out)
    rep = g.add("PreviewAny", "TIMING REPORT — read this", (1560, 60), (420, 240),
                links={"source": (srt, 2, "STRING", False)}, color=GREY)
    g.app_output(rep)
    return g


if __name__ == "__main__":
    WIDGETS.setdefault("PreviewAny", [])
    for v in VARIANTS:
        g = build(v)
        for path, blob in ((os.path.join(HERE, "workflows", f"voiceover-{v}.json"), g.to_ui()),
                           (os.path.join(HERE, "workflows", f"voiceover-{v}.app.json"), g.to_ui()),
                           (os.path.join(HERE, "api", f"voiceover-{v}.api.json"), g.to_api())):
            json.dump(blob, open(path, "w"), indent=1)
        print("wrote voiceover-" + v)
