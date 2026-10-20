#!/usr/bin/env python3
"""An empty Follies stage, drawn locally, as the living-room TV's launcher wallpaper.

    python3 tv_stage_wallpaper.py

Text-to-image through Qwen-Image-Edit with no input image and the f0llie5 LoRA at 1.2, the same
graph as swifton_canon.py. Rendered at 1664x928 (16:9-ish, inside the 12GB envelope) for a 1920x1080
screen; the launcher scales it. The lower third stays dark and plain because the dock sits there.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swifton_canon as C

C.OUT = "/home/wilson/scratch/tv-livingroom/wallpaper"
PROMPT = ("f0llie5, an empty 1930s vaudeville theatre stage seen straight on from the stalls. Heavy "
          "velvet curtains are drawn open and gathered at both sides, with a scalloped swagged valance "
          "across the whole top edge. A bare wooden stage floor with a row of footlights along its front "
          "edge. The centre of the stage is an empty, softly lit backdrop. The lower third of the picture "
          "is dark, plain and calm. No people, no characters, no props, no text, no lettering, no signs.")

if __name__ == "__main__":
    os.makedirs(C.OUT, exist_ok=True)
    for seed in (7, 21, 42):
        C.run(f"stage-s{seed}", PROMPT, seed=seed, w=1664, h=928)
