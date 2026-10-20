#!/usr/bin/env python3
"""Art for the Follies TV launcher theme (heidihowilson/follies-tv, themes/follies).

    python3 tv_follies_art.py

- curtain:  the house curtain fully closed, the frame the launcher raises on wake (curtain.jpg)
- matinee:  the evening stage (tv_stage_wallpaper.py, seed 21) redrawn with the house lights up
- paper:    a blank aged playbill sheet, the ground the tile cards are typeset on
Same graph as swifton_canon.py: Qwen-Image-Edit Q4 + Lightning + f0llie5 at 1.2.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swifton_canon as C

C.OUT = "/home/wilson/scratch/tv-livingroom/art"
EVENING = "/home/wilson/scratch/tv-livingroom/wallpaper/stage-s21.png"

CURTAIN = ("f0llie5, a heavy 1930s vaudeville theatre house curtain, fully closed, filling the entire picture "
           "from edge to edge. Deep vertical velvet folds hang straight down, with a scalloped swagged valance "
           "across the top edge and a fringe along the bottom. No stage, no people, no props, no text, no lettering.")
MATINEE = ("f0llie5, image 1 is the drawing to edit. Redraw exactly this empty theatre stage, the same curtains, "
           "valance, floor, footlights and composition, but with the house lights up for an afternoon matinee: "
           "warm, even, brighter light across the whole stage, the curtains and the floor, with no single spotlight. "
           "The lower third stays plain. No people, no text.")
PAPER = ("f0llie5, a single blank sheet of aged, foxed cream playbill paper filling the whole picture, flat and "
         "evenly lit, with faint stains, fibres and soft edges of wear. No drawing, no border, no ornament, no text.")

if __name__ == "__main__":
    os.makedirs(C.OUT, exist_ok=True)
    for seed in (7, 21):
        C.run(f"curtain-s{seed}", CURTAIN, seed=seed, w=1664, h=928)
        C.run(f"matinee-s{seed}", MATINEE, image1=EVENING, seed=seed, w=1664, h=928)
    C.run("paper-s7", PAPER, seed=7, w=1024, h=576)
