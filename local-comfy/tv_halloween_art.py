#!/usr/bin/env python3
"""Halloween season for the Follies TV theme: the same stage, curtain and paper, redrawn for the
Midnight Spook Show. Edits FROM the base art (tv_follies_art.py, tv_stage_wallpaper.py) so the
composition matches and the season reads as the same theatre dressed up.

    python3 tv_halloween_art.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swifton_canon as C

C.OUT = "/home/wilson/scratch/tv-livingroom/halloween"
BASE = "/home/wilson/scratch/tv-livingroom/"
KEEP = ("f0llie5, image 1 is the drawing to edit. Redraw exactly this picture, the same composition, framing, "
        "lines and sepia palette, dressed for a 1930s Halloween midnight spook show: ")
NO = " No people, no characters, no text, no lettering."
JOBS = {
    "night": (BASE + "wallpaper/stage-s21.png",
              "thick cobwebs draped across the curtains and the valance, the curtain hems tattered, a row of carved "
              "jack-o'-lanterns glowing a warm ochre along the footlights, a pale full moon with a few bats painted on "
              "the backdrop, and drifting fog across the floor. The lower third stays dark and plain."),
    "day": (BASE + "art/matinee-s7.png",
            "cobwebs draped across the curtains and the valance, a row of carved jack-o'-lanterns along the "
            "footlights, a painted harvest moon and a few bats on the backdrop, the house lights still up for a "
            "matinee. The lower third stays plain."),
    "curtain": (BASE + "art/curtain-s21.png",
                "heavy cobwebs hanging from the valance and across the folds, the fringe along the bottom tattered, "
                "and two small bats near the top. The curtain still fills the entire picture."),
    "paper": (BASE + "art/paper-s7.png",
              "a delicate cobweb in each corner and the edges faintly scorched. The middle stays blank and evenly lit."),
}

if __name__ == "__main__":
    os.makedirs(C.OUT, exist_ok=True)
    for name, (src, dress) in JOBS.items():
        w, h = (1024, 576) if name == "paper" else (1664, 928)
        for seed in (7, 21):
            C.run(f"{name}-s{seed}", KEEP + dress + NO, image1=src, seed=seed, w=w, h=h)
