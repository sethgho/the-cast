#!/usr/bin/env python3
"""Thanksgiving and the four December weeks of the Follies TV theme.

    python3 tv_seasons_art.py

Every image is an edit, never a fresh generation, so the theatre stays the same theatre. December
CHAINS: each week edits the previous week's picture, so decorations accumulate instead of being
redrawn - week 1 frost, week 2 greenery, week 3 the tree, week 4 the full spectacular.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swifton_canon as C

ROOT = "/home/wilson/scratch/tv-livingroom/"
OUT = ROOT + "seasons"
BASE = {"night": ROOT + "wallpaper/stage-s21.png", "day": ROOT + "art/matinee-s7.png",
        "curtain": ROOT + "art/curtain-s21.png", "paper": ROOT + "art/paper-s7.png"}
KEEP = ("f0llie5, image 1 is the drawing to edit. Redraw exactly this picture, the same composition, framing, "
        "lines and sepia palette, and add only this: ")
NO = " Keep everything already in the picture. No people, no characters, no text, no lettering."
STAGE_NOTE = " The lower third stays dark and plain."

SEASONS = [
    ("thanksgiving", False, {
        "night": "garlands of autumn leaves along the valance, sheaves of wheat and pumpkins and gourds at both ends of the stage, a harvest moon on the backdrop" + STAGE_NOTE,
        "day": "garlands of autumn leaves along the valance, sheaves of wheat and pumpkins and gourds at both ends of the stage, the house lights still up" + STAGE_NOTE,
        "curtain": "a garland of autumn leaves and acorns along the valance, with a few leaves drifting down the folds",
        "paper": "a small sprig of autumn leaves and wheat in each corner, the middle blank and evenly lit"}),
    ("christmas1", False, {
        "night": "the first frost: a light dusting of snow along the stage edge and the footlights, a few snowflakes falling, frost on the valance" + STAGE_NOTE,
        "day": "the first frost: a light dusting of snow along the stage edge and the footlights, frost on the valance, the house lights still up" + STAGE_NOTE,
        "curtain": "a light frost along the valance and a few snowflakes drifting down the folds",
        "paper": "a sprig of holly with berries in each corner, the middle blank and evenly lit"}),
    ("christmas2", True, {
        "night": "evergreen garlands swagged along the valance tied with ribbon bows, a large wreath hung at the centre of the backdrop, a little more snow along the stage" + STAGE_NOTE,
        "day": "evergreen garlands swagged along the valance tied with ribbon bows, a large wreath at the centre of the backdrop, the house lights still up" + STAGE_NOTE,
        "curtain": "evergreen garland swags along the valance tied with ribbon bows",
        "paper": None}),
    ("christmas3", True, {
        "night": "a tall Christmas tree on the stage at one side, strung with glowing bulbs and topped with a star, wrapped presents beneath it, stockings hung along the proscenium" + STAGE_NOTE,
        "day": "a tall Christmas tree on the stage at one side, strung with bulbs and topped with a star, wrapped presents beneath it, the house lights still up" + STAGE_NOTE,
        "curtain": "strings of glowing Christmas bulbs woven through the garland along the valance",
        "paper": None}),
    ("christmas4", True, {
        "night": "Santa's sleigh and reindeer flying across a full moon on the backdrop, snow falling across the whole stage, candy canes and more presents on the floor, every bulb glowing" + STAGE_NOTE,
        "day": "Santa's sleigh and reindeer painted across the backdrop, snow falling across the whole stage, candy canes and more presents on the floor, the house lights still up" + STAGE_NOTE,
        "curtain": "a sprig of mistletoe and a big ribbon bow at the centre of the valance, and snow falling across the curtain",
        "paper": None}),
]

if __name__ == "__main__":
    prev = dict(BASE)
    for name, chain, jobs in SEASONS:
        C.OUT = f"{OUT}/{name}"
        os.makedirs(C.OUT, exist_ok=True)
        src = prev if chain else BASE
        made = {}
        for part, dress in jobs.items():
            if dress is None:
                continue
            w, h = (1024, 576) if part == "paper" else (1664, 928)
            out = C.run(part, KEEP + dress + NO, image1=src[part], seed=7, w=w, h=h)
            if out:
                made[part] = out
        prev = {**src, **made} if name.startswith("christmas") else dict(BASE)


# The day stage of a late December week, redrawn from that week's own night stage, so a decoration
# the day chain dropped (the tree, the sleigh) can't go missing. Run with: python3 tv_seasons_art.py days
MATINEE_FROM_NIGHT = ("f0llie5, image 1 is the drawing to edit. Redraw exactly this picture, every decoration, the "
                      "same composition, lines and sepia palette, but with the house lights up for an afternoon "
                      "matinee: warm, even, brighter light across the stage, curtains and floor." + STAGE_NOTE + NO)
if __name__ == "__main__" and sys.argv[1:] == ["days"]:
    for name in ("christmas3", "christmas4"):
        C.OUT = f"{OUT}/{name}"
        C.run("day-from-night", MATINEE_FROM_NIGHT, image1=f"{OUT}/{name}/night.png", seed=7, w=1664, h=928)
