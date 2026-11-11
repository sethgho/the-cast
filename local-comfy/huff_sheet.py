"""Huff's sheet.png: retitle Swifton's blank layout, then compose the matted views onto it.

    python3 huff_sheet.py      # after huff_canon.py views + cut
"""
import os
os.chdir("/home/wilson/scratch/huff")
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
T = Image.open("/home/wilson/scratch/swifton/sheet-blank-s7.png").convert("RGB")
BOX = (575, 52, 962, 132)                     # the old lettering, inside the ribbon's outline
reg = T.crop(BOX)
a = np.array(reg).astype(int)
lum = a.mean(axis=2)
dark = lum < 205
paper_rgb = tuple(int(v) for v in np.median(a[lum >= 205], axis=0))
mask = Image.fromarray((dark * 255).astype("uint8")).filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(3))
noise = np.random.default_rng(7).normal(0, 4, a.shape)
paper = Image.fromarray((np.array(paper_rgb) + noise).clip(0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1))
reg.paste(paper, (0, 0), mask)
T.paste(reg, BOX[:2])
d = ImageDraw.Draw(T)
f = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/C059-Bold.otf", 78)
d.text(((BOX[0] + BOX[2]) / 2, (BOX[1] + BOX[3]) / 2 + 2), "HUFF", font=f, fill=(30, 24, 18), anchor="mm")
T.save("sheet-blank-huff.png")
T.crop((480, 20, 1060, 180)).save("/tmp/ribbon.png")
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

T = Image.open("sheet-blank-huff.png").convert("RGB")   # retitle.py: SWIFTON erased, HUFF set
W, H = T.size  # 1536 x 1024
INK = (43, 33, 24)
FONT = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"

# 1. Repaint the expressions panel interior with clean paper (its generated text was garbled).
EX = (1301, 178, 1472, 944)
pw, ph = EX[2] - EX[0], 250
patch = T.crop((740, 700, 740 + pw, 700 + ph))     # empty paper below the view labels
for y in range(EX[1], EX[3], ph):
    T.paste(patch.crop((0, 0, pw, min(ph, EX[3] - y))), (EX[0], y))
d = ImageDraw.Draw(T)
hdr = ImageFont.truetype(FONT, 22)
d.text(((EX[0] + EX[2]) // 2, EX[1] + 22), "EXPRESSIONS", font=hdr, fill=INK, anchor="mm")
d.line((EX[0] + 10, EX[1] + 40, EX[2] - 10, EX[1] + 40), fill=INK, width=2)


def place(img, cx, bottom, max_w, max_h):
    im = img.copy()
    im.thumbnail((max_w, max_h), Image.LANCZOS)
    T.paste(im, (int(cx - im.width / 2), int(bottom - im.height)), im)


# 2. Turnaround above the five printed labels.
LABEL_TOP = 645
for cx, f in zip((166, 412, 668, 902, 1132),
                 ("view-front", "view-3q-front", "view-side", "view-3q-back", "view-back")):
    place(Image.open(f"cut/{f}.png"), cx, LABEL_TOP - 14, 236, 440)

# 3. Expressions: the head and shoulders cut from each full-body expression render.
lab = ImageFont.truetype(FONT, 20)
slot_h = (EX[3] - EX[1] - 50) / 4
for i, (f, name) in enumerate((("expr-smug", "SMUG"), ("expr-pitching", "PITCHING"),
                               ("expr-outraged", "OUTRAGED"), ("expr-triumphant", "TRIUMPHANT"))):
    im = Image.open(f"cut/{f}.png")
    head = im.crop((int(im.width * 0.12), 0, int(im.width * 0.88), int(im.height * 0.36)))
    top = EX[1] + 50 + i * slot_h
    place(head, (EX[0] + EX[2]) / 2, top + slot_h - 34, EX[2] - EX[0] - 16, slot_h - 44)
    d.text(((EX[0] + EX[2]) / 2, top + slot_h - 18), name, font=lab, fill=INK, anchor="mm")

# 4. Props: the fountain pen and the W lapel pin.
PR = (60, 726, 636, 936)
place(Image.open("cut/prop-pen.png"), 250, PR[3] - 14, 190, 190)
place(Image.open("cut/prop-pin.png"), 460, PR[3] - 24, 150, 150)

out = T.resize((2528, 1696), Image.LANCZOS)
out.save("sheet.png")
small = out.copy(); small.thumbnail((1000, 700)); small.save("sheet-sm.jpg", quality=85)
print("sheet.png", out.size)
