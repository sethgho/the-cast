"""Sethmoji post effects, as ComfyUI nodes.

- FillMaskFromSurroundings: paint a masked region with its own surroundings (a clean plate, near
  enough, for slivers of the original performer peeking out from behind the new one).
- BrightTextMask: finds burned-in white meme text, so it can be painted out too.
- BlinkCaption: meme text, white with a black outline, shown only on the listed frames.
- MindBlownFX: freezes the temples pose, blows the head apart from the mouth outward in a spark
  burst, holds on black, then brings him back through a ring of light that drops to his
  shoulders -- the beats of the Tim Heidecker original.
- SlidingDoorFX: an inked van door pulled shut across the frame.

Frames in, frames out; everything is deterministic from the seed. Install by copying this folder
into ComfyUI/custom_nodes and restarting.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "Anton-Regular.ttf")


class FillMaskFromSurroundings:
    """Normalised box blurs at growing radii: each hole pixel takes the average of the known
    pixels around it, smallest radius first, so thin slivers fill from what is right next to
    them and only big holes reach far."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "sethmoji"

    def run(self, images, mask):
        img = images.permute(0, 3, 1, 2).clone()               # B,C,H,W
        hole = (mask > 0.5).float().unsqueeze(1)
        if hole.shape[0] == 1 and img.shape[0] > 1:
            hole = hole.expand(img.shape[0], -1, -1, -1)
        known = 1 - hole
        for r in (2, 4, 8, 16, 32, 64, 128):
            num, den = _box(img * known, r), _box(known, r)
            new = ((1 - known) * (den > 1e-3)).bool()
            img = torch.where(new.expand_as(img), num / den.clamp(min=1e-6), img)
            known = torch.maximum(known, new.float())
        return (img.permute(0, 2, 3, 1).clamp(0, 1),)


def _box(x, r):
    """Sum over a (2r+1)^2 window from an integral image: cost independent of r. Sums rather
    than means, since the caller divides one box by another."""
    p = F.pad(x.double(), (r + 1, r, r + 1, r))
    c = p.cumsum(2).cumsum(3)
    return (c[..., 2 * r + 1:, 2 * r + 1:] - c[..., :-2 * r - 1, 2 * r + 1:] \
        - c[..., 2 * r + 1:, :-2 * r - 1] + c[..., :-2 * r - 1, :-2 * r - 1]).float()


class BrightTextMask:
    """Burned-in meme text: near-white pixels in the top and bottom bands, grown to take in the
    black outline. Add it to the performer's mask before FillMaskFromSurroundings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "threshold": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 1.0, "step": 0.01}),
            "top_band": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 0.5, "step": 0.01}),
            "bottom_band": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.01}),
            "grow": ("INT", {"default": 7, "min": 0, "max": 64}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "sethmoji"

    def run(self, images, threshold, top_band, bottom_band, grow):
        B, H, W, _ = images.shape
        m = (images.min(dim=3).values > threshold).float()
        band = torch.zeros(H, 1)
        band[: int(H * top_band)] = 1
        band[H - int(H * bottom_band):] = 1
        m = (m * band).unsqueeze(1)
        if grow:
            m = F.max_pool2d(m, 2 * grow + 1, stride=1, padding=grow)
        return (m.squeeze(1),)


def _frames(spec):
    on = set()
    for part in spec.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-")
            on.update(range(int(a), int(b) + 1))
        elif part:
            on.add(int(part))
    return on


def _fit(draw, text, width, size):
    while size > 8:
        f = ImageFont.truetype(FONT, size)
        if draw.textlength(text, font=f) <= width:
            return f
        size -= 2
    return ImageFont.truetype(FONT, size)


class BlinkCaption:
    """Top and bottom text on the frames in `frames` ("0-4,10-15"), off on the rest. `width`
    is the fraction of the frame width the text may span, centred -- keep it inside whatever
    crop the emoji takes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "top_text": ("STRING", {"default": "IT'S"}),
            "bottom_text": ("STRING", {"default": "HAPPENING"}),
            "frames": ("STRING", {"default": "0-4,10-15,21-26,32-37,43-49"}),
            "size": ("FLOAT", {"default": 0.17, "min": 0.02, "max": 0.5, "step": 0.01}),
            "width": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.01}),
            "margin": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.3, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "sethmoji"

    def run(self, images, top_text, bottom_text, frames, size, width, margin):
        on = _frames(frames)
        out = []
        for i in range(images.shape[0]):
            arr = (images[i].cpu().numpy() * 255).astype(np.uint8)
            if i not in on:
                out.append(arr)
                continue
            im = Image.fromarray(arr)
            d = ImageDraw.Draw(im)
            W, H = im.size
            px = int(H * size)
            for text, top in ((top_text, True), (bottom_text, False)):
                if not text:
                    continue
                f = _fit(d, text, W * width, px)
                stroke = max(2, f.size // 12)
                l, t, r, b = d.textbbox((0, 0), text, font=f, stroke_width=stroke)
                x = (W - (r - l)) / 2 - l
                y = H * margin - t if top else H * (1 - margin) - b
                d.text((x, y), text, font=f, fill=(255, 255, 255), stroke_width=stroke,
                       stroke_fill=(0, 0, 0))
            out.append(np.asarray(im))
        return (torch.from_numpy(np.stack(out).astype(np.float32) / 255),)


SPARK = np.array([[255, 248, 226], [255, 226, 160], [255, 180, 70], [245, 128, 40],
                  [255, 255, 255]], np.float32) / 255


def _dots(rgb, a, xs, ys, cols, alphas, r):
    """Square dots of half-size r, painted over rgb/a in place."""
    H, W = a.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x = np.round(xs + dx).astype(int)
            y = np.round(ys + dy).astype(int)
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (alphas > 0)
            rgb[y[ok], x[ok]] = cols[ok]
            a[y[ok], x[ok]] = np.maximum(a[y[ok], x[ok]], alphas[ok])


def _ring(rgb, a, cx, cy, rx, ry, thick, alpha, rng, jitter=0.0, n=900):
    th = rng.uniform(0, 2 * np.pi, n)
    rr = 1 + rng.normal(0, thick, n) + rng.normal(0, jitter, n)
    xs, ys = cx + rx * rr * np.cos(th), cy + ry * rr * np.sin(th)
    cols = SPARK[rng.integers(0, 3, n)]
    al = np.full(n, alpha, np.float32) * (rng.random(n) < 0.35 + 0.65 * (1 - jitter * 4))
    _dots(rgb, a, xs, ys, cols, al, 2)


def _head(alpha):
    """Centre and width of the head: the top 30% of the figure's rows, by mask."""
    ys, xs = np.nonzero(alpha > 0.5)
    top, bot = ys.min(), ys.max()
    sel = ys < top + 0.3 * (bot - top)
    return xs[sel].mean(), ys[sel].mean(), xs[sel].max() - xs[sel].min()


class MindBlownFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "alpha": ("MASK",),
            "freeze_frame": ("INT", {"default": 17, "min": 0, "max": 999}),
            "resume_frame": ("INT", {"default": 19, "min": 0, "max": 999}),
            "gap_frames": ("INT", {"default": 40, "min": 8, "max": 400}),
            "burst_x": ("FLOAT", {"default": 0.34, "min": 0, "max": 1, "step": 0.01}),
            "burst_y": ("FLOAT", {"default": 0.64, "min": 0, "max": 1, "step": 0.01}),
            "seed": ("INT", {"default": 7, "min": 0, "max": 2**31}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "run"
    CATEGORY = "sethmoji"

    def run(self, images, alpha, freeze_frame, resume_frame, gap_frames, burst_x, burst_y, seed):
        rng = np.random.default_rng(seed)
        imgs = images.cpu().numpy().astype(np.float32)
        alps = alpha.cpu().numpy().astype(np.float32)
        B, H, W, _ = imgs.shape
        out_rgb, out_a = [], []

        def emit(rgb, a):
            out_rgb.append(rgb * (a[..., None] > 0))
            out_a.append(a)

        for i in range(freeze_frame + 1):
            emit(imgs[i], alps[i])

        base, a0 = imgs[freeze_frame], alps[freeze_frame]
        bx, by = burst_x * W, burst_y * H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dist = np.hypot(xx - bx, yy - by)
        dn = dist / dist[a0 > 0.5].max()
        # blocky noise so the figure crumbles in chunks rather than dissolving smoothly
        noise = np.asarray(Image.fromarray((rng.random((H // 12 + 1, W // 12 + 1)) * 255)
                                           .astype(np.uint8)).resize((W, H), Image.NEAREST),
                           np.float32)[:H, :W] / 255
        # when each pixel lets go, as a fraction of the gap: mouth first, edges last
        t_go = 0.2 + 0.4 * dn + 0.12 * noise

        # the pieces: a grid sample of the figure, flying out from the mouth
        gy, gx = np.nonzero(a0[::3, ::3] > 0.5)
        gy, gx = gy * 3, gx * 3
        pc = base[gy, gx]
        pt = t_go[gy, gx]
        dirx, diry = gx - bx, gy - by
        norm = np.hypot(dirx, diry) + 1e-3
        spd = rng.uniform(0.15, 0.55, gx.size) * W
        pvx = dirx / norm * spd + rng.normal(0, 0.08 * W, gx.size)
        pvy = diry / norm * spd + rng.normal(0, 0.08 * W, gx.size) - 0.1 * W

        # the burst: sparks from the mouth, early in the gap
        ns = 1400
        st = rng.uniform(0.0, 0.3, ns)
        sa = rng.uniform(0, 2 * np.pi, ns)
        ss = rng.uniform(0.2, 1.1, ns) * W
        sc = SPARK[rng.integers(0, len(SPARK), ns)]

        ang = np.arctan2(yy - by, xx - bx)
        hx, hy, hw = _head(alps[resume_frame])
        hd = np.hypot(xx - hx, yy - hy)
        hd = hd / hd[alps[resume_frame] > 0.5].max()
        fine = rng.random((H, W)).astype(np.float32)
        stars_x, stars_y = rng.uniform(0, W, 60), rng.uniform(0, H, 60)

        for k in range(gap_frames):
            t = k / gap_frames
            rgb = np.zeros((H, W, 3), np.float32)
            a = np.zeros((H, W), np.float32)
            keep = (a0 > 0.5) & (t_go > t)
            rgb[keep], a[keep] = base[keep], 1.0
            # flying pieces
            age = t - pt
            live = (age > 0) & (age < 0.35)
            px = gx + pvx * np.maximum(age, 0)
            py = gy + pvy * np.maximum(age, 0) + 0.6 * H * np.maximum(age, 0) ** 2
            _dots(rgb, a, px, py, pc, (live & (rng.random(gx.size) > age / 0.35)).astype(np.float32), 2)
            # sparks, each a short streak back along its path
            sage = t - st
            slive = (sage > 0) & (sage < 0.4) & (rng.random(ns) > sage / 0.4)
            for lag in (0.0, 0.012, 0.024, 0.036):
                ag = np.maximum(sage - lag, 0)
                _dots(rgb, a, bx + np.cos(sa) * ss * ag, by + np.sin(sa) * ss * ag, sc,
                      slive.astype(np.float32), 3 if lag == 0 and t < 0.2 else 2)
            # the flash at the mouth: a spiky star, white inside, amber at the tips
            if t < 0.32:
                r = W * 0.2 * np.sin(np.pi * t / 0.32)
                spike = 0.45 + 0.55 * np.abs(np.sin(5.5 * ang + 1.3)) ** 4 + 0.25 * noise
                core = dist < r * spike
                rgb[core], a[core] = SPARK[2], 1.0
                hot = dist < r * spike * 0.6
                rgb[hot] = SPARK[0]
            # a few twinkles in the dark
            if 0.55 < t < 0.85:
                tw = rng.random(60) < 0.3
                _dots(rgb, a, stars_x, stars_y, SPARK[np.zeros(60, int)], tw.astype(np.float32), 1)
            # the ring flares round his head and he comes back inside it
            if t >= 0.76:
                p = (t - 0.76) / 0.24
                back = alps[resume_frame] > 0.5
                show = back & (hd < p * 1.3 - 0.15 * fine)
                rgb[show], a[show] = imgs[resume_frame][show], 1.0
                _ring(rgb, a, hx, hy, hw * (0.3 + 0.55 * p), hw * (0.3 + 0.55 * p) * 0.45,
                      0.04, 1.0, rng)
                if p < 0.4:
                    disc = np.hypot((xx - hx) / 1.0, (yy - hy) / 0.45) < hw * 0.5 * (1 - p / 0.4)
                    rgb[disc], a[disc] = SPARK[0], 1.0
            emit(rgb, a)

        # the ring drops to his shoulders and scatters
        drop = 16
        for j, i in enumerate(range(resume_frame, B)):
            rgb, a = imgs[i].copy(), alps[i].copy()
            if j < drop:
                p = j / drop
                cx, cy, cw = _head(alps[i])
                _ring(rgb, a, cx, cy + p * 0.35 * H, cw * (0.85 + 0.4 * p),
                      cw * (0.85 + 0.4 * p) * 0.3, 0.04, 1.0, rng, jitter=0.2 * p)
            emit(rgb, a)

        return (torch.from_numpy(np.stack(out_rgb)), torch.from_numpy(np.stack(out_a)))


class SlidingDoorFX:
    """A van's sliding door, drawn in ink in the colour of the door already in the frame, pulled
    shut from the right between start_frame and end_frame (eased). Wan Animate moves people,
    not props, so a door that closes in the source has to be put back afterwards."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "start_frame": ("INT", {"default": 2, "min": 0, "max": 999}),
            "end_frame": ("INT", {"default": 20, "min": 1, "max": 999}),
            "end_x": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 1.0, "step": 0.01}),
            "seed": ("INT", {"default": 7, "min": 0, "max": 2**31}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "sethmoji"

    def run(self, images, start_frame, end_frame, end_x, seed):
        rng = np.random.default_rng(seed)
        B, H, W, _ = images.shape
        f0 = (images[0].cpu().numpy() * 255).astype(np.uint8)
        body = np.median(f0[int(H * .72):int(H * .95), int(W * .55):int(W * .9)].reshape(-1, 3), 0)
        ink = (52, 36, 22)
        # the door, drawn once at full frame size plus a margin, then slid
        DW = int(W * 1.15)
        door = Image.new("RGB", (DW, H), tuple(int(c) for c in body))
        d = ImageDraw.Draw(door)
        lw = max(4, W // 90)
        glass = tuple(int(c * 0.45) for c in body)
        wx0, wx1, wy0, wy1 = int(W * .1), int(W * .92), int(H * .07), int(H * .5)
        d.rounded_rectangle((wx0, wy0, wx1, wy1), radius=W // 25, fill=glass, outline=ink, width=lw)
        hi = tuple(int(min(255, c * 0.8 + 50)) for c in body)
        for off in (0.25, 0.38):
            x = wx0 + int((wx1 - wx0) * off)
            d.line((x, wy1 - 8, x + int(W * .22), wy0 + 8), fill=hi, width=lw * 2)
        d.line((0, int(H * .56), DW, int(H * .56)), fill=ink, width=lw)
        for y in (.72, .78, .84):
            d.rounded_rectangle((int(W * .25), int(H * y), int(W * .65), int(H * y) + H // 70),
                                radius=H // 140, fill=ink)
        d.rounded_rectangle((int(W * .05), int(H * .6), int(W * .16), int(H * .625)),
                            radius=H // 100, fill=tuple(int(c * .7) for c in body), outline=ink, width=lw)
        d.line((lw, 0, lw, H), fill=ink, width=lw * 2)
        d.line((lw * 5, 0, lw * 5, H), fill=ink, width=lw)
        grain = np.asarray(Image.fromarray((rng.random((H // 3 + 1, DW // 3 + 1)) * 255)
                                           .astype(np.uint8)).resize((DW, H), Image.BILINEAR),
                           np.float32)[:H, :DW] / 255
        # shade like the ink drawing: darker toward the bottom and toward the trailing side,
        # with a lit band down the leading edge
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, DW, dtype=np.float32)[None, :]
        shade = 1.08 - 0.22 * yy - 0.12 * xx + 0.1 * np.exp(-((xx * DW - lw * 12) / (W * .03)) ** 2)
        door = np.asarray(door, np.float32) * (shade * (0.88 + 0.12 * grain))[..., None]
        door = np.clip(door, 0, 255)
        out = []
        for i in range(B):
            fr = images[i].cpu().numpy() * 255
            t = np.clip((i - start_frame) / max(1, end_frame - start_frame), 0, 1)
            t = t * t * (3 - 2 * t)
            edge = int(round(W - (W - end_x * W) * t))
            if edge < W:
                fr = fr.copy()
                fr[:, edge:] = door[:, :W - edge]
            out.append(fr)
        return (torch.from_numpy(np.stack(out).astype(np.float32) / 255),)


NODE_CLASS_MAPPINGS = {
    "FillMaskFromSurroundings": FillMaskFromSurroundings,
    "BlinkCaption": BlinkCaption,
    "BrightTextMask": BrightTextMask,
    "MindBlownFX": MindBlownFX,
    "SlidingDoorFX": SlidingDoorFX,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FillMaskFromSurroundings": "Fill Mask From Surroundings",
    "BlinkCaption": "Blink Caption (meme text)",
    "BrightTextMask": "Bright Text Mask",
    "MindBlownFX": "Mind Blown FX",
    "SlidingDoorFX": "Sliding Door FX",
}
