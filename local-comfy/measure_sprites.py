#!/usr/bin/env python3
"""Measure sprite atlas cells: clearance, feet spread, head-width spread, duplicate cells.

Reads the *.png sprite sheets (single-row, 512px cells) in
/home/wilson/artifacts/cast-fighter/sprites/ and prints one table.

Definitions:
- cell count: sheet width / 512
- min top clearance: min over cells of (first non-transparent row from top), i.e.
  smallest gap between the top of the 512px cell and the top of the silhouette.
  Smaller number = figure gets closer to the top of the cell.
- feet spread: max-min of the topmost row >= y=481 that is still non-transparent
  in a narrow band around center, approximated as: for each cell find the lowest
  non-transparent pixel y (foot contact point) restricted to columns within the
  cell, then report spread (max-min) across cells relative to the y=481 feet line.
- head-width spread: for each cell, take the top 1/5 of the figure's bbox rows and
  compute median row-width (count of non-transparent px in that row). Then compute
  this per-cell "head width" as % of the across-cell median, report the spread
  (max% - min%, i.e. range from 100%).
- duplicate cells: count of consecutive cell pairs whose 48x48 downsampled alpha
  silhouettes are pixel-identical.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

SPRITE_DIR = Path("/home/wilson/artifacts/cast-fighter/sprites")
CELL = 512
FEET_Y = 481


def load_cells(png_path):
    im = Image.open(png_path).convert("RGBA")
    w, h = im.size
    assert h == CELL, f"{png_path} height {h} != {CELL}"
    n = w // CELL
    arr = np.array(im)
    cells = [arr[:, i * CELL:(i + 1) * CELL, :] for i in range(n)]
    return cells


def alpha_mask(cell):
    return cell[:, :, 3] > 10


def top_clearance(mask):
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return CELL
    return int(rows[0])


def bbox_rows(mask):
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    return int(rows[0]), int(rows[-1])


def foot_contact_y(mask):
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    return int(rows[-1])


def head_width_metric(mask):
    top, bottom = bbox_rows(mask) or (None, None)
    if top is None:
        return None
    height = bottom - top + 1
    fifth = max(1, height // 5)
    band = mask[top: top + fifth, :]
    widths = band.sum(axis=1)
    widths = widths[widths > 0]
    if len(widths) == 0:
        return None
    return float(np.median(widths))


def silhouette_48(mask):
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.resize((48, 48), Image.NEAREST)
    return np.array(im) > 127


def count_duplicates(masks):
    sils = [silhouette_48(m) for m in masks]
    dup = 0
    for a, b in zip(sils, sils[1:]):
        if np.array_equal(a, b):
            dup += 1
    return dup


def measure_move(png_path):
    cells = load_cells(png_path)
    masks = [alpha_mask(c) for c in cells]

    clearances = [top_clearance(m) for m in masks]
    min_clearance = min(clearances)

    feet_ys = [foot_contact_y(m) for m in masks if foot_contact_y(m) is not None]
    feet_spread = max(feet_ys) - min(feet_ys) if feet_ys else None

    head_widths = [hw for hw in (head_width_metric(m) for m in masks) if hw is not None]
    if head_widths:
        median_hw = float(np.median(head_widths))
        pct = [100.0 * hw / median_hw for hw in head_widths]
        head_spread_pct = max(pct) - min(pct)
    else:
        head_spread_pct = None

    dup_count = count_duplicates(masks)

    return {
        "cells": len(cells),
        "min_top_clearance": min_clearance,
        "feet_spread_px": feet_spread,
        "feet_line_y": FEET_Y,
        "head_width_spread_pct": head_spread_pct,
        "dup_cells": dup_count,
    }


def main():
    chars_moves = {
        "cadbury": ["walk", "idle", "punch", "kick", "jump", "block", "crouch"],
        "seth": ["walk", "idle", "punch", "kick", "jump", "block", "crouch"],
    }

    rows = []
    for char, moves in chars_moves.items():
        for move in moves:
            png = SPRITE_DIR / f"{char}-{move}.png"
            if not png.exists():
                continue
            m = measure_move(png)
            rows.append((f"{char}-{move}", m))

    header = f"{'move':<20} {'cells':>5} {'min_top_clr':>12} {'feet_spread':>12} {'head_w_spread%':>15} {'dup_cells':>10}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        fs = m["feet_spread_px"] if m["feet_spread_px"] is not None else "n/a"
        hw = f"{m['head_width_spread_pct']:.1f}" if m["head_width_spread_pct"] is not None else "n/a"
        print(f"{name:<20} {m['cells']:>5} {m['min_top_clearance']:>12} {fs:>12} {hw:>15} {m['dup_cells']:>10}")


if __name__ == "__main__":
    main()
