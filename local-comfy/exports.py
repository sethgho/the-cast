"""Turn one emitted atlas table into the formats game engines actually load.

The atlas table is `<OUT>/<cid>.json` -- what `sprite_sheet.py`'s `_emit_sheet` writes alongside
`<cid>.png`. Every exporter below is a pure function of that dict: no image bytes, no recomputed
geometry. If a cell gets fixed by a repaint+repack, re-calling an exporter picks up the new
numbers automatically; if an exporter started re-deriving anchors or trims itself, a stale copy of
that logic could quietly disagree with the packer's about the same cell.

This is NOT the character manifest (`sheets/<cid>.json`, read only through
`repaint_cells.load_character_manifest` / `save_character_manifest`). The manifest says which
drawings a tag is made of and is mutated by the editor; the atlas table is read-only build output
-- `sprite_editor.py`'s own `sheet()` already `json.load`s it directly, and this module does the
same.

Frame naming is stable and derivable everywhere: `<cid>-<tag>-<n>`, `n` starting at 1 within its
tag, in atlas order (`_frame_entries`). Every exporter with an animation concept orders and
repeats frames the same way `sprite_editor.py`'s embedded `steps()` plays them back:
`forward` = as recorded, `reverse` = recorded order reversed, `pingpong` = forward then back down
to (not including) the first and last cell. That shared logic lives in `_direction_order` /
`_play_sequence` so every format agrees with the demo page and with each other.
"""
import json
import os

import repaint_cells as RC

# --- atlas table -----------------------------------------------------------------------------


def load_atlas(cid, out_dir=None):
    """The emitted atlas table sprite_sheet.py wrote: `<OUT>/<cid>.json`.

    Raises FileNotFoundError if the character has never been packed -- callers decide how to
    report that (the HTTP endpoint turns it into a 404), so this does not swallow it.
    """
    path = os.path.join(out_dir or RC.OUT, f"{cid}.json")
    return json.load(open(path))


def _atlas_size(atlas):
    """Pixel size of `<cid>.png`, matching `_emit_sheet`'s `Image.new` call exactly.

    The atlas is NOT always `columns * cell` wide -- a character with fewer cells than one row
    (none today, but nothing forbids it) gets a narrower image, and getting `meta.size` wrong in
    an exported atlas is exactly the kind of error a consumer only discovers as UV garbage.
    """
    cell, cols, n = atlas["cell"], atlas["columns"], len(atlas["frames"])
    if n == 0:
        return 0, 0
    rows = -(-n // cols)  # ceil division
    return cell * min(cols, n), cell * rows


def _frame_entries(atlas, cid):
    """One descriptor per named (tagged) frame, in atlas order.

    Every exporter is built off this list instead of touching `atlas["frames"]` directly, so the
    naming rule and the frame/tag index arithmetic exist in exactly one place. Also validates the
    one invariant every exporter below depends on: tags partition the frame list with no gaps and
    no overlaps (true by construction in `_emit_sheet`, but this module reads the file fresh every
    call and a hand-edited manifest could break it -- better a clear error here than a wrong or
    silently-missing frame in an exported atlas).
    """
    cell = atlas["cell"]
    frames = atlas["frames"]
    covered = [False] * len(frames)
    entries = []
    for tag in atlas["tags"]:
        for n, idx in enumerate(range(tag["from"], tag["to"] + 1), start=1):
            if not (0 <= idx < len(frames)):
                raise ValueError(f"{cid}: tag {tag['name']!r} references frame {idx}, but the "
                                  f"atlas only has {len(frames)} frames")
            if covered[idx]:
                raise ValueError(f"{cid}: frame {idx} is claimed by more than one tag")
            covered[idx] = True
            f = frames[idx]
            entries.append({
                "name": f"{cid}-{tag['name']}-{n}",
                "index": idx, "tag": tag["name"], "n": n,
                "x": f["x"], "y": f["y"], "cell": cell,
                "hold": max(1, int(f.get("hold", 1))),
                "pivot": f["pivot"], "trim": f["trim"],
            })
    if not all(covered):
        missing = [i for i, c in enumerate(covered) if not c]
        raise ValueError(f"{cid}: {len(missing)} frame(s) not covered by any tag: {missing}")
    return entries


def _by_tag(entries):
    out = {}
    for e in entries:
        out.setdefault(e["tag"], []).append(e)
    return out


def _direction_order(tag_entries, direction):
    """`tag_entries` reordered for playback -- no hold expansion, no duplicate references.

    Mirrors the embedded `steps()` in sprite_editor.py's PAGE: `reverse` walks the recorded list
    back to front; `pingpong` walks forward then back down to (not including) the first and last
    cell, so a loop does not visibly pause on either end.
    """
    n = len(tag_entries)
    if direction == "reverse":
        idx = range(n - 1, -1, -1)
    elif direction == "pingpong":
        idx = list(range(n)) + list(range(n - 2, 0, -1))
    else:
        idx = range(n)
    return [tag_entries[i] for i in idx]


def _play_sequence(tag_entries, direction):
    """`_direction_order`, with each cell repeated `hold` times.

    For formats with no native per-frame duration (CSS, Phaser's plain frame list), this is how
    `hold` survives: as a repeated reference to the same named frame, exactly what the embedded
    `steps()` already does client-side (`for (let k = 0; k < Math.max(1, c.hold); k++) ...`).
    """
    seq = []
    for e in _direction_order(tag_entries, direction):
        seq.extend([e] * max(1, e["hold"]))
    return seq


def _meta(cid, atlas, comment):
    w, h = _atlas_size(atlas)
    return {
        "app": "the-cast/local-comfy exports.py",
        "version": "1.0",
        "image": f"{cid}.png",
        "format": "RGBA8888",
        "size": {"w": w, "h": h},
        "scale": "1",
        "comment": comment,
    }


# The TexturePacker `frame` rect points at the TRIMMED opaque pixels -- `trim` inside the atlas
# table is exactly that box, offset into the cell's position in the atlas, so cropping the atlas
# image at this rect gives the same pixels a real trimmed pack would. `sourceSize` is the
# untrimmed cell (every cell is emitted full-size so the frame stride stays constant; see
# `_emit_sheet`), and `spriteSourceSize` is `trim` itself -- the offset of those trimmed pixels
# within that untrimmed cell.
_NO_TAG_COMMENT = (
    "Tag/animation data (fps, loop, direction, per-frame hold) is not part of the TexturePacker "
    "atlas schema -- use the phaser3, godot or css export for playback timing. 'holdFrames' below "
    "is a non-standard passthrough of the atlas table's per-frame 'hold'; strict TexturePacker "
    "consumers ignore unknown keys."
)


def _tp_frame(e, cell):
    t = e["trim"]
    return {
        "frame": {"x": e["x"] + t["x"], "y": e["y"] + t["y"], "w": t["w"], "h": t["h"]},
        "rotated": False,
        "trimmed": True,
        "spriteSourceSize": {"x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"]},
        "sourceSize": {"w": cell, "h": cell},
        # Normalised 0..1 against sourceSize. The atlas table's pivot is already in that space --
        # it is the constant SHEET origin recorded once per cell, not shifted by trim (see
        # `_emit_sheet`'s docstring) -- so dividing by `cell` is the whole conversion.
        "pivot": {"x": e["pivot"][0] / cell, "y": e["pivot"][1] / cell},
        "holdFrames": e["hold"],
    }


# --- 1. JSON hash ------------------------------------------------------------------------------


def export_json_hash(atlas, cid):
    entries = _frame_entries(atlas, cid)
    cell = atlas["cell"]
    frames = {e["name"]: _tp_frame(e, cell) for e in entries}
    return json.dumps({"frames": frames, "meta": _meta(cid, atlas, _NO_TAG_COMMENT)}, indent=1)


# --- 2. JSON array -------------------------------------------------------------------------------


def _atlas_array_frames(atlas, cid):
    cell = atlas["cell"]
    out = []
    for e in _frame_entries(atlas, cid):
        d = {"filename": e["name"]}
        d.update(_tp_frame(e, cell))
        out.append(d)
    return out


def export_json_array(atlas, cid):
    frames = _atlas_array_frames(atlas, cid)
    return json.dumps({"frames": frames, "meta": _meta(cid, atlas, _NO_TAG_COMMENT)}, indent=1)


# --- 3. Phaser 3 -----------------------------------------------------------------------------


def export_phaser3(atlas, cid):
    """Phaser 3's own texture-atlas JSON (`this.load.atlas(key, png, json)` takes the array form
    directly) plus its Animation JSON, built from the tags.

    Phaser's `AnimationFrameConfig` is a flat ordered list with no direction flag, so `pingpong`
    and `reverse` tags are pre-expanded into `frames` via `_play_sequence` -- the same repeated-
    reference trick the CSS export uses, since Phaser has no native per-frame duration either.
    """
    entries = _frame_entries(atlas, cid)
    by_tag = _by_tag(entries)
    anims = []
    for tag in atlas["tags"]:
        seq = _play_sequence(by_tag[tag["name"]], tag["direction"])
        anims.append({
            "key": tag["name"],
            "type": "frame",
            "frameRate": tag["fps"],
            "repeat": -1 if tag["loop"] else 0,
            "frames": [{"key": cid, "frame": e["name"]} for e in seq],
        })
    comment = (
        f"`atlas` is a standalone TexturePacker array atlas: this.load.atlas('{cid}', "
        f"'{cid}.png', <this file, atlas key>). `anims` is Phaser's Animation JSON -- adapt to "
        "this.anims.fromJSON(...) or this.load.animation(...). Per-frame 'hold' and pingpong/"
        "reverse direction are pre-expanded into `anims[].frames` as repeated frame references, "
        "since AnimationFrameConfig has no duration-multiplier or direction field of its own."
    )
    return json.dumps({
        "atlas": {"frames": _atlas_array_frames(atlas, cid), "meta": _meta(cid, atlas, comment)},
        "anims": anims,
    }, indent=1)


# --- 4. Godot SpriteFrames ---------------------------------------------------------------------


def export_godot(atlas, cid):
    """Godot 4 `SpriteFrames` resource text (`.tres`): one `AtlasTexture` region per atlas cell,
    one animation per tag.

    Godot 4's SpriteFrames stores a real per-frame `duration` (a float multiplier on the
    animation's `speed`) -- exactly what `hold` already means -- so, unlike the CSS/Phaser
    exports, cells are referenced ONCE each; only `_direction_order` (for reverse/pingpong) is
    used, never `_play_sequence`'s hold-expansion.

    What it cannot carry: a per-frame pivot. `AnimatedSprite2D` applies one node-level `offset`,
    not one per animation frame. Every cell in this atlas shares the same constant sheet pivot
    (`_emit_sheet` records the sheet origin, not a per-frame one), so that value is surfaced as a
    leading comment instead of silently dropped.
    """
    entries = _frame_entries(atlas, cid)
    by_tag = _by_tag(entries)

    # One AtlasTexture per underlying cell, keyed by flat atlas index rather than by (tag, n) --
    # a cell reused by two tags (the schema does not forbid it, even though no character does it
    # today) must not get baked into two textures that could later disagree about the same pixels.
    by_index = {}
    for e in entries:
        by_index.setdefault(e["index"], e)
    ordered_indices = sorted(by_index)
    sub_id = {idx: f"AtlasTexture_{i + 1}" for i, idx in enumerate(ordered_indices)}

    pivots = {tuple(e["pivot"]) for e in entries}
    if len(pivots) == 1:
        px, py = next(iter(pivots))
        pivot_note = f"({px}, {py}) px"
    else:
        # Should never happen -- `_emit_sheet` writes the same constant sheet pivot into every
        # frame -- but this module reads the file fresh, so a hand-edited atlas gets a loud note
        # instead of a silently wrong offset.
        pivot_note = f"VARIES across {len(pivots)} distinct values -- atlas contract broken"

    out = [
        f"; Generated by exports.py from {cid}.json -- do not hand-edit, regenerate instead.",
        f"; Sheet pivot (constant across every cell, cell {atlas['cell']}px): {pivot_note}.",
        "; Godot has no per-frame pivot; apply this as AnimatedSprite2D.offset if pivot-anchored",
        "; placement needs to match the other exporters.",
        f'[gd_resource type="SpriteFrames" load_steps={len(ordered_indices) + 2} format=3]',
        "",
        f'[ext_resource type="Texture2D" path="res://{cid}.png" id="1"]',
        "",
    ]
    for idx in ordered_indices:
        e = by_index[idx]
        t = e["trim"]
        rx, ry = e["x"] + t["x"], e["y"] + t["y"]
        out.append(f'[sub_resource type="AtlasTexture" id="{sub_id[idx]}"]')
        out.append('atlas = ExtResource("1")')
        out.append(f'region = Rect2({rx}, {ry}, {t["w"]}, {t["h"]})')
        out.append("")

    anim_blocks = []
    for tag in atlas["tags"]:
        seq = _direction_order(by_tag[tag["name"]], tag["direction"])
        frame_entries = ", ".join(
            '{\n"duration": %s,\n"texture": SubResource("%s")\n}' % (float(e["hold"]), sub_id[e["index"]])
            for e in seq
        )
        anim_blocks.append(
            '{\n"frames": [%s],\n"loop": %s,\n"name": &"%s",\n"speed": %s\n}' % (
                frame_entries, "true" if tag["loop"] else "false", tag["name"], float(tag["fps"]))
        )
    out.append("[resource]")
    out.append("animations = [%s]" % ", ".join(anim_blocks))
    return "\n".join(out) + "\n"


# --- 5. CSS steps() ----------------------------------------------------------------------------


def _sweep_eligible(ordered, tag):
    """True when a plain two-keyframe `background-position` sweep reads correctly.

    `steps(n)` divides `from -> to` into n EQUALLY spaced landings. That only matches this tag's
    n cell positions in order when: the tag stays in one atlas row (no second axis to move),
    every cell has a constant stride (rules out `pingpong`, whose stride flips sign partway
    through), and `hold == 1` everywhere (a held cell needs two IDENTICAL consecutive landings,
    which equal spacing cannot produce -- it can only ever land on n DISTINCT positions).
    """
    if tag["direction"] not in ("forward", "reverse") or len(ordered) < 2:
        return False
    if len({e["y"] for e in ordered}) != 1:
        return False
    if any(e["hold"] != 1 for e in ordered):
        return False
    stride = ordered[1]["x"] - ordered[0]["x"]
    if stride == 0:
        return False
    return all(b["x"] - a["x"] == stride for a, b in zip(ordered, ordered[1:]))


def _css_rule_shell(cid, cls, cell, body_decl):
    return (
        f".{cls} {{\n"
        f"  width: {cell}px; height: {cell}px;\n"
        f"  background-image: url('{cid}.png'); background-repeat: no-repeat;\n"
        f"  {body_decl}\n"
        f"}}"
    )


def _css_sweep(cid, cls, tag, ordered, cell):
    n = len(ordered)
    y = ordered[0]["y"]
    stride = ordered[1]["x"] - ordered[0]["x"]
    x0 = ordered[0]["x"]
    x1 = x0 + n * stride
    dur = n / tag["fps"]
    iter_count = "infinite" if tag["loop"] else "1"
    fill = "" if tag["loop"] else " forwards"
    note = f"/* {cls}: single row, hold==1 throughout -- background-position sweep. */"
    keyframes = (
        f"@keyframes {cls} {{\n"
        f"  from {{ background-position: -{x0}px -{y}px; }}\n"
        f"  to {{ background-position: -{x1}px -{y}px; }}\n"
        f"}}"
    )
    rule = _css_rule_shell(cid, cls, cell, f"animation: {cls} {dur:.4f}s steps({n}) {iter_count}{fill};")
    return f"{note}\n{keyframes}\n{rule}"


def _css_explicit(cid, cls, tag, tag_entries, cell):
    """Explicit per-step keyframes -- always correct, used whenever `_sweep_eligible` is false.

    One keyframe per entry in the fully hold-expanded playback sequence, each pinned to a
    `step-end` timing so the browser jumps discretely between cells instead of tweening a
    diagonal slide across two rows (or blending toward a held duplicate). This is the only
    technique that can express a tag straddling an atlas row at all, since `background-position`
    has no way to sweep two axes with one `from -> to` pair.
    """
    ordered = _direction_order(tag_entries, tag["direction"])
    rows = sorted({e["y"] // cell for e in ordered})
    if len(rows) > 1:
        reason = f"straddles atlas rows {', '.join(str(r) for r in rows)}"
    else:
        reason = "has a held frame (hold > 1); a uniform sweep would land on the wrong cell mid-hold"
    seq = _play_sequence(tag_entries, tag["direction"])
    total = len(seq)
    dur = total / tag["fps"]
    iter_count = "infinite" if tag["loop"] else "1"
    fill = "" if tag["loop"] else " forwards"
    note = (f"/* {cls}: {reason} -- cannot be a single background-position sweep; "
            "explicit per-frame keyframes instead. */")
    lines = [note, f"@keyframes {cls} {{"]
    for k, e in enumerate(seq):
        pct = 0.0 if total <= 1 else (k * 100.0) / total
        lines.append(f"  {pct:.4f}% {{ background-position: -{e['x']}px -{e['y']}px; "
                      "animation-timing-function: step-end; }")
    lines.append("}")
    rule = _css_rule_shell(cid, cls, cell, f"animation: {cls} {dur:.4f}s step-end {iter_count}{fill};")
    return "\n".join(lines) + "\n" + rule


def export_css(atlas, cid):
    entries = _frame_entries(atlas, cid)
    by_tag = _by_tag(entries)
    cell = atlas["cell"]
    blocks = [f"/* Generated by exports.py from {cid}.json -- do not hand-edit, regenerate "
              "instead. */"]
    for tag in atlas["tags"]:
        tag_entries = by_tag[tag["name"]]
        cls = f"{cid}-{tag['name']}"
        ordered = _direction_order(tag_entries, tag["direction"])
        if _sweep_eligible(ordered, tag):
            blocks.append(_css_sweep(cid, cls, tag, ordered, cell))
        else:
            blocks.append(_css_explicit(cid, cls, tag, tag_entries, cell))
    return "\n\n".join(blocks) + "\n"


# --- dispatch ----------------------------------------------------------------------------------

FORMATS = {
    "json-hash": export_json_hash,
    "json-array": export_json_array,
    "phaser3": export_phaser3,
    "godot": export_godot,
    "css": export_css,
}
