"""What the sprite editor knows about FILES: where the drawings are and what they measure.

Everything in here answers a question the manifest cannot: where the packer actually put a
cell, how tall its silhouette came out, which repaints already exist for a source frame, and
which paths a read-only server is allowed to open at all.

The manifest itself is NOT here. It lives in a Durable Object on the celld `sprites` fleet
(DESIGN-editor.md stage 6), which owns every edit and the single job slot. `sprite_agent.py`
is the only consumer of this module: it long-polls that fleet, runs the packer, and serves
these files to the page. The split exists because numpy, PIL and a GPU will never run in V8.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import repaint_cells as RC  # noqa: E402

# Not a constant, and not a list. Which characters exist is DATA: a character is a manifest in
# `sheets/`, and a new one must be able to exist without editing this file. The Worker answers the
# same question from its own roster (`celld-editor/worker.js`), because the Durable Objects are the
# record; this side answers it from the files, because this side is the one that owns the files.
def characters():
    if not os.path.isdir(RC.MANIFEST_DIR):
        return []
    return sorted(n[:-len(".json")] for n in os.listdir(RC.MANIFEST_DIR) if n.endswith(".json"))


PICK_SKIP = RC.PICK_SKIP         # settle-in frames pick_frames drops; not offerable as a re-pick

# /img serves nothing outside these. The editor only ever needs source frames, repaints, the
# packed output and the plates, and an unrestricted path parameter on a LAN-bound server is a
# file-read hole.
#
# The plates joined the list for the pipeline rail's Plate chip: the plate IS that step's
# artifact, and a step whose artifact cannot be looked at is a chip nobody can check. It is the
# cutout directory alone, not the repo root.
ALLOWED_ROOTS = ("/tmp/sprite-", RC.REPAINT_DIR + "/", os.path.realpath(RC.OUT) + "/",
                 os.path.realpath(os.path.join(RC.HERE, "plates")) + "/")


def sheet_paths(cid):
    return os.path.join(RC.OUT, f"{cid}.png"), os.path.join(RC.OUT, f"{cid}.json")


# One decoded 512px atlas is 10 x 512 wide by six rows: ~63MB of RGBA. Keeping every character's
# would quietly cost more than the packer does, so only the last one asked for is held.
_SHEET = {}


def sheet(cid):
    """The emitted atlas and its frame table, cached against the files' mtimes.

    This is build OUTPUT, not the manifest. The manifest says which drawings a tag is made of; the
    sheet says where the packer actually put them. Both are needed because a cell is no longer a
    file of its own — it is a crop out of the one atlas.
    """
    png, js = sheet_paths(cid)
    try:
        stamp = (os.path.getmtime(png), os.path.getmtime(js))
    except OSError:
        return None
    hit = _SHEET.get(cid)
    if hit and hit[0] == stamp:
        return hit[1]
    from PIL import Image
    meta = json.load(open(js))
    meta["image"] = Image.open(png).convert("RGBA")
    meta["metrics"] = {}
    _SHEET.clear()
    _SHEET[cid] = (stamp, meta)
    return meta


def atlas_base(sh, tag_name, n_frames):
    """Where a manifest tag's cells landed in the atlas, or None if they cannot be located.

    `_prepare_stills` DROPS a frame that keys to nothing, so the atlas can hold fewer cells than
    the manifest tag has frames — and nothing in the emitted schema records WHICH frame went
    missing. When the counts agree the mapping is exact. When they do not, refusing to guess is the
    only correct answer: an off-by-one here shows the wrong drawing beside the wrong source frame,
    which is precisely the judgement this editor exists to make.
    """
    for t in sh["tags"]:
        if t["name"] == tag_name:
            return t["from"] if t["to"] - t["from"] + 1 == n_frames else None
    return None


def cell_image(cid, index):
    """The packed 512px cell, cropped out of the character atlas.

    Shown instead of the raw repaint because it is what actually ships, and because the metrics
    below are only meaningful in packed coordinates.
    """
    sh = sheet(cid)
    if not sh or not 0 <= index < len(sh["frames"]):
        return None
    f, c = sh["frames"][index], sh["cell"]
    return sh["image"].crop((f["x"], f["y"], f["x"] + c, f["y"] + c))


def cell_metrics(cid, index):
    """Silhouette height, top clearance and feet offset for one packed cell.

    These three are the numbers that have caught every bad cell in this project. A repaint that
    came back larger shows as a small clearance; one that came back smaller shows as a short
    silhouette; a keyed-in shadow or a clipped foot shows as a feet offset away from zero, which is
    the character sliding off the ground line mid-animation.

    Measured against THIS FRAME'S OWN emitted pivot (`sh["frames"][index]["pivot"]`), not the
    manifest-level constant `man["pivot"]`. The two agree today because every frame still carries
    the same pivot, but the demo already honours each frame's own pivot — once a later stage makes
    pivots vary per frame, measuring against the manifest constant would silently score a correctly
    placed cell as off the ground line. The cache hangs off the sheet, so a repack invalidates it
    for free.
    """
    sh = sheet(cid)
    if not sh:
        return None
    hit = sh["metrics"].get(index)
    if hit is not None:
        return hit[0]
    im = cell_image(cid, index)
    if im is None:
        return None
    ground = sh["frames"][index]["pivot"][1]
    import numpy as np
    mask = np.asarray(im)[:, :, 3] > 10
    rows = np.where(mask.any(axis=1))[0]
    m = None if len(rows) == 0 else {
        "height": int(rows[-1] - rows[0] + 1),
        "top": int(rows[0]),
        "feet": int(rows[-1] - ground),
        "ground": int(ground),
    }
    sh["metrics"][index] = (m,)
    return m


def source_frames(cid, tag_name):
    d = f"/tmp/sprite-{cid}-{tag_name}-pick"
    if not os.path.isdir(d):
        return []
    names = sorted(n for n in os.listdir(d) if n.endswith(".png"))
    return [os.path.join(d, n) for n in names[PICK_SKIP:]]


def variants(cid, tag_name, src):
    """Every repaint that already exists on disk for this source frame.

    A re-roll used to feel destructive: the previous drawing vanished from the sheet with no way
    back, and getting it back meant remembering the seed. But nothing is ever deleted -- each
    (source frame, seed) pair has its own file. So list them, and let a click swap between them.
    Switching to one that already exists costs no GPU at all.
    """
    stem = os.path.basename(src).replace(".png", "")
    prefix = f"{cid}-{tag_name}-{stem}"
    out = []
    for n in sorted(os.listdir(RC.REPAINT_DIR)):
        if not n.startswith(prefix + "-s") and n != prefix + ".png":
            continue
        if n.endswith("-padded.png") or not n.endswith(".png"):
            continue
        tail = n[len(prefix):-4]
        full = os.path.join(RC.REPAINT_DIR, n)
        # WHEN it was painted, because a seed shelf is a history and a seed is a random 31-bit
        # number: sorting by name orders the shelf by nothing a person can read. The file's mtime
        # is the only record of the order these were drawn in -- nothing else stores it.
        out.append({"png": full, "seed": int(tail[2:]) if tail.startswith("-s") else RC.SEED,
                    "at": int(os.path.getmtime(full))})
    return out


# --- the one thing this side ACCEPTS ------------------------------------------------------------
# A new subject arrives as an uploaded image (DESIGN-pipeline.md, "a new subject"), and it has to
# land on the machine with the GPU and the disk. Everything below exists so that "the editor can
# accept an image" does not also mean "anything on the LAN can write a file of any size and any
# type anywhere on wilson".

UPLOAD_DIR = "/tmp/sprite-uploads"
# 8 MiB. A phone photograph is 2-5MB and the cut-out throws most of it away, so this is generous
# for the one job it has, and small enough that a bad actor -- or a bug in a loop -- cannot fill
# /tmp before anyone notices.
UPLOAD_MAX = 8 << 20
# Below this an image cannot carry a subject worth cutting out; above it, nothing is gained,
# because the plate is written at 1024 square whatever arrives.
UPLOAD_MIN_PX, UPLOAD_MAX_PX = 64, 8192
# Checked against the first bytes, not against the Content-Type header: a header is what the
# client CLAIMS, and PIL is about to be pointed at whatever really arrived.
MAGIC = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpeg"))


def stage_upload(data):
    """Validate one uploaded image and write it where the plate step can read it.

    Returns (path, width, height). Raises ValueError with a sentence a person can act on.

    It is RE-ENCODED to PNG rather than saved as it arrived, and that is the security half of this
    function: a JPEG's EXIF, colour profiles and any appended payload do not survive a decode and a
    fresh encode, so what reaches the disk is pixels and nothing else.
    """
    if len(data) > UPLOAD_MAX:
        raise ValueError(f"that image is {len(data) >> 20}MB and the limit is {UPLOAD_MAX >> 20}MB")
    if not any(data.startswith(m) for m, _ in MAGIC):
        raise ValueError("that is not a PNG or a JPEG — the first bytes say otherwise")
    from PIL import Image
    import io
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:                    # noqa: BLE001 - any decode failure is one answer
        raise ValueError(f"that image would not decode: {type(e).__name__}")
    if not (UPLOAD_MIN_PX <= im.width <= UPLOAD_MAX_PX and
            UPLOAD_MIN_PX <= im.height <= UPLOAD_MAX_PX):
        raise ValueError(f"that image is {im.width}x{im.height}, and the sides have to be between "
                         f"{UPLOAD_MIN_PX} and {UPLOAD_MAX_PX} pixels")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # A name this side chooses, never one the client sends. A client-supplied filename is a path
    # traversal waiting to happen, and the upload is addressed by the path returned from here.
    import secrets
    path = os.path.join(UPLOAD_DIR, f"src-{secrets.token_hex(8)}.png")
    im.convert("RGBA").save(path)
    return path, im.width, im.height
