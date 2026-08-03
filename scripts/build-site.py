#!/usr/bin/env python3
"""Build the public cast.sethgholson.com site into public/.

The registry half renders from cast/*/character.yaml — cast.json and the sheet
URLs cannot drift from the registry. Validation runs against
schema/character.schema.json and FAILS the build on any error, so a bad
character.yaml can't reach production.

The presentation half — the Gholson's Follies pages — is handcrafted in site/
(index.html + one bill page per cast member, art under site/art/) and copied
into public/ verbatim.

    python3 scripts/build-site.py
"""

import json
import shutil
import sys
from pathlib import Path

import yaml
from jsonschema import Draft7Validator
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "public"
SITE = ROOT / "site"
BASE_URL = "https://cast.sethgholson.com"

# Founding four first; new cast members append alphabetically after.
KNOWN_ORDER = ["seth", "wilson", "cadbury", "ake"]

SCHEMA = json.loads((ROOT / "schema" / "character.schema.json").read_text())
VALIDATOR = Draft7Validator(SCHEMA)


def load_cast():
    dirs = sorted(p.name for p in (ROOT / "cast").iterdir() if p.is_dir())
    ordered = [c for c in KNOWN_ORDER if c in dirs] + [c for c in dirs if c not in KNOWN_ORDER]
    cast, errors = [], []
    for cid in ordered:
        path = ROOT / "cast" / cid / "character.yaml"
        if not path.exists():
            errors.append(f"cast/{cid}/ has no character.yaml")
            continue
        c = yaml.safe_load(path.read_text())
        for e in VALIDATOR.iter_errors(c):
            errors.append(f"cast/{cid}: {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}")
        if c.get("id") != cid:
            errors.append(f"cast/{cid}: id '{c.get('id')}' does not match directory")
        if not (ROOT / "cast" / cid / c.get("sheet", "sheet.png")).exists():
            errors.append(f"cast/{cid}: sheet '{c.get('sheet')}' missing")
        head = ROOT / "cast" / cid / c.get("headshot", "headshot.png")
        if not head.exists():
            errors.append(f"cast/{cid}: headshot '{c.get('headshot')}' missing")
        else:
            hw, hh = Image.open(head).size
            if hw != hh:
                errors.append(f"cast/{cid}: headshot must be square, got {hw}x{hh}")
        cast.append(c)
    return cast, errors


def main() -> int:
    cast, errors = load_cast()
    for cid in (c["id"] for c in cast):
        if not (SITE / f"{cid}.html").exists():
            errors.append(f"site/{cid}.html missing — every cast member needs a bill page")
    if errors:
        for e in errors:
            print(f"FAIL {e}", file=sys.stderr)
        return 1

    if PUB.exists():
        shutil.rmtree(PUB)
    PUB.mkdir()

    registry = []
    for c in cast:
        cid = c["id"]
        (PUB / cid).mkdir()
        src = ROOT / "cast" / cid / c["sheet"]
        shutil.copy2(src, PUB / cid / "sheet.png")
        w, h = Image.open(src).size
        head_src = ROOT / "cast" / cid / c["headshot"]
        shutil.copy2(head_src, PUB / cid / "headshot.png")
        hw, hh = Image.open(head_src).size
        entry = {
            "id": cid,
            "name": c["name"],
            "role": c.get("role"),
            "tokens": c["tokens"].strip(),
            "props": c.get("props", []),
            "accent": (c.get("accent") or "").strip() or None,
            "notes": (c.get("notes") or "").strip() or None,
            "sheet_url": f"{BASE_URL}/{cid}/sheet.png",
            "sheet_size": [w, h],
            "headshot_url": f"{BASE_URL}/{cid}/headshot.png",
            "headshot_size": [hw, hh],
        }
        art = SITE / "art" / cid
        if art.is_dir():
            for f in art.iterdir():
                shutil.copy2(f, PUB / cid / f.name)
            if (art / "portrait.png").exists():
                entry["portrait_url"] = f"{BASE_URL}/{cid}/portrait.png"
        registry.append(entry)

    strip = ROOT / "showcase" / "comfy-setup-strip.png"
    if strip.exists():
        shutil.copy2(strip, PUB / "showcase-latest.png")

    hero = SITE / "art" / "hero.png"
    if hero.exists():
        shutil.copy2(hero, PUB / "hero.png")

    pages = 0
    for page in list(SITE.glob("*.html")) + list(SITE.glob("*.css")):
        shutil.copy2(page, PUB / page.name)
        pages += 1

    (PUB / "cast.json").write_text(json.dumps({
        "source": "https://github.com/sethgho/the-cast",
        "style": "vaudeville-1933",
        "base_url": BASE_URL,
        "cast": registry,
    }, indent=2) + "\n")

    print(f"built public/ — {len(registry)} cast members, cast.json, {pages} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
