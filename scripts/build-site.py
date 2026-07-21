#!/usr/bin/env python3
"""Build the public cast.sethgholson.com site into public/.

Everything renders from cast/*/character.yaml — the page cannot drift from the
registry. Validation runs against schema/character.schema.json and FAILS the
build on any error, so a bad character.yaml can't reach production.

    python3 scripts/build-site.py
"""

import html
import json
import shutil
import sys
from pathlib import Path

import yaml
from jsonschema import Draft7Validator
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "public"
BASE_URL = "https://cast.sethgholson.com"

# Founding four first; new cast members append alphabetically after.
KNOWN_ORDER = ["seth", "wilson", "cadbury", "ake"]

SCHEMA = json.loads((ROOT / "schema" / "character.schema.json").read_text())
VALIDATOR = Draft7Validator(SCHEMA)

CSS = """
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; padding:2rem 1.25rem 5rem; background:#14120f; color:#e8e0d0;
         font:16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width:1400px; margin:0 auto; }
  header { text-align:center; margin-bottom:3rem; }
  h1 { font-size:clamp(2rem,6vw,3.2rem); margin:0; letter-spacing:.08em; }
  .sub { color:#b8a882; font-style:italic; margin:.4rem 0 0; }
  h2 { font-size:1.35rem; margin:0; letter-spacing:.04em; text-transform:uppercase; }
  .role { color:#b8a882; font-style:italic; margin:.15rem 0 .8rem; }
  .card { background:#1c1915; border:1px solid #2f2a23; border-radius:10px;
          padding:1.1rem; margin-bottom:2rem; }
  .meta { color:#9c9284; font-size:.875rem; margin:0 0 1rem; }
  code { background:#2a251e; padding:.1rem .35rem; border-radius:4px; font-size:.85em;
         word-break:break-all; }
  img { width:100%; height:auto; display:block; border-radius:6px; background:#ddd; }
  a { color:#d9b56a; }
  a.zoom:focus-visible { outline:2px solid #b8823c; outline-offset:3px; }
  .cols { display:grid; gap:1.5rem; grid-template-columns:1fr; margin-top:1.1rem; }
  @media (min-width:1000px) { .cols { grid-template-columns:1.6fr 1fr; } }
  dl { margin:0; }
  dt { color:#8f8677; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase;
       margin-top:1rem; }
  dt:first-child { margin-top:0; }
  dd { margin:.3rem 0 0; }
  .tokens { background:#221d16; border-radius:6px; padding:.7rem .85rem; font-size:.92rem;
            color:#d9cfbc; }
  ul { margin:.3rem 0 0; padding-left:1.15rem; }
  li { margin:.15rem 0; }
  .note { border-left:3px solid #4a6f7c; padding:.5rem .85rem; margin:.4rem 0 0;
          background:#1a2024; color:#c3d2d8; font-size:.9rem; border-radius:0 6px 6px 0; }
  .accent { border-left:3px solid #c47a2e; background:#221d16; color:#d6c9b0; }
  footer { color:#6f675c; font-size:.85rem; margin-top:3rem; border-top:1px solid #2a251e;
           padding-top:1rem; text-align:center; }
"""


def esc(s):
    return html.escape(str(s).strip())


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
        cast.append(c)
    return cast, errors


def card(c, dims):
    cid = c["id"]
    dl = [f'<dt>Prompt tokens</dt><dd><div class="tokens">{esc(c["tokens"])}</div></dd>']
    if c.get("props"):
        items = "".join(f"<li>{esc(p)}</li>" for p in c["props"])
        dl.append(f"<dt>Props</dt><dd><ul>{items}</ul></dd>")
    if c.get("accent"):
        dl.append(f'<dt>Palette accent</dt><dd><div class="note accent">{esc(c["accent"])}</div></dd>')
    if c.get("notes"):
        dl.append(f'<dt>Production notes</dt><dd><div class="note">{esc(c["notes"])}</div></dd>')
    dl.append(f'<dt>Canonical sheet</dt><dd><code>{BASE_URL}/{cid}/sheet.png</code></dd>')
    return f"""<div class="card" id="{cid}">
<h2>{esc(c["name"])}</h2>
<p class="role">{esc(c.get("role", ""))}</p>
<p class="meta">{dims}</p>
<div class="cols">
<a class="zoom" href="{cid}/sheet.png"><img src="{cid}/sheet.png" alt="{esc(c["name"])} character sheet" loading="lazy"></a>
<dl>{''.join(dl)}</dl>
</div>
</div>"""


def main() -> int:
    cast, errors = load_cast()
    if errors:
        for e in errors:
            print(f"FAIL {e}", file=sys.stderr)
        return 1

    if PUB.exists():
        shutil.rmtree(PUB)
    PUB.mkdir()

    cards, registry = [], []
    for c in cast:
        cid = c["id"]
        (PUB / cid).mkdir()
        src = ROOT / "cast" / cid / c["sheet"]
        shutil.copy2(src, PUB / cid / "sheet.png")
        w, h = Image.open(src).size
        cards.append(card(c, f"{w}&times;{h}"))
        registry.append({
            "id": cid,
            "name": c["name"],
            "role": c.get("role"),
            "tokens": c["tokens"].strip(),
            "props": c.get("props", []),
            "accent": (c.get("accent") or "").strip() or None,
            "notes": (c.get("notes") or "").strip() or None,
            "sheet_url": f"{BASE_URL}/{cid}/sheet.png",
            "sheet_size": [w, h],
        })

    showcase = ""
    strip = ROOT / "showcase" / "comfy-setup-strip.png"
    if strip.exists():
        shutil.copy2(strip, PUB / "showcase-latest.png")
        showcase = """<h1 style="font-size:1.6rem;margin-top:3.5rem">From the funnies</h1>
<div class="card"><a class="zoom" href="showcase-latest.png">
<img src="showcase-latest.png" alt="Latest comic strip" loading="lazy"></a></div>"""

    (PUB / "cast.json").write_text(json.dumps({
        "source": "https://github.com/sethgho/the-cast",
        "style": "vaudeville-1933",
        "base_url": BASE_URL,
        "cast": registry,
    }, indent=2) + "\n")

    (PUB / "index.html").write_text(f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Cast — A Vaudeville Production</title>
<meta name="description" content="The canonical cast of sethgholson.com — character sheets, prompt tokens, and props.">
<style>{CSS}</style>
<main>
  <header>
    <h1>THE CAST</h1>
    <p class="sub">A Vaudeville Production &middot; Est. 1933</p>
  </header>
  {''.join(cards)}
  {showcase}
  <footer>
    machine-readable registry: <a href="cast.json"><code>cast.json</code></a> &middot;
    source: <a href="https://github.com/sethgho/the-cast">sethgho/the-cast</a>
  </footer>
</main>
</html>
""")
    print(f"built public/ — {len(registry)} cast members, cast.json, index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
