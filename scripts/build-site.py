#!/usr/bin/env python3
"""Build the public cast.sethgholson.com site into public/.

cast.sethgholson.com is the data hub for the cast. It serves identity (who a
character is), assets (what they look like, in every form a renderer needs) and
voice (how they talk, in prose and in audio) — as static JSON with stable URLs.
There is no server and no database. "The API" is generated files.

Three halves:

  1. The REGISTRY renders from cast/*/character.yaml, cast/*/assets.yaml,
     cast/*/voice.yaml and cast/*/lora.yaml. Every manifest is validated against
     its schema and a bad one FAILS THE BUILD. Stale canon is the failure mode
     that has actually hurt here; a red build has not.

  2. The ASSET CONTRACT gives every asset one shape (schema/asset.schema.json)
     whatever its kind. Sizes, dimensions, durations and hashes are computed
     here from the bytes on disk — there is deliberately no way to hand-write
     them, because hand-written metadata is how a registry starts lying.

  3. The PRESENTATION half — the Gholson's Follies pages — is handcrafted in
     site/ (index.html + one bill page per cast member, art under site/art/) and
     copied into public/ verbatim.

    python3 scripts/build-site.py
"""

import hashlib
import json
import shutil
import sys
import wave
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

# Top-level paths the build owns. A cast id colliding with one of these would
# have its assets silently overwritten by the site, so it is a build failure.
RESERVED_IDS = {"assets", "schema", "style", "showcase", "api", "hero"}

# Explicit, not guessed. An unknown extension fails the build: mimetypes' answer
# for an unrecognised suffix is None, and a null mime in a published record is
# worse than a red build. Audio is WAV-only on purpose — duration comes from the
# stdlib `wave` module, so the Docker build needs no ffmpeg.
MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".wav": "audio/wav",
}
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_SUFFIXES = {".wav"}

# The canonical sheet/headshot/portrait predate the asset manifest, so there is no
# authored date for them. This is the day the asset contract landed and they first
# got records — not a claim about when the art was drawn.
CANON_ASSET_DATE = "2026-08-06"

SCHEMA_DIR = ROOT / "schema"
SCHEMA_NAMES = ["character", "asset", "voice", "lora"]
SCHEMAS = {n: json.loads((SCHEMA_DIR / f"{n}.schema.json").read_text()) for n in SCHEMA_NAMES}


def _inline_refs(node, store):
    """Replace {"$ref": "<known $id>"} with the referenced schema, in place.

    voice.schema.json $refs asset.schema.json by absolute URL so that published
    schemas compose for consumers. Resolving that at validation time would mean
    jsonschema's RefResolver, which is deprecated as of 4.18 and would either
    warn or break under the unpinned pip install in the Dockerfile. Inlining is
    two lines and has no version opinion.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref in store:
            return dict(store[ref])
        return {k: _inline_refs(v, store) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, store) for v in node]
    return node


_STORE = {s["$id"]: s for s in SCHEMAS.values() if "$id" in s}
VALIDATORS = {n: Draft7Validator(_inline_refs(s, _STORE)) for n, s in SCHEMAS.items()}


def validate(name, doc, label, errors):
    """Collect every schema error with a locating label. Returns True if clean."""
    before = len(errors)
    for e in sorted(VALIDATORS[name].iter_errors(doc), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in e.path) or "<root>"
        errors.append(f"{label}: {where}: {e.message}")
    return len(errors) == before


def read_yaml(path, label, errors):
    """Parse a manifest, turning a syntax error into a FAIL line not a traceback.

    A stack trace is a build failure that looks like a build BUG. Every other
    problem in this file reports as 'FAIL <file>: <what>'; unparseable YAML is
    the most likely problem of all and had no business being the exception.
    """
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        where = ""
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            where = f" at line {mark.line + 1}, column {mark.column + 1}"
        errors.append(f"{label}: not valid YAML{where} — {getattr(exc, 'problem', exc)}")
        return None
    return doc


def sha256_of(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measure(path, errors, label):
    """(width, height, duration_s) for a file, or (None, None, None) on error."""
    suffix = path.suffix.lower()
    if suffix in RASTER_SUFFIXES:
        try:
            w, h = Image.open(path).size
        except Exception as exc:  # noqa: BLE001 — any decode failure is a build failure
            errors.append(f"{label}: cannot read image ({exc})")
            return None, None, None
        return w, h, None
    if suffix in AUDIO_SUFFIXES:
        try:
            with wave.open(str(path)) as wav:
                duration = round(wav.getnframes() / float(wav.getframerate()), 3)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: cannot read WAV ({exc})")
            return None, None, None
        return None, None, duration
    return None, None, None


def make_asset(cid, kind, src, rel_url_path, errors, *, label, tags=None, caption=None,
               created=None, source=None, notes=None, slug=None):
    """Compose one uniform asset record from a file on disk.

    rel_url_path is the path under /<id>/ where the file gets published; it is
    the repo path with the assets/ prefix stripped, so cast/wilson/assets/
    poses/arms-crossed.png publishes to /wilson/poses/arms-crossed.png. Stable
    filenames in, stable URLs out — consumers never chase a rename.
    """
    suffix = src.suffix.lower()
    mime = MIME.get(suffix)
    if mime is None:
        errors.append(f"{label}: unsupported extension '{suffix}' — add it to MIME in build-site.py")
        return None

    slug = slug or src.stem
    width, height, duration = measure(src, errors, label)
    record = {
        "id": f"{cid}/{kind}/{slug}",
        "character": cid,
        "kind": kind,
        "url": f"{BASE_URL}/{cid}/{rel_url_path}",
        "mime": mime,
        "bytes": src.stat().st_size,
        "sha256": sha256_of(src),
        "width": width,
        "height": height,
        "duration_s": duration,
        "tags": sorted(tags or []),
        "caption": caption or "",
        "created": created or "",
    }
    if source:
        record["source"] = source
    if notes:
        record["notes"] = notes

    # created is required by the schema; say so in the manifest's language rather
    # than making the author reverse-engineer a jsonschema message.
    if not created:
        errors.append(f"{label}: 'created' is required (YYYY-MM-DD)")
    validate("asset", record, label, errors)
    return record


def load_cast(errors):
    """Read and validate every character.yaml. Order is canon: founding four first."""
    dirs = sorted(p.name for p in (ROOT / "cast").iterdir() if p.is_dir())
    ordered = [c for c in KNOWN_ORDER if c in dirs] + [c for c in dirs if c not in KNOWN_ORDER]
    cast = []
    for cid in ordered:
        if cid in RESERVED_IDS:
            errors.append(f"cast/{cid}: '{cid}' is a reserved top-level path — rename the character")
            continue
        path = ROOT / "cast" / cid / "character.yaml"
        if not path.exists():
            errors.append(f"cast/{cid}/ has no character.yaml")
            continue
        c = read_yaml(path, f"cast/{cid}/character.yaml", errors)
        if not isinstance(c, dict):
            if c is not None:
                errors.append(f"cast/{cid}/character.yaml: must be a mapping")
            continue
        validate("character", c, f"cast/{cid}", errors)
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
    return cast


def load_manifest(cid, errors):
    """Read cast/<id>/assets.yaml into (entry, src_path, rel_url_path) triples.

    The manifest carries only what a human knows: which file, what kind, what it
    shows. Everything measurable is measured later, from the bytes.
    """
    path = ROOT / "cast" / cid / "assets.yaml"
    if not path.exists():
        return []
    label = f"cast/{cid}/assets.yaml"
    doc = read_yaml(path, label, errors) or {}
    if not isinstance(doc, dict) or not isinstance(doc.get("assets"), list):
        errors.append(f"{label}: must be a mapping with an 'assets' list")
        return []

    assets_root = (ROOT / "cast" / cid / "assets").resolve()
    out = []
    for i, entry in enumerate(doc["assets"]):
        item = f"{label}[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{item}: each entry must be a mapping")
            continue
        rel = entry.get("file")
        if not rel:
            errors.append(f"{item}: 'file' is required")
            continue
        src = (ROOT / "cast" / cid / rel).resolve()
        try:
            under = src.relative_to(assets_root)
        except ValueError:
            errors.append(f"{item}: '{rel}' must live under cast/{cid}/assets/")
            continue
        if not src.is_file():
            errors.append(f"{item}: '{rel}' does not exist")
            continue
        out.append((entry, src, under.as_posix(), item))
    return out


def build_assets(cid, character, errors):
    """Every asset record for one character: the canon three, then the manifest."""
    records, copies = [], []

    # The canonical identity assets are not in assets.yaml — they are declared in
    # character.yaml and predate this contract. Synthesise their records here so
    # /assets/index.json is genuinely complete and one filter reaches everything.
    canon = [
        ("turnaround", "sheet", ROOT / "cast" / cid / character["sheet"], "sheet.png",
         ["turnaround", "reference", "multi-view"],
         f"{character['name']} turnaround sheet — the canonical reference conditioning"),
        ("headshot", "square", ROOT / "cast" / cid / character["headshot"], "headshot.png",
         ["square", "avatar", "close-up"],
         f"{character['name']} square avatar headshot, centred with even margin"),
    ]
    portrait = SITE / "art" / cid / "portrait.png"
    if portrait.exists():
        canon.append(("portrait", "bill", portrait, "portrait.png", ["bill-page", "full-body"],
                      f"{character['name']} bill-page portrait"))

    for kind, slug, src, rel, tags, caption in canon:
        if not src.exists():
            continue
        rec = make_asset(cid, kind, src, rel, errors, label=f"cast/{cid}:{rel}", slug=slug,
                         tags=tags, caption=caption, created=CANON_ASSET_DATE)
        if rec:
            records.append(rec)
            copies.append((src, rel))

    for entry, src, rel, item in load_manifest(cid, errors):
        rec = make_asset(
            cid, entry.get("kind", ""), src, rel, errors, label=item,
            tags=entry.get("tags"), caption=entry.get("caption"),
            created=str(entry.get("created") or ""), source=entry.get("source"),
            notes=entry.get("notes"),
        )
        # Anything left in the manifest entry that isn't a known input key is a
        # typo — 'tag:' instead of 'tags:' would otherwise vanish silently.
        unknown = set(entry) - {"file", "kind", "tags", "caption", "created", "source", "notes"}
        if unknown:
            errors.append(f"{item}: unknown key(s) {sorted(unknown)}")
        if rec:
            records.append(rec)
            copies.append((src, rel))

    seen = {}
    for rec, (_, rel) in zip(records, copies):
        if rec["id"] in seen:
            errors.append(
                f"cast/{cid}: duplicate asset id '{rec['id']}' — file stems must be "
                f"unique within a kind ({seen[rec['id']]} and {rel})")
        seen[rec["id"]] = rel
    return records, copies


def load_voice(cid, samples, errors):
    """cast/<id>/voice.yaml -> the /<id>/voice.json document, or None."""
    path = ROOT / "cast" / cid / "voice.yaml"
    if not path.exists():
        return None
    label = f"cast/{cid}/voice.yaml"
    doc = read_yaml(path, label, errors)
    if doc is None:  # already reported as a syntax error; don't pile on six more
        return None
    if not isinstance(doc, dict):
        errors.append(f"{label}: must be a mapping")
        return None
    if doc.get("id") != cid:
        errors.append(f"{label}: id '{doc.get('id')}' does not match directory")
    # Samples are declared once, in assets.yaml, and land here by kind. Two places
    # to list the same WAV is two places for it to go stale.
    doc["samples"] = samples
    doc["voice_url"] = f"{BASE_URL}/{cid}/voice.json"
    validate("voice", doc, label, errors)
    return doc


def load_loras(cid, asset_ids, errors):
    """cast/<id>/lora.yaml -> a list of LoRA cards. Weights are not in this repo."""
    path = ROOT / "cast" / cid / "lora.yaml"
    if not path.exists():
        return []
    label = f"cast/{cid}/lora.yaml"
    doc = read_yaml(path, label, errors) or {}
    if not isinstance(doc, dict) or not isinstance(doc.get("loras"), list):
        errors.append(f"{label}: must be a mapping with a 'loras' list")
        return []
    cards = []
    for i, card in enumerate(doc["loras"]):
        item = f"{label}[{i}]"
        if not isinstance(card, dict):
            errors.append(f"{item}: each entry must be a mapping")
            continue
        if validate("lora", card, item, errors):
            # A trained_on id that resolves to nothing breaks the one link between
            # the weights (which left the repo) and the images (which did not).
            missing = [a for a in card.get("trained_on", []) if a not in asset_ids]
            if missing:
                errors.append(f"{item}: trained_on references unknown asset id(s) {missing}")
        cards.append(card)
    return cards


def fenced_block(md_text, heading_needle, label, errors):
    """The first fenced code block under the '## ' heading containing `needle`.

    The .md stays the source of truth for the house style; this makes the JSON a
    projection of it rather than a copy. A copy would drift the first time the
    style changed — it already did once, which is why the character-tokens
    section of that file is now a pointer instead of a duplicate.
    """
    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        if not (line.startswith("## ") and heading_needle in line):
            continue
        j = i + 1
        while j < len(lines) and not lines[j].startswith("```"):
            if lines[j].startswith("## "):
                break
            j += 1
        if j >= len(lines) or not lines[j].startswith("```"):
            break
        body = []
        k = j + 1
        while k < len(lines) and not lines[k].startswith("```"):
            body.append(lines[k])
            k += 1
        if k >= len(lines):
            break
        # Hard-wrapped in the .md for reading; a prompt wants one paragraph.
        return " ".join("\n".join(body).split())
    errors.append(f"{label}: no fenced block under a '## …{heading_needle}…' heading")
    return None


def build_style(cast, errors):
    src = ROOT / "style" / "vaudeville-1933.md"
    if not src.exists():
        errors.append("style/vaudeville-1933.md missing")
        return None
    md = src.read_text()
    label = "style/vaudeville-1933.md"
    accents = [{"character": c["id"], "accent": " ".join(c["accent"].split())}
               for c in cast if (c.get("accent") or "").strip()]
    return {
        "id": "vaudeville-1933",
        "name": "Vaudeville 1933",
        "source_url": f"{BASE_URL}/style/vaudeville-1933.md",
        "rule": ("Shared by every panel and never varied per render. Vary staging and "
                 "camera only; varying the style per panel is the fastest way to make a "
                 "strip look like four unrelated drawings."),
        "style_block": fenced_block(md, "STYLE_BLOCK", label, errors),
        "exclusion_block": fenced_block(md, "EXCLUSION_BLOCK", label, errors),
        "exclusion_note": ("FLUX.2 ignores negative prompts entirely and Nano Banana treats "
                           "them loosely, so exclusions are stated positively inside the "
                           "prompt. Diegetic text — lettering on a prop — is wanted and is "
                           "often the joke; non-diegetic furniture (balloons, borders, "
                           "captions) is composited afterwards and must not be drawn."),
        "palette": {"mode": "sepia duotone on aged newsprint", "accents": accents},
        "character_tokens": ("Not duplicated here. Read `tokens` from /cast.json — a copy "
                             "would drift the first time a character changed. It did."),
    }


def main() -> int:
    errors = []
    cast = load_cast(errors)
    for cid in (c["id"] for c in cast):
        if not (SITE / f"{cid}.html").exists():
            errors.append(f"site/{cid}.html missing — every cast member needs a bill page")

    per_character = {}
    for c in cast:
        cid = c["id"]
        records, copies = build_assets(cid, c, errors)
        per_character[cid] = {"records": records, "copies": copies}

    all_ids = {r["id"] for d in per_character.values() for r in d["records"]}
    for c in cast:
        cid = c["id"]
        samples = [r for r in per_character[cid]["records"] if r["kind"] == "voice-sample"]
        per_character[cid]["voice"] = load_voice(cid, samples, errors)
        per_character[cid]["loras"] = load_loras(cid, all_ids, errors)

    style = build_style(cast, errors)

    if errors:
        for e in errors:
            print(f"FAIL {e}", file=sys.stderr)
        return 1

    if PUB.exists():
        shutil.rmtree(PUB)
    PUB.mkdir()
    (PUB / "assets").mkdir()
    (PUB / "schema").mkdir()
    (PUB / "style").mkdir()

    registry, index = [], []
    for c in cast:
        cid = c["id"]
        (PUB / cid).mkdir()
        bundle = per_character[cid]

        for src, rel in bundle["copies"]:
            dest = PUB / cid / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        # site/art holds presentation extras (reference photos) alongside the
        # portrait; copy the rest verbatim, the portrait is already an asset.
        art = SITE / "art" / cid
        if art.is_dir():
            for f in art.iterdir():
                if f.is_file() and not (PUB / cid / f.name).exists():
                    shutil.copy2(f, PUB / cid / f.name)

        records = bundle["records"]
        index.extend(records)
        counts = {}
        for r in records:
            counts[r["kind"]] = counts.get(r["kind"], 0) + 1

        sheet = next(r for r in records if r["kind"] == "turnaround")
        head = next(r for r in records if r["kind"] == "headshot")
        entry = {
            "id": cid,
            "name": c["name"],
            "role": c.get("role"),
            "tokens": c["tokens"].strip(),
            "props": c.get("props", []),
            "accent": (c.get("accent") or "").strip() or None,
            "notes": (c.get("notes") or "").strip() or None,
            "sheet_url": sheet["url"],
            "sheet_size": [sheet["width"], sheet["height"]],
            "headshot_url": head["url"],
            "headshot_size": [head["width"], head["height"]],
        }
        if (art / "portrait.png").exists():
            entry["portrait_url"] = f"{BASE_URL}/{cid}/portrait.png"
        # New fields only append. Everything above is what consumers already read.
        entry["assets"] = dict(sorted(counts.items()))
        entry["assets_url"] = f"{BASE_URL}/{cid}/assets.json"
        entry["voice_url"] = f"{BASE_URL}/{cid}/voice.json" if bundle["voice"] else None
        entry["lora_count"] = len(bundle["loras"])
        registry.append(entry)

        write_json(PUB / cid / "assets.json", {
            "character": cid,
            "name": c["name"],
            "base_url": BASE_URL,
            "schema": f"{BASE_URL}/schema/asset.schema.json",
            "style_url": f"{BASE_URL}/style/vaudeville-1933.json",
            "counts": dict(sorted(counts.items())),
            "assets": records,
            "voice": bundle["voice"],
            "lora": bundle["loras"],
        })
        if bundle["voice"]:
            write_json(PUB / cid / "voice.json", bundle["voice"])

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

    for name in SCHEMA_NAMES:
        shutil.copy2(SCHEMA_DIR / f"{name}.schema.json", PUB / "schema" / f"{name}.schema.json")
    shutil.copy2(ROOT / "style" / "vaudeville-1933.md", PUB / "style" / "vaudeville-1933.md")
    if (ROOT / "API.md").exists():
        shutil.copy2(ROOT / "API.md", PUB / "API.md")

    write_json(PUB / "style" / "vaudeville-1933.json", style)
    write_json(PUB / "assets" / "index.json", {
        "base_url": BASE_URL,
        "schema": f"{BASE_URL}/schema/asset.schema.json",
        "count": len(index),
        "assets": index,
    })
    write_json(PUB / "cast.json", {
        "source": "https://github.com/sethgho/the-cast",
        "style": "vaudeville-1933",
        "base_url": BASE_URL,
        "endpoints": {
            "cast": f"{BASE_URL}/cast.json",
            "assets_index": f"{BASE_URL}/assets/index.json",
            "character_assets": f"{BASE_URL}/{{id}}/assets.json",
            "character_voice": f"{BASE_URL}/{{id}}/voice.json",
            "style": f"{BASE_URL}/style/vaudeville-1933.json",
            "schemas": {n: f"{BASE_URL}/schema/{n}.schema.json" for n in SCHEMA_NAMES},
            "docs": f"{BASE_URL}/API.md",
        },
        "cast": registry,
    })

    voices = sum(1 for d in per_character.values() if d["voice"])
    loras = sum(len(d["loras"]) for d in per_character.values())
    for c in cast:
        if not per_character[c["id"]]["voice"]:
            print(f"WARN cast/{c['id']}/voice.yaml missing — no /{c['id']}/voice.json", file=sys.stderr)
    print(f"built public/ — {len(registry)} cast members, {len(index)} assets, "
          f"{voices} voice profiles, {loras} lora cards, {pages} pages")
    return 0


def write_json(path, doc):
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
