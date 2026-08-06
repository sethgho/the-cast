#!/usr/bin/env bash
#
# Publish a LoRA: weights to a GitHub Release, card to the registry.
#
# The hub serves the CARD; GitHub serves the WEIGHTS. .safetensors files run
# 50-400 MB — in git they bloat every clone forever and land inside the Docker
# image the site ships from, for bytes nginx should never have been asked to
# serve. The training IMAGES stay in the repo (they are small, and they are the
# only link back from the weights to what made them); the derived weights do not.
#
# This script is the only supported way to write a card's sha256 and bytes.
# Everywhere else in this registry those are recomputed at build time from the
# file on disk — here the file is not in the tree to hash, so they are captured
# at upload time instead. Hand-editing them is how the card starts lying.
#
#   scripts/publish-lora.sh weights.safetensors \
#     --character wilson --id wilson-character-v1 --kind character \
#     --base-model sdxl-1.0 --trigger wlsnfnce --strength 0.8
#
# Add --dry-run to see the card and the release plan without uploading anything.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '3,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Required:
  <file>                  the .safetensors to publish
  --character <id>        cast member the LoRA belongs to (a cast/<id>/ directory)
  --id <slug>             card id, kebab-case and versioned, e.g. wilson-character-v1
  --kind <character|style>
  --base-model <name>     exactly what it was trained against, e.g. sdxl-1.0
  --trigger <token>       the word that summons it
  --strength <float>      measured starting weight

Optional:
  --release-tag <tag>     defaults to lora-<id>
  --trained-on <kinds>    comma-separated asset kinds to record as training data.
                          Defaults to training-set for a character LoRA, and
                          training-set,style-plate for a style LoRA — a style
                          LoRA MUST include character-free plates or it
                          memorises the leads and every character drifts.
  --trainer <name>        e.g. "kohya sd-scripts"
  --notes <text>
  --dry-run               print the card and the plan; upload nothing
USAGE
}

FILE="" CHARACTER="" CARD_ID="" KIND="" BASE_MODEL="" TRIGGER="" STRENGTH=""
RELEASE_TAG="" TRAINED_ON_KINDS="" TRAINER="" NOTES="" DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)      usage; exit 0 ;;
        --character)    CHARACTER="${2:-}"; shift 2 ;;
        --id)           CARD_ID="${2:-}"; shift 2 ;;
        --kind)         KIND="${2:-}"; shift 2 ;;
        --base-model)   BASE_MODEL="${2:-}"; shift 2 ;;
        --trigger)      TRIGGER="${2:-}"; shift 2 ;;
        --strength)     STRENGTH="${2:-}"; shift 2 ;;
        --release-tag)  RELEASE_TAG="${2:-}"; shift 2 ;;
        --trained-on)   TRAINED_ON_KINDS="${2:-}"; shift 2 ;;
        --trainer)      TRAINER="${2:-}"; shift 2 ;;
        --notes)        NOTES="${2:-}"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -*)             die "unknown flag '$1' (--help for usage)" ;;
        *)              [ -z "$FILE" ] || die "more than one file given"; FILE="$1"; shift ;;
    esac
done

for pair in "file:$FILE" "--character:$CHARACTER" "--id:$CARD_ID" "--kind:$KIND" \
            "--base-model:$BASE_MODEL" "--trigger:$TRIGGER" "--strength:$STRENGTH"; do
    [ -n "${pair#*:}" ] || die "missing ${pair%%:*} (--help for usage)"
done

[ -f "$FILE" ] || die "no such file: $FILE"
[ -d "$ROOT/cast/$CHARACTER" ] || die "no cast member '$CHARACTER' (cast/$CHARACTER/ does not exist)"
case "$KIND" in character|style) ;; *) die "--kind must be 'character' or 'style', got '$KIND'" ;; esac
command -v gh >/dev/null || die "gh CLI not found"
command -v python3 >/dev/null || die "python3 not found"

if [ -z "$TRAINED_ON_KINDS" ]; then
    if [ "$KIND" = "style" ]; then TRAINED_ON_KINDS="training-set,style-plate"
    else TRAINED_ON_KINDS="training-set"; fi
fi
[ -n "$RELEASE_TAG" ] || RELEASE_TAG="lora-$CARD_ID"

# The registry has to be valid BEFORE weights get published against it. A card
# whose trained_on points at assets that do not exist is worse than no card, and
# the build is the thing that knows.
printf '==> validating the registry (scripts/build-site.py)\n'
python3 "$ROOT/scripts/build-site.py" >/dev/null || die "build failed — fix the registry before publishing weights"

ASSET_JSON="$ROOT/public/$CHARACTER/assets.json"
[ -f "$ASSET_JSON" ] || die "no $ASSET_JSON after a successful build"

BYTES="$(wc -c < "$FILE" | tr -d ' ')"
if command -v sha256sum >/dev/null; then SHA="$(sha256sum "$FILE" | cut -d' ' -f1)"
else SHA="$(shasum -a 256 "$FILE" | cut -d' ' -f1)"; fi

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo sethgho/the-cast)"
ASSET_NAME="$(basename "$FILE")"
DOWNLOAD_URL="https://github.com/$REPO/releases/download/$RELEASE_TAG/$ASSET_NAME"
TRAINED_AT="$(date -u +%Y-%m-%d)"

printf '==> %s  %s bytes  sha256 %s\n' "$ASSET_NAME" "$BYTES" "${SHA:0:16}…"
printf '==> release %s on %s\n' "$RELEASE_TAG" "$REPO"

# Card written by python: bash cannot edit YAML safely, and python3 is already a
# hard dependency of this repo. Replaces an existing card with the same id rather
# than appending a duplicate, so re-running after a failed upload is safe.
CARD_ARGS=("$ROOT/cast/$CHARACTER/lora.yaml" "$ASSET_JSON" "$CARD_ID" "$KIND" "$BASE_MODEL"
           "$DOWNLOAD_URL" "$SHA" "$BYTES" "$TRIGGER" "$STRENGTH" "$TRAINED_AT"
           "$TRAINED_ON_KINDS" "$RELEASE_TAG" "$TRAINER" "$NOTES" "$DRY_RUN")

python3 - "${CARD_ARGS[@]}" <<'PY'
import json, sys, yaml

(path, asset_json, cid, kind, base_model, url, sha, nbytes, trigger, strength,
 trained_at, kinds, tag, trainer, notes, dry) = sys.argv[1:17]

wanted = {k.strip() for k in kinds.split(",") if k.strip()}
assets = json.load(open(asset_json))["assets"]
trained_on = sorted(a["id"] for a in assets if a["kind"] in wanted)
if not trained_on:
    sys.exit(f"error: no assets of kind(s) {sorted(wanted)} in {asset_json} — "
             "add the training images to assets.yaml before publishing weights")

card = {
    "id": cid,
    "kind": kind,
    "base_model": base_model,
    "download_url": url,
    "sha256": sha,
    "bytes": int(nbytes),
    "trigger": trigger,
    "recommended_strength": float(strength),
    "trained_on": trained_on,
    "trained_at": trained_at,
    "release_tag": tag,
}
if trainer:
    card["trainer"] = trainer
if notes:
    card["notes"] = notes

try:
    doc = yaml.safe_load(open(path).read()) or {}
except FileNotFoundError:
    doc = {}
doc.setdefault("loras", [])
doc["loras"] = [c for c in doc["loras"] if c.get("id") != cid] + [card]

text = ("# LoRA cards for this character. WEIGHTS ARE NOT IN THIS REPO — they are\n"
        "# GitHub Release assets; this file is the card that points at them.\n"
        "# Written by scripts/publish-lora.sh. sha256 and bytes are captured at\n"
        "# upload time because the file is not in the tree to hash at build time;\n"
        "# do not hand-edit them.\n"
        + yaml.safe_dump(doc, sort_keys=False, width=88, allow_unicode=True))

print(f"--- card ({len(trained_on)} training assets) ---")
print(yaml.safe_dump(card, sort_keys=False, width=88, allow_unicode=True))
if dry == "1":
    print(f"[dry-run] would write {path}")
else:
    open(path, "w").write(text)
    print(f"wrote {path}")
PY

if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run] would upload %s to release %s\n' "$ASSET_NAME" "$RELEASE_TAG"
    printf '[dry-run] nothing was uploaded and no file was written\n'
    exit 0
fi

if ! gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
    printf '==> creating release %s\n' "$RELEASE_TAG"
    gh release create "$RELEASE_TAG" \
        --title "LoRA: $CARD_ID" \
        --notes "$KIND LoRA for $CHARACTER, trained against $BASE_MODEL. Trigger: \`$TRIGGER\`, recommended strength $STRENGTH. Card: https://cast.sethgholson.com/$CHARACTER/assets.json"
fi

printf '==> uploading %s\n' "$ASSET_NAME"
gh release upload "$RELEASE_TAG" "$FILE" --clobber

# The card is now in the registry, so the gate must still pass with it in place.
printf '==> re-validating with the new card\n'
python3 "$ROOT/scripts/build-site.py" >/dev/null || die "build failed with the new card — the card is written but broken"

printf '\npublished.\n  weights  %s\n  card     https://cast.sethgholson.com/%s/assets.json\n' \
    "$DOWNLOAD_URL" "$CHARACTER"
printf '  commit   cast/%s/lora.yaml\n' "$CHARACTER"
