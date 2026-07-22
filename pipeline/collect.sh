#!/usr/bin/env bash
# Stage 0 collector — runs on wilson (nightly, or by hand).
# Pushes the current ledger.py + per-host allowlist to each machine, runs it
# there (redaction/aliasing happens at source), and pulls back only the
# text-free JSON ledger. Output: ~/follies/ledgers/YYYY-MM-DD.<host>.json
#
# The repo is PUBLIC — ledgers are never committed; they live only here.
set -euo pipefail

PIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$HOME/follies/ledgers"
MAC="sethgho@100.64.185.78"
DATE="${1:-$(date -u -d yesterday +%F)}"
mkdir -p "$OUT_DIR"

echo "== wilson =="
python3 "$PIPE_DIR/ledger.py" --allowlist "$PIPE_DIR/allowlist.wilson.json" \
  --date "$DATE" --out "$OUT_DIR/$DATE.wilson.json"

echo "== mac =="
scp -q "$PIPE_DIR/ledger.py" "$PIPE_DIR/allowlist.mac.json" "$MAC:/tmp/"
ssh "$MAC" "python3 /tmp/ledger.py --allowlist /tmp/allowlist.mac.json --date $DATE" \
  > "$OUT_DIR/$DATE.mac.json"

echo "== akebot (via mac) =="
scp -q "$PIPE_DIR/allowlist.akebot.json" "$MAC:/tmp/"
ssh "$MAC" "scp -q /tmp/ledger.py /tmp/allowlist.akebot.json akebot:/tmp/ && \
  ssh akebot 'python3 /tmp/ledger.py --allowlist /tmp/allowlist.akebot.json --date $DATE'" \
  > "$OUT_DIR/$DATE.akebot.json"

for h in wilson mac akebot; do
  f="$OUT_DIR/$DATE.$h.json"
  s=$(jq -r '.sessions' "$f")
  echo "$h: $s sessions -> $f"
done
