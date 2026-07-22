#!/usr/bin/env python3
"""Stage 0-1a: harvest + deterministic metrics ledger (derivation.md).

Runs ON the source machine. Reads ~/.claude/projects/*/[uuid].jsonl for sessions
active in the window, filtered by an opt-in allowlist, and emits ONE small JSON
ledger of counts/ratios/durations. No transcript text is ever written to the
output — privacy by construction, not by scrubbing.

Usage:
  ledger.py --allowlist allowlist.wilson.json [--date 2026-07-22] [--out ledger.json]

The ledger is the only thing that leaves the machine (plus Vein B event records,
which are templated from these same numbers — never quoted from a session).
"""

import argparse
import fnmatch
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECTS_DIR = Path.home() / ".claude" / "projects"

CONFIDENCE_RE = re.compile(
    r"that should fix|should work now|you'?re absolutely right|this will fix|"
    r"the fix is simple|one small change|quick fix",
    re.I,
)
SENTIMENT_RE = re.compile(
    r"\bwtf\b|\bffs\b|why is this|why does this|still broken|again\?|\bugh\b|"
    r"goddamn|dammit|fuck|\bshit\b|are you kidding",
    re.I,
)
LIMIT_RE = re.compile(r"session limit|rate limit|usage limit|out of memory|oom", re.I)

LONG_SILENCE_MIN = 20


def parse_ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def allowed(name, cfg):
    if any(fnmatch.fnmatch(name, p) for p in cfg["deny"]):
        return False
    return any(fnmatch.fnmatch(name, p) for p in cfg["allow"])


def text_of(content):
    """Concatenated text blocks of a message content field (str or block array)."""
    if isinstance(content, str):
        return content
    out = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
    return "\n".join(out)


def wc(s):
    return len(s.split())


def scan_session(path, day_start, day_end):
    """One pass over a session file; returns metrics or None if outside window."""
    m = {
        "human_turns": 0, "human_words": 0, "assistant_msgs": 0, "agent_words": 0,
        "output_tokens": 0, "tool_calls": 0, "tool_errors": 0,
        "confidence_phrases": 0, "sentiment_spikes": 0, "limit_events": 0,
        "long_silences": 0, "waiting_min": 0.0, "compactions": 0,
        "tools": Counter(), "models": Counter(), "bash_repeats": 0,
        "first_ts": None, "last_ts": None,
    }
    bash_cmds = Counter()
    prev_ts, prev_was_assistant = None, False
    in_window = False

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            ts = parse_ts(rec.get("timestamp", ""))
            if ts:
                if day_start <= ts < day_end:
                    in_window = True
                m["first_ts"] = m["first_ts"] or ts
                m["last_ts"] = ts

            rtype = rec.get("type")
            msg = rec.get("message") or {}
            content = msg.get("content")

            if rtype == "user":
                if isinstance(content, str):  # a human actually typed this
                    m["human_turns"] += 1
                    m["human_words"] += wc(content)
                    if SENTIMENT_RE.search(content):
                        m["sentiment_spikes"] += 1
                    if ts and prev_ts and prev_was_assistant:
                        gap = (ts - prev_ts).total_seconds() / 60
                        if gap > LONG_SILENCE_MIN:
                            m["long_silences"] += 1
                        if gap < 12 * 60:  # ignore overnight gaps
                            m["waiting_min"] += gap
                    prev_was_assistant = False
                elif isinstance(content, list):  # tool results
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            if block.get("is_error"):
                                m["tool_errors"] += 1
            elif rtype == "assistant":
                m["assistant_msgs"] += 1
                model = msg.get("model")
                if model:
                    m["models"][model] += 1
                usage = msg.get("usage") or {}
                m["output_tokens"] += usage.get("output_tokens", 0) or 0
                if isinstance(content, list):
                    for block in content:
                        btype = block.get("type") if isinstance(block, dict) else None
                        if btype == "text":
                            t = block.get("text", "")
                            m["agent_words"] += wc(t)
                            if CONFIDENCE_RE.search(t):
                                m["confidence_phrases"] += 1
                        elif btype == "tool_use":
                            m["tool_calls"] += 1
                            name = block.get("name", "?")
                            m["tools"][name] += 1
                            if name == "Bash":
                                cmd = (block.get("input") or {}).get("command", "")
                                bash_cmds[cmd] += 1
                prev_was_assistant = True
            elif rtype == "system":
                stext = text_of(content) or json.dumps(rec.get("content", ""), default=str)
                if "compact" in stext.lower():
                    m["compactions"] += 1
                if LIMIT_RE.search(stext):
                    m["limit_events"] += 1
            if ts:
                prev_ts = ts

    if not in_window:
        return None
    m["bash_repeats"] = sum(1 for c in bash_cmds.values() if c >= 3)
    m["tools"] = dict(m["tools"].most_common(10))
    m["models"] = dict(m["models"])
    for k in ("first_ts", "last_ts"):
        m[k] = m[k].isoformat() if m[k] else None
    m["waiting_min"] = round(m["waiting_min"], 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allowlist", required=True)
    ap.add_argument("--date", help="UTC day to ledger (default: yesterday)")
    ap.add_argument("--out", help="output path (default: stdout)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.allowlist).read_text())
    if args.date:
        day_start = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
    else:
        day_start = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    day_end = day_start + timedelta(days=1)

    sessions, skipped_projects = [], 0
    for proj in sorted(PROJECTS_DIR.iterdir()):
        if not proj.is_dir():
            continue
        if not allowed(proj.name, cfg):
            skipped_projects += 1
            continue
        for f in proj.glob("*.jsonl"):
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            if mtime < day_start:  # untouched since before the window
                continue
            m = scan_session(f, day_start, day_end)
            if m:
                m["project"] = proj.name
                m["session"] = f.stem
                sessions.append(m)

    totals = Counter()
    for s in sessions:
        for k in ("human_turns", "human_words", "assistant_msgs", "agent_words",
                  "output_tokens", "tool_calls", "tool_errors", "confidence_phrases",
                  "sentiment_spikes", "limit_events", "long_silences", "compactions",
                  "bash_repeats"):
            totals[k] += s[k]
    totals["waiting_min"] = round(sum(s["waiting_min"] for s in sessions), 1)

    ledger = {
        "schema": 1,
        "host": cfg["host"],
        "agent": cfg["agent"],
        "date": day_start.date().isoformat(),
        "sessions": len(sessions),
        "projects": len({s["project"] for s in sessions}),
        "projects_denied_or_unlisted": skipped_projects,
        "totals": dict(totals),
        "per_session": sessions,
    }
    out = json.dumps(ledger, indent=1)
    if args.out:
        Path(args.out).write_text(out)
        print(f"wrote {args.out}: {len(sessions)} sessions", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
