#!/usr/bin/env python3
"""Check that scan_session counts each reply once and only inside the --date window.

    python3 test_ledger.py
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ledger import scan_session

DAY_21 = datetime(2026, 10, 21, tzinfo=timezone.utc)
DAY_22 = datetime(2026, 10, 22, tzinfo=timezone.utc)
DAY_23 = datetime(2026, 10, 23, tzinfo=timezone.utc)


def human(ts, text):
    return {"type": "user", "timestamp": ts, "message": {"content": text}}


def block(ts, msg_id, req, content_block, tokens=100):
    return {"type": "assistant", "timestamp": ts, "requestId": req, "message": {
        "id": msg_id, "model": "claude-test", "usage": {"output_tokens": tokens},
        "content": [content_block]}}


def scan(records, start, end):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(r) for r in records))
    try:
        return scan_session(Path(f.name), start, end)
    finally:
        Path(f.name).unlink()


def test_reply_split_into_blocks_counts_once():
    ts = "2026-10-22T10:00:00Z"
    m = scan([
        human(ts, "fix it"),
        block(ts, "msg_1", "req_1", {"type": "thinking", "thinking": "hm"}),
        block(ts, "msg_1", "req_1", {"type": "text", "text": "on it"}),
        block(ts, "msg_1", "req_1", {"type": "tool_use", "name": "Bash", "input": {}}),
    ], DAY_22, DAY_23)
    assert m["assistant_msgs"] == 1, m["assistant_msgs"]
    assert m["output_tokens"] == 100, m["output_tokens"]
    assert m["models"] == {"claude-test": 1}, m["models"]
    assert m["tool_calls"] == 1 and m["agent_words"] == 2, m


def test_session_across_midnight_counts_each_day_once():
    records = [
        human("2026-10-21T23:00:00Z", "one"),
        block("2026-10-21T23:01:00Z", "msg_1", "req_1", {"type": "text", "text": "a"}),
        human("2026-10-21T23:30:00Z", "two"),
        block("2026-10-21T23:31:00Z", "msg_2", "req_2", {"type": "text", "text": "b"}),
        human("2026-10-22T00:30:00Z", "three"),
    ]
    day21 = scan(records, DAY_21, DAY_22)
    day22 = scan(records, DAY_22, DAY_23)
    assert day21["human_turns"] == 2 and day21["output_tokens"] == 200, day21
    assert day22["human_turns"] == 1 and day22["output_tokens"] == 0, day22
    assert day22["first_ts"] == day22["last_ts"] == "2026-10-22T00:30:00+00:00", day22
    # The wait from the 23:31 reply to the 00:30 turn crosses midnight and still counts.
    assert day22["long_silences"] == 1 and day22["waiting_min"] == 59, day22


def test_session_outside_window_is_skipped():
    assert scan([human("2026-10-21T12:00:00Z", "hi")], DAY_22, DAY_23) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
