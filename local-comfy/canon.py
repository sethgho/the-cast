"""One unambiguous text for a JSON-shaped value, and the digest taken over it.

This is the bottom of the pipeline's cache key (`pipeline.py`) and it exists as its own module for
one reason: `celld-editor/worker.js` has to compute the SAME digests inside the Durable Object,
because the DO owns every mutation and must know what an edit invalidated without asking a machine
that may be asleep. There is a hand-written copy of `canon()` in that file, and the two must agree
byte for byte or a healthy character reads as entirely stale.

`test_pipeline_keys.py` runs both implementations over the real manifests and fails on one digit.

Written by hand rather than left to `json.dumps`, because the two runtimes disagree in two ways
that would each be silent:

- **Key order.** Objects are emitted with their keys SORTED, so insertion order cannot leak in.
- **Numbers.** `json.dumps(1.0)` is `"1.0"` and `JSON.stringify(1.0)` is `"1"`. A float that is a
  whole number is emitted as an integer here, on both sides. cfg=1.0 is a real repaint param, so
  this is not academic.

Everything else round-trips through the shortest representation, which Python's `repr` and
JavaScript's Number-to-String both produce.
"""
import hashlib
import json

DIGEST_CHARS = 16   # 64 bits of SHA-256; these keys address ~130 steps, not the internet


def canon(value):
    if value is None:
        return "null"
    # bool before int on purpose: in Python, True is an int and would serialise as 1.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canon(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{canon(k)}:{canon(value[k])}" for k in sorted(value)) + "}"
    raise TypeError(f"{type(value).__name__} has no canonical form: {value!r}")


def digest(value):
    return hashlib.sha256(canon(value).encode("utf-8")).hexdigest()[:DIGEST_CHARS]
