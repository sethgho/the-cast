// The sprite cell editor's state, on celld: one CharacterDO per character.
//
// The split is deliberate and is the whole design (DESIGN-editor.md, stage 6). The packer is
// numpy and the repaint drives ComfyUI on a GPU, so neither will ever run in V8; they stay on
// wilson, behind `sprite_agent.py`, which long-polls this Worker for work. What moves here is
// the part that is pure state: the manifest, and the job QUEUE that stops two browsers from
// queueing the same 45-second GPU job.
//
// The mutations below are MESSAGES, not manifest writes. The Python editor loaded the manifest,
// spliced it in the client of the file and saved it back; against a Durable Object that same
// shape would be a non-atomic read-modify-write and the DO's single-threadedness would buy
// nothing. So "drop frame 3 of tag punch" crosses the wire, and the splice happens in here.
//
// This object used to allow ONE in-flight job per character and 409 everything else, which made
// the page a modal wait: every edit blocked the next one for the length of a paint. It is a
// queue now. Three rules make that safe on one GPU:
//
//   1. Only the HEAD of the queue is ever claimable, so the card runs one job at a time.
//   2. Cheap mutations (`generates: false`) coalesce into ONE trailing repack. They edit the
//      manifest immediately and the agent re-reads the manifest when it starts, so one repack
//      absorbs any number of state edits — twenty holds and reorders cost one repack, not twenty.
//   3. A queued job can be cancelled, and its stored INVERSE is re-spliced. A re-roll is an
//      experiment; backing out of one must not need a second GPU job.

// Which characters exist is DATA, not a constant. It was a two-name array here, so adding a
// character meant editing and redeploying a Worker — and the whole point of DESIGN-pipeline.md's
// new-subject flow is that a person uploads a plate and the character exists. The roster is one
// reserved cell in this same Durable Object namespace, appended to whenever a character is seeded.
//
// The name is unusable as a character id (a NUL byte), so it cannot collide with a real one, and
// it keeps the roster inside the namespace it describes rather than in a second store that could
// disagree with it.
const ROSTER = "\u0000roster";

// A character id has to be safe in a file name, a URL and a repaint path. Membership is NOT
// checked on the hot path on purpose: an id that names nothing simply routes to an unseeded cell,
// which already answers "this character has not been seeded yet", and checking the roster on every
// request would put a second Durable Object hop in front of every read.
const CHARACTER_ID = /^[a-z][a-z0-9-]{0,31}$/;

// Mirrors repaint_cells.SEED: the one seed every cell starts on, and the only one whose repaint
// file carries no `-s<seed>` suffix.
const BASE_SEED = 77;

const DIRECTIONS = ["forward", "reverse", "pingpong"];

// A CLAIMED job whose agent has gone silent for this long is handed back to the queue. It is
// generous because a claim is not a start: the agent serialises the two characters against the
// one card, so cadbury's job can sit claimed for the length of seth's paint with nothing to
// report.
const LEASE_MS = 10 * 60 * 1000;

// A RUNNING job is different — the agent heartbeats every few seconds while it works, so silence
// here means the process died. Without the shorter bound a wilson reboot mid-repaint stalled the
// whole queue behind it for ten minutes.
const RUNNING_LEASE_MS = 60 * 1000;

// The longest a claim request is held open before answering "no work". Kept well under any
// proxy's idle timeout, and short enough that an agent restart cannot leave a request hanging
// against a DO that has since been evicted.
const MAX_CLAIM_WAIT_MS = 20000;

// The queue depth past which a mutation is refused. Deep enough that a person editing quickly
// never meets it, shallow enough that a stuck agent cannot bank an hour of GPU work nobody
// remembers asking for.
const MAX_PENDING = 8;

// A finished job stays visible this long so the page can show that it completed. Any longer and
// the job list becomes a log; the manifest is the record, not this.
const HISTORY_MS = 30000;

// Which operations cost GPU time. This is static, not something an op handler decides, because
// the queue has to know whether a mutation coalesces BEFORE it takes the lock and splices.
const GENERATES = {
  reroll: true,
  repick: true,
  use: false,
  drop: false,
  reorder: false,
  pivot: false,
  hold: false,
  tag: false,
  // A step's own params (DESIGN-pipeline.md, "what each step exposes"). Never GPU: the edit is
  // recorded and whatever it invalidated is drawn stale. Whether it queues a REPACK is decided
  // by the pack's key afterwards, in `mutate` — see the norun branch there.
  step: false,
  // Reattaching an orphaned edit to a live cell, and hand-collecting a tray entry. Both are free:
  // a restore carries the HOLD and the NUDGE, which are playback metadata the pack reads and no
  // drawing depends on. It deliberately does NOT carry the seed — a seed names a drawing of one
  // particular source frame, and pushing it onto a different frame would queue a 45-second paint
  // behind a button whose whole point is that it is one free click. The tray shows the seed so a
  // re-roll to it stays an explicit, priced act.
  restore: false,
  forget: false,
};
const MUTATIONS = Object.keys(GENERATES);

// Which agent job type each runnable step kind becomes. The step kind is what the rail draws; the
// job kind is what the agent dispatches on, and the two differ for `frames`, whose job is an
// ffmpeg re-EXTRACTION rather than a step called "frames".
const JOB_KIND = {
  clip: "clip",
  frames: "extract",
  picks: "picks",
  repaint: "repaint",
  pack: "pack",
};

// How near, in source-frame index within the clip, a new pick has to land to inherit an old
// frame's identity and its hand edits. DESIGN-pipeline.md fixes it at ±3: the extraction runs at
// 24fps, so three frames is an eighth of a second and no pose changes meaningfully inside it,
// while a fourth frame is far enough that a pivot nudge measured on one pose lands on another.
const MATCH_FRAMES = 3;

const LIVE = ["preparing", "queued", "claimed", "running"];

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {
      "content-type": "application/json",
      // Every value here is overwritten by the next edit, and a cached state response is the
      // most likely way this page lies about what is on disk.
      "cache-control": "no-store",
    },
  });
}

// `crypto.subtle.timingSafeEqual` is a workerd extension and does not exist on celld — it threw
// on every CORRECT token in tincan while wrong-length guesses still failed cleanly, which hid
// the bug. Fold the difference instead, with no early return on length.
function tokenMatches(given, expected) {
  if (typeof given !== "string" || typeof expected !== "string" || expected === "") return false;
  const enc = new TextEncoder();
  const a = enc.encode(given);
  const b = enc.encode(expected);
  let diff = a.length ^ b.length;
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) diff |= (a[i] ?? 0) ^ (b[i] ?? 0);
  return diff === 0;
}

function bearer(request) {
  const h = request.headers.get("authorization") || "";
  return h.startsWith("Bearer ") ? h.slice(7) : "";
}

function shortId() {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 12);
}

// --- manifest surgery ------------------------------------------------------------------------
// The key order here matches repaint_cells._frame() exactly, and `png` is omitted when absent.
// The agent writes whatever the DO hands it straight to sheets/<cid>.json, so a reordered key
// would rewrite every manifest on disk for no change in meaning.
//
// `_fid` and `_rev` are this object's own bookkeeping and are stripped from every copy that
// leaves it (`stripped()`). They are stored ON the frame rather than in a parallel array
// because a parallel array has to be spliced in lockstep with `man.frames` at six call sites,
// and the one that got missed would silently mis-key a cancel or a paint report.
//   _fid — a frame's identity, stable across reorder and re-insert.
//   _rev — the manifest revision at which this frame's DRAWING (src, seed) last changed. A
//          paint report is accepted per frame against this, so an edit made during a 45s paint
//          no longer fails a paint that was correct when it started.
function frameRecord(f, rev) {
  const out = { src: f.src, seed: f.seed };
  if (f.png) out.png = f.png;
  out.hold = f.hold ?? 1;
  out.pivot_nudge = f.pivot_nudge ?? [0, 0];
  out._fid = f._fid ?? shortId();
  out._rev = f._rev ?? rev;
  return out;
}

function stripped(man) {
  return {
    ...man,
    frames: man.frames.map(({ _fid, _rev, ...f }) => f),
  };
}

function frameMeta(man) {
  return man.frames.map((f) => ({ fid: f._fid, rev: f._rev }));
}

function tagIndex(man, name) {
  return man.tags.findIndex((t) => t.name === name);
}

function whole(value, what) {
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new HttpError(400, `${what} must be a whole number, not ${JSON.stringify(value)}`);
  }
  return value;
}

function xy(value, what) {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new HttpError(400, `${what} must be a pair [x, y]`);
  }
  return [whole(value[0], `${what} x`), whole(value[1], `${what} y`)];
}

function inCell(x, y, cell, what) {
  if (!(x >= 0 && x < cell && y >= 0 && y < cell)) {
    throw new HttpError(400, `${what} ${x},${y} is outside the ${cell}px cell`);
  }
}

// Which params of which step a person may type into, and nothing else. It is a WHITELIST because
// the alternative is a mutation that can set any key of any tag: `from`, `to` and `clip` are in
// the same objects, and a typo in one of those does not fail, it silently packs another move's
// drawings under this tag's name.
//
// Everything absent from here is either LOCKED — measured, with its reason on the chip, and
// overriding one is a named deviation this stage deliberately does not build — or it is not
// manifest data at all yet. `recipe`, and therefore the tag NAME, is absent on purpose: the
// extracted-frames directory is named after the tag, so renaming one orphans it, and nothing
// re-extracts until the stage-4 job types land.
const STEP_EDITABLE = {
  clip: ["recipe_text", "trait", "cyclic"],
  picks: ["cells", "cyclic"],
  pack: ["loop", "hold_key", "unify"],
};

// The three forms `sprite_sheet._unify_factors` actually reads. `true` is total height and is only
// for a move that must not change height at all; `"head"` normalises drift without touching a
// stride's rise and fall; `false` is none. A fourth spelling would silently mean "none".
const UNIFY = [false, true, "head"];

// A recipe is a paragraph, not a document. The cap is here rather than at the page because the
// manifest is written verbatim to sheets/<cid>.json and posted to a model.
const TEXT_MAX = 2000;

function text(value, what) {
  if (typeof value !== "string") throw new HttpError(400, `${what} must be text`);
  const s = value.trim();
  if (!s) throw new HttpError(400, `${what} cannot be empty`);
  if (s.length > TEXT_MAX) {
    throw new HttpError(400, `${what} is ${s.length} characters and the limit is ${TEXT_MAX}`);
  }
  // A control character survives JSON, reaches the prompt and lands in the manifest on disk,
  // where it is invisible in every view of it.
  if (/[\u0000-\u0008\u000b-\u001f\u007f]/.test(s)) {
    throw new HttpError(400, `${what} contains a control character`);
  }
  return s;
}

function flag(value, what) {
  if (typeof value !== "boolean") throw new HttpError(400, `${what} must be true or false`);
  return value;
}

function stepValue(field, value) {
  if (field === "recipe_text") return text(value, "the recipe");
  if (field === "trait") return text(value, "the trait line");
  if (field === "cyclic" || field === "loop" || field === "hold_key") return flag(value, field);
  if (field === "unify") {
    if (!UNIFY.some((u) => u === value)) {
      throw new HttpError(400, `unify must be false, true or "head"`);
    }
    return value;
  }
  // cells. The floor is 2 because one cell is not an animation; the ceiling is what the skill
  // measured — asked for 16, the repaint returns near-duplicates and the QC report says so.
  const n = whole(value, "the cell count");
  if (n < 2 || n > 24) throw new HttpError(400, "the cell count must be between 2 and 24");
  return n;
}

// The pack's cache key, which is the only thing that decides whether a param edit owes a repack.
// Every field the upstream keys deliberately exclude — fps, loop, hold_key, unify, the pivot,
// every hold and nudge — lands in it, so asking it is asking "did the atlas just change".
function packKey(man) {
  const s = (man.steps || []).find((x) => x.id === "pack");
  return s ? s.key : null;
}

// A cell's position WITHIN ITS TAG, which is what the page counts and what every endpoint takes.
// The atlas index is a different number and is never accepted here.
function cellIndex(man, tagName, req) {
  const tag = man.tags[tagIndex(man, tagName)];
  const n = tag.to - tag.from + 1;
  const i = whole(req.index, "index");
  if (!(i >= 0 && i < n)) {
    throw new HttpError(400, `cell ${i + 1} is not in the ${tagName} tag — it has ${n}`);
  }
  return i;
}

// The seed a repaint filename encodes. repaint_cells.repaint_path() writes the base seed with no
// suffix at all, so an unsuffixed name is seed 77 and not "unknown".
function seedOfRepaint(png) {
  const stem = png.split("/").pop().replace(/\.png$/, "").split("-s");
  return stem.length > 1 ? parseInt(stem[stem.length - 1], 10) : BASE_SEED;
}

// Where a source frame sits IN THE CLIP, read off the file name `pick_frames` extracted it under
// (`f_0011.png`). This is the only address a hand edit can survive a re-pick by: the manifest
// position moves whenever the cell count changes, and the file HASH changes whenever the clip is
// re-rendered, but the time a pose happens at is the one thing a re-pick of the same clip keeps.
// A name that does not parse returns null and is then matched by nothing, which is the safe
// answer — an unmatched frame is orphaned visibly rather than paired with a guess.
function frameTime(src) {
  const m = /(?:^|\/)f_(\d+)\.png$/.exec(src || "");
  return m ? parseInt(m[1], 10) : null;
}

// One frame parked in the tray or retired to the dropped ledger. It carries the whole frame, not
// just the edit, because the tray has to say WHICH frame an edit came from before a person can
// decide where it should go — and because a restore must be able to show the drawing it is
// offering to move.
function atticEntry(f, tag, why, at) {
  return {
    fid: f._fid,
    tag,
    src: f.src,
    seed: f.seed,
    png: f.png ?? null,
    hold: f.hold ?? 1,
    pivot_nudge: f.pivot_nudge ?? [0, 0],
    why,
    at,
  };
}

// Does this frame carry a HAND EDIT, as opposed to being exactly what the picker chose? The tray
// keeps every unmatched frame either way — a frame is identity, not just its edits — but the
// receipt and the tray both need to say how many of them a person actually touched.
function isHandEdited(f) {
  return (f.seed ?? BASE_SEED) !== BASE_SEED || (f.hold ?? 1) > 1 ||
    (f.pivot_nudge ?? [0, 0])[0] !== 0 || (f.pivot_nudge ?? [0, 0])[1] !== 0;
}

// Splice one tag's frame list and renumber every tag after it. Dropping or adding a frame moves
// the whole flat list along; a stale range points at the neighbouring tag's cells and the packer
// then emits another move's drawings under this tag's name.
function editTagFrames(man, tagName, mutate, rev) {
  const i = tagIndex(man, tagName);
  const tag = man.tags[i];
  const span = tag.to + 1 - tag.from;
  const frames = mutate(man.frames.slice(tag.from, tag.to + 1).map((f) => ({ ...f })));
  if (!frames.length) throw new HttpError(400, "a tag cannot have zero frames");
  const next = frames.map((f) => frameRecord(f, rev));
  man.frames.splice(tag.from, span, ...next);
  const shift = next.length - span;
  tag.to += shift;
  for (const later of man.tags.slice(i + 1)) {
    later.from += shift;
    later.to += shift;
  }
}

// The frame a cell index names, as it stands before an edit. Captured as the edit's INVERSE, so a
// cancel can put the drawing back without a second trip to the GPU.
function frameInverse(f) {
  return {
    k: "frame",
    fid: f._fid,
    src: f.src,
    seed: f.seed,
    png: f.png,
    hold: f.hold ?? 1,
    pivot_nudge: f.pivot_nudge ?? [0, 0],
  };
}

// Undo one recorded edit. Everything is addressed by `_fid`, never by index: by the time a
// cancel arrives, later edits in the same queue may have dropped or reordered the frames around
// it, and an index-addressed undo would restore the wrong cell.
//
// `attic` is the side channel for the two buckets that are NOT in the manifest — the orphan tray
// and the dropped ledger. They live in Durable Object storage, this function is synchronous, and
// an undo that put a frame back without also taking it out of the dropped ledger would leave that
// id live AND dropped, which is the exact corruption the identity assertion refuses to write.
function applyInverse(man, inv, rev, attic) {
  if (inv.k === "attic") {
    // A restore or a hand-collection moved a tray entry into the dropped ledger and changed
    // nothing else. Putting it back is the whole undo.
    attic.dropRemove.push(inv.fid);
    attic.trayAdd.push(inv.entry);
    return;
  }
  if (inv.k === "frame") {
    const at = man.frames.findIndex((f) => f._fid === inv.fid);
    // A later edit dropped the frame this one touched. There is nothing to restore, and
    // re-inserting it would resurrect a cell the person deliberately removed.
    if (at < 0) return;
    man.frames[at] = frameRecord({ ...inv, _fid: inv.fid, _rev: rev }, rev);
    return;
  }
  if (inv.k === "insert") {
    editTagFrames(
      man,
      inv.tag,
      (frames) => {
        frames.splice(Math.min(inv.at, frames.length), 0, inv.frame);
        return frames;
      },
      rev,
    );
    // The drop that this undoes wrote the frame into the dropped ledger, so that its id stayed
    // accounted for while it was out of the manifest. It is live again now.
    attic.dropRemove.push(inv.frame._fid);
    return;
  }
  if (inv.k === "order") {
    editTagFrames(
      man,
      inv.tag,
      (frames) => {
        const by = new Map(frames.map((f) => [f._fid, f]));
        const back = inv.fids.map((id) => by.get(id)).filter(Boolean);
        // A later drop or insert changed which frames the tag holds, so the recorded order is no
        // longer a permutation of it. Leaving the current order alone is the only safe answer.
        return back.length === frames.length ? back : frames;
      },
      rev,
    );
    return;
  }
  if (inv.k === "sheetPivot") {
    man.pivot = inv.pivot;
    return;
  }
  // One step param, put back where it came from. Addressed by tag NAME rather than by index for
  // the same reason a frame is addressed by `_fid`: an edit queued behind this one can have
  // spliced the frame list, and nothing here may assume a position survived it.
  if (inv.k === "stepParams") {
    if (inv.field === "trait") { man.trait = inv.value; return; }
    const at = man.tags[tagIndex(man, inv.tag)];
    if (at) at[inv.field] = inv.value;
    return;
  }
  const t = man.tags[tagIndex(man, inv.tag)];
  if (t) {
    t.fps = inv.fps;
    t.direction = inv.direction;
  }
}


// --- the pipeline's step records ---------------------------------------------------------------
// A mirror of pipeline.py, and it has to live here: this object owns every mutation, so it is the
// only thing that can say what an edit invalidated without a round trip to a machine that may be
// asleep. `canon()` and `digest()` are copied from canon.py rule for rule, and
// test_pipeline_keys.py runs the two implementations over the real manifests and fails on one
// digit of disagreement.
//
// What this side CANNOT do is read a file, so every artifact hash is carried forward from the
// record the agent reported. A step with no previous record therefore has no hash and no
// built_key, and reads "never built" — which for a re-roll's brand-new drawing is exactly true.
//
// The four functions below are exported only so that test can import them. celld loads this module
// for its `default` export and the `CharacterDO` class; the rest is invisible to it.

export function canon(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    // Python's json cannot emit these and neither should we: a NaN in a key would be a key that
    // never equals itself, so every step carrying it would read stale forever.
    if (!Number.isFinite(value)) throw new HttpError(500, `${value} has no canonical form`);
    // Number.isInteger(1.0) is true, so 1.0 emits as "1" here and str(int(1.0)) emits "1" there.
    // Left to JSON.stringify this would be "1" against Python's "1.0", and cfg=1.0 is a real
    // repaint param — the two sides would have disagreed about every repaint's key.
    return String(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(canon).join(",") + "]";
  if (typeof value === "object") {
    return "{" + Object.keys(value).sort()
      .map((k) => `${canon(k)}:${canon(value[k])}`).join(",") + "}";
  }
  throw new HttpError(500, `${typeof value} has no canonical form`);
}

export async function digest(value) {
  const bytes = new TextEncoder().encode(canon(value));
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...hash].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

// One incoming edge: which step it came from, which file was read, and that file's hash. Only the
// hash is key material — `from` and `path` both carry the TAG NAME, and a tag name in a cache key
// would mean renaming `walk` re-keys its picks, its ten repaints and the pack.
function edge(from, path, hash) {
  return { from, path: path ?? null, hash: hash ?? null };
}

async function stepKey(version, kind, params, inputs) {
  return await digest([version, kind, params, inputs.map((e) => e.hash)]);
}

// pipeline.frames_dir / pipeline.repaint_id, duplicated for the same reason repaintPath() below
// already is: a mutation has to name the step it just invalidated without asking the agent.
function framesDir(cid, move) {
  return `/tmp/sprite-${cid}-${move}-pick`;
}

function repaintId(png) {
  return `repaint:${png.split("/").pop().replace(/\.png$/, "")}`;
}

// Every step instance for one character, recomputed from the manifest it is called with. `prior`
// is the last set, and it is where every artifact hash and every built_key comes from: this side
// recomputes what an edit MEANS, never what was built.
export async function buildSteps(man, prior) {
  const was = new Map((prior || []).map((s) => [s.id, s]));
  const build = man.build;
  const version = build.template_version;
  const steps = [];

  // Source-frame hashes, recovered from the repaint steps that already consumed them. A re-roll
  // keeps the source frame and only changes the seed, so its new step finds the hash here and
  // keys correctly; a re-pick to a frame nothing has painted yet finds nothing, keys against null,
  // and reads "never built" until the agent reports the real hash back.
  const srcHash = new Map();
  for (const s of was.values()) {
    if (s.kind === "repaint" && s.inputs[0] && s.inputs[0].path) {
      srcHash.set(s.inputs[0].path, s.inputs[0].hash);
    }
  }

  const add = async (id, kind, tag, params, inputs, artifact, artifactHash, current) => {
    const had = was.get(id);
    const rec = {
      id, kind, tag, params, inputs,
      key: await stepKey(version, kind, params, inputs),
      built_key: null,
      artifact: artifact ?? had?.artifact ?? null,
      artifact_hash: artifactHash ?? null,
      cost_s: build.cost_s[kind],
    };
    if (current !== undefined) rec.built_key = current ? rec.key : null;
    else if (had) rec.built_key = had.built_key ?? null;
    steps.push(rec);
    return rec;
  };

  const plate = was.get("plate");
  const plateHash = plate?.artifact_hash ?? null;
  const plateStep = await add("plate", "plate", null, {},
    [edge(null, plate?.artifact, plateHash)], plate?.artifact, plateHash, plateHash !== null);

  for (const tag of man.tags) {
    const move = tag.name;
    const clipId = `clip:${move}`, framesId = `frames:${move}`, picksId = `picks:${move}`;
    const frames = man.frames.slice(tag.from, tag.to + 1);

    const clip = await add(clipId, "clip", move,
      { recipe: tag.recipe, recipe_text: tag.recipe_text, trait: man.trait, cyclic: tag.cyclic,
        ...build.clip },
      [edge("plate", plateStep.artifact, plateHash)], tag.clip, was.get(clipId)?.artifact_hash);

    const dir = was.get(framesId)?.artifact ?? framesDir(man.character, move);
    const fr = await add(framesId, "frames", move, { skip: build.picks.skip },
      [edge(clipId, tag.clip, clip.artifact_hash)], dir, was.get(framesId)?.artifact_hash);

    // `cycle` and `stop` are facts about the MOTION — is there a gait to detect, and does the move
    // end on a held pose. They are not the tag's `loop` and `hold_key`, which are what the game
    // does at playback, and keeping them apart is what lets a re-tag leave the picks alone.
    const srcs = [...new Set(frames.map((f) => f.src))].sort();
    await add(picksId, "picks", move,
      { cells: tag.cells, cycle: tag.cyclic, stop: tag.hold_key,
        anchor: build.picks.anchor, smooth: build.picks.smooth },
      [edge(framesId, fr.artifact, fr.artifact_hash)],
      was.get(picksId)?.artifact, await digest(srcs));

    for (const f of frames) {
      const png = f.png ?? repaintPath(man.character, move, f.src, f.seed);
      const id = repaintId(png);
      await add(id, "repaint", move, { seed: f.seed, ...build.repaint },
        [edge(picksId, f.src, srcHash.get(f.src) ?? null)], png,
        was.get(id)?.artifact_hash);
    }
  }

  // Every field the key deliberately keeps out of the steps above lands here, because the pack is
  // the step that actually reads it: fps, direction, loop, hold_key, unify, the pivot and every
  // cell's hold and nudge.
  const repaints = steps.filter((s) => s.kind === "repaint");
  const pack = await add("pack", "pack", null,
    { cell: man.cell, pivot: man.pivot, ...build.pack,
      tags: man.tags.map((t) => ({
        name: t.name, fps: t.fps, direction: t.direction, loop: t.loop, hold_key: t.hold_key,
        unify: t.unify,
        cells: man.frames.slice(t.from, t.to + 1).map((f) => ({
          hold: f.hold ?? 1, pivot_nudge: f.pivot_nudge ?? [0, 0],
        })),
      })) },
    repaints.map((s) => edge(s.id, s.artifact, s.artifact_hash)),
    was.get("pack")?.artifact, was.get("pack")?.artifact_hash);

  // Which export formats exist is the agent's knowledge (exports.FORMATS), not this object's, so
  // the instances are carried from the last report rather than invented here. An export is
  // computed on request straight off the atlas and streamed, so it is built exactly when the
  // atlas exists and goes stale only by inheriting a stale pack.
  for (const id of (prior || []).filter((s) => s.kind === "export").map((s) => s.id)) {
    await add(id, "export", null, { format: id.slice("export:".length) },
      [edge("pack", pack.artifact, pack.artifact_hash)], null, null,
      pack.artifact_hash !== null);
  }
  return steps;
}

// `stale` and why, for every step, propagated down every edge. Two independent reasons and both
// are needed: a step goes stale on its OWN key when a param or an input's content changed, and by
// INHERITANCE when an upstream step is stale but has not been rebuilt — a re-worded recipe leaves
// the old mp4 byte-identical, so no hash downstream of it moves and only the propagation carries
// the news. Derived on read, never stored: a stored flag is a second answer free to disagree.
export function staleness(steps) {
  const out = {};
  const known = new Set(steps.map((s) => s.id));
  for (const s of steps) {
    if (s.built_key === null) out[s.id] = { stale: true, reason: "never built" };
    else if (s.key !== s.built_key) out[s.id] = { stale: true, reason: "params or inputs changed" };
    else {
      const behind = s.inputs.map((e) => e.from)
        .filter((u) => u && known.has(u) && out[u].stale);
      out[s.id] = behind.length
        ? { stale: true, reason: `upstream is stale: ${behind[0]}` }
        : { stale: false, reason: null };
    }
  }
  return out;
}


// --- the job queue ---------------------------------------------------------------------------

function isLive(job) {
  return LIVE.includes(job.state);
}

// What every job-shaped response says. `queuePos` is the job's place among the LIVE jobs, so 0 is
// the one the agent is allowed to work on and a finished job has none.
function view(job, pos, now) {
  const started = job.startedAt ?? null;
  return {
    id: job.id,
    label: job.label,
    // `preparing` is this object's internal lock and lasts for the length of one splice. It is
    // reported as `queued` because it is one: the edit is going in and no agent can take it.
    state: job.state === "preparing" ? "queued" : job.state,
    generates: job.generates,
    // Which agent job type this is. The page reads it to decide which rail chip is running, and
    // the agent reads it to decide which of the five things it knows how to do it should do.
    // A job written before stage 4 existed carries none and is a repack, which is what it was.
    kind: job.kind ?? "pack",
    tag: job.tag ?? null,
    index: job.index ?? null,
    queuePos: pos,
    startedAt: started,
    elapsedMs: started === null ? null : (job.finishedAt ?? now) - started,
    error: job.error ?? null,
  };
}

function viewAll(jobs, now) {
  let pos = 0;
  return jobs.map((j) => view(j, isLive(j) ? pos++ : null, now));
}

// --- the character cell ------------------------------------------------------------------------

export class CharacterDO {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
    // Agents parked on a claim. In memory only, and deliberately so: a lost waiter costs one
    // idle agent one poll interval, whereas persisting it would mean a job could be handed to
    // an agent that is no longer listening.
    this.waiters = [];
  }

  async manifest() {
    const man = (await this.ctx.storage.get("manifest")) || null;
    // Cells seeded before frames carried an identity have none. Stamping them on first read is
    // the whole migration, and it has to happen HERE rather than by re-seeding from disk: the
    // Durable Object is the record, and `sheets/<cid>.json` is only ever a copy of it.
    if (man && man.frames.some((f) => !f._fid)) {
      const rev = await this.revision();
      man.frames = man.frames.map((f) => frameRecord(f, rev));
      await this.saveManifest(man);
      // The single-slot lock this queue replaced. Left behind it would be a second, invisible
      // record of "what is running" that nothing reads and nothing clears.
      await this.ctx.storage.delete("job");
    }
    return man;
  }

  // --- frame identity, and the two places a frame can be when it is not live -------------------
  // DESIGN-pipeline.md's risk 1: "the orphan tray silently eats a hand edit on a half-matching
  // repick". Everything below exists to make that impossible to do quietly.
  //
  // Three buckets, and every `_fid` this object has ever written is in exactly one of them:
  //   LIVE     — a frame in `man.frames`, packed into the atlas.
  //   TRAY     — orphaned by a re-run that could not match it. Visible, restorable, and never
  //              collected by anything but a person.
  //   DROPPED  — retired on purpose, with the reason recorded: dropped by hand, restored onto
  //              another cell, or collected out of the tray. A re-pick consults this so a cell a
  //              person deliberately removed does not come back the next time the picker likes
  //              that pose again.

  async tray() {
    return (await this.ctx.storage.get("tray")) || [];
  }

  async dropped() {
    return (await this.ctx.storage.get("dropped")) || [];
  }

  // Apply one batch of attic moves. Batched rather than per-entry because a reconcile moves a
  // whole tag at once and two storage round trips per frame would be the slowest thing in the
  // object for no gain.
  async applyAttic(a) {
    if (!a) return;
    const gone = new Set([...(a.trayRemove || []), ...(a.dropRemove || [])]);
    const tray = (await this.tray()).filter((e) => !gone.has(e.fid)).concat(a.trayAdd || []);
    const dropped = (await this.dropped()).filter((e) => !gone.has(e.fid))
      .concat(a.dropAdd || []);
    await this.ctx.storage.put("tray", tray);
    await this.ctx.storage.put("dropped", dropped);
  }

  // THE INVARIANT, and the only door the manifest can be written through.
  //
  // Every id this object has ever persisted is remembered in `fids`. A write that leaves one of
  // them neither live, nor in the tray, nor dropped means a mutation has quietly deleted a frame
  // — and with it whatever hold, nudge or seed a person had put on it. That write is REFUSED,
  // loudly, with the ids named: a 500 on one edit is recoverable, an edit that vanished is not.
  //
  // The check runs before the ids of this write are folded in, so a brand-new frame never trips
  // it and a disappeared one always does.
  async assertIdentity(man) {
    const live = new Set();
    for (const f of man.frames) {
      if (!f._fid) throw new HttpError(500, "a frame reached storage with no identity");
      if (live.has(f._fid)) {
        throw new HttpError(500, `frame ${f._fid} is in the manifest twice — refusing the write`);
      }
      live.add(f._fid);
    }
    const held = new Map();
    for (const e of await this.tray()) held.set(e.fid, "the tray");
    for (const e of await this.dropped()) held.set(e.fid, "the dropped ledger");
    for (const [fid, where] of held) {
      if (live.has(fid)) {
        throw new HttpError(500, `frame ${fid} is live AND in ${where} — refusing the write`);
      }
    }
    const known = (await this.ctx.storage.get("fids")) || [];
    const lost = known.filter((id) => !live.has(id) && !held.has(id));
    if (lost.length) {
      throw new HttpError(
        500,
        `refusing the write: ${lost.length} frame${lost.length === 1 ? "" : "s"} ` +
        `(${lost.slice(0, 6).join(", ")}) would be neither live, nor in the tray, nor dropped — ` +
        "a hand edit was about to be lost",
      );
    }
    // Only rewrite the ledger when it actually grew. Every hold typed into the page comes through
    // here, and a storage write per keystroke would buy nothing.
    const next = [...new Set([...known, ...live])];
    if (next.length !== known.length) await this.ctx.storage.put("fids", next);
  }

  // Forget the DRAWINGS of one tag's tray entries, because they are no longer on disk.
  //
  // A re-extraction that changed the bytes deletes every repaint of that move: the packer caches
  // by file name, so `f_0011.png` of the new clip would otherwise be served the old clip's
  // picture. The tray outlives those files on purpose — the facts are the payload, and a hold of
  // 4 is still restorable with nothing left to look at — but a tray entry still naming a deleted
  // file makes the page ask the agent for it and take a 404 on every open.
  async clearTrayArt(tag) {
    const tray = await this.tray();
    let hit = false;
    for (const e of tray) {
      if (e.tag === tag && e.png) { e.png = null; hit = true; }
    }
    if (hit) await this.ctx.storage.put("tray", tray);
  }

  async saveManifest(man) {
    await this.assertIdentity(man);
    await this.ctx.storage.put("manifest", man);
  }

  // Recompute every step's params, inputs and key from the manifest as it now stands. `built_key`
  // and every artifact hash are carried, never invented: this object cannot read a file, and
  // stamping "built" from an edit would erase the staleness the edit just created.
  async refreshSteps(man) {
    if (!man.build) return;   // seeded before the pipeline records existed; the agent tops it up
    man.steps = await buildSteps(man, man.steps);
  }

  // --- the roster ------------------------------------------------------------------------
  // Only the reserved ROSTER cell answers these. It is an ordinary CharacterDO with no manifest,
  // used as the namespace's own index, because a Durable Object namespace cannot be listed and a
  // second store would be free to disagree with the cells it claims to describe.

  async roster() {
    return (await this.ctx.storage.get("roster")) || [];
  }

  // Append-only, and ordered by when each character was first seeded, so the page's first
  // character does not move under it when another is added.
  async rosterAdd(cid) {
    const list = await this.roster();
    if (!list.includes(cid)) {
      list.push(cid);
      await this.ctx.storage.put("roster", list);
    }
    return list;
  }

  async catalogue() {
    return (await this.ctx.storage.get("catalogue")) || { sources: {}, variants: {} };
  }

  async revision() {
    return (await this.ctx.storage.get("revision")) || 0;
  }

  // Every read of the queue ages it first: leases are enforced and finished jobs fall out of
  // history. Doing it on READ rather than from an alarm means a dead agent's job is reaped by
  // the next person who looks, which is the only moment it matters.
  async jobs() {
    const jobs = (await this.ctx.storage.get("jobs")) || [];
    const now = Date.now();
    const head = jobs.find(isLive);
    let changed = false;
    if (head && now > head.deadline) {
      if (head.state === "claimed") {
        // Claimed and then silent means the agent never STARTED it — `run_job` reports progress
        // within a second of claiming. So the work was not done, and the edit is already in the
        // manifest: queue it again rather than fail it, and the next agent to poll packs it.
        head.state = "queued";
        head.startedAt = null;
        head.deadline = now + LEASE_MS;
      } else if (head.state === "running") {
        head.state = "failed";
        head.error = "the agent stopped reporting; lease expired";
        head.finishedAt = now;
      } else if (head.state === "preparing") {
        // One splice long. Reaching here means the request that took the lock died inside it, so
        // the manifest was never written and the slot must not be held for the whole lease.
        head.state = "failed";
        head.error = "the edit was never completed";
        head.finishedAt = now;
      }
      changed = true;
    }
    const kept = jobs.filter((j) => isLive(j) || now - j.finishedAt < HISTORY_MS);
    if (changed || kept.length !== jobs.length) await this.ctx.storage.put("jobs", kept);
    return kept;
  }

  async putJobs(jobs) {
    await this.ctx.storage.put("jobs", jobs);
  }

  // Read-modify-write of the queue with no await between the read and the write, which is what
  // makes it a lock rather than a suggestion.
  async editJobs(fn) {
    const jobs = await this.jobs();
    const out = fn(jobs);
    await this.putJobs(jobs);
    return out;
  }

  liveWaiter() {
    // Sweep waiters that have outlived their own wait. Their `setTimeout` should have done it,
    // but celld does not run the timers of a request whose client has gone: an agent that
    // restarts while parked leaves a waiter that looks live and can never answer, and the next
    // job handed to it disappeared — the character then sat locked for the whole lease with
    // nobody running anything. Measured on this fleet, twice, during the cutover. Every claim
    // and every handoff sweeps, so a leak can outlive at most one poll.
    const now = Date.now();
    this.waiters = this.waiters.filter((w) => w.until > now);
    return this.waiters.shift() || null;
  }

  // Hand the head of the queue to a parked agent, if there is one and it is claimable. Called
  // whenever the head can have changed, so an edit does not wait out the poll interval it landed
  // in the middle of, and the next job starts the instant the last one reports.
  async pump() {
    const now = Date.now();
    const jobs = await this.jobs();
    const head = jobs.find(isLive);
    if (!head || head.state !== "queued") return;
    const waiter = this.liveWaiter();
    if (!waiter) return;
    this.give(head, waiter.agent, now);
    await this.putJobs(jobs);
    waiter.resolve(view(head, 0, now));
  }

  give(job, agent, now) {
    job.state = "claimed";
    job.agent = agent;
    job.startedAt = now;
    job.deadline = now + LEASE_MS;
  }

  async claim(waitMs, agent) {
    const now = Date.now();
    this.waiters = this.waiters.filter((w) => w.until > now);
    const jobs = await this.jobs();
    const head = jobs.find(isLive);
    // A job claimed by a DIFFERENT agent process than the one now asking cannot still be running:
    // there is one agent for this fleet, so a claim from a new process id means the old one died
    // holding it. Without this the queue stalled for the full ten-minute lease after every agent
    // restart — the handoff had gone to a waiter whose request had already been dropped, which
    // celld does not tell the object about. The work was never started, so it is requeued, not
    // failed. Note the assumption this rests on: exactly one agent process per fleet.
    if (head && head.state === "claimed" && head.agent !== agent) {
      if (head.cancelRequested) {
        // Asked to stop, and the agent that would have been told is gone. Running it now would
        // be the queue doing the one thing it was told not to.
        head.state = "cancelled";
        head.finishedAt = now;
      } else {
        head.state = "queued";
        head.startedAt = null;
        head.deadline = now + LEASE_MS;
      }
      await this.putJobs(jobs);
    }
    // Strictly the head, and only if it is queued. That single rule is what keeps one GPU job
    // running at a time while everything behind it stays visible and cancellable.
    const next = jobs.find(isLive);
    if (next && next.state === "queued") {
      this.give(next, agent, now);
      await this.putJobs(jobs);
      return view(next, 0, now);
    }
    if (waitMs <= 0) return null;
    return await new Promise((resolve) => {
      const waiter = { until: Date.now() + waitMs, agent };
      waiter.timer = setTimeout(() => {
        this.waiters = this.waiters.filter((w) => w !== waiter);
        resolve(null);
      }, waitMs);
      waiter.resolve = (j) => {
        clearTimeout(waiter.timer);
        this.waiters = this.waiters.filter((w) => w !== waiter);
        resolve(j);
      };
      this.waiters.push(waiter);
    });
  }

  async finish(id, state, error) {
    const now = Date.now();
    await this.editJobs((jobs) => {
      const job = jobs.find((j) => j.id === id);
      if (!job || !isLive(job)) return;
      // The agent reports a plain failure either way; whether that failure was ASKED FOR is this
      // object's own knowledge, so it decides the word rather than trusting the report.
      job.state = state === "failed" && job.cancelRequested ? "cancelled" : state;
      job.error = error ?? null;
      job.finishedAt = now;
      if (job.startedAt === null) job.startedAt = now;
    });
    await this.pump();
  }

  // The agent reports the manifest it actually packed, and the frame list it was handed when it
  // started. Only `png` is taken from it: repaint_cells.main() fills those in as it repaints, and
  // dropping them would leave the DO pointing at drawings it thinks do not exist yet.
  //
  // The acceptance is PER FRAME, keyed on that frame's identity and drawing revision. The old
  // check compared the whole manifest and refused the report if anything at all differed — which
  // meant a hold typed during a 45-second paint failed a paint that was perfectly correct when it
  // started, and the drawing on disk was then invisible to the editor. A frame the agent packed
  // and nobody has touched since is still exactly the frame it packed, wherever it has since been
  // moved to; a frame that HAS been re-rolled or re-picked since has a newer drawing coming and
  // must not take the old one's path.
  async absorbPacked(packed, meta) {
    const man = await this.manifest();
    if (!man) throw new HttpError(409, "no manifest to absorb into");
    if (!packed || !Array.isArray(packed.frames) || !Array.isArray(meta) ||
        meta.length !== packed.frames.length) {
      throw new HttpError(409, "the packed report does not match the frame list it was given");
    }
    const painted = new Map();
    packed.frames.forEach((f, i) => painted.set(meta[i].fid, { rev: meta[i].rev, png: f.png }));
    for (const f of man.frames) {
      const p = painted.get(f._fid);
      if (p && p.rev === f._rev && typeof p.png === "string" && p.png) f.png = p.png;
    }
    // Rebuilt, not just assigned: a frame that had no `png` would otherwise carry it as its LAST
    // key, and this manifest is written verbatim to sheets/<cid>.json — the whole file would
    // re-order for no change in meaning.
    man.frames = man.frames.map((f) => frameRecord(f, f._rev));
    // The BUILD REPORT. Every artifact hash and every built_key is the packer's knowledge — it is
    // the half of this system that can open a file — so those are taken as given and the keys are
    // recomputed on top of them. The locked constants come with it because they are owned by
    // repaint_cells.py; the trait line is NOT, because that is data this object owns.
    if (packed.build) man.build = packed.build;
    if (Array.isArray(packed.steps)) man.steps = packed.steps;
    await this.refreshSteps(man);
    // Which extraction each tag's live cells were picked out of, backfilled from the build report
    // the first time this object ever sees one. Before stage 4 there was no job that could
    // re-extract a clip, so a tag's picks were necessarily made against the frames directory as
    // it stands — and without this backfill the FIRST re-pick of every existing move would decide
    // it could not match anything and orphan a whole tag that nobody had changed.
    const from = (await this.ctx.storage.get("pickedFrom")) || {};
    let learned = false;
    for (const s of man.steps || []) {
      if (s.kind !== "frames" || from[s.tag] !== undefined || s.artifact_hash === null) continue;
      from[s.tag] = s.artifact_hash;
      learned = true;
    }
    if (learned) await this.ctx.storage.put("pickedFrom", from);
    await this.saveManifest(man);
  }

  // --- a re-pick, and what happens to the hand edits under it ---------------------------------
  // The heart of DESIGN-pipeline.md's "hand edits under a re-run". The agent has just asked the
  // picker for a fresh set of cells off this tag's clip; it can choose frames but it cannot
  // decide what that means for the edits already on them, because the edits are this object's.
  //
  // THE MATCHING RULE, exactly as implemented:
  //
  //   * The picker's choice is addressed by the source frame's TIME in the clip — the `f_NNNN`
  //     index `pick_frames` extracted it under. Nothing else survives a re-pick: the manifest
  //     position moves with the cell count and the file hash moves with the clip.
  //   * A re-pick off the SAME extraction matches. Candidate pairs are every (new pick, old
  //     frame) within ±3 frames, taken smallest-distance-first, each side used once. That order
  //     matters: nearest-first per new pick in sequence lets an early pick take a frame a later
  //     one was a better fit for, and the edit then lands one pose off.
  //   * A match onto a LIVE frame carries its id, seed, hold and nudge onto the new pick. Same
  //     source frame means the same drawing too, so nothing is repainted; a moved one re-keys its
  //     repaint and the rail says so.
  //   * A match onto a DROPPED frame suppresses the new pick. A cell someone deliberately removed
  //     does not come back because the picker liked that pose again.
  //   * Everything unmatched goes to the TRAY. Nothing is deleted, ever.
  //   * A re-pick off a DIFFERENT extraction matches NOTHING. A re-rendered clip is a new
  //     performance and frame times no longer name the same poses, so the whole tag is orphaned
  //     to the tray and rebuilt from the fresh pick. Pretending otherwise is how a pivot nudge
  //     lands on the wrong pose.
  //
  // "The same extraction" is decided by content, not by a flag: `pickedFrom` records the hash of
  // the frames directory the live cells were picked out of, and the agent reports the hash it has
  // just picked from. Equal means the bytes on disk are the frames these edits were made against.
  async reconcile(tagName, srcs, framesHash) {
    const man = await this.manifest();
    if (!man) throw new HttpError(409, "this character has not been seeded yet");
    const ti = tagIndex(man, tagName);
    if (ti < 0) throw new HttpError(400, `no ${tagName} tag for ${man.character}`);
    if (!Array.isArray(srcs) || !srcs.length || srcs.some((s) => typeof s !== "string" || !s)) {
      throw new HttpError(400, "the pick report must be a non-empty list of source frame paths");
    }
    if (typeof framesHash !== "string" || !framesHash) {
      throw new HttpError(400, "the pick report must name the extraction it picked from");
    }
    const now = Date.now();
    const rev = (await this.revision()) + 1;
    const tag = man.tags[ti];
    const old = man.frames.slice(tag.from, tag.to + 1).map((f) => ({ ...f }));
    const pickedFrom = (await this.ctx.storage.get("pickedFrom")) || {};
    const same = pickedFrom[tagName] === framesHash;
    const retired = same
      ? (await this.dropped()).filter((e) => e.tag === tagName)
      : [];

    // One flat candidate list so both kinds of old frame compete on the same footing: a new pick
    // nearer a dropped frame than a live one must take the drop's answer, or the suppression is
    // decided by which list happened to be searched first.
    const cand = same
      ? [...old.map((f, k) => ({ live: f, t: frameTime(f.src), k })),
         ...retired.map((e, k) => ({ dead: e, t: frameTime(e.src), k: old.length + k }))]
        .filter((c) => c.t !== null)
      : [];
    const want = srcs.map((s, n) => ({ src: s, t: frameTime(s), n }));
    const pairs = [];
    for (const w of want) {
      if (w.t === null) continue;
      for (const c of cand) {
        const d = Math.abs(c.t - w.t);
        if (d <= MATCH_FRAMES) pairs.push({ n: w.n, k: c.k, d, c });
      }
    }
    pairs.sort((a, b) => a.d - b.d || a.n - b.n || a.k - b.k);
    const byNew = new Map(), usedOld = new Set();
    for (const p of pairs) {
      if (byNew.has(p.n) || usedOld.has(p.k)) continue;
      byNew.set(p.n, p.c);
      usedOld.add(p.k);
    }

    const next = [], kept = new Set(), suppressed = [];
    for (const w of want) {
      const m = byNew.get(w.n);
      if (m && m.dead) { suppressed.push(w.src); continue; }
      if (m && m.live) {
        const f = m.live;
        kept.add(f._fid);
        const moved = f.src !== w.src;
        next.push({
          src: w.src,
          seed: f.seed,
          // A pick that did not move keeps the drawing it already has, byte for byte. One that
          // moved names the drawing of its NEW source frame, which does not exist yet — so the
          // repaint reads never built and the rail prices it before anything runs.
          png: moved ? repaintPath(man.character, tagName, w.src, f.seed) : f.png,
          hold: f.hold,
          pivot_nudge: f.pivot_nudge,
          _fid: f._fid,
          _rev: moved ? rev : f._rev,
        });
        continue;
      }
      next.push({ src: w.src, seed: BASE_SEED, hold: 1, pivot_nudge: [0, 0], _rev: rev });
    }
    // Every dropped frame in the tag can out-compete every new pick, which would leave the tag
    // with nothing and `editTagFrames` refusing the splice. A tag with no cells is not a state
    // this editor has, so the first suppression is released and said out loud instead.
    let unsuppressed = null;
    if (!next.length) {
      unsuppressed = suppressed[0];
      next.push({ src: unsuppressed, seed: BASE_SEED, hold: 1, pivot_nudge: [0, 0], _rev: rev });
    }

    const orphans = old.filter((f) => !kept.has(f._fid));
    const why = same ? "unmatched by a re-pick of the same clip"
                     : "orphaned by a clip re-run — a new clip is a new performance";
    editTagFrames(man, tagName, () => next, rev);
    await this.applyAttic({ trayAdd: orphans.map((f) => atticEntry(f, tagName, why, now)) });
    pickedFrom[tagName] = framesHash;
    await this.ctx.storage.put("pickedFrom", pickedFrom);
    await this.refreshSteps(man);
    await this.saveManifest(man);
    await this.ctx.storage.put("revision", rev);
    return {
      matched: kept.size,
      cells: next.length,
      orphaned: orphans.length,
      edits_orphaned: orphans.filter(isHandEdited).length,
      suppressed: suppressed.length,
      unsuppressed,
      carried_over: same,
      revision: rev,
    };
  }

  // --- mutations -------------------------------------------------------------------------
  // Each one validates, splices in place, saves, and only then publishes the job. The manifest
  // is written before the repack for the reason the Python editor kept the same order: a repack
  // can be killed, and if it is, the edit must still be recorded so the next run picks it up.

  async mutate(op, req) {
    const generates = GENERATES[op];
    // The lock is genuinely HELD across the splice, not merely checked before it. An earlier
    // version checked here and took the slot at the end, and the gap between the two was only
    // safe under Cloudflare's input-gate semantics, which cannot be proven on celld from
    // outside: two mutations fired at once gave one job and TWO edits, and the loser's edit rode
    // along on the winner's repack under a 409 that said it had been refused. Measured. So the
    // queue record is written FIRST, in the unclaimable `preparing` state, and every later await
    // is inside it.
    const taken = await this.editJobs((jobs) => {
      const live = jobs.filter(isLive);
      // A cheap edit joins a repack that is queued but not yet claimed. The agent re-reads the
      // manifest when it starts a job, so that one repack will carry this edit and every other
      // one that lands before it is claimed. Joining a CLAIMED repack would be a lie — its agent
      // already has its copy of the manifest.
      const join = generates ? null : live.find((j) => !j.generates && j.state === "queued");
      if (join) {
        join.state = "preparing";
        return { job: join, joined: true };
      }
      if (live.length >= MAX_PENDING) {
        throw new HttpError(
          409,
          `refused — ${live.length} jobs are already queued on this character, which is the limit`,
        );
      }
      const job = {
        id: shortId(),
        label: `preparing ${op}`,
        generates,
        // Every mutation is answered by a repack, whether it painted a cell first or not: the
        // atlas is what the edit was for.
        kind: "pack",
        tag: req.tag ?? null,
        index: null,
        state: "preparing",
        agent: null,
        error: null,
        edits: [],
        cancelRequested: false,
        startedAt: null,
        deadline: Date.now() + LEASE_MS,
        finishedAt: null,
      };
      jobs.push(job);
      return { job, joined: false };
    });

    const rev = (await this.revision()) + 1;
    try {
      const man = await this.manifest();
      if (!man) throw new HttpError(409, "this character has not been seeded yet");
      const wasPack = packKey(man);
      const tagName = req.tag;
      if (tagIndex(man, tagName) < 0) {
        throw new HttpError(400, `no ${tagName} tag for ${man.character}`);
      }
      const done = await this[op](man, tagName, req, rev);
      // The tray and the dropped ledger move BEFORE the manifest is written, because the identity
      // assertion in `saveManifest` reads them: a drop that spliced the frame out and recorded it
      // afterwards would be refused by its own bookkeeping halfway through.
      await this.applyAttic(done.attic);
      // The step records are recomputed from the manifest this edit just produced. Nothing here
      // runs, queues or prices anything as a result — a step that has gone stale is drawn stale
      // and stays that way until someone asks for it (DESIGN-pipeline.md, stage 1).
      await this.refreshSteps(man);
      await this.saveManifest(man);
      await this.ctx.storage.put("revision", rev);
      // An edit that only made things STALE queues nothing. Re-wording a recipe invalidates a
      // 170-second render this fleet has no job type for, and a repack would not answer it — it
      // would just be the one-click cascade DESIGN-pipeline.md forbids, wearing a repack's
      // clothes. The pack's key is the test because it is where every field a pack really reads
      // lands: unchanged means nothing runnable changed.
      if (done.norun && packKey(man) === wasPack) {
        await this.releaseJob(taken);
        await this.pump();
        return { job: null, revision: rev };
      }
      return await this.publish(taken, done, tagName, rev);
    } catch (e) {
      // A rejected edit must not leave a job in the queue, and must not leave a repack that other
      // edits have already joined half-cancelled. `man` is a fresh copy out of storage and was
      // never written, so backing the record out drops the whole attempt.
      await this.releaseJob(taken);
      throw e;
    }
  }

  // Give the queue slot back, whether the edit was refused or simply owed no work. A slot this
  // object took and then kept would be a job nobody can cancel, because it names no edit.
  async releaseJob(taken) {
    await this.editJobs((jobs) => {
      const at = jobs.findIndex((j) => j.id === taken.job.id);
      if (at < 0) return;
      if (taken.joined) jobs[at].state = "queued";
      else jobs.splice(at, 1);
    });
  }

  async publish(taken, done, tagName, rev) {
    const now = Date.now();
    const id = taken.job.id;
    done.inverse.label = done.text;
    await this.editJobs((jobs) => {
      const job = jobs.find((j) => j.id === id);
      job.state = "queued";
      job.tag = tagName;
      job.index = done.index ?? null;
      job.edits.push(done.inverse);
      // A restore moves a tray entry as well as a cell, so its undo is two inverses and not one.
      // They are pushed in the order they must be UNDONE last-first, which `cancel` reverses.
      for (const extra of done.extraInverses || []) job.edits.push({ ...extra, label: done.text });
      // A coalesced repack is named after the edit that created it, plus a count. The individual
      // texts are kept on the inverses, which is where a cancel needs them anyway.
      job.label = taken.joined ? `${job.edits[0].label} +${job.edits.length - 1} more` : done.text;
      job.deadline = now + LEASE_MS;
    });
    // Read the job back only AFTER the handoff. An agent parked on a claim takes it inside
    // `pump()`, so a view built before this call would tell the page "queued" about a job that is
    // already running, and the page would then show the wrong thing until its next poll.
    await this.pump();
    const jobs = await this.jobs();
    const at = jobs.findIndex((j) => j.id === id);
    return { job: viewAll(jobs, Date.now())[at], revision: rev };
  }

  // Cancel is first-class because a re-roll is an experiment. A job that has not started yet is
  // removed and its recorded edits are re-spliced in reverse; one that is already with the agent
  // can only be ASKED to stop, which it reads from its next heartbeat.
  async cancel(id) {
    const now = Date.now();
    const outcome = await this.editJobs((jobs) => {
      const job = jobs.find((j) => j.id === id);
      if (!job) throw new HttpError(404, "no job with that id");
      if (!isLive(job)) throw new HttpError(409, `that job has already ${job.state}`);
      if (job.state === "preparing") throw new HttpError(409, "that job is still being prepared");
      if (job.state !== "queued") {
        job.cancelRequested = true;
        return null;
      }
      // Marked finished BEFORE the manifest is touched, for the same reason a mutation takes its
      // slot first: from this instant no agent can claim it and no second cancel can double-apply
      // the inverse.
      job.state = "cancelled";
      job.error = null;
      job.finishedAt = now;
      if (job.startedAt === null) job.startedAt = now;
      return job.edits;
    });
    if (!outcome) return { ok: true, reverted: false };
    const rev = (await this.revision()) + 1;
    const man = await this.manifest();
    const attic = { trayAdd: [], trayRemove: [], dropAdd: [], dropRemove: [] };
    for (const inv of [...outcome].reverse()) applyInverse(man, inv, rev, attic);
    await this.applyAttic(attic);
    await this.saveManifest(man);
    await this.ctx.storage.put("revision", rev);
    await this.pump();
    return { ok: true, reverted: true, revision: rev };
  }

  async reroll(man, tagName, req, rev) {
    const i = cellIndex(man, tagName, req);
    const seed =
      req.seed === undefined || req.seed === null
        ? crypto.getRandomValues(new Uint32Array(1))[0] % 2147483647 + 1
        : whole(req.seed, "seed");
    let inverse;
    editTagFrames(
      man,
      tagName,
      (frames) => {
        const f = frames[i];
        if (seed === f.seed) throw new HttpError(400, "that is the seed the cell already has");
        inverse = frameInverse(f);
        f.seed = seed;
        f.png = repaintPath(man.character, tagName, f.src, seed);
        f._rev = rev;
        return frames;
      },
      rev,
    );
    return { text: `${man.character}-${tagName} cell ${i + 1} re-roll`, index: i, inverse };
  }

  async repick(man, tagName, req, rev) {
    const i = cellIndex(man, tagName, req);
    const src = req.src;
    const cat = await this.catalogue();
    const known = cat.sources[tagName] || [];
    if (!known.includes(src)) {
      throw new HttpError(400, "that source frame is not in this tag's pick directory");
    }
    let inverse;
    editTagFrames(
      man,
      tagName,
      (frames) => {
        const f = frames[i];
        inverse = frameInverse(f);
        f.src = src;
        f.png = repaintPath(man.character, tagName, src, f.seed);
        f._rev = rev;
        return frames;
      },
      rev,
    );
    return { text: `${man.character}-${tagName} cell ${i + 1} re-pick`, index: i, inverse };
  }

  async use(man, tagName, req, rev) {
    const i = cellIndex(man, tagName, req);
    const png = req.png;
    const cat = await this.catalogue();
    let inverse;
    editTagFrames(
      man,
      tagName,
      (frames) => {
        const f = frames[i];
        const known = (cat.variants[f.src] || []).map((v) => v.png);
        if (!known.includes(png)) {
          throw new HttpError(400, "that is not a variant of this cell's source frame");
        }
        inverse = frameInverse(f);
        f.png = png;
        f.seed = seedOfRepaint(png);
        f._rev = rev;
        return frames;
      },
      rev,
    );
    return { text: `${man.character}-${tagName} cell ${i + 1} use variant`, index: i, inverse };
  }

  async drop(man, tagName, req, rev) {
    const i = cellIndex(man, tagName, req);
    let inverse, gone;
    editTagFrames(
      man,
      tagName,
      (frames) => {
        if (frames.length <= 1) throw new HttpError(400, "refusing to drop the last cell of a tag");
        // The whole frame, keeping its `_fid`, so an undo puts the same cell back rather than a
        // copy of it — every other edit queued behind this one addresses it by that id.
        inverse = { k: "insert", tag: tagName, at: i, frame: frames[i] };
        gone = frames[i];
        frames.splice(i, 1);
        return frames;
      },
      rev,
    );
    return {
      text: `${man.character}-${tagName} drop cell ${i + 1}`,
      index: i,
      inverse,
      // The id is retired, not forgotten. Two things need it later: the identity assertion, which
      // refuses any write that loses an id it has seen, and the re-pick reconcile, which will not
      // resurrect a cell a person deliberately removed just because the picker likes that pose
      // again (DESIGN-pipeline.md: "a drop is recorded against a frame id, so a re-matched frame
      // stays dropped").
      attic: { dropAdd: [atticEntry(gone, tagName, "dropped by hand", Date.now())] },
    };
  }

  // Reattach one orphaned edit to a cell that is live now. The tray is never emptied by anything
  // but a person, so the entry does not disappear — it moves to the dropped ledger with the cell
  // it landed on recorded, which is both the audit trail and what keeps its id accounted for.
  async restore(man, tagName, req, rev) {
    const i = cellIndex(man, tagName, req);
    const entry = (await this.tray()).find((e) => e.fid === req.fid && e.tag === tagName);
    if (!entry) throw new HttpError(404, `nothing in the ${tagName} tray with id ${req.fid}`);
    let inverse;
    editTagFrames(
      man,
      tagName,
      (frames) => {
        inverse = frameInverse(frames[i]);
        frames[i].hold = entry.hold;
        frames[i].pivot_nudge = [...entry.pivot_nudge];
        return frames;
      },
      rev,
    );
    const nudge = entry.pivot_nudge.join(",");
    return {
      text: `${man.character}-${tagName} cell ${i + 1} restore hold ${entry.hold} nudge ${nudge}`,
      index: i,
      inverse,
      attic: {
        trayRemove: [entry.fid],
        dropAdd: [{ ...entry, why: `restored onto cell ${i + 1}`, at: Date.now() }],
      },
      // The undo has to put the tray entry back as well as the cell it landed on; `mutate` only
      // records ONE inverse per edit, so the attic half rides on the frame inverse's own list.
      extraInverses: [{ k: "attic", fid: entry.fid, entry }],
    };
  }

  // Hand-collect one tray entry. The design says the tray is collected only by hand, and this is
  // that hand: the entry stops being offered, its id stays accounted for, and the reason says a
  // person did it rather than a re-run.
  async forget(man, tagName, req) {
    const entry = (await this.tray()).find((e) => e.fid === req.fid && e.tag === tagName);
    if (!entry) throw new HttpError(404, `nothing in the ${tagName} tray with id ${req.fid}`);
    return {
      text: `${man.character}-${tagName} collect ${req.fid} from the tray`,
      index: null,
      attic: {
        trayRemove: [entry.fid],
        dropAdd: [{ ...entry, why: "collected from the tray by hand", at: Date.now() }],
      },
      // Nothing in the manifest moved, so nothing needs packing. The revision still advances, so
      // every open page redraws the tray it just changed.
      norun: true,
    };
  }

  async reorder(man, tagName, req, rev) {
    if (!Array.isArray(req.order)) throw new HttpError(400, "order must be a list of indices");
    const order = req.order.map((v, k) => whole(v, `order[${k}]`));
    let inverse;
    editTagFrames(
      man,
      tagName,
      (frames) => {
        const want = frames.map((_, k) => k).join(",");
        if ([...order].sort((a, b) => a - b).join(",") !== want) {
          throw new HttpError(400, "order must be a permutation of the current cell indices");
        }
        inverse = { k: "order", tag: tagName, fids: frames.map((f) => f._fid) };
        return order.map((k) => frames[k]);
      },
      rev,
    );
    return { text: `${man.character}-${tagName} reorder`, index: null, inverse };
  }

  // `scope` says WHICH origin moves, because the two are different operations: the sheet pivot
  // is the ground line the whole character is packed against and moving it re-seats every cell,
  // while a frame's nudge moves one drawing relative to that shared line.
  async pivot(man, tagName, req, rev) {
    if (req.scope === "sheet") {
      const [x, y] = xy(req.pivot, "pivot");
      inCell(x, y, man.cell, "pivot");
      const inverse = { k: "sheetPivot", pivot: man.pivot };
      man.pivot = [x, y];
      return { text: `${man.character} sheet pivot -> ${x},${y}`, index: null, inverse };
    }
    if (req.scope === "frame") {
      const i = cellIndex(man, tagName, req);
      const [dx, dy] = xy(req.pivot_nudge, "pivot_nudge");
      // The nudge is an OFFSET, so it is the RESULT that must land inside the cell: the artwork
      // is pasted at pivot+nudge, and a nudge past the edge paints it into a neighbouring cell.
      inCell(man.pivot[0] + dx, man.pivot[1] + dy, man.cell, "nudged pivot");
      let inverse;
      editTagFrames(
        man,
        tagName,
        (frames) => {
          inverse = frameInverse(frames[i]);
          frames[i].pivot_nudge = [dx, dy];
          return frames;
        },
        rev,
      );
      const sign = (v) => (v > 0 ? `+${v}` : `${v}`);
      return {
        text: `${man.character}-${tagName} cell ${i + 1} nudge ${sign(dx)},${sign(dy)}`,
        index: i,
        inverse,
      };
    }
    throw new HttpError(400, 'scope must be "sheet" or "frame"');
  }

  // One cell's duration multiplier over its tag's fps — an animator sitting on an extreme. Not
  // the tag's `hold_key`, which is the game freezing on the last cell while a key is down.
  async hold(man, tagName, req, rev) {
    const i = cellIndex(man, tagName, req);
    const beats = whole(req.hold, "hold");
    if (beats < 1) throw new HttpError(400, "hold is a count of beats, so it cannot be less than 1");
    let inverse;
    editTagFrames(
      man,
      tagName,
      (frames) => {
        inverse = frameInverse(frames[i]);
        frames[i].hold = beats;
        return frames;
      },
      rev,
    );
    return { text: `${man.character}-${tagName} cell ${i + 1} hold ${beats}`, index: i, inverse };
  }

  async tag(man, tagName, req) {
    const fps = whole(req.fps, "fps");
    if (fps <= 0) throw new HttpError(400, "fps must be greater than zero — both consumers divide by it");
    if (!DIRECTIONS.includes(req.direction)) {
      throw new HttpError(400, `direction must be one of ${DIRECTIONS.join(", ")}`);
    }
    const t = man.tags[tagIndex(man, tagName)];
    const inverse = { k: "tagTiming", tag: tagName, fps: t.fps, direction: t.direction };
    t.fps = fps;
    t.direction = req.direction;
    return {
      text: `${man.character}-${tagName} ${fps}fps ${req.direction}`,
      index: null,
      inverse,
    };
  }

  // One step param. The editable column of DESIGN-pipeline.md's table, and only that column: the
  // locked constants are shown on the chip with their measured reason and have no control here at
  // all, because overriding one is meant to be an explicit named deviation and this stage does
  // not build that.
  //
  // Validated to the last field before one byte of the manifest is touched, so a refusal cannot
  // leave half an edit behind. `mutate` re-reads the manifest from storage on every call and only
  // writes it after this returns, which is what makes that guarantee hold rather than hope.
  async step(man, tagName, req, rev) {
    const allowed = STEP_EDITABLE[req.kind];
    if (!allowed) {
      throw new HttpError(400, `the ${req.kind} step has no editable params`);
    }
    if (!allowed.includes(req.field)) {
      throw new HttpError(
        400,
        `${req.field} is not editable on the ${req.kind} step — ${allowed.join(", ")} are`,
      );
    }
    const value = stepValue(req.field, req.value);
    const tag = man.tags[tagIndex(man, tagName)];
    const now = req.field === "trait" ? man.trait : tag[req.field];
    if (canon(now) === canon(value)) {
      throw new HttpError(400, `that is the ${req.field} it already has`);
    }
    const inverse = { k: "stepParams", tag: tagName, field: req.field, value: now };
    if (req.field === "trait") man.trait = value;
    else tag[req.field] = value;
    const what = typeof value === "string" && value.length > 24
      ? `${value.slice(0, 24)}…` : JSON.stringify(value);
    return {
      // The trait line is the character's, not the tag's, and saying otherwise in the queue would
      // be the one edit on this page whose blast radius is wider than it reads.
      text: req.field === "trait"
        ? `${man.character} trait ${what}`
        : `${man.character}-${tagName} ${req.field} ${what}`,
      index: null,
      inverse,
      norun: true,
    };
  }

  // Run a stale step. Not a mutation: nothing in the manifest changes, so there is no inverse and
  // no revision — this only puts work on the queue that the person has just priced and confirmed.
  //
  // Five kinds. `plate` is uploaded rather than computed and `export` is streamed on request, so
  // neither has anything to run; every other step is an agent job type as of stage 4.
  async run(req) {
    const man = await this.manifest();
    if (!man) throw new HttpError(409, "this character has not been seeded yet");
    const jobKind = JOB_KIND[req.kind];
    if (!jobKind) {
      throw new HttpError(400,
        `${req.kind} has no agent job type — a plate is uploaded and an export is streamed`);
    }
    if (jobKind !== "pack" && tagIndex(man, req.tag) < 0) {
      throw new HttpError(400, `no ${req.tag} tag for ${man.character}`);
    }
    const steps = man.steps || [];
    const stale = staleness(steps);
    const scope = steps.filter((s) =>
      s.kind === req.kind && (req.kind === "pack" || s.tag === req.tag));
    if (!scope.length) {
      throw new HttpError(400, `no ${req.kind} step for ${req.tag ?? man.character}`);
    }
    const todo = scope.filter((s) => stale[s.id].stale);
    if (!todo.length) throw new HttpError(409, `the ${req.kind} step is already fresh`);
    // The agent paints a cell only when its drawing is MISSING from disk — the packer caches by
    // file name. So a repaint that is stale for any other reason cannot be answered by running,
    // and the count in the label says what will really be painted.
    const paints = todo.filter((s) => s.built_key === null).length;
    // `generates` is the COALESCING flag, not "costs GPU". A job with it false is a plain repack
    // and any cheap edit may join it; a clip, an extraction and a re-pick are none of those, so
    // all three are true even though only the clip touches the card. Letting a hold join an
    // extraction would have run ffmpeg and silently never packed the hold.
    const generates = jobKind === "clip" || jobKind === "extract" || jobKind === "picks" ||
      (jobKind === "repaint" && paints > 0);
    // The pack is not tag-scoped — it packs every move in one atlas, because the cell scale is
    // shared across the set. Carrying the page's selected tag onto it would name a scope the job
    // does not have.
    const tag = jobKind === "pack" ? null : req.tag;
    const LABEL = {
      clip: `run clip(${tag}) — one 170s render`,
      extract: `run extract(${tag}) — re-extract the frames`,
      picks: `run picks(${tag}) — choose ${man.tags[tagIndex(man, tag)]?.cells} cells`,
      repaint: `run repaint(${tag}) — ${paints} cell${paints === 1 ? "" : "s"}`,
      pack: "run pack — repack",
    };
    const now = Date.now();
    const job = {
      id: shortId(),
      label: generates ? LABEL[jobKind] : `run ${req.kind} — repack`,
      generates,
      kind: jobKind,
      tag,
      index: null,
      state: "queued",
      agent: null,
      error: null,
      edits: [],
      cancelRequested: false,
      startedAt: null,
      deadline: now + LEASE_MS,
      finishedAt: null,
    };
    await this.editJobs((jobs) => {
      const live = jobs.filter(isLive);
      if (live.length >= MAX_PENDING) {
        throw new HttpError(
          409,
          `refused — ${live.length} jobs are already queued on this character, which is the limit`,
        );
      }
      jobs.push(job);
    });
    await this.pump();
    const jobs = await this.jobs();
    const at = jobs.findIndex((j) => j.id === job.id);
    return { job: viewAll(jobs, Date.now())[at], steps: todo.length, paints };
  }

  // --- the page's view -------------------------------------------------------------------
  // Only the manifest half. Where the packer actually put each cell, what it measures, and which
  // repaints exist on disk all come from the agent, because all three are answers about files.

  async state() {
    const man = await this.manifest();
    if (!man) throw new HttpError(409, "this character has not been seeded yet");
    const tags = man.tags.map((tag) => ({
      name: tag.name,
      fps: tag.fps,
      loop: tag.loop,
      direction: tag.direction,
      hold_key: tag.hold_key,
      cells: man.frames.slice(tag.from, tag.to + 1).map((f, i) => ({
        index: i,
        // The frame's identity, which the page needs for exactly one thing: saying which live
        // cell an orphaned edit came from, and which one a restore is about to land on. It is
        // still stripped from every copy of the MANIFEST that leaves this object.
        fid: f._fid,
        src: f.src,
        seed: f.seed,
        png: f.png ?? null,
        hold: f.hold ?? 1,
        // Always the OFFSET from the sheet pivot, never an absolute point, so a later change to
        // the sheet pivot carries every nudged cell with it.
        pivot_nudge: f.pivot_nudge ?? [0, 0],
      })),
    }));
    const steps = man.steps || [];
    const stale = staleness(steps);
    return {
      character: man.character,
      cell: man.cell,
      pivot: man.pivot,
      trait: man.trait ?? null,
      revision: await this.revision(),
      tags,
      // The pipeline's state, made explicit. Nothing reads this yet and nothing runs off it; the
      // rail that draws it is stage 3.
      steps: steps.map((s) => ({ ...s, ...stale[s.id] })),
      // The orphan tray, and the ledger of ids that are neither live nor in it. Both travel with
      // the state rather than behind a second request, because the tray is the one thing on this
      // page a person MUST see without going looking: an edit that is only visible to whoever
      // thinks to open a panel is an edit that was silently eaten.
      tray: await this.tray(),
      dropped: (await this.dropped()).map((e) => ({ fid: e.fid, tag: e.tag, why: e.why })),
    };
  }

  async fetch(request) {
    const url = new URL(request.url);
    const body = request.method === "POST" ? await request.json() : {};
    try {
      const op = url.pathname.slice(1);
      if (MUTATIONS.includes(op)) {
        // 202: the edit IS in the manifest, and the drawing that follows from it is not made yet.
        return json(await this.mutate(op, body), 202);
      }
      switch (url.pathname) {
        case "/roster":
          return json({ characters: await this.roster() });
        case "/roster-add":
          return json({ characters: await this.rosterAdd(body.cid) });
        case "/health": {
          // Reads storage, so a 200 here means the route AND the cell's state are live. It must
          // never 404: celld-release counts 404 as healthy, and DO routing is broken for about
          // ten seconds after every restart.
          const man = await this.manifest();
          return json({ ok: true, seeded: man !== null, frames: man ? man.frames.length : 0 });
        }
        case "/state":
          return json(await this.state());
        case "/jobs": {
          const now = Date.now();
          return json({ revision: await this.revision(), jobs: viewAll(await this.jobs(), now) });
        }
        case "/job": {
          const id = url.searchParams.get("id");
          const jobs = await this.jobs();
          const at = jobs.findIndex((j) => j.id === id);
          if (at < 0) throw new HttpError(404, "no job with that id");
          return json({
            revision: await this.revision(),
            job: viewAll(jobs, Date.now())[at],
          });
        }
        case "/cancel":
          return json(await this.cancel(body.id));
        case "/picked":
          // The agent has picked; this object decides what that costs the hand edits. It is a
          // separate call from `/done` because the reconcile has to land BEFORE the agent
          // recomputes the step records — those key off the chosen source frames, and stamping
          // built_key against a set of picks the object has not accepted yet would say "fresh"
          // about a manifest that never existed.
          return json(await this.reconcile(body.tag, body.srcs, body.frames_hash));
        case "/run":
          // 202 like a mutation, and for the same reason: the queue has taken it and the work
          // itself has not happened yet.
          return json(await this.run(body), 202);
        case "/manifest": {
          const man = await this.manifest();
          if (!man) throw new HttpError(404, "not seeded");
          // The frame list travels beside the manifest and not inside it: the agent writes what
          // it is given straight to sheets/<cid>.json, so a bookkeeping key in there would end up
          // in every manifest on disk.
          return json({
            manifest: stripped(man),
            revision: await this.revision(),
            frames: frameMeta(man),
          });
        }
        case "/seed": {
          const live = (await this.jobs()).filter(isLive);
          if (live.length) throw new HttpError(409, `refusing to seed with ${live.length} jobs queued`);
          const rev = (await this.revision()) + 1;
          const man = body.manifest;
          man.frames = man.frames.map((f) => frameRecord(f, rev));
          // Seeding replaces every frame with a freshly-identified one, so the old ids are
          // genuinely gone and the identity ledger has to go with them — otherwise the very next
          // write is refused for losing frames that a cutover deliberately discarded. This is the
          // one operation allowed to reset the three buckets, and it already refuses to run with
          // anything queued.
          await this.ctx.storage.delete("fids");
          await this.ctx.storage.delete("tray");
          await this.ctx.storage.delete("dropped");
          await this.ctx.storage.delete("pickedFrom");
          await this.saveManifest(man);
          await this.ctx.storage.put("revision", rev);
          if (body.catalogue) await this.ctx.storage.put("catalogue", body.catalogue);
          return json({ ok: true, frames: man.frames.length, revision: rev });
        }
        case "/claim": {
          const wait = Math.min(MAX_CLAIM_WAIT_MS, Math.max(0, body.waitMs ?? 0));
          return json({ job: await this.claim(wait, body.agent ?? "") });
        }
        case "/progress": {
          const now = Date.now();
          const cancel = await this.editJobs((jobs) => {
            const job = jobs.find((j) => j.id === body.id);
            if (!job || !isLive(job)) throw new HttpError(404, "no such job");
            job.state = "running";
            job.label = body.message ?? job.label;
            if (job.startedAt === null) job.startedAt = now;
            job.deadline = now + RUNNING_LEASE_MS;
            return job.cancelRequested === true;
          });
          // The heartbeat is also how a cancel reaches a job that is already on the GPU. There is
          // no other channel: the agent is behind a long poll and this Worker cannot call it.
          return json({ ok: true, cancel });
        }
        case "/done": {
          // An extraction that really changed took this tag's drawings with it. Said before the
          // report is absorbed so that a rejected report cannot leave the tray pointing at files
          // the agent has already deleted.
          if (body.cleared_tag) await this.clearTrayArt(body.cleared_tag);
          // Validate the report BEFORE storing anything from it. Storing the catalogue first
          // meant a REJECTED report still replaced the repick and variant allowlists, so the
          // next edit was validated against a packing run this object had refused to believe.
          await this.absorbPacked(body.manifest, body.frames);
          if (body.catalogue) await this.ctx.storage.put("catalogue", body.catalogue);
          await this.finish(body.id, "done", null);
          return json({ ok: true });
        }
        case "/failed":
          await this.finish(body.id, "failed", body.message);
          return json({ ok: true });
        default:
          return json({ error: "not found" }, 404);
      }
    } catch (e) {
      if (e instanceof HttpError) return json({ error: e.message }, e.status);
      return json({ error: `${e.name}: ${e.message}` }, 500);
    }
  }
}

// repaint_cells.repaint_path(), which is what names the file the packer will look for. It is
// duplicated here rather than asked of the agent because a mutation must be able to say which
// drawing it now wants without a round trip to a machine that may be asleep.
function repaintPath(cid, move, src, seed) {
  const stem = src.split("/").pop().replace(/\.png$/, "");
  return `/tmp/repaint/${cid}-${move}-${stem}${seed === BASE_SEED ? "" : `-s${seed}`}.png`;
}

function characterStub(env, cid) {
  // The SHAPE of the id, not membership of a list. An id that names no character routes to an
  // unseeded cell, which already answers "this character has not been seeded yet" — and checking
  // the roster here would put a second Durable Object hop in front of every single request.
  if (!CHARACTER_ID.test(cid)) throw new HttpError(400, `not a character id: ${cid}`);
  return env.CHARACTER.get(env.CHARACTER.idFromName(cid));
}

function rosterStub(env) {
  return env.CHARACTER.get(env.CHARACTER.idFromName(ROSTER));
}

async function roster(env) {
  const r = await rosterStub(env).fetch(new Request("http://character/roster"));
  return (await r.json()).characters;
}

async function callDO(env, cid, path, init) {
  const stub = characterStub(env, cid);
  return await stub.fetch(new Request(`http://character${path}`, init));
}

function post(body) {
  return { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    try {
      if (url.pathname === "/api/health") {
        // Touches a DO on purpose. The asset path answers 200 all through a restart while DO
        // routing is still broken, so health-checking "/" would report green on a broken app.
        // Any failure in here has to surface as 5xx, never as 404.
        //
        // It also digests a fixed probe, because the whole staleness model rests on
        // crypto.subtle.digest existing on this runtime. If it ever does not, this says so at
        // deploy time instead of on the first edit somebody makes.
        const names = await roster(env);
        const probe = await digest("celld");
        // The roster cell itself when there is no character yet: an empty fleet must still be
        // able to say it is healthy, and it is a Durable Object, so it proves routing just as well.
        const stub = names.length ? characterStub(env, names[0]) : rosterStub(env);
        const r = await stub.fetch(new Request("http://character/health"));
        if (!r.ok) return json({ error: "durable object unhealthy" }, 503);
        return json({ ok: true, characters: names, digest: probe, ...(await r.json()) });
      }

      if (url.pathname.startsWith("/api/agent/")) {
        if (!tokenMatches(bearer(request), env.AGENT_TOKEN || "")) {
          return json({ error: "bad agent token" }, 401);
        }
        const body = await request.json();
        const verb = url.pathname.slice("/api/agent/".length);
        if (verb === "manifest") return await callDO(env, body.cid, "/manifest");
        if (!["seed", "claim", "progress", "picked", "done", "failed"].includes(verb)) {
          return json({ error: "not found" }, 404);
        }
        const r = await callDO(env, body.cid, `/${verb}`, post(body));
        // Seeding is what makes a character EXIST, so it is what puts him on the roster — and
        // only once the cell has actually taken the manifest, or a failed seed would leave a name
        // the page offers and nothing can answer for.
        if (verb === "seed" && r.ok) {
          await rosterStub(env).fetch(new Request("http://character/roster-add", post(body)));
        }
        return r;
      }

      // Every route below is the PAGE's, and the page may arrive without naming a character, so
      // this is where the roster is read. It is deliberately after the agent block: the agent
      // always names its own character, and its claim is a long poll, so putting a second Durable
      // Object hop in front of it would cost one on every poll of every character forever.
      const names = await roster(env);
      const cid = url.searchParams.get("cid") || names[0];
      if (!cid) throw new HttpError(409, "no character has been seeded on this fleet yet");
      if (url.pathname === "/api/state") {
        const r = await callDO(env, cid, "/state");
        const body = await r.json();
        // The page needs a second origin: the cells, the source frames and the atlas are files
        // on wilson and are served from there, never through this bucket.
        return json({ ...body, characters: names, agent: env.AGENT_BASE }, r.status);
      }

      if (url.pathname === "/api/jobs") return await callDO(env, cid, "/jobs");

      if (url.pathname.startsWith("/api/job/")) {
        const rest = url.pathname.slice("/api/job/".length).split("/");
        const id = rest[0];
        if (rest.length === 2 && rest[1] === "cancel") {
          if (request.method !== "POST") return json({ error: "POST only" }, 405);
          return await callDO(env, cid, "/cancel", post({ id }));
        }
        if (rest.length === 1) {
          return await callDO(env, cid, `/job?id=${encodeURIComponent(id)}`);
        }
        return json({ error: "not found" }, 404);
      }

      if (url.pathname === "/api/run") {
        if (request.method !== "POST") return json({ error: "POST only" }, 405);
        const body = await request.json();
        return await callDO(env, body.cid || cid, "/run", post(body));
      }

      const mutation = url.pathname.startsWith("/api/") ? url.pathname.slice(5) : null;
      if (mutation && MUTATIONS.includes(mutation)) {
        if (request.method !== "POST") return json({ error: "POST only" }, 405);
        const body = await request.json();
        return await callDO(env, body.cid || cid, `/${mutation}`, post(body));
      }

      if (url.pathname === "/" || url.pathname === "/index.html") {
        const page = new URL(url);
        page.pathname = "/index.html";
        return await env.ASSETS.fetch(new Request(page, request));
      }
      return json({ error: "not found" }, 404);
    } catch (e) {
      if (e instanceof HttpError) return json({ error: e.message }, e.status);
      return json({ error: `${e.name}: ${e.message}` }, 500);
    }
  },
};
