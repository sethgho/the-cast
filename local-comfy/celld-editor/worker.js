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

const CHARACTERS = ["seth", "cadbury"];

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
};
const MUTATIONS = Object.keys(GENERATES);

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
function applyInverse(man, inv, rev) {
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
  const t = man.tags[tagIndex(man, inv.tag)];
  if (t) {
    t.fps = inv.fps;
    t.direction = inv.direction;
  }
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
      await this.ctx.storage.put("manifest", man);
      // The single-slot lock this queue replaced. Left behind it would be a second, invisible
      // record of "what is running" that nothing reads and nothing clears.
      await this.ctx.storage.delete("job");
    }
    return man;
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
    await this.ctx.storage.put("manifest", man);
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
      const tagName = req.tag;
      if (tagIndex(man, tagName) < 0) {
        throw new HttpError(400, `no ${tagName} tag for ${man.character}`);
      }
      const done = await this[op](man, tagName, req, rev);
      await this.ctx.storage.put("manifest", man);
      await this.ctx.storage.put("revision", rev);
      return await this.publish(taken, done, tagName, rev);
    } catch (e) {
      // A rejected edit must not leave a job in the queue, and must not leave a repack that other
      // edits have already joined half-cancelled. `man` is a fresh copy out of storage and was
      // never written, so backing the record out drops the whole attempt.
      await this.editJobs((jobs) => {
        const at = jobs.findIndex((j) => j.id === taken.job.id);
        if (at < 0) return;
        if (taken.joined) jobs[at].state = "queued";
        else jobs.splice(at, 1);
      });
      throw e;
    }
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
    for (const inv of [...outcome].reverse()) applyInverse(man, inv, rev);
    await this.ctx.storage.put("manifest", man);
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
    let inverse;
    editTagFrames(
      man,
      tagName,
      (frames) => {
        if (frames.length <= 1) throw new HttpError(400, "refusing to drop the last cell of a tag");
        // The whole frame, keeping its `_fid`, so an undo puts the same cell back rather than a
        // copy of it — every other edit queued behind this one addresses it by that id.
        inverse = { k: "insert", tag: tagName, at: i, frame: frames[i] };
        frames.splice(i, 1);
        return frames;
      },
      rev,
    );
    return { text: `${man.character}-${tagName} drop cell ${i + 1}`, index: i, inverse };
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
        src: f.src,
        seed: f.seed,
        png: f.png ?? null,
        hold: f.hold ?? 1,
        // Always the OFFSET from the sheet pivot, never an absolute point, so a later change to
        // the sheet pivot carries every nudged cell with it.
        pivot_nudge: f.pivot_nudge ?? [0, 0],
      })),
    }));
    return {
      character: man.character,
      characters: CHARACTERS,
      cell: man.cell,
      pivot: man.pivot,
      revision: await this.revision(),
      tags,
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
          await this.ctx.storage.put("manifest", man);
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
  if (!CHARACTERS.includes(cid)) throw new HttpError(400, `unknown character: ${cid}`);
  return env.CHARACTER.get(env.CHARACTER.idFromName(cid));
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
    const cid = url.searchParams.get("cid") || CHARACTERS[0];
    try {
      if (url.pathname === "/api/health") {
        // Touches a DO on purpose. The asset path answers 200 all through a restart while DO
        // routing is still broken, so health-checking "/" would report green on a broken app.
        // Any failure in here has to surface as 5xx, never as 404.
        const r = await callDO(env, CHARACTERS[0], "/health");
        if (!r.ok) return json({ error: "durable object unhealthy" }, 503);
        return json({ ok: true, character: CHARACTERS[0], ...(await r.json()) });
      }

      if (url.pathname === "/api/state") {
        const r = await callDO(env, cid, "/state");
        const body = await r.json();
        // The page needs a second origin: the cells, the source frames and the atlas are files
        // on wilson and are served from there, never through this bucket.
        return json({ ...body, agent: env.AGENT_BASE }, r.status);
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

      if (url.pathname.startsWith("/api/agent/")) {
        if (!tokenMatches(bearer(request), env.AGENT_TOKEN || "")) {
          return json({ error: "bad agent token" }, 401);
        }
        const body = await request.json();
        const verb = url.pathname.slice("/api/agent/".length);
        if (verb === "manifest") return await callDO(env, body.cid, "/manifest");
        if (!["seed", "claim", "progress", "done", "failed"].includes(verb)) {
          return json({ error: "not found" }, 404);
        }
        return await callDO(env, body.cid, `/${verb}`, post(body));
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
