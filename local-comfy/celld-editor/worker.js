// The sprite cell editor's state, on celld: one CharacterDO per character.
//
// The split is deliberate and is the whole design (DESIGN-editor.md, stage 6). The packer is
// numpy and the repaint drives ComfyUI on a GPU, so neither will ever run in V8; they stay on
// wilson, behind `sprite_agent.py`, which long-polls this Worker for work. What moves here is
// the part that is pure state: the manifest, and the single job slot that stops two browsers
// from queueing the same 45-second GPU job.
//
// The mutations below are MESSAGES, not manifest writes. The Python editor loaded the manifest,
// spliced it in the client of the file and saved it back; against a Durable Object that same
// shape would be a non-atomic read-modify-write and the DO's single-threadedness would buy
// nothing. So "drop frame 3 of tag punch" crosses the wire, and the splice happens in here.

const CHARACTERS = ["seth", "cadbury"];

// Mirrors repaint_cells.SEED: the one seed every cell starts on, and the only one whose repaint
// file carries no `-s<seed>` suffix.
const BASE_SEED = 77;

const DIRECTIONS = ["forward", "reverse", "pingpong"];

// A claimed job whose agent has gone silent for this long is declared failed and the character
// unlocks. Without it a wilson reboot mid-repaint would lock that character until someone
// noticed; a repaint of a whole tag is minutes, so the lease is generous and every progress
// report renews it.
const LEASE_MS = 10 * 60 * 1000;

// The longest a claim request is held open before answering "no work". Kept well under any
// proxy's idle timeout, and short enough that an agent restart cannot leave a request hanging
// against a DO that has since been evicted.
const MAX_CLAIM_WAIT_MS = 20000;

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

// --- manifest surgery ------------------------------------------------------------------------
// The key order here matches repaint_cells._frame() exactly, and `png` is omitted when absent.
// The agent writes whatever the DO hands it straight to sheets/<cid>.json, so a reordered key
// would rewrite every manifest on disk for no change in meaning.
function frameRecord(src, seed, png, hold, nudge) {
  const f = { src, seed };
  if (png) f.png = png;
  f.hold = hold;
  f.pivot_nudge = nudge;
  return f;
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
function editTagFrames(man, tagName, mutate) {
  const i = tagIndex(man, tagName);
  const tag = man.tags[i];
  const span = tag.to + 1 - tag.from;
  const frames = mutate(man.frames.slice(tag.from, tag.to + 1).map((f) => ({ ...f })));
  if (!frames.length) throw new HttpError(400, "a tag cannot have zero frames");
  const next = frames.map((f) =>
    frameRecord(f.src, f.seed, f.png, f.hold ?? 1, f.pivot_nudge ?? [0, 0]),
  );
  man.frames.splice(tag.from, span, ...next);
  const shift = next.length - span;
  tag.to += shift;
  for (const later of man.tags.slice(i + 1)) {
    later.from += shift;
    later.to += shift;
  }
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
    return (await this.ctx.storage.get("manifest")) || null;
  }

  async catalogue() {
    return (await this.ctx.storage.get("catalogue")) || { sources: {}, variants: {} };
  }

  async job() {
    return (await this.ctx.storage.get("job")) || null;
  }

  // The lock. A job holds the character until it finishes, fails, or its lease runs out — and
  // the lease check happens on READ, so a dead agent's job is reaped by the next person who
  // tries to edit rather than needing an alarm to come round.
  async activeJob() {
    const job = await this.job();
    if (!job || job.state === "done" || job.state === "error") return null;
    if (Date.now() <= job.deadline) return job;
    if (job.state === "claimed") {
      // Claimed and then silent means the agent never STARTED it — `run_job` reports progress
      // within a second of claiming. So the work was not done, and the edit is already in the
      // manifest: queue it again rather than fail it, and the next agent to poll packs it.
      const requeued = {
        ...job,
        state: "queued",
        message: `${job.label} — the agent never started it; queued again`,
        deadline: Date.now() + LEASE_MS,
      };
      await this.setJob(requeued);
      return requeued;
    }
    await this.finish(job.id, "error", `${job.label} — the agent stopped reporting; lease expired`);
    return null;
  }

  async setJob(job) {
    await this.ctx.storage.put("job", job);
  }

  async finish(id, state, message) {
    const job = await this.job();
    if (!job || job.id !== id) return;
    await this.setJob({ ...job, state, message, deadline: Date.now() + LEASE_MS });
  }

  // Take the lock, before anything is spliced. `state: "preparing"` and not "queued" is the
  // whole trick: `activeJob` counts it, so a second mutation is refused from this moment on,
  // while `claim` only ever takes a "queued" job — so no agent can run off with a job whose
  // edit has not been written yet. The lease applies from here, so a crash mid-mutation unlocks
  // the character on its own.
  async reserve(op, tagName, character) {
    const id = crypto.randomUUID().replace(/-/g, "").slice(0, 12);
    const job = {
      id,
      label: `${character}-${tagName} ${op}`,
      generates: false,
      state: "preparing",
      message: `preparing: ${character}-${tagName} ${op}`,
      deadline: Date.now() + LEASE_MS,
    };
    await this.setJob(job);
    return job;
  }

  async publish(job) {
    await this.setJob(job);
    // Hand it straight to a parked agent if one is listening, so an edit does not wait out the
    // poll interval it happened to land in the middle of.
    const waiter = this.waiters.shift();
    if (waiter) {
      const claimed = { ...job, state: "claimed", message: `${job.label} — claimed by the agent` };
      await this.setJob(claimed);
      waiter.resolve(claimed);
      return claimed;
    }
    return job;
  }

  async claim(waitMs) {
    // Sweep waiters that have outlived their own wait. Their `setTimeout` should have done it,
    // but celld does not run the timers of a request whose client has gone: an agent that
    // restarts while parked leaves a waiter that looks live and can never answer, and the next
    // job handed to it disappeared — the character then sat locked for the whole lease with
    // nobody running anything. Measured on this fleet, twice, during the cutover. Every claim
    // sweeps, so a leak can outlive at most one poll.
    const now = Date.now();
    this.waiters = this.waiters.filter((w) => w.until > now);
    const job = await this.activeJob();
    if (job && job.state === "queued") {
      const claimed = { ...job, state: "claimed", message: `${job.label} — claimed by the agent`, deadline: Date.now() + LEASE_MS };
      await this.setJob(claimed);
      return claimed;
    }
    if (waitMs <= 0) return null;
    return await new Promise((resolve) => {
      const waiter = { until: Date.now() + waitMs };
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

  // The agent reports the manifest it actually packed. Only the `png` fields are taken from it:
  // repaint_cells.main() fills those in as it repaints, and dropping them would leave the DO
  // pointing at drawings it thinks do not exist yet. Everything else differing means the agent
  // edited the manifest, which is this object's job alone — so that is reported as a failure
  // rather than absorbed.
  async absorbPacked(packed) {
    const man = await this.manifest();
    if (!man) throw new HttpError(409, "no manifest to absorb into");
    const same =
      packed &&
      Array.isArray(packed.frames) &&
      packed.frames.length === man.frames.length &&
      JSON.stringify(packed.tags) === JSON.stringify(man.tags) &&
      JSON.stringify(packed.pivot) === JSON.stringify(man.pivot) &&
      packed.frames.every((f, i) => {
        const m = man.frames[i];
        return (
          f.src === m.src &&
          f.seed === m.seed &&
          (f.hold ?? 1) === (m.hold ?? 1) &&
          JSON.stringify(f.pivot_nudge ?? [0, 0]) === JSON.stringify(m.pivot_nudge ?? [0, 0])
        );
      });
    if (!same) throw new HttpError(409, "the packed manifest disagrees with this character's own");
    man.frames = man.frames.map((f, i) =>
      frameRecord(f.src, f.seed, packed.frames[i].png, f.hold ?? 1, f.pivot_nudge ?? [0, 0]),
    );
    await this.ctx.storage.put("manifest", man);
  }

  // --- mutations -------------------------------------------------------------------------
  // Each one validates, splices in place, saves, and only then publishes the job. The manifest
  // is written before the repack for the reason the Python editor kept the same order: a repack
  // can be killed, and if it is, the edit must still be recorded so the next run picks it up.

  async mutate(op, req) {
    // The lock is genuinely HELD across the splice, not merely checked before it. An earlier
    // version checked here and took the slot at the end, and the gap between the two was only
    // safe under Cloudflare's input-gate semantics, which cannot be proven on celld from
    // outside: two mutations fired at once gave one job and TWO edits, and the loser's edit rode
    // along on the winner's repack under a 409 that said it had been refused. Measured. So a
    // provisional record is written FIRST, and every later await is inside the lock.
    const held = await this.activeJob();
    if (held) {
      throw new HttpError(409, `refused — "${held.label}" is still ${held.state} on this character`);
    }
    const man = await this.manifest();
    if (!man) throw new HttpError(409, "this character has not been seeded yet");
    const tagName = req.tag;
    if (tagIndex(man, tagName) < 0) {
      throw new HttpError(400, `no ${tagName} tag for ${man.character}`);
    }
    const job = await this.reserve(op, tagName, man.character);
    let label;
    try {
      label = await this[op](man, tagName, req);
      await this.ctx.storage.put("manifest", man);
    } catch (e) {
      // A rejected edit must not leave the character locked. `man` is a fresh copy out of
      // storage and was never written, so dropping the reservation drops the whole attempt.
      await this.ctx.storage.delete("job");
      throw e;
    }
    return await this.publish({
      ...job,
      label: label.text,
      generates: label.generates,
      state: "queued",
      message: `queued: ${label.text}`,
    });
  }

  async reroll(man, tagName, req) {
    const i = cellIndex(man, tagName, req);
    const seed =
      req.seed === undefined || req.seed === null
        ? crypto.getRandomValues(new Uint32Array(1))[0] % 2147483647 + 1
        : whole(req.seed, "seed");
    editTagFrames(man, tagName, (frames) => {
      const f = frames[i];
      if (seed === f.seed) throw new HttpError(400, "that is the seed the cell already has");
      f.seed = seed;
      f.png = repaintPath(man.character, tagName, f.src, seed);
      return frames;
    });
    return { text: `${man.character}-${tagName} cell ${i + 1} re-roll`, generates: true };
  }

  async repick(man, tagName, req) {
    const i = cellIndex(man, tagName, req);
    const src = req.src;
    const cat = await this.catalogue();
    const known = cat.sources[tagName] || [];
    if (!known.includes(src)) {
      throw new HttpError(400, "that source frame is not in this tag's pick directory");
    }
    editTagFrames(man, tagName, (frames) => {
      const f = frames[i];
      f.src = src;
      f.png = repaintPath(man.character, tagName, src, f.seed);
      return frames;
    });
    return { text: `${man.character}-${tagName} cell ${i + 1} re-pick`, generates: true };
  }

  async use(man, tagName, req) {
    const i = cellIndex(man, tagName, req);
    const png = req.png;
    const cat = await this.catalogue();
    editTagFrames(man, tagName, (frames) => {
      const f = frames[i];
      const known = (cat.variants[f.src] || []).map((v) => v.png);
      if (!known.includes(png)) {
        throw new HttpError(400, "that is not a variant of this cell's source frame");
      }
      f.png = png;
      f.seed = seedOfRepaint(png);
      return frames;
    });
    return { text: `${man.character}-${tagName} cell ${i + 1} use variant`, generates: false };
  }

  async drop(man, tagName, req) {
    const i = cellIndex(man, tagName, req);
    editTagFrames(man, tagName, (frames) => {
      if (frames.length <= 1) throw new HttpError(400, "refusing to drop the last cell of a tag");
      frames.splice(i, 1);
      return frames;
    });
    return { text: `${man.character}-${tagName} drop cell ${i + 1}`, generates: false };
  }

  async reorder(man, tagName, req) {
    if (!Array.isArray(req.order)) throw new HttpError(400, "order must be a list of indices");
    const order = req.order.map((v, k) => whole(v, `order[${k}]`));
    editTagFrames(man, tagName, (frames) => {
      const want = frames.map((_, k) => k).join(",");
      if ([...order].sort((a, b) => a - b).join(",") !== want) {
        throw new HttpError(400, "order must be a permutation of the current cell indices");
      }
      return order.map((k) => frames[k]);
    });
    return { text: `${man.character}-${tagName} reorder`, generates: false };
  }

  // `scope` says WHICH origin moves, because the two are different operations: the sheet pivot
  // is the ground line the whole character is packed against and moving it re-seats every cell,
  // while a frame's nudge moves one drawing relative to that shared line.
  async pivot(man, tagName, req) {
    if (req.scope === "sheet") {
      const [x, y] = xy(req.pivot, "pivot");
      inCell(x, y, man.cell, "pivot");
      man.pivot = [x, y];
      return { text: `${man.character} sheet pivot -> ${x},${y}`, generates: false };
    }
    if (req.scope === "frame") {
      const i = cellIndex(man, tagName, req);
      const [dx, dy] = xy(req.pivot_nudge, "pivot_nudge");
      // The nudge is an OFFSET, so it is the RESULT that must land inside the cell: the artwork
      // is pasted at pivot+nudge, and a nudge past the edge paints it into a neighbouring cell.
      inCell(man.pivot[0] + dx, man.pivot[1] + dy, man.cell, "nudged pivot");
      editTagFrames(man, tagName, (frames) => {
        frames[i].pivot_nudge = [dx, dy];
        return frames;
      });
      const sign = (v) => (v > 0 ? `+${v}` : `${v}`);
      return {
        text: `${man.character}-${tagName} cell ${i + 1} nudge ${sign(dx)},${sign(dy)}`,
        generates: false,
      };
    }
    throw new HttpError(400, 'scope must be "sheet" or "frame"');
  }

  // One cell's duration multiplier over its tag's fps — an animator sitting on an extreme. Not
  // the tag's `hold_key`, which is the game freezing on the last cell while a key is down.
  async hold(man, tagName, req) {
    const i = cellIndex(man, tagName, req);
    const beats = whole(req.hold, "hold");
    if (beats < 1) throw new HttpError(400, "hold is a count of beats, so it cannot be less than 1");
    editTagFrames(man, tagName, (frames) => {
      frames[i].hold = beats;
      return frames;
    });
    return { text: `${man.character}-${tagName} cell ${i + 1} hold ${beats}`, generates: false };
  }

  async tag(man, tagName, req) {
    const fps = whole(req.fps, "fps");
    if (fps <= 0) throw new HttpError(400, "fps must be greater than zero — both consumers divide by it");
    if (!DIRECTIONS.includes(req.direction)) {
      throw new HttpError(400, `direction must be one of ${DIRECTIONS.join(", ")}`);
    }
    const t = man.tags[tagIndex(man, tagName)];
    t.fps = fps;
    t.direction = req.direction;
    return { text: `${man.character}-${tagName} ${fps}fps ${req.direction}`, generates: false };
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
      tags,
    };
  }

  async fetch(request) {
    const url = new URL(request.url);
    const body = request.method === "POST" ? await request.json() : {};
    try {
      const op = url.pathname.slice(1);
      if (MUTATIONS.includes(op)) return json({ job: await this.mutate(op, body) });
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
        case "/manifest": {
          const man = await this.manifest();
          if (!man) throw new HttpError(404, "not seeded");
          return json(man);
        }
        case "/job": {
          const job = await this.job();
          if (!job || job.id !== url.searchParams.get("id")) {
            return json({ state: "error", message: "no such job" }, 404);
          }
          return json(job);
        }
        case "/seed": {
          const held = await this.activeJob();
          if (held) throw new HttpError(409, `refusing to seed while "${held.label}" is ${held.state}`);
          await this.ctx.storage.put("manifest", body.manifest);
          if (body.catalogue) await this.ctx.storage.put("catalogue", body.catalogue);
          return json({ ok: true, frames: body.manifest.frames.length });
        }
        case "/claim": {
          const wait = Math.min(MAX_CLAIM_WAIT_MS, Math.max(0, body.waitMs ?? 0));
          return json({ job: await this.claim(wait) });
        }
        case "/progress": {
          const job = await this.job();
          if (!job || job.id !== body.id) throw new HttpError(404, "no such job");
          await this.setJob({
            ...job,
            state: "running",
            message: body.message,
            deadline: Date.now() + LEASE_MS,
          });
          return json({ ok: true });
        }
        case "/done": {
          // Validate the report BEFORE storing anything from it. Storing the catalogue first
          // meant a REJECTED report still replaced the repick and variant allowlists, so the
          // next edit was validated against a packing run this object had refused to believe.
          await this.absorbPacked(body.manifest);
          if (body.catalogue) await this.ctx.storage.put("catalogue", body.catalogue);
          await this.finish(body.id, "done", body.message);
          return json({ ok: true });
        }
        case "/failed":
          await this.finish(body.id, "error", body.message);
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

const MUTATIONS = ["reroll", "repick", "use", "drop", "reorder", "pivot", "hold", "tag"];

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
        const cid = url.searchParams.get("cid") || CHARACTERS[0];
        const r = await callDO(env, cid, "/state");
        const body = await r.json();
        // The page needs a second origin: the cells, the source frames and the atlas are files
        // on wilson and are served from there, never through this bucket.
        return json({ ...body, agent: env.AGENT_BASE }, r.status);
      }

      if (url.pathname.startsWith("/api/job/")) {
        const cid = url.searchParams.get("cid") || CHARACTERS[0];
        const id = url.pathname.split("/").pop();
        return await callDO(env, cid, `/job?id=${encodeURIComponent(id)}`);
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
        return await callDO(env, body.cid, `/${mutation}`, post(body));
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
