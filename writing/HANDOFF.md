# Handoff — Follies writing system

Written 2026-07-22 by the outgoing session (a Discord thread that was, embarrassingly,
titled after a Superset daemon incident — hence this promotion to a real Superset
project). Everything below is current as of commit `96f0280`.

## Where things stand

**Live and done:**
- `cast.sethgholson.com` serves **Gholson's Follies** — Seth's Claude Design newspaper
  layout, converted to static pages in `site/`, autodeploying from master via GitHub
  Actions → Coolify (app `nw0sswocksk44gowgsk08ws0`).
- All 9 image slots filled with generated art (4 bill portraits, 4 circle thumbs, 1
  curtain-call hero), made with Nano Banana Pro conditioned on the character sheets.
- The site is **responsive** — `site/styles.css`, reflowed hierarchy below 740px (full
  uncropped hero, 2×2 playbill company grid, stacked bill pages at true 8:9).
- **The writing target is defined**: `writing/bible.md` (the craft rules + 12 quality
  gates) and `writing/derivation.md` (the session-logs→strip pipeline spec).

**Not built:** any of the pipeline. `derivation.md` is a spec, not code. Nothing in
Stages 0–7 exists yet.

## What the next agent should pick up

Seth's stated sequence: *define the target, then gather the right data.* The target is
defined. **The next work is Stage 0–1: harvesting and incident extraction.**

Before writing code, note two things:

1. **Seth has not read `bible.md` or `derivation.md` yet.** He answered the open questions
   but explicitly said he hadn't reviewed the docs. **Do not treat the spec as ratified.**
   Expect revisions. The thing he was most likely to overrule, flagged honestly at the
   time: the **word budgets** (≤12 words/balloon, ≤50/strip) are house caps invented by
   the outgoing session, *not* sourced from the research — everything else in the bible
   traces to a verified primary source, those don't.
2. **Privacy is the hard part, and it is not optional.** See Decision 2 and Stage 1.
   Opt-in allowlist by project, redaction on the source machine before transport. The logs
   contain family, finances, health and credentials. Get this wrong once and the project
   is over.

## Decisions already made (don't relitigate)

1. **Human in the loop** until tone is dialed in. Stage 5 ends in a review queue; Seth
   approves every strip. **Log every rejection + reason** — that's the tuning signal.
2. **Privacy: opt-in allowlist, redact at source.**
3. **The cast may be wrong about themselves** — self-deprecation fully in bounds.
4. **Standalone strips now; recurring gags mined from the archive later** (Stage 7, ~50
   strips out). Engines get promoted from what landed, never invented upfront.
5. **Brand line:** no Ethos, its vendors, clients or internal systems — generalize, keep
   the shape, drop identifiers. Zeitgeist-scale brands (Anthropic pricing, Fable releases)
   are fair game.

## Facts worth not rediscovering

- **Harvest volume, measured:** ~240 sessions / ~230 MB per day across three machines
  (wilson 31/119MB, Mac 38/72MB, akebot 172/38MB). Too big for an LLM — Stage 1 must be
  deterministic scripts.
- **akebot is not directly SSH-able from wilson.** Jump: `ssh sethgho@100.64.185.78` then
  `ssh akebot`. akebot has passwordless sudo. Cadbury is not online yet; the incident
  schema already carries an `agent` field for him.
- **Art generation — use the async graph path, never `partner_generate`, for anything
  reference-conditioned.** (Corrected 2026-07-22; an earlier version of this note blamed
  payload size, which was wrong.) `partner_generate` with image inputs uses *direct
  dispatch*: one synchronous HTTP call held open for the whole generation, and
  `cloud.comfy.org` is behind Cloudflare's ~100 s ceiling. Long generations get the
  connection cut **while the job keeps running and still bills**, and direct-dispatch
  results persist in neither `/api/jobs` nor the asset library — so the spend is
  unrecoverable. Duration is the whole story: single-character portraits run 37–54 s and
  survive; a 4-panel 2K comic ran 64 s of GPU time (~90–120 s wall) and dropped 4/4,
  burning ~120 credits. The async path returns a `prompt_id` in under a second and
  persists the result, so drops cost nothing:
  `upload_file → LoadImage → ImageBatch → GeminiImage2Node (gemini-3-pro-image-preview)
  → SaveImage → submit_workflow → poll → get_output`.
  Check true cost/duration for free via `/api/jobs`
  (`execution_start_time`/`execution_end_time`). Small `/<id>/ref.jpg` files (~140 KB) exist
  on the live site and are still worth using as refs — just for speed, not because size
  was the bug. Nano Banana also draws its own panel border despite the exclusion block;
  crop inside it.
- **Workflow model policy:** the built-in `deep-research` workflow set no per-agent model,
  so 104 agents inherited the session model and burned 3M tokens / hit the session limit
  twice. A tuned copy lives at `~/.claude/workflows/deep-research.js` (verify+search on
  sonnet, synthesis on the frontier model). **Model/effort opts are part of an agent's
  cache key** — retuning mid-run invalidates resume caching and re-runs everything.

## Research provenance

The bible's rules came from a deep-research pass: 22 sources, 107 claims extracted, 25
verified under 3-vote adversarial review → **21 confirmed, 1 refuted, 3 unverified**. The
refuted one is called out inline in bible §2 (ensemble characters should *not* be reduced
to one-word archetypes — that's Watterson's "the plastic comes through"). The 3 unverified
are Adams' "write from annoyance" claims; they are **not** used as load-bearing rules and
should stay that way unless re-verified.
