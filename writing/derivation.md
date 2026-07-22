# Session Logs → Strip: the derivation spec

How real Claude Code session data becomes one gated strip. `bible.md` defines what a
good strip *is*; this defines the pipeline that produces one.

**What we're mining (Seth, 2026-07-22):** not incidents. The subject is **the condition
of being an agent-assisted software developer** — the recurring, recognizable truths of
this way of working, observed across many sessions and days. A specific event earns a
strip only when it is genuinely extraordinary. This is the Dilbert register, not the
sitcom register: the audience laughs because *their* week looks like this, not because
our Tuesday did.

**Design stance:** the pipeline's job is **not** to be funny. Its job is to (a) surface
genuinely strip-worthy observations about this life, (b) route them through fixed
characters, and (c) **reject relentlessly**. Per the bible's discard gate, publishing
nothing is an acceptable outcome. A pipeline that always ships is broken.

---

## Stage 0 — Harvest

**Input:** Claude Code transcripts across the three machines.

| Host | Path | Reachable via |
|---|---|---|
| wilson | `~/.claude/projects/*/[uuid].jsonl` | local |
| akebot | `~/.claude/projects/*/[uuid].jsonl` | Mac jump (`ssh sethgho@100.64.185.78` → `ssh akebot`) |
| Seth's Mac | `~/.claude/projects/*/[uuid].jsonl` | `ssh sethgho@100.64.185.78` |

*(Cadbury is not online yet. The schema below carries an `agent` field so he slots in
without a rewrite.)*

**Measured volume (2026-07-22, 24h window):** wilson 31 sessions / 119 MB · Mac 38 / 72 MB
· akebot 172 / 38 MB. **~240 sessions, ~230 MB/day.** Far too much to hand an LLM raw —
Stage 1 exists to get this down by ~3 orders of magnitude.

**Record types in the JSONL:** `user`, `assistant`, `attachment`, `queue-operation`,
`ai-title`, `last-prompt`, `mode`. Useful signal lives in `user`, `assistant` (tool calls
and their results), and `ai-title` (a free one-line session summary).

**Privacy — non-negotiable.** These logs contain family, finances, health, and
credentials. Harvest must:
- **Redact before anything leaves the machine it came from.** Secrets, tokens, keys, IPs,
  full paths under `~/Obsidian`, `#family`/`#finances` project dirs.
- **Hard-exclude by project path**, with an allowlist rather than a blocklist —
  infrastructure/dev projects opt *in*.
- Never place raw transcript text in the published strip. The strip carries a *joke about*
  an incident, never a quote from a session.

---

## Stage 1 — Observation mining (two veins)

Reduce ~230 MB/day to material an LLM can reason about. Per the `scheduled-workflows`
principle, **scripts collect, the LLM analyses** — but what the scripts collect changed
with the reframe: the primary output is no longer a list of events, it's a **ledger of
aggregate signals** from which higher-level observations get drawn.

### Vein A — the condition (primary)

**1a. Deterministic ledger (scripts, free).** Nightly, per machine, over the allowlisted
+ redacted logs, compute a small metrics ledger — counts and ratios, no transcript text:

- Sessions, hours-of-day active, human-turns vs agent-turns, words typed by the human vs
  words produced by the agents
- Retry loops (same command ≥3×), tool-failure rates, permission prompts answered
- Confidence phrases (`"that should fix it"`, `"you're absolutely right"`) per day
- Waiting: total gap time between agent-finished and human-responded
- Scope drift: intent size (first user message) vs diff size, distribution not anecdote
- Cost/limit events, compactions, session restarts
- Rolling 7/30-day deltas on all of the above — trends, not snapshots

The ledger is tiny (KB, not MB) and is the **only thing that crosses machines** besides
Vein B records.

**1b. Observation drafting (LLM, on the ledger only).** **Model policy: any stage that
summarizes or reads session-derived text uses the smallest available model** (Haiku-class,
explicitly pinned — never inherited from the session, per the deep-research 3M-token
lesson in HANDOFF.md). Frontier models are reserved for Stage 4 joke construction and the
Stage 5 judge panel, where the craft actually lives. Here, a small model reads the
aggregated ledgers + trend deltas and proposes **observations about the condition** — expectation/
reality gaps at the level of the *life*, not the day. The register to aim for:

- "He types two sentences a day and reviews ten thousand words of confidence."
- "Three agents, three machines, and the human's job is now saying 'yes' to prompts."
- "The more help he has, the later he's up."

Each observation must still carry a Dean pair — what this way of working *promised* vs
what it *is* — because that's the Connector-bearing gap Stage 4 needs.

### Vein B — the extraordinary event (secondary, high bar)

The old incident extractor survives, with the threshold moved way up: a specific event
gets a record **only if it's an outlier** (the 28 GB OOM kill, a 3M-token workflow, a
session limit hit twice in a day). Signals: reversals, blast radius, wrong-thing-fixed,
cost blowups — but expect **<1 qualifying event per week**, not dozens per day. Ordinary
mishaps feed the ledger as counts; they do not get their own strip.

**Output — one record per observation (both veins):**

```json
{
  "id": "2026-07-22-approval-economy",
  "date_range": "2026-07-16..2026-07-22",
  "kind": "condition",
  "agents_involved": ["wilson", "akebot", "mac"],
  "observation": "The human's main output this week was consent",
  "evidence": {"permission_prompts": 214, "human_words": 3100, "agent_words": 41000},
  "the_expectation": "Agents free the developer to do the real work",
  "the_reality": "The real work is now approving other people's work",
  "recognizability": "high"
}
```

`kind` is `"condition"` or `"event"`; event records keep the old per-incident fields
(`host`, `signal`, `duration_min`, `severity`). `the_expectation` / `the_reality` remain
the load-bearing fields — a pre-dug **Target Assumption / Reinterpretation** pair, which
is exactly what Dean's model needs. `evidence` holds the ledger numbers that back the
claim; an observation with no numbers behind it is a guess, not an observation.

---

## Stage 2 — Premise scoring (gate #1)

Score each observation *before* any writing. Smallest available model (same policy as
Stage 1b), structured output.

| Criterion | Why |
|---|---|
| **Recognizable** | Adams' near-mandatory dimension. Would any dev living the agent-assisted life nod — not just someone who was there? |
| **Reversal strength** | Is there a real gap between what this way of working promises and what it is? |
| **Legibility** | Can the *emotion* land without domain knowledge? (bible §3) |
| **Engine fit** | Does it run on a standing character engine? |
| **Freshness** | Have we published this shape in the last 30 strips? |
| **Mission fit** | Does the strip leave Seth reading as a frontier engineer reporting back, not a victim of his tools? (bible, "The mission") |

**Gate:** drop anything without a genuine expectation/reality gap. A metric that's merely
interesting, or an event that simply broke with no irony, is not a premise. Condition
observations accrue over days — expect **a handful of viable premises per week**, with
Vein B events rarer still.

---

## Stage 3 — Routing (the rescue mechanism)

For each surviving premise, decide the cast configuration — **before** writing balloons.

1. **Who owns the flaw?** Whose standing engine does this incident run on?
2. **Who plays straight, who plays comic?** (bible §2 — name both; never two comics.)
3. **Try re-casting a weak premise before discarding it.** Schulz's routing insight: flat
   material becomes funny assigned to the character who doesn't realise it's flat. Our Ake
   is the Snoopy slot — his alarm makes dry material funny.

Generate **the same premise routed 3 different ways**, then pick. Cheap, and it directly
implements the repertory-company principle.

---

## Stage 4 — Joke construction (structured, not freeform)

Do **not** ask a model for "a funny 4-panel strip." Ask it to fill Dean's skeleton, which
is checkable:

```json
{
  "connector": "「uptime」 — the thing that's up, and the thing that's over",
  "target_assumption": "Wilson is reporting good news about the server",
  "reinterpretation": "He's reporting how long it has been down",
  "straight": "cadbury", "comic": "ake", "punchline_owner": "wilson",
  "technique": "misunderstanding",
  "dimensions": ["recognizable", "clever"],
  "panels": [
    {"n":1,"staging":"...","balloons":[{"who":"seth","text":"..."}]},
    {"n":2,"staging":"...","balloons":[]},
    {"n":3,"staging":"...","balloons":[]},
    {"n":4,"staging":"...","balloons":[{"who":"wilson","text":"..."}]}
  ]
}
```

**Word budget** (compression is the whole craft; these are house caps, not sourced law):
- **≤ 20 words per panel**, **≤ 12 words per balloon**, **≤ 2 balloons per panel**
- **≤ 50 words total** across the strip
- Panel 4's final balloon is the punchline — and should be the **shortest** thing in it.

**Placement defaults, from the Peanuts measurements** (bible §1):
- Panel 1 has text (~98%). Panel 4 has text (~92.5%) and carries the reveal.
- Panels 2–3 are the quiet middle. **Panel 3 defaults to silent** unless it's the turn.

---

## Stage 5 — The gate (the part that matters)

Run the bible's 11-point checklist as an **automated rejection pass**, with the mechanical
ones as hard code and the judgment ones as an adversarial LLM panel:

**Hard-coded (free, deterministic):**
- Word/balloon caps
- Punchline is in panel 4, last balloon
- ≥2 of Adams' six dimensions declared
- Technique is one of Bushmiller's six
- Engine referenced is a standing engine
- Freshness: shape not repeated within 30 strips

**Adversarial panel (independent judges, prompted to *reject*):**
- **Connector judge** — "name the double meaning; if you can't, reject."
- **Swap judge** — "rewrite the punchline in another character's mouth. Did the joke
  survive? If yes, reject."
- **Symmetry judge** — "state how panel 4 answers panel 1. If it doesn't, reject."
- **Outsider judge** — "you have never written software. Is this legible?"

**Ship only on unanimous survival.** Per the discard gate: **an uncertain gag is a no.**

---

## Stage 6 — Render & publish

Hand the approved script to the existing art pipeline (`comfy-cloud-workflows`,
`comic-panel` recipe): character sheets as reference conditioning, house style block
verbatim, one fixed camera for the whole strip (bible §1), balloons composited by
`compose-strip.py` — **never** drawn by the model.

Publish to `the-cast` → `showcase-latest.png` → cast.sethgholson.com, which already
autodeploys.

---

---

## Stage 7 — Engine mining (deferred; the recurring-gag play)

**Not built during calibration.** This is the medium/long-term arc, and it only works once
there's a corpus: **target ~50 published strips before the first mining pass.**

The premise: Schulz's twelve devices were credited retrospectively — they accreted from
strips that worked. So we don't invent running bits; we **discover which ones we already
have.**

Inputs, all of which the earlier stages produce for free if we retain them:

- Published strips + their structured scripts (technique, dimensions, engine, Connector)
- **Seth's approve/reject decisions and reasons** (Decision 1's rejection log)
- Whatever audience signal exists by then

What the pass looks for:

1. **Recurring shapes** — Connectors, techniques, or premise types that keep reappearing
   in *approved* strips. A shape that survives the gate repeatedly is a latent engine.
2. **Character gravity** — which character keeps owning the punchline for a given premise
   type. That's a routing rule discovering itself.
3. **Callback candidates** — a specific prop, phrase or failure the archive has used ≥3
   times. Per the rule of three, the third appearance is where a thing stops being a
   coincidence and starts being a bit.
4. **Dead engines** — seed engines from `bible.md` §2 that never actually generated an
   approved strip. Retire them; they were guesses.

Output: a proposed amendment to the bible's engine table — **promotions from the archive,
retirements of guesses** — for Seth to approve. Only then do strips start referencing each
other.

**Guard:** the football rule still applies to any promoted engine. If mining surfaces a
bit that only works by resolving, it's a one-off, not an engine.

---

## Cadence

Daily harvest, **not** daily publication. The pipeline runs every night; it publishes only
when something clears the gate. A 2–3 strip week of strips that land beats seven that
don't — and the "coming soon" tiles on the site are already built for an irregular
archive.

---

## Decisions (Seth, 2026-07-22)

1. **Human in the loop, to start.** Seth approves every strip before publish until the
   tone and humor are dialed in. The gate does not stand alone yet — it *proposes*, he
   disposes. Revisit after the calibration window (~20 strips); the gate earns autonomy by
   demonstrably matching his taste, not by elapsed time.
   → Stage 5 ends in a **review queue**, not a publish. Stage 6 fires only on approval.
   → **Log every rejection and why.** His no's are the training signal for tuning the
   gate's thresholds — a rejected strip is more informative than an approved one.
2. **Privacy: allowlist, redact at source.** Confirmed. Projects opt *in* to being mined;
   nothing is mined by default. Redaction happens on the machine the log came from, before
   transport.
3. **Agents can be wrong about themselves.** Confirmed — the cast may fail in ways the
   real agent would dispute. Self-deprecation is fully in bounds; see bible §2.

4. **Standalone now; recurring gags derived later.** Every strip stands alone through the
   calibration period. Recurring bits are a **medium/long-term play, mined from the
   archive rather than designed** — we need a corpus of things that landed before we can
   see which shapes recur. See Stage 7.
5. **The brand line.** No Ethos, its vendors, clients or internal systems — generalize the
   incident, keep the shape, drop the identifiers. Zeitgeist-scale brands (Anthropic
   pricing, Fable releases, the discourse of the week) are fair game. Full rule and the
   reasoning in `bible.md` §3.
   → Stage 1 **redacts organizational identity at harvest**, alongside secrets. By the
   time a premise reaches scoring, "which vendor" is already gone.
   → Stage 5 gets a hard **brand-line check**; a zeitgeist reference must be a
   *public* matter, not a private one involving a public company.

## Still open

- Nothing blocking. Revisit the calibration window (Decision 1) after ~20 strips and the
  engine-mining threshold (Stage 7) after ~50.
