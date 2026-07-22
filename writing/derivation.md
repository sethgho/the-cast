# Session Logs → Strip: the derivation spec

How a day of real Claude Code session data becomes one gated strip. `bible.md` defines
what a good strip *is*; this defines the pipeline that produces one.

**Design stance:** the pipeline's job is **not** to be funny. Its job is to (a) surface
the day's genuinely strip-worthy incidents, (b) route them through fixed characters, and
(c) **reject relentlessly**. Per the bible's discard gate, publishing nothing is an
acceptable daily outcome. A pipeline that always ships is broken.

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

## Stage 1 — Incident extraction (deterministic, cheap)

Reduce ~230 MB/day to a few dozen structured **incidents**. This stage is scripts, not a
model — per the `scheduled-workflows` principle that scripts collect and the LLM analyses.

**Mine for signals that correlate with comedy** — which, per Adams, means starting from
annoyance and recognizability:

- **Reversals** — a confident assertion followed by its contradiction. (`"that should
  fix it"` → same error again). The single richest vein; this *is* the Connector pattern.
- **Repetition** — the same error ≥3 times in a session. Rule-of-three material, free.
- **Escalation** — a task whose scope grew: 1-file intent → 40-file diff.
- **The long silence** — a >20-minute gap between tool call and next action.
- **Blast radius** — a small change that broke something unrelated (the "one small change"
  engine, verbatim).
- **Wrong-thing-fixed** — time spent on X when the cause was Y (this session's Superset
  "power blip" that was actually a second org appearing: textbook).
- **Human sentiment spikes** — profanity, all-caps, `"why is this"`, `"wtf"`.
- **Cost/limit events** — OOM kills, session limits, runaway spend.

**Output — one record per incident:**

```json
{
  "id": "2026-07-22-akebot-oom",
  "date": "2026-07-22",
  "host": "akebot",
  "agent": "wilson",
  "project": "the-cast",
  "signal": "wrong-thing-fixed",
  "one_line": "Diagnosed vanished workspaces as a power blip; was actually a 28GB OOM kill",
  "the_expectation": "A reboot took the daemon down",
  "the_reality": "One agent ate 28 of 31 GB and the kernel shot it",
  "human_reaction": "asked if superset died again",
  "duration_min": 45,
  "severity": "medium",
  "recognizability": "high"
}
```

`the_expectation` / `the_reality` are the important fields — they're a pre-dug
**Target Assumption / Reinterpretation** pair, which is exactly what Dean's model needs.

---

## Stage 2 — Premise scoring (gate #1)

Score each incident *before* any writing. Cheap model, structured output.

| Criterion | Why |
|---|---|
| **Recognizable** | Adams' near-mandatory dimension. Would a dev who wasn't there nod? |
| **Reversal strength** | Is there a real gap between expectation and reality? |
| **Legibility** | Can the *emotion* land without domain knowledge? (bible §3) |
| **Engine fit** | Does it run on a standing character engine? |
| **Freshness** | Have we published this shape in the last 30 strips? |

**Gate:** drop anything without a genuine expectation/reality gap. An incident where
things simply worked, or simply broke with no irony, is not a premise. Expect **most days
to yield 0–3 survivors from a few dozen incidents.**

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

## Output modes — the spec assumed one, there are two (2026-07-22)

Stages 1–5 are identical for both. Only the format and the gate emphasis differ.

**A. The strip (default).** 4 panels, 1×4, published to the site. Must stand entirely on
its own — the reader gets no accompanying text. Full 12-gate checklist applies.

**B. The companion.** A strip made to run *with* a post, where the text carries the thesis
and **the strip's job is to earn attention and be liked, not to make the argument.** This
mode is real and proven — the first one shipped 2026-07-22 alongside a thread about
agentic coding. Differences that matter:

- **2×2 square, not 1×4.** A four-panel row is unreadable at tweet size; the square fits
  the card and the panels stay legible. Verified.
- **The gate relaxes on "does it make the point" and tightens on "is it funny."** In this
  mode a strip that restates the thesis is *worse* — the reader already has the argument
  in text. Redundancy reads as a lecture.
- **Tonal pairing is the real craft decision.** A humble strip under a sharp post works
  (the picture disarms, the text provokes). Two sharp registers together read as smug, and
  the strip gets scrolled past. Decide which of the two is carrying the humility.
- Word budgets and the Connector gate still apply unchanged.

**Production note:** for a one-off companion, generating the balloons *in* the image
monolithically is acceptable and works — Nano Banana Pro rendered four balloons of correct
lettering in one pass. For the site pipeline, balloons stay composited by
`compose-strip.py`. Also: the model draws its own outer panel border regardless of
instruction; crop inside it, or keep it, since it reads as a clipped newspaper square.

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
