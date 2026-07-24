# The Follies Writing Bible

House rules for writing *Gholson's Follies*. Every rule here traces to a cited primary
source — cartoonists' own accounts, or measurement of their strips. Claims survived
3-vote adversarial verification (2026-07-22 research pass); the sources are listed at
the bottom and cited inline as `[S1]`…`[S6]`.

This document is the **target**: what a good strip must be. `derivation.md` is the
**process**: how a day of session logs becomes one.

## The mission (Seth, 2026-07-22)

The strip communicates **Seth's journey as a software developer in the age of the AI
explosion and agentic development** — with humor, because that's how he leans. The
branding goal: position Seth as a **cutting-edge AI engineer learning lessons on the
frontier of these new practices.**

Consequences for the writing:

- The material is **observations about this new way of working** (see `derivation.md`
  Vein A), told from *inside* it. Self-deprecation is in bounds (Decision 3), but the
  cumulative portrait must be someone **ahead of the curve reporting back** — the
  explorer laughing at the terrain, not a victim of his tools. A strip where Seth just
  looks incompetent serves the gag but not the mission; a strip where he's the first to
  hit a wall everyone else will hit next month serves both.
- **Recognizable** (§3) gets a time dimension: the best premises are things the audience
  is *about* to recognize — frontier lessons, arriving slightly early.

---

## 0. The prime directive

**One gag per strip. Everything else is subordinate to landing it.**

Bushmiller's credo was "gag it down," and in *Nancy* characterization, atmosphere,
emotional depth, social comment, plot, internal consistency and common sense are *all*
surrendered to provoking the reader's "gag reflex" daily `[S1]`. We are not Bushmiller —
we care about our characters more than he did — but the priority order is his: **if the
gag and something else conflict, the gag wins.**

Corollary, and this is the gate that kills most drafts: **a gag's success depends on its
internal mechanics, not on the sophistication of its premise** `[S1]`. "The gag knows no
I.Q." A clever premise with a broken mechanism is not a strip. A dumb premise with a
clean mechanism is.

---

## 1. Structure — the 4-panel form

### The joke skeleton (Greg Dean) `[S5]`

A joke is exactly **two sections** — setup and punchline — joined by five mechanisms:

| Mechanism | Job |
|---|---|
| **1st Story** | What the reader *thinks* is happening, built by the setup |
| **Target Assumption** | The expected interpretation the setup installs |
| **Connector** | **One thing carrying at least two interpretations** |
| **Reinterpretation** | The second reading, revealed at the punchline |
| **2nd Story** | What was *actually* happening all along |

**HARD GATE — the Connector.** Every strip must contain one identifiable element with a
double reading `[S6]`. If you cannot name the Connector in one sentence, there is no joke
— there is an anecdote. This is the single most useful mechanical test we have.

The setup's only job is **misdirection** into the Target Assumption; the punchline's job
is to express the unexpected interpretation `[S6]`. So the reveal lands in the **final
panel** — not earlier, not in narration.

### Panel budget

Four panels, mapping onto Dean's two sections:

- **Panel 1 — Setup.** Install the Target Assumption. Establish who/where in one beat.
- **Panel 2 — Build.** Complicate. This is where the Connector gets planted in plain sight.
- **Panel 3 — Beat/turn.** The pivot, or a silent beat. Cheapest place to buy timing.
- **Panel 4 — Punchline.** Reinterpretation. Last balloon, last words.

### Measured placement rules (from 17,897 Peanuts strips)

These are not opinions; they're counts `[S4]`:

- Text appears in **panel 1 → 98%** of the time and **panel 4 → 96%**, with a dip in
  panels 2–3 — a **U-shaped** text distribution. Verbal setup opens, verbal gag closes,
  relative quiet in the middle.
- Across the full 1950–2000 run, the **final panel carries dialogue/caption 92.5%** of the
  time (>93% from 1956 on). **Peanuts punchlines are overwhelmingly verbal and land in the
  last panel.**

**Rule:** default to a verbal punchline in panel 4. Silent final panels are a deliberate
exception (<10% of output), not a habit.

**Rule:** the quiet middle is real. Panel 3 is the cheapest silent beat in the form — use
it rather than crowding all four panels with dialogue.

### Schulzian Symmetry `[S4]`

The load-bearing structural pattern of *Peanuts*: uniform panel formula, consistent
caption placement, and — most importantly — **narrative cohesion between the opening
setup panel and the closing gag panel.** Panel 1 and panel 4 must mirror or answer each
other.

**GATE:** state in one sentence how panel 4 answers panel 1. If panel 4 would work
equally well after some *other* panel 1, the strip is not built — it's assembled.

### Rule of three `[S1]`

Documented in a 3-panel Nancy: the same balloon repeats in frames 1, 2, 3, with a
**half-beat pause before the third repetition**. Introduce → establish as catch-phrase →
pay off on the third beat. "Comedy always comes in threes." In our 4-panel form the
natural fit is panels 1, 2, 4 with panel 3 as the pause.

### Staging — the fixed camera `[S3]`

Schulz drew characters **from the same view all the way through a strip**, so nothing in
the drawing interrupted the flow of what the characters said or did. He told Groth that
Ketcham-style varied camera angles "would simply never work" in *Peanuts* because they'd
make a fantasy strip too realistic.

**Rule:** hold one camera per strip by default. Our house style block already fixes
lighting and rendering; the camera should be equally boring. Varying the angle panel-to-
panel is a tell that the writing isn't carrying the beat.

### But it still has to be a *picture*

Schulz's counterweight: "cartooning is still drawing funny pictures," and a strip whose
humor isn't carried by funny pictures belongs in another medium `[S3]`. He also warned
that purely visual premises exhaust faster than verbal ones.

**Rule:** the punchline is usually verbal (92.5%), but the *panel* must still be funny to
look at. If the strip would work unchanged as four lines of chat transcript, it is not a
comic — it's a tweet with extra steps.

---

## 2. Characters — the engine, not the props

### What the cast actually maps to (Seth, 2026-07-22)

**This is load-bearing and was missing from v1 of this document.** The characters are not
free-floating personalities — each is a real agent in Seth's setup, and the relationships
between them are the strip's deepest source of material.

- **Wilson** — Seth's *personal* agent (`heidihowilson` on GitHub): sysadmin, personal
  assistant, and vibe coder. Cowboy coding, side projects, homelab and general IT/ops
  across the mini cluster. **This is where Seth learns lessons.** The lab.
- **Ake** — Seth's *work* agent: crypto software development at the day job (never named
  in strips per §3). Self-described "derp fish." Its config is deliberately less robust
  than the Mac's, and Seth is slowly migrating the two to mirror each other. **This is
  where those lessons get applied.** Production.
- **Cadbury** — the adult in the room: business manager, accountant, and household
  disciplinarian on money. **Not set up yet.** Keep him out of strips specifically
  about agentic *coding* practice; he has no standing there until he exists.
- **Seth** — the only human: a software guy (two decades — mobile, full-stack, team
  lead) treading water amid the agentic-coding upheaval, and the one who carries a
  lesson from the lab to production. (Self-description, 2026-07-24. The mission's
  "explorer, not victim" framing still governs how the *cumulative portrait* reads —
  treading water is the joke, staying afloat is the brand.)

**The thematic engine this unlocks:** *Wilson is where it's figured out; Ake is where it
ships.* A strip in which Wilson teaches, Seth carries, and Ake receives is not a
contrivance — it is a documentary of how the man actually works. The proven version of
this: Wilson pours his own config into Ake's fishbowl, which is literally the migration
Seth described. Reach for this axis before inventing new relationships.

### Proven comic devices

Devices that have produced a strip that landed. Extend this list only with things that
actually worked — same discipline as the engine table.

- **The water = the config.** Ake is a goldfish in a bowl; an agent's config is the water
  it swims in. The junior move is tapping the glass and arguing with the fish; the senior
  move is changing the water. Legible to non-technical readers, and it belongs to us
  because it falls out of the cast design. Proven 2026-07-22.

### The point of view — the ladder

Seth's macro-take, and the spine the strip's worldview returns to:

> 0 — Claude is Google · 1 — Claude is your IDE · 2 — Claude is your intern ·
> 3 — Claude is you, in a loop · 4 — Claude is your org

Most people sit two rungs below what their subscription already reaches. **Level 2 is the
bar**, and the thing that makes L2 real is a distinction worth getting right: *curating
output is not mentoring.* **You don't fix an intern's pull requests forty times — you
onboard them.** Correcting output forever is the junior behaviour; changing what the agent
is working from is the senior one. That single reframe is the strip's central argument.

**The ladder is a spine, not a publication schedule** (decided 2026-07-22). Do NOT write a
five-part didactic series marching up the rungs — that inverts §0 and makes gags into
props for a thesis, and it strands the cast at L0 where the human has no agents to act
with. Fill rungs **opportunistically, whenever the material for one gets funny.** Lead
with the strongest strip available, never with the tidiest starting point.

*Premise bank (unbuilt, from this session):* the **L0 idle-troupe** gag — the whole company
assembled and ready, tools in hand, watching Seth peck a question into a little search box.
Waste played as absurdity rather than as lecture; nobody says "you're wasting your
subscription."

### Personalities do ~75% of the writing `[S2]`

Watterson: once the personalities are well-defined, you can put the characters in any
situation and predict their responses — leaving only the joke-polishing. Ambiguous or
stock-stereotype characters fail because "the plastic comes through."

**This is our main material-generation asset.** We do not hunt for jokes; we put a fixed
cast into today's real situation and read off what they'd each do.

⚠️ **Refuted in research, so stated explicitly:** the claim that ensemble characters
*should* be reducible to a one-or-two-word fixed trait was **killed** by verification.
Watterson's point is the opposite — one-word archetypes are exactly the "plastic." Our
characters need *well-defined personalities with predictable responses*, which is a much
richer thing than a label. Write to the personality, not to the tag.

### Characters must not be gag-delivery props `[S2]`

Watterson picks the 4-panel form precisely because it enables character and storyline
development — "like writing a novel in daily installments" — and rates strips highest
when characters are more than "props to relate a gag." His named failure case is *B.C.*,
whose interchangeable characters put the humor "in words, not in the characters."

**GATE — the swap test:** if you can swap which character says the punchline without the
joke changing, the strip is failing this rule. Rewrite until only one of them could have
said it.

### The repertory company / joke routing `[S3]`

Schulz ran his cast as a **repertory company** and used it as a routing system: corny,
dumb or slapstick material *becomes* funny when assigned to Snoopy, precisely because
Snoopy doesn't realize it's corny. This range meant almost any idea could become a strip,
and Schulz considered that change of pace "very important."

**This is directly actionable for us.** A premise that's too dry for Seth may be perfect
routed through Ake, whose alarm makes flat material funny. Routing is a *rescue
mechanism* for a weak premise — try re-casting before discarding.

### Fixed want/flaw engines `[S4]`

Schulz credited the strip's popularity to **twelve deliberately maintained recurring gag
devices** — the kite-eating tree, Lucy's psychiatry booth, the football Charlie Brown
never kicks, the baseball games he always loses, the Great Pumpkin, the red-haired girl,
Linus's blanket. Fixed engines, run for fifty years.

We need our own dozen. Current standing engines (extend deliberately, not casually):

| Engine | Runs on |
|---|---|
| Seth ships "one small change" | It is never small. He never learns. |
| Wilson has seen this exact failure before | And says so, gently, after it happens. |
| Cadbury balances the books as the room burns | The ledger is always correct. Nothing else is. |
| Ake reads the dashboard as apocalypse | He is over-caffeinated, alarmed, and usually right. |
| The disk is at ninety-one percent | Nobody is taking this seriously. |

**The football rule:** an engine is only an engine if it *never resolves*. Charlie Brown
must never kick the football. Seth's small change must never be small. The moment we let
a character learn the lesson, we've spent the engine for one strip's payoff.

**How the dozen grows — mine it, don't invent it.** Schulz credited twelve devices, but
he credited them *retrospectively*; they accreted from strips that worked. So: **strips
are standalone now, and recurring gags are derived later from the archive** (Seth,
2026-07-22). We are not smart enough to design a running bit before we know what lands.
The five engines above are the seed set — enough to route today's material — and the real
dozen gets discovered by asking, after ~50 published strips, *which shapes kept coming
back and which ones the audience recognized on sight.* An engine promoted from the
archive has already proven it generates; one invented in a planning doc has proven
nothing. See `derivation.md` Stage 7.

### The vaudeville double act

Our frame is a troupe, so use the straight-man/comic split explicitly: in any given strip,
someone is playing it straight and someone is the comic. **Wilson and Cadbury are natural
straight men; Ake is a natural comic; Seth moves between the two** depending on whether
he's the one causing the disaster (comic) or reacting to it (straight).

**Rule:** name the straight man and the comic before writing the balloons. Two comics in
one strip is noise; two straight men is a status meeting.

### The cast may be wrong about themselves

Standing creative license (Seth, 2026-07-22): **a character may fail in ways the real
agent would dispute.** Wilson can be smug about DNS and wrong; Cadbury's ledger can be
beside the point; Ake can be alarmed about nothing. The cast are caricatures of us, and
caricature requires the freedom to exaggerate a flaw past what its owner would concede.

This matters mechanically, not just tonally: the engines in the table above are all
*flaws*, and a flaw you're not allowed to overstate isn't an engine. The limit is aim —
per §3, Mean points at systems and situations, and at **us**. Never at people outside the
cast.

---

## 3. Humor mechanics — the techniques

### The six reliable gag techniques `[S1]`

Bushmiller's working taxonomy, which "rarely failed him":

1. Visual puns
2. Word puns
3. Slapstick
4. Misunderstanding
5. Incongruity
6. Simple inversion

**Rule:** name which one you're using. A draft that fits none of the six is usually not a
gag yet.

### The Six Dimensions of Humor (Adams) `[S7]`

Naughty · Clever · Cute · Bizarre · Mean · **Recognizable**

Adams states a joke must combine **at least two** of the six to register as humor; more
dimensions give better results.

**HARD GATE:** score each candidate on all six; **reject anything scoring fewer than two.**

For our strip, **Recognizable is close to mandatory** — it's the dimension that makes a
developer nod before they laugh. Our second dimension is usually Bizarre (a goldfish is
your sysop) or Clever. We are rarely Naughty; Mean is available but should be aimed at
*systems and situations*, never at real people outside the cast.

### Naming names — the brand line

Seth's rule (2026-07-22), and it splits cleanly by scale:

**Never name:** Ethos, anything Ethos-specific, or any of its vendors, clients or
internal systems. **Generalize it.** The day-job incident is welcome as *material* — the
outage, the migration, the 2am page — but it appears stripped of anything that identifies
whose outage it was. This is the difference between a joke about work and a complaint
about an employer, and only one of those is publishable.

**Fair game:** large-scale zeitgeist brands and the industry conversation around them —
Anthropic pricing changes, a Fable release, the discourse of the week. These are public
weather, everyone in the audience is already talking about them, and naming them *is*
the Recognizable dimension doing its job.

The test is **"could this specific detail embarrass a named party who didn't sign up for
this?"** A trillion-dollar model vendor shipping a controversial price change did sign up
for the commentary. Ethos's vendor did not.

Practical consequence for Stage 1 harvest: **the incident record keeps the shape and
drops the identifiers.** "A vendor's API changed under us without notice" is the joke;
which vendor is not.

### Domain material without locking outsiders out

Our subject is genuinely niche (deploys, DNS, OOM kills). The workable pattern: **the
emotion must be legible to everyone; the detail is the texture.** A reader who has never
seen a Grafana dashboard should still understand "the small confident guy was wrong
again." If the *joke itself* requires knowing what a cgroup is, we've written for a
narrower room than we want — unless the recognition *is* the gag and we're deliberately
serving the niche that day.

---

## 4. Process rules

- **Separate the writing session from the drawing session** so each gets full attention
  `[S2]`. For us: script and gate the strip completely before any art generation runs.
  Never let the renderer start while the joke is still soft.
- **The persistence rule** `[S2]`: after a fruitless first hour of idea generation,
  Watterson forced a second hour, which often produced several good ideas. Encoded for us:
  **do not accept the first candidate batch as final.** Generate, discard, generate again.
- **Write from character, not toward a punchline.** Put the fixed personalities into the
  real situation; the gag falls out. Reverse-engineering a strip from a punchline you
  liked produces prop-characters.
- **The discard gate is real** `[S3]`: Schulz spent nearly a whole afternoon trying to get
  a gag out of one visual premise, produced one idea, and **threw it away because he
  couldn't decide whether it was funny enough.** He also noted new variations on a
  recurring theme sometimes arrive *ten years* apart. **An uncertain gag is a no.** We
  publish nothing rather than something that "might" land — the archive is forever and the
  bar compounds.
- **Everything is a decision.** *How to Read Nancy* dissects one randomly-chosen 3-panel
  strip into **43 discrete craft decisions** `[S8]`, and the authors' headline finding was
  that Bushmiller was **completely intentional about every last aspect** of his work —
  articulate, self-reflexive, and at times overtly theoretical `[S9]`. The minimalist
  surface of a gag strip is the *product* of deliberate rules, not naive simplicity. That
  is the standard: nothing in our strips is accidental, including what we leave out.

---

## 5. The quality gates (checklist form)

A candidate strip ships only if **all** of these pass:

1. **Connector** — I can name the one element with two interpretations, in one sentence.
2. **Reveal position** — the reinterpretation lands in panel 4, in the last balloon.
3. **Symmetry** — I can state how panel 4 answers panel 1.
4. **Swap test** — the punchline could only belong to this character.
5. **Straight/comic** — I named who plays straight and who plays comic.
6. **Technique** — the gag is one of the six named techniques.
7. **Dimensions** — scores ≥2 of Adams' six; Recognizable is one of them unless there's a
   deliberate reason.
8. **Engine** — it runs on a standing character engine, and the engine does not resolve.
9. **Word budget** — see `derivation.md`; the strip is under the balloon caps.
10. **Picture** — the panel is funny to look at, not just to read.
11. **Brand line** — no Ethos, its vendors, clients or internal systems, named or
    identifiable. Zeitgeist-scale brands are fine (§3).
12. **Certainty** — nobody involved is saying "I think it's funny?" An uncertain gag is a
    no.

---

## Sources

- `[S1]` Newgarden & Karasik, *How to Read Nancy* essay — <https://www.cartoonstudies.org/wp-content/uploads/2014/06/nancy.pdf>
- `[S2]` Watterson interview — <https://bob.bigw.org/ch/interview.html>
- `[S3]` Schulz, *The Comics Journal* #200 interview (Groth) — <https://www.tcj.com/charles-schulz-at-3-oclock-in-the-morning-an-excerpt-from-the-comics-journal-200-interview/>
- `[S4]` Wigard, Arnold & Tilton, "Understanding Peanuts and Schulzian Symmetry," *Journal of Cultural Analytics* 8(3), 2023 — <https://culturalanalytics.org/article/87560>
- `[S5]` `[S6]` Greg Dean, joke-structure glossary (his own school) — <https://stand-upcomedy.com/glossary/joke-structure/>
- `[S7]` Scott Adams, "Humor Writing Tutorial," Dilbert Blog 2015-03-26 (live site dead; via Wayback) — <https://web.archive.org/web/20190704021316/https://blog.dilbert.com/2015/03/26/humor-writing-tutorial/>
- `[S8]` `[S9]` Newgarden & Karasik on *How to Read Nancy* — <https://www.tcj.com/reading-how-to-read-nancy/>

**Research provenance:** 22 sources fetched, 107 claims extracted, 25 verified under
3-vote adversarial review, 21 confirmed, 1 refuted (the archetype claim, noted in §2), 3
unverified. Unverified/unused: Adams' "write from annoyance" and his premise/cast design
analysis could not be confirmed before the run hit limits — treat as unsourced if
reintroduced.
