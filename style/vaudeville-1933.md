# Style anchor — "Vaudeville 1933"

This block is **shared by every panel and never varied**. Varying it per panel is the
fastest way to make a strip look like four unrelated drawings. Vary `staging` and `camera`
only.

## On the word "rotoscoped"

Worth naming the tension: the character sheets are **rubber-hose cartoon**, not rotoscoped.
True Fleischer rotoscoping (1933 — Cab Calloway in *Minnie the Moocher*) meant tracing live
film, which reads as naturalistic weight and timing under cartoon inking.

For a still strip that's mostly a *posing* instruction, not a rendering one. So the block
below asks for rubber-hose designs posed with rotoscoped naturalism: real weight, real
balance, follow-through and drag in the limbs, contact shadows — rather than the floaty
symmetrical poses generic "1930s cartoon" prompting produces. That's the useful half of
"rotoscoped" for this format.

## STYLE_BLOCK (verbatim, prepend to every panel prompt)

```
A single panel from a 1933 newspaper comic strip, drawn in Fleischer-era rubber hose
cartoon style but posed with rotoscoped naturalism: believable weight, balance and
follow-through, as if traced from live vaudeville stage footage. Big simple rounded
cartoon forms with confident smooth ink outlines of even weight, large areas left open
with minimal interior rendering, shaded only with soft halftone dots. Printed in warm
sepia duotone on aged newsprint — muted grey-brown paper tone, soft foxing and age
spots, slightly uneven ink density. Theatrical stage lighting with simple deep cast
shadows. High contrast, period charm.
```

**History, so nobody reintroduces it:** this block originally asked for "dense
cross-hatching" and "vintage engraving sensibility." Those two phrases pulled Wilson and
Cadbury into detailed engraving-realism while Seth stayed cartoonish (2026-07-21 style
pass). "Shaded only with soft halftone dots" is the positive-phrasing replacement — it
excludes crosshatch without using a negation that FLUX-family models would ignore.

## EXCLUSION_BLOCK (append to every panel prompt)

FLUX.2 **ignores negative prompts entirely** and Nano Banana treats them loosely, so
exclusions have to be stated positively inside the prompt. This is not optional — if the
model draws its own balloons or borders, the compositor's lettering lands on top of them.

```
The panel art fills the entire frame edge to edge with no border, no frame, no margin.
The image contains no speech balloons, no word balloons, no caption boxes, no narration
boxes, no title lettering, no signature and no watermark. Any writing that appears is
only lettering physically painted or printed on an object within the scene itself.
```

### Diegetic vs non-diegetic text — the distinction that matters

Do **not** blanket-ban text. Two different things:

- **Diegetic** — words that exist as objects in the world: a mug lettered CODE FUEL, the
  SYSOP cap, a scroll labelled PR 3.5, a CI board reading GREEN, a shipping box stencilled
  SLOPSHOP. These are **props, and often the joke**. They belong in the render; describe
  them explicitly in `staging` with the exact string in quotes.
- **Non-diegetic** — the comic's own furniture: speech balloons, caption boxes, panel
  borders, the masthead. These are **composited** by `bin/compose-strip.py`. If the model
  draws one, the composited version lands on top of it and the panel is wasted.

Model-rendered diegetic text is imperfect at small sizes — expect to re-roll a panel whose
prop lettering comes out garbled. That's a known cost of keeping the joke in the art.

## Character tokens

**Do not copy character descriptions into this file.** The single source of truth is
`cast/<id>/character.yaml` (`tokens`, `props`, `notes`) — also served machine-readable at
https://cast.sethgholson.com/cast.json. Read them from there at render time; a copy here
would drift the first time a character changes. (It did. That's why this section is now a
pointer.)

The rule that stays: tokens ride WITH the reference sheet in every prompt. The sheet
carries silhouette and costume; the tokens stop the model inventing a different design
when reference conditioning runs weak. Both halves, always.

## Palette

Sepia duotone throughout. **Ake is the only warm accent** — his body carries a muted ochre
orange. Everything else stays in the grey-brown range. That single spot of colour is the
strip's visual signature; don't spread it to the other characters.
