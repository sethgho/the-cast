# the-cast

The canonical cast of [sethgholson.com](https://sethgholson.com) — a vaudeville troupe,
est. 1933. This repo is the single source of truth for who these characters are: their
turnaround sheets, the prompt tokens that render them, their props, and (eventually) their
personalities.

Live at **[cast.sethgholson.com](https://cast.sethgholson.com)**, with a machine-readable
registry at [`/cast.json`](https://cast.sethgholson.com/cast.json).

## The cast

| id | who |
|---|---|
| `seth` | The operator. Optimistic, mid-deploy, one small change from disaster. |
| `wilson` | The neighbour. Steady, dry, faceless — emotes entirely through posture. |
| `cadbury` | The butler. Unflappable. Arrives with tea precisely as things collapse. |
| `ake` | The sysop. A goldfish. Over-caffeinated, alarmed, usually right. |

## Layout

```
cast/<id>/
  character.yaml   # identity: name, role, prompt tokens, props, accent, notes
  sheet.png        # canonical turnaround sheet
  headshot.png     # canonical square avatar — 1024x1024, purpose-rendered
style/vaudeville-1933.md    # the locked house style — shared by every renderer
schema/character.schema.json
scripts/build-site.py       # registry -> public/ (validates first; bad YAML fails the build)
showcase/                   # strips worth showing off
```

## How this gets consumed

- **Generation pipelines** feed `sheet.png` as reference conditioning and inject `tokens`
  verbatim into prompts. The sheet carries silhouette and costume; the tokens stop a model
  inventing a different design when conditioning runs weak. Both halves are required —
  dropping either one measurably breaks identity (we tested).
- **Avatars** use `headshot.png` — a rendered square portrait, never a crop of the sheet.
  Cropping the sheet is what we did first and it cut off chins and caught stray arms. Each
  headshot is generated from the sheet as reference plus the character's `tokens`, centred
  with even margin and bleeding to all four edges, so a consumer can mask it to a circle or
  a square without losing the face.
- **Scripts and agents** read `cast.json` — stable URLs, no YAML parsing, CORS enabled.
- **Sites** hotlink `https://cast.sethgholson.com/<id>/sheet.png` behind Cloudflare caching.

The generation machinery itself (ComfyUI Cloud recipes, comic-strip compositor, agent
skill) lives in a separate private repo; this one stays clean: identity and assets only.

## Adding or changing a cast member

1. Edit or add `cast/<id>/character.yaml` (schema-validated), `sheet.png` and `headshot.png`.
2. `python3 scripts/build-site.py` — fails loudly on schema or missing-sheet errors.
3. Commit, push. Deploy picks it up; the site and `cast.json` regenerate from the registry.

Sheets keep stable URLs when regenerated — consumers never chase filenames.

## Style

One rule above all: the style block in `style/vaudeville-1933.md` is **shared and never
varied per render**. Vary staging and camera, not style. Ake is the only colour accent in
an otherwise sepia world; keep it that way.
