# local-comfy — posing the cast on our own GPU

These are ComfyUI workflows for redrawing a cast member in a new pose, expression or scene
**locally on gpu-worker**, from that character's reference plate. No cloud credits.

The model is **Qwen-Image-Edit-2511** (Q4_K_S GGUF) with a **Lightning** step-distill LoRA.
Measured on the RTX 3080 Ti (12GB): **~48s at 8 steps**, **~25s at 4 steps**, 1024×1024.

## Files

| Path | What it is |
|---|---|
| `workflows/<id>-<kind>.json` | Load this in the ComfyUI builder. |
| `workflows/<id>-<kind>.app.json` | The same file under the name the Apps list reads. |
| `api/<id>-<kind>.api.json` | Same graph, API format — for scripts and CI. |
| `plates/` | The reference images the workflows expect in `ComfyUI/input/`. |
| `build_workflows.py` | Generates every file from one character spec. Edit here, not the JSON. |
| `smoke_test.py` | Runs the API graph through all four switch combinations. |

Both kinds carry the same `SCENE` / `TRANSPARENT PNG` / size / seed controls, numbered after
their own fields.

Two kinds today, both off one `build()`:

- **`pose`** — full-body figure from the character plate. Field: `YOUR SHOT`.
- **`headshot`** — head-and-shoulders portrait from the headshot plate. Fields:
  `HIS EXPRESSION`, `FRAMING`. Expression control is strong — "weary, heavy lids, one brow up,
  mouth a flat line" reads as a different performance, not a different person.

## App Mode

The workflow ships with `extra.linearMode: true`, so opening it drops you straight into
ComfyUI's [App Mode](https://docs.comfy.org/interface/app-mode) — a plain form, no graph. The
form is defined by `extra.linearData` and is generated from the same script:

    1 · YOUR SHOT   2 · SCENE   3 · YOUR SCENE   4 · SCENE PLATE
    5 · TRANSPARENT PNG   6 · OUTPUT SIZE (w/h)   7 · SEED + STEPS  →  RESULT

Field labels are the `label` on each promoted widget's input slot — the same field the UI's
right-click → Rename writes. Two traps when authoring them by hand:

- A top-level `"definitions": {"subgraphs": []}` key sends the loader down the subgraph path,
  which rebuilds every widget input slot and **throws the labels away**. Omit the key.
- ComfyUI keeps an already-open workflow tab in memory. After overwriting the file on disk,
  close the tab and reopen it, or the old label-less copy is what you see.

Use the breadcrumb menu at the top to leave App Mode and see the graph. App Mode has no
conditional fields, so `3 · YOUR SCENE` and `4 · SCENE PLATE` stay on screen even when `SCENE`
is off — they are simply ignored. Share links are Comfy-Cloud-only; on our box App Mode is
local-only.

## The control surface (graph view)

Three groups, colour-coded:

- **① blue — CHARACTER & STYLE LOCK.** Who he is and how the house draws him. Leave alone.
- **② green — YOUR SHOT.** Two text boxes and two switches. This is the whole job.
  - `▶ YOUR SHOT` — what he is doing, one sentence.
  - `▶ YOUR SCENE` — where he is. Only read when `SCENE` is on.
  - `SCENE` — off = flat mid-grey plate; on = the scene plate (image 2) plus your scene text.
  - `TRANSPARENT PNG` — cuts the figure out with BiRefNet and saves RGBA.
- **③ amber — OUTPUT.** Size lives on `OUTPUT SIZE`; seed and steps on the sampler.

The final prompt is assembled as:

    CHARACTER LOCK + YOUR SHOT + (plate clause | scene lead-in + YOUR SCENE) + STYLE LOCK

Read it off the `FINAL PROMPT` node any time you want to see exactly what the model got.

## Setup on a fresh box

Models, all under `ComfyUI/models/`:

| Folder | File |
|---|---|
| `unet/` | `qwen-image-edit-2511-Q4_K_S.gguf` — [unsloth/Qwen-Image-Edit-2511-GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) |
| `loras/` | `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors` — [lightx2v](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) |
| `text_encoders/` | `qwen_2.5_vl_7b_fp8_scaled.safetensors` — [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) |
| `vae/` | `qwen_image_vae.safetensors` — same repo |
| `background_removal/` | `birefnet.safetensors` — [Comfy-Org/BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet) |

Custom node: [`ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) (city96). Everything else
is comfy-core.

Then copy `plates/*.png` into `ComfyUI/input/`, and the workflow JSON into
`ComfyUI/user/default/workflows/` so it shows up in the sidebar.

On gpu-worker: `sudo systemctl start comfyui` (it evicts texture-api, whisper and kokoro to
claim the card), and `sudo systemctl stop comfyui` to hand the GPU back.

## Two rules learned the hard way

1. **Never wire a second character plate into `image2`.** Qwen treats every reference as a
   *subject*, not a style — two character plates get merged into one creature. `image2` is for
   a scene, nothing else.
2. **State the style, and the colours, in words.** With only a reference image the palette
   drifts modern — Seth's near-black t-shirt came back mid-teal until the lock said
   "very dark olive charcoal, near-black". Positive phrasing works; "never bright green" does not.

On headshots, **SCENE only works if `FRAMING` leaves room** — at a tight crop the scene has
nowhere to go and comes back as speckled paper. "head and shoulders with room to breathe, the
place visible behind his shoulders" puts the auditorium in.

A third, smaller one: `JoinImageWithAlpha` inverts its mask (ComfyUI's convention is 1 = masked
out), so `RemoveBackground` needs an `InvertMask` after it or the *figure* goes transparent.
