# local-comfy — posing the cast on our own GPU

These are ComfyUI workflows for redrawing a cast member in a new pose, expression or scene
**locally on gpu-worker**, from that character's reference plate. No cloud credits.

The model is **Qwen-Image-Edit-2511** (Q4_K_S GGUF) with a **Lightning** step-distill LoRA.
Measured on the RTX 3080 Ti (12GB): **~48s at 8 steps**, **~25s at 4 steps**, 1024×1024.

## Files

| Path | What it is |
|---|---|
| `workflows/<id>-pose-and-scene.json` | Load this in the ComfyUI builder. |
| `api/<id>-pose-and-scene.api.json` | Same graph, API format — for scripts and CI. |
| `plates/` | The reference images the workflow expects in `ComfyUI/input/`. |
| `build_workflows.py` | Generates both files from one character spec. Edit here, not the JSON. |
| `smoke_test.py` | Runs the API graph through all four switch combinations. |

## The control surface

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

A third, smaller one: `JoinImageWithAlpha` inverts its mask (ComfyUI's convention is 1 = masked
out), so `RemoveBackground` needs an `InvertMask` after it or the *figure* goes transparent.
