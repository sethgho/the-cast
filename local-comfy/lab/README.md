# Lab ComfyUI graphs

Since 2026-10-30 these run on the **main ComfyUI** (:8188, upgraded to v0.43.2, with VideoHelperSuite,
`onetake_vfx` and `lidar-lab` installed). They were first proven in the lab ComfyUI (`/srv/models/comfy-lab`,
:8190), which is now a sandbox for future version bumps. The runbook is the `gpu-experiments` skill.

- `ltx25_i2v_audio_api.json`: LTX-2.5 image-to-video with generated speech and sound. OpenField's `video-i2v`
  graph, plus the audio decode it lacked and a temporally tiled VAE decode for 12 GB. Base 640×352 →
  2× latent upscale → 1280×704, 121 frames (5 s), about 100 s.
- `onetake_vfx_core_api.json`: One-take VFX (a clip's motion through Canny + IC-LoRA union control, frame 0 locked).
  It is rewired to core nodes (`LoraLoaderModelOnly` + `GetICLoRAParameters` + `LTXVAddGuide`), because the nodes
  it names are not in ComfyUI 0.43.2. It needs VideoHelperSuite and the repo's `onetake_vfx` pack in the lab.
  768×320, 73 frames, 56 s.
- `lab_run.py <graph> <out.mp4> node.input=value ...`: queues a graph on main (`COMFY_HOST` to override), fetches the video, and logs peak
  VRAM, host RAM and swap.
