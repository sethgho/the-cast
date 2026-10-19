"""Wilson in Halloween costumes, drawn as pumpkin-carving stencils. Orchestration only:
the graph draws the stencil and posterizes it to cut / shave / leave in-graph."""
import json, subprocess, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sethmoji import submit, GPU
from build_workflows import NEGATIVE, STEPS

OUT = "/home/wilson/scratch/figurines/pumpkin/costume"
COSTUMES = {
    "vampire": "a Dracula vampire costume: a black cape with a tall pointed collar rising behind his head, two long sharp fangs under his glowing eyes, and a bat flying overhead",
    "witch": "a witch costume: a tall crooked pointed witch hat with a buckle in place of his bucket hat, a broomstick leaning on the fence, and a crescent moon behind him",
    "skeleton": "a skeleton costume: his bucket hat on a grinning skull mask, bones painted on the fence pickets like ribs, and a spider hanging from the brim on a thread",
    "reaper": "a grim reaper costume: a ragged hooded cloak pulled up over his bucket hat, a long curved scythe held beside the fence, and a few bats behind him",
}

def graph(prompt, seed, tag):
    return {
     "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_S.gguf"}},
     "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "strength_model": 1.0, "model": ["2", 0]}},
     "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
     "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
     "6": {"class_type": "LoadImage", "inputs": {"image": "pumpkin-wilson.png"}},
     "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 896, "height": 1152, "batch_size": 1}},
     "20": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["4", 0], "vae": ["5", 0], "image1": ["6", 0]}},
     "21": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": NEGATIVE + ", photograph, text, lettering, colour, gradients, fine hatching", "clip": ["4", 0]}},
     "22": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["3", 0], "positive": ["20", 0], "negative": ["21", 0], "latent_image": ["11", 0]}},
     "23": {"class_type": "VAEDecode", "inputs": {"samples": ["22", 0], "vae": ["5", 0]}},
     "99": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"pumpkin/{tag}", "images": ["23", 0]}},
    }

PROMPT = ("Image 1 shows Wilson, a cartoon character: a shadowy figure whose face is a dark void with two big "
          "round eyes, wearing a floppy bucket hat, peeking over a white picket fence that hides his body. "
          "Draw Wilson dressed in {c}, still peeking over his picket fence, as a printable jack-o'-lantern "
          "carving pattern. He looks spooky and menacing, with sinister narrowed glowing eyes. "
          "Front view, centred, the whole figure and fence in frame on plain white paper. "
          "The pattern is bold flat solid black shapes and a few flat mid-grey shapes on plain white paper, "
          "with thick white gaps separating the shapes like a pumpkin carving stencil.")

subprocess.run(["scp", "-q", "/home/wilson/dev/the-cast/cast/wilson/assets/expressions/attentive.png", f"{GPU}:comfyui/input/pumpkin-wilson.png"], check=True)
for name, c in COSTUMES.items():
    for seed in (11, 23):
        tag = f"{name}-{seed}"
        submit(graph(PROMPT.format(c=c), seed, tag), "99", "images", f"{OUT}/{tag}.png")
        print(tag, flush=True)
