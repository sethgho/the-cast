"""Render one H3 sprite clip for one character and one move.

    python3 hires_sprite.py <cid> <move> 832 20

## The idle pin, and why cycles are exempt

gary149/h3-game-sprites reached the same Mortal-Kombat method independently and pins BOTH ends of
every clip to the idle still. For a one-shot that is straightforwardly right: H3 holds identity
when it is anchored in a locked frame, the move returns exactly to neutral, and every move sharing
a first and last pose is what lets a state machine cut between them without a jump.

For a CYCLE it backfires, measured on Cadbury's walk. Pinning the end to a standing plate turns the
clip into stand -> walk -> stand, so the footage now contains a start-up and a settle that are not
part of the gait, and the period detector has to see through them. Self-similarity went from 0.0061
to 0.1223 against a random-pair baseline of 0.139 -- from an unmistakable cycle to no better than
chance. Their skill solves this the other way, with a loop mode that hunts the best-matching frame
pair inside the sustained middle; our clips are already briefed to loop, so leaving the end
unpinned costs nothing and measures 20x better.

So: pin one-shots, leave cycles open. A move that ends somewhere else entirely -- a KO, a faint --
pins its own end frame.
"""
import json, sys, time, os, subprocess
sys.path.insert(0,'/home/wilson/dev/the-cast/local-comfy')
import smoke_test as S
import repaint_cells as RC
from build_sprite import (SPRITE_LOCK, MOVES, WHO_LEAD, KEY_MAGENTA, STAGE_RESTATE, LENGTH,
                          CLIP_SEED)
from build_transition import TRAITS, SOUND_LOCK, H3_UNET, H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE

SIZE, STEPS = int(sys.argv[3]), int(sys.argv[4])
cid, move = sys.argv[1], sys.argv[2]


def brief(cid, recipe):
    """The recipe text, the trait line and the cyclic flag for one clip.

    The character MANIFEST wins wherever it has a tag for this recipe: a move is data now, and its
    text is editable, so re-rendering a clip has to use the words the tag actually carries. MOVES
    and TRAITS are the bootstrap defaults a recipe that no character has ever built falls back to.
    A cycle is briefed to end where it began, so its end must stay unpinned; see the docstring.
    """
    # Read straight off disk rather than through load_character_manifest(): that one tops a
    # manifest up with any move it is missing, which picks cells off clips -- and this script is
    # what RENDERS a clip, so it must not depend on one already being there.
    path = RC.manifest_path(cid)
    man = json.load(open(path)) if os.path.exists(path) else {}
    for tag in man.get("tags", []):
        if tag["recipe"] == recipe:
            return tag["recipe_text"], man["trait"], tag["cyclic"]
    return MOVES[recipe], TRAITS[cid], RC.is_cyclic(recipe)


RECIPE_TEXT, TRAIT, CYCLIC = brief(cid, move)
prompt = f"{SPRITE_LOCK} {RECIPE_TEXT}. {WHO_LEAD} {TRAIT}. {STAGE_RESTATE} {SOUND_LOCK}"
name = f"{cid}-{move}-{SIZE}-{STEPS}"

g = {
 "6":{"class_type":"UNETLoader","inputs":{"unet_name":H3_UNET,"weight_dtype":"default"}},
 "50":{"class_type":"EasyCache","inputs":{"model":["6",0],"reuse_threshold":0.2,"start_percent":0.15,"end_percent":0.95,"verbose":True}},
 "13":{"class_type":"CLIPLoader","inputs":{"clip_name":H3_CLIP,"type":"minimax","device":"default"}},
 "11":{"class_type":"VAELoader","inputs":{"vae_name":H3_VIDEO_VAE}},
 "24":{"class_type":"VAELoader","inputs":{"vae_name":H3_AUDIO_VAE}},
 "screen":{"class_type":"EmptyImage","inputs":{"width":SIZE,"height":SIZE,"batch_size":1,"color":KEY_MAGENTA}},
 "plate":{"class_type":"LoadImage","inputs":{"image":f"cast-cutout-{cid}.png"}},
 "fit":{"class_type":"ImageScale","inputs":{"image":["plate",0],"upscale_method":"lanczos","width":SIZE,"height":SIZE,"crop":"disabled"}},
 "alpha":{"class_type":"InvertMask","inputs":{"mask":["plate",1]}},
 "aimg":{"class_type":"MaskToImage","inputs":{"mask":["alpha",0]}},
 "afit":{"class_type":"ImageScale","inputs":{"image":["aimg",0],"upscale_method":"lanczos","width":SIZE,"height":SIZE,"crop":"disabled"}},
 "amask":{"class_type":"ImageToMask","inputs":{"image":["afit",0],"channel":"red"}},
 "first":{"class_type":"ImageCompositeMasked","inputs":{"destination":["screen",0],"source":["fit",0],"mask":["amask",0],"x":0,"y":0,"resize_source":False}},
 "104":{"class_type":"MiniMaxH3ImageToVideo","inputs":{"clip":["13",0],"vae":["11",0],"prompt":prompt,
        "width":SIZE,"height":SIZE,"length":LENGTH,"first_frame":["first",0], **({} if CYCLIC else {"last_frame":["first",0]})}},
 "15":{"class_type":"RandomNoise","inputs":{"noise_seed":CLIP_SEED}},
 "16":{"class_type":"BasicGuider","inputs":{"model":["50",0],"conditioning":["104",0]}},
 "17":{"class_type":"KSamplerSelect","inputs":{"sampler_name":"res_multistep"}},
 "9":{"class_type":"BasicScheduler","inputs":{"model":["50",0],"scheduler":"simple","steps":STEPS,"denoise":1.0}},
 "14":{"class_type":"SamplerCustomAdvanced","inputs":{"noise":["15",0],"guider":["16",0],"sampler":["17",0],"sigmas":["9",0],"latent_image":["104",1]}},
 "10":{"class_type":"VAEDecode","inputs":{"samples":["14",0],"vae":["11",0]}},
 "23":{"class_type":"VAEDecodeAudio","inputs":{"samples":["14",0],"vae":["24",0]}},
 "91":{"class_type":"CreateVideo","inputs":{"images":["10",0],"audio":["23",0],"fps":24,"bit_depth":8}},
 "92":{"class_type":"SaveVideo","inputs":{"video":["91",0],"filename_prefix":f"video/{name}","format":"auto","codec":"auto"},"_meta":{"title":"RESULT"}},
}
for n,d in g.items(): d.setdefault("_meta",{"title":n})
t0=time.time()
pid=S.api("/prompt",{"prompt":g,"client_id":"hires"})["prompt_id"]
while True:
    h=S.api(f"/history/{pid}")
    if pid in h: break
    time.sleep(10)
st=h[pid]['status']
if st.get('status_str')!='success':
    print("FAILED", json.dumps(st)[:500]); sys.exit(1)
o=h[pid]['outputs']['92']['images'][0]
raw=f"/tmp/{name}.mp4"
subprocess.run(["curl","-s",f"{S.HOST}/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output","-o",raw],check=True)
print(f"{name}: {time.time()-t0:.0f}s -> {raw}")
