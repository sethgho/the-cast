#!/usr/bin/env python3
"""Render one still per shot of a storyboard. Orchestration only: it fills two fields and queues.

    python3 previz_run.py french-again
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smoke_test as S
from build_transition import TRAITS
from build_previz import PLATES

# (character, shot prose). Camera position is part of the prose -- see the h3-storyboard skill:
# if what the character looks at is off-screen, say where the camera is or he stares at the lens.
BOARDS = {
 "french-again": [
  ("seth", "Wide shot of a family living room at night. The camera is beside the television, at "
           "seated height, looking back across the room at him on the sofa; the television is out "
           "of frame and its light flickers on his face. He is mid-bite of a crisp, a bowl beside "
           "him. He is not looking at the camera."),
  ("seth", "A closer shot of him on the sofa from the same side, television light on his face. He "
           "is caught in a hard wince: eyes squeezed shut, teeth bared, shoulders hunched up "
           "around his ears."),
  ("seth", "The same closer shot. His face has settled into flat, unimpressed annoyance, and his "
           "eyes have travelled all the way to the far left of their sockets, looking off past the "
           "edge of the frame at something he is fed up with. His head has not turned."),
  ("wilson", "A small studio room. He stands in profile at a wooden easel with a paintbrush held "
             "up, wearing a smock. Beside him, a tall black server rack reclines along a red "
             "velvet chaise longue like an artist's model, one thick cable draped over the top of "
             "it, its little red status lights glowing. He has turned his head to look back over "
             "his shoulder toward the left edge of the frame."),
  ("wilson", "A closer shot of him at the easel, still turned to look off to the left, eyes wide "
             "and mid-blink, caught with the eyelids partly lowered. The easel and the reclining "
             "server rack are behind him."),
  ("wilson", "A close shot of the tall black server rack reclining on the red velvet chaise "
             "longue, cable draped over it. Its row of small red status lights has flared bright "
             "and hot, glowing far more strongly than before, washing red light onto the velvet."),
 ],
}


def main():
    board = sys.argv[1]
    shots = BOARDS[board]
    base = json.load(open(os.path.join(HERE, "api", "previz.api.json")))

    def find(g, title):
        for nid, n in g.items():
            if n.get("_meta", {}).get("title") == title:
                return nid
        raise SystemExit(f"no node titled {title!r}")

    outdir = f"/home/wilson/scratch/previz/{board}"
    os.makedirs(outdir, exist_ok=True)
    for i, (cid, prose) in enumerate(shots, 1):
        dest = f"{outdir}/shot{i}.png"
        if os.path.exists(dest):
            print(f"shot{i} cached"); continue
        g = json.loads(json.dumps(base))
        g[find(g, "▶ 1 · PLATE")]["inputs"]["image"] = PLATES[cid]
        g[find(g, "▶ 2 · WHO")]["inputs"]["value"] = (
            "Image 1 is the drawing to edit, and the character in image 1 is the only character in "
            f"the finished picture. Redraw him exactly as he is: {TRAITS[cid]}.")
        g[find(g, "▶ 3 · SHOT")]["inputs"]["value"] = prose
        g[find(g, "RESULT")]["inputs"]["filename_prefix"] = f"previz/{board}-{i}"
        t0 = time.time()
        r = S.api("/prompt", {"prompt": g, "client_id": "previz"})
        if "prompt_id" not in r:
            print(f"shot{i} REJECTED", json.dumps(r)[:400]); continue
        pid = r["prompt_id"]
        while True:
            h = S.api(f"/history/{pid}")
            if pid in h: break
            time.sleep(4)
        st = h[pid]["status"]
        if st.get("status_str") != "success":
            print(f"shot{i} FAILED", json.dumps(st)[:300]); continue
        o = h[pid]["outputs"][find(g, "RESULT")]["images"][0]
        q = f"/view?filename={o['filename']}&subfolder={o.get('subfolder','')}&type=output"
        with urllib.request.urlopen(S.HOST + q, timeout=300) as rr:
            open(dest, "wb").write(rr.read())
        print(f"shot{i} ({cid}) {time.time()-t0:.0f}s -> {dest}")
    print("DONE")


if __name__ == "__main__":
    main()
