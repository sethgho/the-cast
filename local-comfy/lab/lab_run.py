"""Queue an API-format graph on ComfyUI (main :8188 by default; COMFY_HOST for the lab :8190) with node.input=value overrides; fetch outputs; log peak VRAM/RSS."""
import json, os, subprocess, sys, threading, time, urllib.request
HOST = os.environ.get("COMFY_HOST", "http://192.168.0.210:8188")
wf, out = sys.argv[1], sys.argv[2]
g = json.load(open(wf))
for a in sys.argv[3:]:
    k, v = a.split("=", 1)
    node, key = k.split(".", 1)
    try:
        v = json.loads(v)
    except ValueError:
        pass
    g[node]["inputs"][key] = v
def api(p, body=None):
    req = urllib.request.Request(HOST + p, json.dumps(body).encode() if body else None, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))
peak = {"vram": 0, "rss": 0, "swap": 0}
stop = False
def poll():
    while not stop:
        r = subprocess.run(["ssh", "wilson@192.168.0.210",
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; free -m | awk '/Mem/{print $3} /Swap/{print $3}'"],
            capture_output=True, text=True).stdout.split()
        if len(r) >= 3:
            peak["vram"] = max(peak["vram"], int(r[0])); peak["rss"] = max(peak["rss"], int(r[1])); peak["swap"] = max(peak["swap"], int(r[2]))
        time.sleep(5)
t = threading.Thread(target=poll, daemon=True); t.start()
r = api("/prompt", {"prompt": g})
if "prompt_id" not in r:
    sys.exit("REJECTED " + json.dumps(r)[:2000])
pid, t0 = r["prompt_id"], time.time()
while pid not in (h := api(f"/history/{pid}")):
    time.sleep(3)
stop = True
st = h[pid]["status"]
print("peak host-used MB", peak, f"{time.time() - t0:.0f}s")
if st.get("status_str") != "success":
    sys.exit("FAILED " + json.dumps(st)[-2500:])
for o in h[pid]["outputs"].values():
    for kind in ("videos", "images", "gifs"):
        for f in o.get(kind, []):
            if f.get("type") != "output":
                continue
            q = f"/view?filename={f['filename']}&subfolder={f.get('subfolder','')}&type=output"
            open(out, "wb").write(urllib.request.urlopen(HOST + q).read())
print(out)
