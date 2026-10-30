"""Queue a LiDAR Lab API workflow on the lab ComfyUI (:8190) with overrides; fetch the video."""
import json, sys, time, urllib.request
HOST = "http://192.168.0.210:8190"
wf, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]
sets = [a.split("=", 1) for a in sys.argv[4:]]
g = json.load(open(wf))
g["1"]["inputs"]["file"] = inp
for k, v in sets:
    node, key = k.split(".", 1)
    try:
        v = json.loads(v)
    except ValueError:
        pass
    g[node]["inputs"][key] = v
def api(p, body=None):
    req = urllib.request.Request(HOST + p, json.dumps(body).encode() if body else None, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))
r = api("/prompt", {"prompt": g})
if "prompt_id" not in r:
    sys.exit("REJECTED " + json.dumps(r)[:1500])
pid, t0 = r["prompt_id"], time.time()
while pid not in (h := api(f"/history/{pid}")):
    time.sleep(3)
st = h[pid]["status"]
if st.get("status_str") != "success":
    sys.exit("FAILED " + json.dumps(st)[:2500])
for o in h[pid]["outputs"].values():
    for kind in ("videos", "images", "gifs"):
        for f in o.get(kind, []):
            q = f"/view?filename={f['filename']}&subfolder={f.get('subfolder','')}&type={f.get('type','output')}"
            open(out, "wb").write(urllib.request.urlopen(HOST + q).read())
print(out, f"{time.time() - t0:.1f}s")
