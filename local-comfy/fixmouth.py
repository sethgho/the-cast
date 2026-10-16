"""Repaint Seth's mouth in the texture: Meshy baked lips, teeth and mouth interior as one white band.
Top of the band -> teeth, middle -> dark mouth, bottom -> skin lower lip, edge -> ink line."""
import io, sys, numpy as np, pygltflib
from PIL import Image, ImageDraw
TEETH, MOUTH, INK = (246, 244, 236), (92, 38, 36), (52, 34, 28)
def fix(src, dst):
    g = pygltflib.GLTF2().load(src); blob = bytearray(g.binary_blob())
    def view(i):
        a = g.accessors[i]; bv = g.bufferViews[a.bufferView]
        dt = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}[a.componentType]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a.type]
        return np.frombuffer(blob, dt, a.count * n, (bv.byteOffset or 0) + (a.byteOffset or 0)).reshape(a.count, n).copy()
    p = g.meshes[0].primitives[0]; P = view(p.attributes.POSITION); UV = view(p.attributes.TEXCOORD_0)
    F = view(p.indices).reshape(-1, 3).astype(int)
    img = g.images[0]; bvi = g.bufferViews[img.bufferView]
    off = bvi.byteOffset or 0
    tex = Image.open(io.BytesIO(bytes(blob[off:off + bvi.byteLength]))).convert("RGB"); W, H = tex.size
    T = np.asarray(tex).astype(int)
    col = T[np.clip((UV[:, 1] * H).astype(int), 0, H - 1), np.clip((UV[:, 0] * W).astype(int), 0, W - 1)]
    lum = col.mean(1); sat = col.max(1) - col.min(1)
    region = (P[:, 1] > 1.40) & (P[:, 1] < 1.448) & (np.abs(P[:, 0]) < 0.10) & (P[:, 2] > 0.12)
    white = region & (lum > 190) & (sat < 30)
    skin = np.median(col[region & (sat >= 35) & (lum > 150)], 0).astype(int)
    # smooth band edges: quadratic fits of the band's top and bottom over x
    xs = P[white, 0]; ys = P[white, 1]
    bins = np.linspace(xs.min(), xs.max(), 15); tx, ty, bx, by = [], [], [], []
    for i in range(14):
        m = (xs >= bins[i]) & (xs < bins[i + 1])
        if m.sum() >= 5: c = (bins[i] + bins[i + 1]) / 2; tx.append(c); ty.append(ys[m].max()); bx.append(c); by.append(ys[m].min())
    ftop = np.poly1d(np.polyfit(tx, ty, 2)); fbot = np.poly1d(np.polyfit(bx, by, 2))
    x0, x1 = xs.min(), xs.max()
    # per-vertex band coordinate t (0 = top edge, 1 = bottom edge); outside the band's x span -> skin
    t = (ftop(P[:, 0]) - P[:, 1]) / np.maximum(ftop(P[:, 0]) - fbot(P[:, 0]), 1e-4)
    ex = np.clip(np.maximum(x0 - P[:, 0], P[:, 0] - x1) / 0.006, 0, None)   # fade to skin past the corners
    t = np.where(ex > 0, 2.0, t)
    zone = region | (white)
    faces = F[zone[F].any(1)]
    out = T.astype(np.float32).copy(); skinf = skin.astype(np.float32)
    def colour(tt):
        c = np.empty(tt.shape + (3,), np.float32)
        c[:] = skinf
        c[(tt >= 0.0) & (tt < 0.45)] = TEETH
        c[(tt >= 0.45) & (tt < 0.82)] = MOUTH
        ink = (np.abs(tt - 0.0) < 0.07) | (np.abs(tt - 0.82) < 0.06) | ((tt > -0.07) & (tt < 0.82) & (np.abs(tt - 0.45) < 0.025))
        c[ink & (tt > -0.07) & (tt < 0.9)] = INK
        return c
    for f in faces:
        uv = UV[f] * [W, H]; tv = t[f]
        lo = np.floor(uv.min(0)).astype(int); hi = np.ceil(uv.max(0)).astype(int) + 1
        if (hi - lo).max() > 200: continue
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
        a, b, c = uv; v0, v1 = b - a, c - a; den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-9: continue
        px = gx + 0.5 - a[0]; py = gy + 0.5 - a[1]
        w1 = (px * v1[1] - v1[0] * py) / den; w2 = (v0[0] * py - px * v0[1]) / den; w0 = 1 - w1 - w2
        ins = (w0 >= -0.12) & (w1 >= -0.12) & (w2 >= -0.12)
        if not ins.any(): continue
        tt = w0 * tv[0] + w1 * tv[1] + w2 * tv[2]
        cc = colour(tt[ins]); yy = np.clip(gy[ins], 0, H - 1); xx = np.clip(gx[ins], 0, W - 1)
        keep = tt[ins] < 1.6          # leave real skin texture alone outside the band
        out[yy[keep], xx[keep]] = cc[keep]
    canvas = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    vc = np.where(zone)[0]
    buf = io.BytesIO(); canvas.save(buf, "JPEG", quality=92); data = buf.getvalue()
    # append new image bytes and repoint the image's bufferView
    pad = (-len(blob)) % 4; blob.extend(b"\0" * pad); newoff = len(blob); blob.extend(data)
    g.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=newoff, byteLength=len(data)))
    img.bufferView = len(g.bufferViews) - 1; img.mimeType = "image/jpeg"
    g.buffers[0].byteLength = len(blob); g.set_binary_blob(bytes(blob)); g.save(dst)
    print(dst.split("/")[-1], "painted verts", len(vc))
for n in sys.argv[1:]:
    fix(f"/home/wilson/artifacts/figurines/seth-v2raw/{n}.glb", f"/home/wilson/artifacts/figurines/{n}.glb")
