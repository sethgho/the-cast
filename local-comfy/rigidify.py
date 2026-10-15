"""Wilson rigid parts: fence panel -> Spine02, head piece (dark face blob + eyes + hat) -> Head,
both 100% (no blend, so neither shears). The head piece is found by texture colour, not height:
the dark blob, plus any light island it encloses (eyes), plus everything above the brim line."""
import io, sys, numpy as np, pygltflib
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
PANEL_Y0, HALF_W, HAT_Y = 0.27, 0.205, 0.94
# Meshy fuses face, pickets and grain into one surface, so a separately turning head tears.
# Off: fence + face + hat are one rigid piece on the chest bone, as on the character sheet.
HEAD_TURNS = False
ANKLE_Y = 0.115   # boot top / cuff bottom, just above the ankle joint (0.11)

def fix(src, dst):
    g = pygltflib.GLTF2().load(src); blob = bytearray(g.binary_blob())
    def view(i):
        a = g.accessors[i]; bv = g.bufferViews[a.bufferView]
        dt = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}[a.componentType]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a.type]
        off = (bv.byteOffset or 0) + (a.byteOffset or 0)
        return np.frombuffer(blob, dt, a.count * n, off).reshape(a.count, n), off
    names = [g.nodes[j].name for j in g.skins[0].joints]
    chest, head = names.index("Spine02"), names.index("Head")
    p = g.meshes[0].primitives[0]
    P, _ = view(p.attributes.POSITION); UV, _ = view(p.attributes.TEXCOORD_0)
    F = view(p.indices)[0].reshape(-1, 3).astype(np.int64)
    mat = g.materials[p.material]; ti = mat.pbrMetallicRoughness.baseColorTexture.index
    img = g.images[g.textures[ti].source]; bv = g.bufferViews[img.bufferView]
    tex = np.asarray(Image.open(io.BytesIO(bytes(blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength]))).convert("L"), np.float32)
    h, w = tex.shape
    u = np.clip((UV[:, 0] % 1) * (w - 1), 0, w - 1).astype(int); v = np.clip((UV[:, 1] % 1) * (h - 1), 0, h - 1).astype(int)
    lum = tex[v, u]
    band = (P[:, 1] >= 0.78) & (P[:, 1] < HAT_Y) & (np.abs(P[:, 0]) <= HALF_W)
    dark = lum < 80
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    n = len(P); A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr(); A = A + A.T
    def comps(mask):
        idx = np.where(mask)[0]; k, lab = connected_components(A[idx][:, idx], directed=False)
        return [idx[lab == c] for c in range(k)]
    headv = np.zeros(n, bool)
    for c in comps(band & dark):          # the face blob, not grain strokes on the pickets
        if len(c) >= 40: headv[c] = True
    for c in comps(band & ~dark):         # eyes: light islands that never reach down into the fence
        if P[c, 1].min() > 0.84 and np.abs(P[c, 0]).max() < 0.14 and len(c) < 3000: headv[c] = True
    headv |= (P[:, 1] >= HAT_Y) & (np.abs(P[:, 0]) <= HALF_W)
    body = (P[:, 1] >= PANEL_Y0) & (np.abs(P[:, 0]) <= HALF_W)
    J, jo = view(p.attributes.JOINTS_0); W, wo = view(p.attributes.WEIGHTS_0)
    J = J.copy(); W = W.copy()
    J[body] = 0; W[body] = 0; J[body, 0] = chest; W[body, 0] = 1
    if HEAD_TURNS:
        J[headv, 0] = head
    # stubby legs: each whole leg -- the stem from inside the fence down through cuff and boot --
    # is one rigid piece on the hip bone, so it swings like a toy's leg and its top stays hidden
    # inside the panel. Human knee/ankle bends kinked the stem and tore it where it enters the fence.
    for side, sgn in (("Left", 1), ("Right", -1)):
        onside = np.sign(P[:, 0]) == sgn
        below = onside & (P[:, 1] < PANEL_Y0) & (np.abs(P[:, 0]) <= 0.24)
        inside = onside & (P[:, 1] >= PANEL_Y0) & (P[:, 1] < 0.40) & (np.abs(P[:, 2]) < 0.05) \
                 & (np.abs(P[:, 0]) > 0.02) & (np.abs(P[:, 0]) < 0.16)
        leg = below | inside
        boot = leg & (P[:, 1] < ANKLE_Y); stem = leg & ~boot
        J[stem] = 0; W[stem] = 0; J[stem, 0] = names.index(f"{side}UpLeg"); W[stem, 0] = 1
        # the boot rides the ankle so a custom walk can roll it; the knee is never bent, so the
        # ankle stays at the end of the rigid stem
        J[boot] = 0; W[boot] = 0; J[boot, 0] = names.index(f"{side}Foot"); W[boot, 0] = 1
    blob[jo:jo + J.nbytes] = J.tobytes(); blob[wo:wo + W.nbytes] = W.tobytes()
    g.set_binary_blob(bytes(blob)); g.save(dst)
    print(f"{dst.split('/')[-1]}: body {body.sum()}, head piece {headv.sum()} (face {(headv & band).sum()})")

for name in sys.argv[1:]:
    fix(f"/home/wilson/artifacts/figurines/{name}.glb", f"/home/wilson/artifacts/figurines/{name}-rigid.glb")
