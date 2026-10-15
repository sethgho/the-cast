"""Make Wilson's fence panel rigid: panel -> Spine02, hat -> Head (blended under the brim);
arms and legs keep Meshy's weights. Rewrites JOINTS_0/WEIGHTS_0 in place."""
import numpy as np, pygltflib, sys
PANEL_Y0, BLEND_Y0, BLEND_Y1, HALF_W = 0.27, 0.90, 0.96, 0.205

def fix(src, dst):
    g = pygltflib.GLTF2().load(src); blob = bytearray(g.binary_blob())
    def view(i):
        a = g.accessors[i]; bv = g.bufferViews[a.bufferView]
        dt = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}[a.componentType]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a.type]
        off = (bv.byteOffset or 0) + (a.byteOffset or 0)
        assert not bv.byteStride or bv.byteStride == n * np.dtype(dt).itemsize, "strided"
        return np.frombuffer(blob, dt, a.count * n, off).reshape(a.count, n), a
    names = [g.nodes[j].name for j in g.skins[0].joints]
    chest, head = names.index("Spine02"), names.index("Head")
    for m in g.meshes:
        for p in m.primitives:
            if p.attributes.JOINTS_0 is None: continue
            P, _ = view(p.attributes.POSITION); J, _ = view(p.attributes.JOINTS_0); W, wa = view(p.attributes.WEIGHTS_0)
            assert W.dtype == np.float32, "normalized weights not handled"
            J = J.copy(); W = W.copy()
            body = (P[:, 1] >= PANEL_Y0) & (np.abs(P[:, 0]) <= HALF_W)
            t = np.clip((P[:, 1] - BLEND_Y0) / (BLEND_Y1 - BLEND_Y0), 0, 1)   # 0 = chest, 1 = head
            J[body] = 0; W[body] = 0
            J[body, 0] = chest; W[body, 0] = 1 - t[body]
            J[body, 1] = head;  W[body, 1] = t[body]
            # write back
            for arr, acc_i in ((J, p.attributes.JOINTS_0), (W, p.attributes.WEIGHTS_0)):
                a = g.accessors[acc_i]; bv = g.bufferViews[a.bufferView]
                off = (bv.byteOffset or 0) + (a.byteOffset or 0)
                b = arr.astype(arr.dtype).tobytes(); blob[off:off + len(b)] = b
            print(f"{dst}: {body.sum()} of {len(P)} verts made rigid")
    g.set_binary_blob(bytes(blob)); g.save(dst)

for name in sys.argv[1:]:
    fix(f"/home/wilson/artifacts/figurines/{name}.glb", f"/home/wilson/artifacts/figurines/{name}-rigid.glb")
