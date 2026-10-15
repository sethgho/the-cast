"""Wilson's own walk cycle, keyframed onto Meshy's skeleton (no mocap).
Rotations are authored as WORLD-space deltas on the rest pose, so bone axis conventions do not matter.
Stiff legs swing from the hip and hike up into the fence while swinging; ankles roll; hips bob and
waddle; arms hang lower than the A-pose and swing opposite the legs; wrists stay at rest."""
import numpy as np, pygltflib, struct, sys
from scipy.spatial.transform import Rotation as R

SRC, DST = sys.argv[1], sys.argv[2]
FRAMES, DUR = 32, 1.1                 # one full stride
LEG_SWING, FOOT_INHERIT = 22, 0.55    # degrees; ankle keeps 55% of the leg angle -> heel/toe roll
HIKE_CM, BOB_CM, WADDLE, TWIST = 3.0, 1.2, 4, 5
ARM_DOWN, ARM_SWING = 35, 18
PALM_TWIST = float(sys.argv[3]) if len(sys.argv) > 3 else 90   # turns A-pose palms (facing out) to face the legs

g = pygltflib.GLTF2().load(SRC)
nodes = g.nodes; idx = {n.name: i for i, n in enumerate(nodes)}
parent = {c: i for i, n in enumerate(nodes) for c in (n.children or [])}
rest_r = {i: R.from_quat(n.rotation) if n.rotation else R.identity() for i, n in enumerate(nodes)}
rest_t = {i: np.array(n.translation if n.translation else [0, 0, 0], float) for i, n in enumerate(nodes)}
scale = {i: np.array(n.scale if n.scale else [1, 1, 1], float) for i, n in enumerate(nodes)}

def chain(i):
    c = [];
    while True:
        c.append(i)
        if i not in parent: return c[::-1]
        i = parent[i]

def world_rot(i, local):
    r = R.identity()
    for j in chain(i): r = r * local[j]
    return r

def world_pos(i, local_r, local_t):
    m = np.eye(4)
    for j in chain(i):
        t = np.eye(4); t[:3, :3] = local_r[j].as_matrix() * scale[j]; t[:3, 3] = local_t[j]; m = m @ t
    return m[:3, 3]

rest_world = {i: world_rot(i, rest_r) for i in range(len(nodes))}

def solve(deltas, trans):
    """deltas: node -> world-space rotation delta; returns local rotations/translations."""
    lr = dict(rest_r); lt = dict(rest_t)
    for name, d in deltas.items():
        pass
    # top-down so parents are final first
    order = sorted(range(len(nodes)), key=lambda i: len(chain(i)))
    world = {}
    for i in order:
        pw = world[parent[i]] if i in parent else R.identity()
        if nodes[i].name in deltas:
            w = deltas[nodes[i].name] * rest_world[i]; lr[i] = pw.inv() * w
        world[i] = pw * lr[i]
    for name, v in trans.items():
        i = idx[name]; pw = world[parent[i]] if i in parent else R.identity()
        s = np.prod([scale[j][0] for j in chain(parent[i])]) if i in parent else 1.0
        lt[i] = rest_t[i] + pw.inv().apply(np.array(v) * 0.01) / s
    return lr, lt

X = lambda deg: R.from_rotvec(np.radians(deg) * np.array([1, 0, 0]))
Y = lambda deg: R.from_rotvec(np.radians(deg) * np.array([0, 1, 0]))
Z = lambda deg: R.from_rotvec(np.radians(deg) * np.array([0, 0, 1]))

# pick signs from the rest pose: "forward" moves the left boot to +z, "down" brings the left hand in
def probe(name, rot, target):
    lr, lt = solve({name: rot}, {}); return world_pos(idx[target], lr, lt)
fwd = 1 if probe("LeftUpLeg", X(10), "LeftFoot")[2] > world_pos(idx["LeftFoot"], rest_r, rest_t)[2] else -1
down = 1 if abs(probe("LeftArm", Z(10), "LeftHand")[0]) < abs(world_pos(idx["LeftHand"], rest_r, rest_t)[0]) else -1
print("signs fwd", fwd, "down", down)

times = np.linspace(0, DUR, FRAMES + 1)
chans = {k: [] for k in ("Hips", "LeftUpLeg", "RightUpLeg", "LeftFoot", "RightFoot", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm")}
tchans = {k: [] for k in ("Hips", "LeftUpLeg", "RightUpLeg")}
for t in times:
    ph = 2 * np.pi * t / DUR
    aL, aR = LEG_SWING * np.sin(ph), -LEG_SWING * np.sin(ph)
    d = {
        "Hips": Z(WADDLE * np.sin(ph)) * Y(TWIST * np.sin(ph)),
        "LeftUpLeg": X(fwd * aL), "RightUpLeg": X(fwd * aR),
        "LeftFoot": X(fwd * aL * FOOT_INHERIT), "RightFoot": X(fwd * aR * FOOT_INHERIT),
        "LeftArm": X(-fwd * ARM_SWING * np.sin(ph)) * Z(down * ARM_DOWN),
        "RightArm": X(fwd * ARM_SWING * np.sin(ph)) * Z(-down * ARM_DOWN),
    }
    # palms in: twist each forearm about its own rest axis, on top of the arm's motion
    for side, sgn in (("Left", 1), ("Right", -1)):
        fa, hd = idx[f"{side}ForeArm"], idx[f"{side}Hand"]
        axis = world_pos(hd, rest_r, rest_t) - world_pos(fa, rest_r, rest_t); axis /= np.linalg.norm(axis)
        d[f"{side}ForeArm"] = d[f"{side}Arm"] * R.from_rotvec(np.radians(sgn * PALM_TWIST) * axis)
    tr = {"Hips": [0, BOB_CM * abs(np.cos(ph)), 0],
          "LeftUpLeg": [0, HIKE_CM * max(0, np.cos(ph)), 0],
          "RightUpLeg": [0, HIKE_CM * max(0, -np.cos(ph)), 0]}
    lr, lt = solve(d, tr)
    for k in chans: chans[k].append(lr[idx[k]].as_quat())
    for k in tchans: tchans[k].append(lt[idx[k]])

# write: replace animations with one "Wilson_Walk"
blob = bytearray(g.binary_blob())
def add(arr, typ):
    arr = np.asarray(arr, np.float32); off = len(blob); pad = (-off) % 4; blob.extend(b"\0" * pad); off += pad
    blob.extend(arr.tobytes())
    g.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=off, byteLength=arr.nbytes))
    acc = pygltflib.Accessor(bufferView=len(g.bufferViews) - 1, componentType=5126, count=len(arr), type=typ)
    if typ == "SCALAR": acc.min = [float(arr.min())]; acc.max = [float(arr.max())]
    g.accessors.append(acc); return len(g.accessors) - 1
tin = add(times, "SCALAR")
anim = pygltflib.Animation(name="Wilson_Walk", samplers=[], channels=[])
for k, q in chans.items():
    anim.samplers.append(pygltflib.AnimationSampler(input=tin, output=add(q, "VEC4"), interpolation="LINEAR"))
    anim.channels.append(pygltflib.AnimationChannel(sampler=len(anim.samplers) - 1, target=pygltflib.AnimationChannelTarget(node=idx[k], path="rotation")))
for k, v in tchans.items():
    anim.samplers.append(pygltflib.AnimationSampler(input=tin, output=add(v, "VEC3"), interpolation="LINEAR"))
    anim.channels.append(pygltflib.AnimationChannel(sampler=len(anim.samplers) - 1, target=pygltflib.AnimationChannelTarget(node=idx[k], path="translation")))
g.animations = [anim]
g.buffers[0].byteLength = len(blob); g.set_binary_blob(bytes(blob)); g.save(DST)
print("wrote", DST)
