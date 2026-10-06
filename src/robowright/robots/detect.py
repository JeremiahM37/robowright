"""Any robot from its model file: work out what a person would otherwise write down.

A model file (MJCF, or a URDF: see :mod:`.from_urdf`) describes every robot the same way -
bodies, joints, actuators, limits - but not what the parts are *for*: which joints form the
arm, which bodies are the fingers,
which actuator drives the gripper and which end of its range is open. This module reads those
off the model's structure and motion, the way Playwright finds a button by its role rather
than by a hand-written selector:

* a model with a free joint is a mobile robot: legged if chains of actuated joints reach the
  ground, its actuated joints being what it moves; otherwise, carrying a gripper, a mobile
  manipulator, whose base is held where it stands so its arm is tested as on a fixed one;
* the gripper is the actuator that moves several coupled joints, a slide, or parts named like
  a gripper (finger, jaw, claw...), at the end of the chain;
* the hand is the body the gripper's joints hang from, and the arm is every actuated joint on
  the chain from the base to it;
* the fingers are the two moving parts that reach furthest along the tool axis and close
  towards each other (or one moving jaw against the hand, as on the SO-101);
* "open" is whichever end of the gripper's range leaves the fingers further apart;
* the robot is mounted where top-down grasps reach the whole task area of the default scene.

Every decision is recorded in :attr:`Detection.notes`, so ``robowright robots --inspect FILE``
can show why, and any field can be overridden by passing it to :func:`load`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from .model import _PRINCIPAL, Attachment, RobotModel, _collision_geoms, _corners, _surface, fix_gripper, reset_data

_FREE, _HINGE, _SLIDE = (int(t) for t in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE))
_GRIP_WORDS = re.compile(r"grip|finger|jaw|claw|pinch|knuckle|thumb|tong|hand", re.I)
_LEFT = re.compile(r"(^|[^a-z])(left|l)([^a-z]|$)|left", re.I)

# The default tabletop's task area (see robowright.scene.tabletop). A robot is mounted where
# top-down grasps reach what pick-and-place needs - grasp and lift the cube, carry it over the
# bin - and, among those positions, where they also reach most of the area around it.
_NEEDED = np.array(
    [
        (0.22, -0.06, 0.02),  # grasp the cube
        (0.22, -0.06, 0.07),  # lifted
        (0.2, 0.12, 0.08),  # over the bin
        (0.2, 0.0, 0.08),  # the lowest home pose
    ]
)
_AROUND = np.array([(x, y, z) for x in (0.12, 0.3) for y in (-0.12, 0.18) for z in (0.03, 0.1)])
_HOME_HEIGHTS = (0.2, 0.15, 0.12, 0.1, 0.08)
# Where the default scenes put objects (the cube, the bin, a second cube), with 5 mm to spare:
# no part of the robot's base may stand on them.
_OBJECTS = (
    ((0.2025, -0.0775), (0.2375, -0.0425)),
    ((0.1425, -0.1175), (0.1775, -0.0825)),
    ((0.145, 0.065), (0.255, 0.175)),
)


@dataclass
class Detection:
    """What was worked out about a robot, and why."""

    fields: dict
    notes: list[str] = field(default_factory=list)


class DetectionError(ValueError):
    """The model's structure did not say what robowright needed; the message says which field to pass."""


def load(path: str | Path, name: str | None = None, **overrides) -> RobotModel:
    """A :class:`RobotModel` for the robot in ``path`` (MJCF ``.xml`` or ``.urdf``), registered under ``name``.

    Whatever :func:`detect` cannot work out, or gets wrong, can be passed as a keyword, using
    :class:`RobotModel`'s field names (``arm_joints``, ``hand``, ``left_finger``,
    ``gripper_actuator``, ``gripper_open``, ``base_pos``, ``home``...).
    """
    from . import REGISTRY, register

    path = Path(path).expanduser().resolve()
    key = (str(path), name, tuple(sorted((k, repr(v)) for k, v in overrides.items())))
    if key not in _LOADED:
        model = build(path, name or _name_for(path, REGISTRY), **overrides).model
        _LOADED[key] = register(model)
    return _LOADED[key]


_LOADED: dict[tuple, RobotModel] = {}


@dataclass
class Built:
    model: RobotModel
    notes: list[str]


def build(path: str | Path, name: str | None = None, **overrides) -> Built:
    """Detect, apply ``overrides``, and place the robot; nothing is registered."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"no robot model at {path}")
    first = []
    if path.suffix.lower() in (".urdf", ".xacro"):
        from .from_urdf import URDFError, to_mjcf
        from .urdf import cache_root

        try:
            path_mjcf, first = to_mjcf(path, cache_root().parent / "from_urdf")
        except URDFError as e:
            raise DetectionError(str(e)) from e
    else:
        path_mjcf = path
    mjcf = _named(path_mjcf)
    mjcf, held = _hold_mobile_base(mjcf)
    det = detect(mjcf, overrides.get("attach"), given=overrides)
    det.notes[:0] = first + held
    if first and det.fields.get("family", "arm") == "arm" and "attach" not in det.fields:
        # From a URDF: grasping needs the jaws' real shapes where their hulls would not do.
        from .from_urdf import split_jaws

        f = {**det.fields, **overrides}
        mjcf, more = split_jaws(mjcf, f["hand"], (f["left_finger"][0], f["right_finger"][0]), cache_root().parent / "from_urdf")
        det.notes.extend(more)
    fields = {**det.fields, **overrides}
    for k in overrides:
        det.notes.append(f"{k}: given ({overrides[k]!r})")
    name = name or path.stem
    model = RobotModel(
        name,
        fields.pop("title", path.stem),
        _const(mjcf),
        tuple(fields.pop("arm_joints")),
        **fields,
        extra={"file": str(path)},
    )
    if model.family == "arm" and ("base_pos" not in overrides or "home" not in overrides):
        model = _place(model, det.notes, keep_base="base_pos" in overrides, keep_home="home" in overrides)
    return Built(model, det.notes)


def _named(path: Path) -> Path:
    """``path``, or a copy whose unnamed bodies, joints and actuators have names robowright can address."""
    spec = mujoco.MjSpec.from_file(str(path))
    groups = (("body", spec.bodies[1:]), ("joint", spec.joints), ("actuator", spec.actuators))
    if all(e.name for _, es in groups for e in es):
        return path
    for kind, es in groups:
        taken = {e.name for e in es}
        for i, e in enumerate(es):
            if not e.name:
                n = f"{kind}{i}"
                while n in taken:
                    n += "_"
                e.name = n
                taken.add(n)
    return _copy(spec, path, "named")


def _copy(spec: mujoco.MjSpec, path: Path, kind: str) -> Path:
    """``spec``, a changed copy of the model at ``path``, written to robowright's cache."""
    # Asset paths are resolved against the original's directory, wherever the copy is.
    for attr in ("meshdir", "texturedir"):
        setattr(spec, attr, str((path.parent / (getattr(spec, attr) or "")).resolve()))
    import hashlib

    from .urdf import cache_root

    out = cache_root().parent / kind / f"{path.stem}-{hashlib.sha1(str(path).encode()).hexdigest()[:10]}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    xml = spec.to_xml()
    if not out.exists() or out.read_text() != xml:  # test workers write it at once: replace atomically
        import os
        import tempfile

        fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".xml")
        with os.fdopen(fd, "w") as f:
            f.write(xml)
        os.replace(tmp, out)
    return out


def _hold_mobile_base(path: Path) -> tuple[Path, list[str]]:
    """A mobile manipulator (a free-floating base carrying a gripper) with its base held still
    where it stands, so its arm is tested like any other; anything else as it is."""
    spec = mujoco.MjSpec.from_file(str(path))
    m = spec.compile()
    free = [j for j in range(m.njnt) if m.jnt_type[j] == _FREE]
    moved = _moved_joints(m)
    if not free or not any(_is_gripper(m, a, j) for a, j in moved.items()) or _legs(m, int(m.jnt_bodyid[free[0]]), moved) >= 2:
        return path, []  # fixed already, nothing to hold, or a legged robot (a humanoid's hands are not a mobile base)
    base = m.body(int(m.jnt_bodyid[free[0]])).name
    qa, va = int(m.jnt_qposadr[free[0]]), int(m.jnt_dofadr[free[0]])
    for key in spec.keys:  # keyframes lose the free joint's position and velocity
        if len(key.qpos):
            key.qpos = np.delete(np.asarray(key.qpos), np.s_[qa : qa + 7])
        if len(key.qvel):
            key.qvel = np.delete(np.asarray(key.qvel), np.s_[va : va + 6])
    spec.delete(spec.joint(m.joint(free[0]).name))
    spec.compiler.fusestatic = False  # a jointless base would otherwise be merged into the world
    # Standing on the floor: the lowest part of the robot at the floor's height.
    m = spec.compile()
    d = mujoco.MjData(m)
    reset_data(m, d)
    low = min(_corners(m, d, g)[:, 2].min() for g in _collision_geoms(m, set(range(1, m.nbody))))
    spec.body(base).pos[2] -= low
    note = f"family: arm on a mobile base; {base!r} is held where it stands, so the arm is tested as on a fixed one"
    return _copy(spec, path, "mobile"), [note]


def _const(p: Path):
    return lambda: p


def _name_for(path: Path, registry) -> str:
    stem = path.stem if path.stem.lower() not in ("robot", "model", "scene", "main", "urdf") else path.parent.name
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_").lower() or "robot"
    name, i = stem, 2
    while name in registry and registry[name].extra.get("file") != str(path):
        name, i = f"{stem}_{i}", i + 1
    return name


# ---------------------------------------------------------------------------------------------
def detect(mjcf: str | Path, attach: Attachment | None = None, given: dict | None = None) -> Detection:
    """Work out the :class:`RobotModel` fields for the robot in ``mjcf``.

    Fields in ``given`` are taken as they are, and what depends on them is worked out from them:
    a model whose gripper range is a placeholder, say, is read once ``gripper_open`` and
    ``gripper_closed`` say where it opens.
    """
    given = given or {}
    spec = mujoco.MjSpec.from_file(str(mjcf))
    m = spec.compile()
    notes: list[str] = []
    free = [j for j in range(m.njnt) if m.jnt_type[j] == _FREE]
    out = {"timestep": float(m.opt.timestep)}
    if (why := _unstable_integration(m)) is not None:
        out["integrator"] = "implicitfast"
        notes.append(f"integrator: implicitfast, not the model's Euler ({why})")
    if floor := _armature_floor(m):
        out["armature"] = floor
        notes.append(f"armature: motor inertia added to {[j for j, _ in floor]} so their force-limited servos do not shake")
    if free:
        moved = _moved_joints(m)
        if any(_is_gripper(m, a, j) for a, j in moved.items()) and _legs(m, int(m.jnt_bodyid[free[0]]), moved) < 2:
            # (build() holds such a base still first; this is detect() called on the model as it is.)
            raise DetectionError("a free-floating base with a gripper and no legs is a mobile manipulator: load it with robots.load")
        out.update(_legged(m, free[0], notes))
        return Detection(out, notes)
    if attach is None:
        attach = _default_attachment(m, spec, notes)
    if attach is not None:
        out["attach"] = attach
        m = _with_attachment(spec, attach)
    elif m.opt.cone == int(mujoco.mjtCone.mjCONE_PYRAMIDAL) and m.opt.impratio < 2:
        # MuJoCo's default friction (a pyramidal cone, impratio 1) lets a held object creep down
        # through the fingers (TIAGo's slid out of a 4 N grip within seconds); MuJoCo's own
        # gripper models hold with an elliptic cone and impratio 10.
        out["grip_contacts"] = True
        notes.append("grip_contacts: elliptic friction cone and impratio 10, as the model's defaults let a held object creep")
    out.update(_arm(m, spec, notes, given))
    return Detection(out, notes)


def _unstable_integration(m: mujoco.MjModel) -> str | None:
    """Why the model's explicit (Euler) integration of its servos' damping would oscillate, or None.

    Euler integrates joint damping implicitly but an actuator's velocity gain explicitly, which
    is stable only while kv * dt / inertia < 2. The FR3 v2's servos (kv 450 on 0.2 kg m^2 of
    armature at 2 ms) are at 4.6: a joint chatters at the step rate and never settles.
    """
    if m.opt.integrator != int(mujoco.mjtIntegrator.mjINT_EULER):
        return None
    d = mujoco.MjData(m)
    reset_data(m, d)
    M = np.zeros((m.nv, m.nv))
    try:
        mujoco.mj_fullM(m, d, M)  # MuJoCo 3.4+
    except TypeError:
        mujoco.mj_fullM(m, M, d.qM)
    for a in range(m.nu):
        kv = -float(m.actuator_biasprm[a, 2]) * float(m.actuator_gear[a, 0]) ** 2
        if m.actuator_trntype[a] != int(mujoco.mjtTrn.mjTRN_JOINT) or kv <= 0:
            continue
        dof = m.jnt_dofadr[m.actuator_trnid[a, 0]]
        ratio = kv * m.opt.timestep / M[dof, dof]
        if ratio > 1.0:
            return f"{m.actuator(a).name!r} damps at kv*dt/inertia = {ratio:.1f}"
    return None


def _armature_floor(m: mujoco.MjModel) -> tuple:
    """Joints whose force-limited servo damps faster than a step can integrate, with the inertia that fixes it.

    Implicit integrators take a servo's damping implicitly only while its force is inside its
    limit; at the limit (a joint far from its target) the damping is explicit, and a light joint
    whose kv * dt / inertia is above ~2 shakes at the step rate and barely moves. The Unitree Z1's
    jaw (3e-4 kg m^2, kv 100 at 2 ms) is at 625. Geared motors add reflected inertia that such
    models leave out; this adds back the least that keeps kv * dt / inertia at 1.
    """
    d = mujoco.MjData(m)
    reset_data(m, d)
    M = np.zeros((m.nv, m.nv))
    try:
        mujoco.mj_fullM(m, d, M)
    except TypeError:
        mujoco.mj_fullM(m, M, d.qM)
    out = []
    for a in range(m.nu):
        if m.actuator_trntype[a] != int(mujoco.mjtTrn.mjTRN_JOINT) or not m.actuator_forcelimited[a]:
            continue
        kv = -float(m.actuator_biasprm[a, 2]) * float(m.actuator_gear[a, 0]) ** 2
        j = int(m.actuator_trnid[a, 0])
        dof = m.jnt_dofadr[j]
        if kv > 0 and kv * m.opt.timestep / M[dof, dof] > 2.0 and _shakes(m, a, j):
            out.append((m.joint(j).name, round(float(m.dof_armature[dof] + kv * m.opt.timestep - M[dof, dof]), 6)))
    return tuple(out)


def _shakes(m: mujoco.MjModel, a: int, j: int) -> bool:
    """Whether servo ``a``, sent across its joint's range with the rest of the robot held, fails
    to get even a fifth of the way in half a second (it shakes in place instead)."""
    if not m.jnt_limited[j]:
        return False
    flags, gravity = m.opt.disableflags, m.opt.gravity.copy()
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    m.opt.gravity[:] = 0
    try:
        d = mujoco.MjData(m)
        reset_data(m, d)
        for i in range(m.nu):
            if m.actuator_trntype[i] == int(mujoco.mjtTrn.mjTRN_JOINT):
                d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        q0 = float(d.qpos[m.jnt_qposadr[j]])
        lo, hi = m.jnt_range[j]
        goal = lo if q0 - lo > hi - q0 else hi
        d.ctrl[a] = goal
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        return abs(float(d.qpos[m.jnt_qposadr[j]]) - q0) < 0.2 * abs(goal - q0)
    finally:
        m.opt.disableflags, m.opt.gravity[:] = flags, gravity


def _with_attachment(spec: mujoco.MjSpec, attach: Attachment) -> mujoco.MjModel:
    g = mujoco.MjSpec.from_file(str(attach.mjcf()))
    for k in list(g.keys):
        g.delete(k)
    for k in list(spec.keys):
        spec.delete(k)
    spec.attach(g, prefix=attach.prefix, site=attach.at(spec))
    return spec.compile()


def _moved_joints(m: mujoco.MjModel) -> dict[int, set[int]]:
    """For each actuator, the joints it moves directly (through a joint, a tendon, or a slider-crank)."""
    d = mujoco.MjData(m)
    reset_data(m, d)
    out = {}
    for a in range(m.nu):
        adr = d.moment_rowadr[a]
        out[a] = {int(m.dof_jntid[d.moment_colind[i]]) for i in range(adr, adr + d.moment_rownnz[a]) if abs(d.actuator_moment[i]) > 1e-9}
    return out


def _coupled(m: mujoco.MjModel, joints: set[int]) -> set[int]:
    """``joints`` plus every joint tied to them by joint equality constraints."""
    out, grew = set(joints), True
    while grew:
        grew = False
        for e in range(m.neq):
            if m.eq_type[e] != int(mujoco.mjtEq.mjEQ_JOINT):
                continue
            a, b = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            if (a in out) != (b in out) and b >= 0:
                out |= {a, b}
                grew = True
    return out


def _ancestors(m, b) -> list[int]:
    path = [b]
    while b > 0:
        b = int(m.body_parentid[b])
        path.append(b)
    return path  # b, parent, ..., world


def _subtree(m, b) -> set[int]:
    return {c for c in range(m.nbody) if b in _ancestors(m, c)}


def _names(m, a, joints) -> str:
    parts = [m.actuator(a).name] + [m.joint(j).name for j in joints] + [m.body(m.jnt_bodyid[j]).name for j in joints]
    if m.actuator_trntype[a] == int(mujoco.mjtTrn.mjTRN_TENDON):
        parts.append(m.tendon(m.actuator_trnid[a, 0]).name)
    return " ".join(parts)


def _is_gripper(m, a, joints) -> str | None:
    """Why actuator ``a`` looks like a gripper, or None."""
    if not joints:
        return None
    if len(_coupled(m, joints)) > 1:
        return "moves several coupled joints"
    if any(m.jnt_type[j] == _SLIDE for j in joints):
        return "drives a slide"
    if _GRIP_WORDS.search(_names(m, a, joints)):
        return "is named like a gripper part"
    return None


def _default_attachment(m: mujoco.MjModel, spec: mujoco.MjSpec, notes) -> Attachment | None:
    """A bare arm (no gripper) gets the Robotiq 2F-85 on its flange, as the built-in arms do."""
    moved = _moved_joints(m)
    if any(_is_gripper(m, a, j) for a, j in moved.items()):
        return None
    from . import menagerie

    gripper = lambda: menagerie.path("robotiq_2f85/2f85.xml")  # noqa: E731
    sites = [s.name for s in spec.sites]
    site = next((s for s in sites if "attach" in s.lower()), None)
    site = site or next((s for s in sites if re.search(r"flange|tool|pinch|ee|tcp", s, re.I)), None)
    if site is not None:
        notes.append(f"gripper: none in the model, so a Robotiq 2F-85 is attached at site {site!r}")
        return Attachment(gripper, site)
    # No site: the flange is the end of the actuated chain, past any bodies fixed to the last link.
    joints = {j for js in moved.values() for j in js}
    if len(joints) < 3:
        raise DetectionError(f"no gripper found, and {len(joints)} actuated joint(s) is not an arm: pass gripper_actuator=..., hand=...")
    last = max(joints, key=lambda j: len(_ancestors(m, int(m.jnt_bodyid[j]))))
    b = int(m.jnt_bodyid[last])
    while len(kids := [c for c in range(1, m.nbody) if m.body_parentid[c] == b]) == 1:
        b = kids[0]
    flange = m.body(b).name
    if not flange:
        raise DetectionError("no gripper found, and the arm's last link is unnamed: pass attach=Attachment(...)")
    # The flange faces along the last joint's axis (a wrist roll turns the tool about its own
    # axis), whichever axis of the link's frame that is: z on most arms, x on the Unitree Z1.
    d = mujoco.MjData(m)
    reset_data(m, d)
    R = d.xmat[b].reshape(3, 3)
    n = R.T @ d.xaxis[last]
    if (R.T @ (d.xpos[b] - d.xanchor[last])) @ n < -1e-3 or (abs(n[2]) > 0.99 and n[2] < 0):
        n = -n  # pointing away from the arm
    quat = None
    if n[2] < 0.99:
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, n)
        quat = tuple(round(float(v), 9) for v in quat)
    how = "" if quat is None else f", turned to face along the last joint's axis {np.round(n, 3).tolist()}"
    notes.append(f"gripper: none in the model, so a Robotiq 2F-85 is attached at the origin of {flange!r}, the end of the arm{how}")
    return Attachment(gripper, "robowright_flange", body=flange, quat=quat)


def _arm(m: mujoco.MjModel, spec: mujoco.MjSpec, notes, given: dict) -> dict:
    moved = _moved_joints(m)
    grippers = [(a, why) for a, j in moved.items() if (why := _is_gripper(m, a, j))]
    if "gripper_actuator" in given:
        act, why = m.actuator(given["gripper_actuator"]).id, "given"
        grippers = [(act, why), *((a, w) for a, w in grippers if a != act)]
    elif not grippers:
        raise DetectionError("no gripper actuator found: pass gripper_actuator=... (and hand=..., left_finger=..., right_finger=...)")
    else:
        # The deepest one: on a robot with a tool changer or a second gripper, the one at the tip.
        depth = lambda a: max(len(_ancestors(m, m.jnt_bodyid[j])) for j in moved[a])  # noqa: E731
        act, why = max(grippers, key=lambda g: (depth(g[0]), -g[0]))
    fixes = _gripper_fixes(m, act, [a for a, _ in grippers if a != act], moved, notes)
    if any(fixes.values()):
        name = m.actuator(act).name
        fix_gripper(spec, name, fixes["gripper_servo"], fixes["gripper_mirrors"])
        m = spec.compile()
        act = m.actuator(name).id
        moved = _moved_joints(m)
    elif len(grippers) > 1:
        notes.append(f"gripper: {len(grippers)} candidates ({', '.join(m.actuator(a).name for a, _ in grippers)}); took the deepest")
    notes.append(f"gripper_actuator: {m.actuator(act).name!r} ({why})")
    gj = _coupled(m, moved[act])
    gbodies = {int(m.jnt_bodyid[j]) for j in gj}

    # The hand: the deepest body above every gripper joint.
    chains = [_ancestors(m, int(m.body_parentid[b])) for b in gbodies]
    common = set(chains[0]).intersection(*chains[1:])
    hand = m.body(given["hand"]).id if "hand" in given else next(b for b in chains[0] if b in common)
    if "hand" not in given:
        notes.append(f"hand: {m.body(hand).name!r} (the body the gripper's joints hang from)")

    # The arm: every joint from the base to the hand, each driven by its own actuator.
    path = [b for b in reversed(_ancestors(m, hand)) if b > 0]
    arm = [j for b in path for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]) if m.jnt_type[j] in (_HINGE, _SLIDE)]
    driven = {next(iter(js)) for a, js in moved.items() if len(js) == 1 and a != act}
    if "arm_joints" in given:
        arm = [m.joint(n).id for n in given["arm_joints"]]
    undriven = [m.joint(j).name for j in arm if j not in driven]
    if undriven:
        raise DetectionError(f"arm joints {undriven} have no actuator of their own; robowright drives every arm joint: pass arm_joints=...")
    if not arm:
        raise DetectionError("no arm joints between the base and the hand: pass arm_joints=...")
    others = sorted({j for js in moved.values() for j in js} - set(arm) - _coupled(m, moved[act]))
    if others:
        notes.append(f"other actuated joints, held still: {[m.joint(j).name for j in others]}")
    notes.append(f"arm_joints: {len(arm)} on the chain from the base to the hand")

    soft = _soft_servos(m, arm, moved)
    if soft:
        notes.append(f"warning: the model's servos can only hold {soft} to within that of a target (joint friction / stiffness)")
    if "gripper_open" in given and "gripper_closed" in given:
        lo, hi = sorted((float(given["gripper_open"]), float(given["gripper_closed"])))
    else:
        lo, hi = _ctrl_range(m, act, moved[act])
    if all(k in given for k in ("left_finger", "right_finger", "gripper_open", "gripper_closed")):
        left, right = m.body(given["left_finger"][0]).id, m.body(given["right_finger"][0]).id
        opening = 1 if given["gripper_open"] > given["gripper_closed"] else -1
    else:
        left, right, opening = _fingers(m, hand, gj, act, (lo, hi), arm, notes)
    open_, closed = (hi, lo) if opening > 0 else (lo, hi)
    if "gripper_open" not in given:
        notes.append(f"gripper_open: {open_:g}, gripper_closed: {closed:g} (open leaves the fingers further apart)")
    return {
        "arm_joints": tuple(m.joint(j).name for j in arm),
        "hand": m.body(hand).name,
        "left_finger": (m.body(left).name,),
        "right_finger": (m.body(right).name,),
        "gripper_actuator": m.actuator(act).name,
        "gripper_open": float(open_),
        "gripper_closed": float(closed),
        "tags": (f"{len(arm)}dof",),
        **{k: v for k, v in fixes.items() if v},
    }


def _gripper_fixes(m, act, others, moved, notes) -> dict:
    """What it takes to drive the gripper as one position-controlled actuator (``fix_gripper``)."""
    out = {"gripper_servo": None, "gripper_mirrors": ()}
    (lead,) = moved[act] if len(moved[act]) == 1 else (None,)
    if lead is not None:
        # Fingers with motors of their own: the others hang from the same body as the driven one.
        parent = m.body_parentid[m.jnt_bodyid[lead]]
        mirrors = []
        for b in others:
            if len(moved[b]) == 1 and m.body_parentid[m.jnt_bodyid[(j := next(iter(moved[b])))]] == parent:
                mirrors.append((m.actuator(b).name, m.joint(j).name, m.joint(lead).name, _mirror_ratio(m, lead, j)))
        if mirrors:
            out["gripper_mirrors"] = tuple(mirrors)
            notes.append(f"gripper_mirrors: fingers {[x[1] for x in mirrors]} have motors of their own; they follow {m.joint(lead).name!r}")
    if m.actuator_biastype[act] == int(mujoco.mjtBias.mjBIAS_NONE) and m.actuator_trntype[act] == int(mujoco.mjtTrn.mjTRN_JOINT):
        j = lead if lead is not None else next(iter(moved[act]))
        if not m.jnt_limited[j]:
            raise DetectionError(f"gripper {m.actuator(act).name!r} is a force motor on an unlimited joint: pass gripper_servo=...")
        force = float(np.max(np.abs(m.actuator_forcerange[act] if m.actuator_forcelimited[act] else m.actuator_ctrlrange[act])))
        span = float(np.diff(m.jnt_range[j])[0])
        mass = float(m.body_subtreemass[m.jnt_bodyid[j]])
        # Stiff enough to reach the motor's force 5% of the travel short of its target, but no
        # stiffer than the timestep keeps stable.
        kp = min(force / (0.05 * span), (0.3 / m.opt.timestep) ** 2 * mass)
        kv = 2.0 * np.sqrt(kp * mass)
        out["gripper_servo"] = (round(kp, 3), round(kv, 4))
        notes.append(f"gripper_servo: {m.actuator(act).name!r} is a {force:g} N force motor; driven as a position servo (kp {kp:.4g})")
    return out


def _mirror_ratio(m, lead: int, follow: int) -> float:
    """How far ``follow`` moves per unit of ``lead`` for the two fingers to open alike."""
    d = mujoco.MjData(m)
    reset_data(m, d)

    def velocity(j):  # how the finger's body moves per unit of its joint
        b, eps = int(m.jnt_bodyid[j]), 1e-4
        d.qpos[m.jnt_qposadr[j]] += eps
        mujoco.mj_kinematics(m, d)
        p = d.xpos[b].copy()
        d.qpos[m.jnt_qposadr[j]] -= 2 * eps
        mujoco.mj_kinematics(m, d)
        d.qpos[m.jnt_qposadr[j]] += eps
        return (p - d.xpos[b]) / (2 * eps)

    # The follower moves opposite to the leader. (Not from the gap between the two bodies: two
    # fingers whose frames start at the same point, as on the AgileX PiPER, have no gap to vary.)
    va, vb = velocity(lead), velocity(follow)
    return round(-float(va @ vb) / float(vb @ vb), 6) if float(vb @ vb) > 1e-12 else 1.0


def _soft_servos(m, arm, moved) -> str:
    """Arm joints whose position servo is too soft for the joint's dry friction to settle precisely."""
    out = []
    for a, js in moved.items():
        if len(js) == 1 and (j := next(iter(js))) in arm and m.actuator_biastype[a] == int(mujoco.mjtBias.mjBIAS_AFFINE):
            kp = -float(m.actuator_biasprm[a, 1])
            error = float(m.dof_frictionloss[m.jnt_dofadr[j]]) / kp if kp > 0 else np.inf
            if error > 0.01:
                out.append(f"{m.joint(j).name} {error:.2f} rad")
    return ", ".join(out)


def _ctrl_range(m, act, joints) -> tuple[float, float]:
    if m.actuator_ctrllimited[act]:
        return tuple(float(v) for v in m.actuator_ctrlrange[act])
    j = next(iter(joints))
    if m.jnt_limited[j] and m.actuator_biastype[act] != 0:  # a position servo: its control is a joint target
        return tuple(float(v) for v in m.jnt_range[j])
    raise DetectionError(f"gripper actuator {m.actuator(act).name!r} has no control range: pass gripper_open=... and gripper_closed=...")


def _settled(m, d, act, value, arm):
    """Drive the gripper to ``value`` with the arm held, contacts off, until it stops."""
    d.ctrl[act] = value
    still = 0
    for _ in range(int(5.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        still = still + 1 if np.abs(d.qvel).max() < 1e-4 else 0
        if still * m.opt.timestep > 0.2:
            break


def _fingers(m, hand, gj, act, rng, arm, notes):
    """The two finger bodies, and +1 if the top of the control range opens the gripper (-1 if it closes it)."""
    # Every jointed part below the hand belongs to the gripper, driven or not: a linkage's
    # passive joints (the Robotiq 2F-85's followers) are closed by connect constraints.
    below = {j for j in range(m.njnt) if m.jnt_type[j] in (_HINGE, _SLIDE) and m.jnt_bodyid[j] in _subtree(m, hand) - {hand}}
    branches = [c for c in range(1, m.nbody) if m.body_parentid[c] == hand and any(m.jnt_bodyid[j] in _subtree(m, c) for j in below | gj)]
    # A model with no collision geometry on the fingers cannot grasp in any engine.
    sub = {c: _subtree(m, c) for c in branches}
    if not any(_collision_geoms(m, s) for s in sub.values()):
        raise DetectionError("the gripper's fingers have no collision geometry: pass left_finger=... and right_finger=...")

    # Fingers move freely for this: contacts and gravity off, both restored on the way out.
    opt_flags = m.opt.disableflags
    gravity = m.opt.gravity.copy()
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    m.opt.gravity[:] = 0
    try:
        d = mujoco.MjData(m)
        reset_data(m, d)
        for i in range(m.nu):  # hold the arm where it is
            if m.actuator_trntype[i] == int(mujoco.mjtTrn.mjTRN_JOINT) and m.actuator_trnid[i, 0] in arm:
                d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]

        def local(bodies):
            R, p = d.xmat[hand].reshape(3, 3), d.xpos[hand]
            gs = _collision_geoms(m, bodies)
            return np.vstack([(_corners(m, d, g) - p) @ R for g in gs]) if gs else np.zeros((0, 3))

        _settled(m, d, act, rng[1], arm)
        all_pts = local(set().union(*sub.values()))
        axis = _PRINCIPAL[int(np.argmax(_PRINCIPAL @ (all_pts.mean(axis=0) / np.linalg.norm(all_pts.mean(axis=0)))))]

        def tip_body(bodies):
            """The jointed body carrying the collision geometry that reaches furthest along the tool axis."""
            best, reach = None, -np.inf
            for b in bodies:
                pts = local({b})
                if len(pts) and (pts @ axis).max() > reach:
                    best, reach = b, float((pts @ axis).max())
            while best is not None and m.body_jntnum[best] == 0 and best not in branches:
                best = int(m.body_parentid[best])
            return best, reach

        tips = {}
        for c in branches:
            b, reach = tip_body(sub[c])
            if b is not None:
                tips[b] = reach

        def centre(b):
            pts = local({b} | (_subtree(m, b) if b != hand else set()))
            near = pts[pts @ axis > (pts @ axis).max() - 0.02]
            return near.mean(axis=0)

        at_hi = {b: centre(b) for b in tips}
        _settled(m, d, act, rng[0], arm)
        at_lo = {b: centre(b) for b in tips}
        moves = {b: at_hi[b] - at_lo[b] for b in tips}
        moving = {b: v for b, v in moves.items() if np.linalg.norm(v - axis * (v @ axis)) > 1e-3}
        if not moving:
            if rng[1] - rng[0] > 2 * np.pi - 1e-3:  # both ends of a full turn are the same place
                raise DetectionError(
                    f"the gripper {m.actuator(act).name!r} has a full turn for its range ({rng[0]:g} to {rng[1]:g}), a placeholder"
                    " that says nothing of where it opens: pass gripper_open=... and gripper_closed=..."
                )
            raise DetectionError(f"the gripper actuator {m.actuator(act).name!r} does not move the fingers: pass gripper_actuator=...")
        pairs = [(a, b) for a in moving for b in moving if a < b and moving[a] @ moving[b] < 0]
        if pairs:
            a, b = max(pairs, key=lambda p: min(tips[p[0]], tips[p[1]]))
            fixed = False
        else:  # one moving jaw closing against the hand (or a part fixed to it)
            a, b = max(moving, key=lambda x: tips[x]), hand
            fixed = True
            at_hi[hand] = at_lo[hand] = centre(hand) if _collision_geoms(m, {hand}) else np.zeros(3)
        hi_ap = np.linalg.norm(at_hi[a] - at_hi[b])
        lo_ap = np.linalg.norm(at_lo[a] - at_lo[b])
        opening = 1 if hi_ap > lo_ap else -1
    finally:
        m.opt.disableflags = opt_flags
        m.opt.gravity[:] = gravity

    names = (m.body(a).name, m.body(b).name)
    if _LEFT.search(names[1]) and not _LEFT.search(names[0]):
        a, b = b, a
    elif not (_LEFT.search(names[0]) or _LEFT.search(names[1])) and at_hi[a][1] < at_hi[b][1]:
        a, b = b, a
    how = "one moving jaw against the hand" if fixed else "the moving parts that reach furthest and close towards each other"
    notes.append(f"fingers: {m.body(a).name!r} / {m.body(b).name!r} ({how})")
    return a, b, opening


def _legged(m: mujoco.MjModel, free: int, notes) -> dict:
    base = int(m.jnt_bodyid[free])
    moved = _moved_joints(m)
    joints = [next(iter(js)) for a, js in moved.items() if len(js) == 1]
    if not joints:
        raise DetectionError("a free-floating robot with no actuated joints: nothing to drive")
    out = {"family": "legged", "base_body": m.body(base).name, "arm_joints": tuple(m.joint(j).name for j in joints)}
    notes.append(f"family: legged (free-floating base {m.body(base).name!r}, {len(joints)} actuated joints)")
    motors = [a for a in moved if m.actuator_biastype[a] == 0]
    if motors and len(motors) == m.nu:
        mass = float(m.body_subtreemass[base])
        kp = round(4.0 * mass, 1)
        out["servo"] = (kp, round(kp / 20, 2))
        notes.append(f"servo: torque motors, so driven as joint servos (kp {kp}, scaled by the robot's {mass:.1f} kg)")
    feet = _feet(m, base)
    tag = "humanoid" if feet == 2 else "quadruped" if feet == 4 else "legged"
    out["tags"] = (tag,)
    notes.append(f"tags: {tag} ({feet} feet)")
    return out


def _legs(m, base, moved) -> int:
    """Chains from the base down to the ground that move on two or more actuated joints: legs,
    where a wheel turns on one and a caster on none."""
    actuated = {j for js in moved.values() for j in js}
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    tree = _subtree(m, base)
    leaves = [b for b in tree if not any(m.body_parentid[c] == b for c in range(m.nbody))]
    low = min(d.xpos[b][2] for b in leaves)
    span = max(d.xpos[base][2] - low, 0.05)
    legs = 0
    for b in leaves:
        if d.xpos[b][2] < low + 0.15 * span and d.xpos[b][2] < d.xpos[base][2] - 0.05:  # down at the ground, below the base
            chain = [c for c in _ancestors(m, b) if c in tree]
            if sum(1 for c in chain for j in range(m.body_jntadr[c], m.body_jntadr[c] + m.body_jntnum[c]) if j in actuated) >= 2:
                legs += 1
    return legs


def _feet(m, base) -> int:
    """Leaf chains of the base that end near the ground in the model's first keyframe (or zero pose)."""
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    tree = _subtree(m, base)
    leaves = [b for b in tree if not any(m.body_parentid[c] == b for c in range(m.nbody))]
    zs = {b: d.xpos[b][2] for b in leaves}
    low = min(zs.values())
    span = d.xpos[base][2] - low
    return sum(1 for z in zs.values() if z < low + 0.15 * max(span, 0.05))


def _base_footprint(model: RobotModel) -> list[tuple[np.ndarray, bool]]:
    """For each collision shape fixed to the robot's base that comes down to the table (a
    mounting plate, say): points (x, y) on it, with the robot at the origin facing +x, and
    whether they are a mesh's vertices (else the corners of its box)."""
    m = model.robot_spec(calibrated=False).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = []
    for g in _collision_geoms(m, {b for b in range(1, m.nbody) if m.body_weldid[b] == 0}):
        pts = _surface(m, d, g)
        if pts[:, 2].min() < 0.05:
            out.append((pts[:, :2], m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH)))
    return out


# ---------------------------------------------------------------------------------------------
def _place(model: RobotModel, notes, keep_base=False, keep_home=False) -> RobotModel:
    """Mount the arm where top-down grasps reach the whole task area, and pick a home height."""
    from ..robot import DOWN
    from .model import Kinematics

    kin = Kinematics(replace(model, base_pos=(0.0, 0.0, 0.0), base_yaw=0.0))
    seed = model.keyframe_q()
    if seed is None:
        seed = np.clip(np.zeros(model.n_arm), kin.lower, kin.upper)
    seed = np.clip(seed, kin.lower, kin.upper)

    # IK is local: start it from the model's own pose, mid-range, and a few fixed random poses.
    rng = np.random.default_rng(0)
    lo, hi = np.maximum(kin.lower, -np.pi), np.minimum(kin.upper, np.pi)
    seeds = [seed, (lo + hi) / 2, *(rng.uniform(lo, hi) for _ in range(4))]

    any_yaw = False  # set for an arm with no wrist roll, which grips at whatever angle it reaches with

    def solve(p, turned):
        """Joint angles reaching ``p`` top-down, or None. Grasps are square to the world (yaw 0),
        which on a robot turned by ``turned`` is -turned in its own frame."""
        p = np.asarray(p, float)
        for y in (-turned, None) if any_yaw else (-turned,):
            for s in seeds:
                q, err = kin.ik(p, s, DOWN, yaw=y, rest=seed)
                if err < 1e-3:
                    return q
        return None

    def reaches(p, turned):
        return solve(p, turned) is not None

    def local(p, x, yaw):  # a world point in the frame of a robot mounted at (x, 0, 0) facing yaw
        c, sn = np.cos(yaw), np.sin(yaw)
        v = np.asarray(p, float) - (x, 0.0, 0.0)
        return np.array([c * v[0] + sn * v[1], -sn * v[0] + c * v[1], v[2]])

    base, yaw = np.asarray(model.base_pos, float), float(model.base_yaw)
    footprint = _base_footprint(model)

    def clear(x, turned):  # no part of the base stands on an object
        c, sn = np.cos(turned), np.sin(turned)
        for pts, is_mesh in footprint:
            xy = pts @ np.array([[c, sn], [-sn, c]]) + (x, 0.0)
            for lo, hi in _OBJECTS:
                if is_mesh:  # a mesh's own vertices: its bounding box can be far bigger than it
                    if np.any(np.all((xy > lo) & (xy < hi), axis=1)):
                        return False
                elif xy[:, 0].max() > lo[0] and xy[:, 0].min() < hi[0] and xy[:, 1].max() > lo[1] and xy[:, 1].min() < hi[1]:
                    return False
        return True

    def search():
        xs = np.round(np.arange(0.15, -1.2, -0.025), 3)
        best = (-1, [], 0.0)
        # Facing the task area first: a model whose arm points along another axis is turned.
        for y in (0.0, np.pi / 2, -np.pi / 2, np.pi):
            score = {
                x: sum(reaches(local(p, x, y), y) for p in _AROUND)
                for x in xs
                if clear(x, y) and all(reaches(local(p, x, y), y) for p in _NEEDED)
            }
            if not score:
                continue
            # Of the positions that reach the most, the middle of the longest run leaves the most margin.
            top = max(score.values())
            ok = [x for x in xs if score.get(x) == top]
            runs, run = [], [ok[0]]
            for x in ok[1:]:
                if abs(run[-1] - x - 0.025) < 1e-6:
                    run.append(x)
                else:
                    runs.append(run)
                    run = [x]
            runs.append(run)
            longest = max(runs, key=len)
            if (top, len(longest)) > (best[0], len(best[1])):
                best = (top, longest, y)
            if top == len(_AROUND):
                break  # facing this way already reaches everything
        return best

    if not keep_base:
        top, run, yaw = search()
        if not run:
            # An arm without a wrist roll (four joints, say) cannot square its grip to the
            # world; picks then grip at the angle the arm reaches with (see robot.solve_ik).
            any_yaw = True
            top, run, yaw = search()
            if run:
                notes.append("yaw: the arm cannot turn its grip square to the task area, so it grips at the angle it reaches with")
        if not run:
            raise DetectionError("no mounting position lets top-down grasps reach the task area: pass base_pos=... and home=...")
        base = np.array([float(run[len(run) // 2]), 0.0, 0.0])
        turned = f", turned {np.degrees(yaw):+.0f} deg" if yaw else ""
        notes.append(
            f"base_pos: x = {base[0]:+.3f} m{turned} (grasps reach the task area, and {top}/{len(_AROUND)} points around it,"
            f" from {run[-1]:+.3f} to {run[0]:+.3f})"
        )
    home = model.home
    if not keep_home:
        home = next(((0.2, 0.0, h) for h in _HOME_HEIGHTS if reaches(local((0.2, 0.0, h), base[0], yaw), yaw)), None)
        if home is None:
            raise DetectionError("no reachable home pose above the task area: pass home=...")
        notes.append(f"home: {home}")
    start = model.start
    if start is None and _zero_pose_touches_table(model):
        q = solve(local(home, base[0], yaw), yaw)
        if q is not None:
            start = tuple(round(float(v), 6) for v in q)
            notes.append("start: the home pose, as the model's zero pose puts the arm into the table")
    return replace(model, base_pos=tuple(float(v) for v in base), base_yaw=yaw, home=home, start=start)


def _zero_pose_touches_table(model: RobotModel) -> bool:
    """Whether a moving part of the arm, in the model's zero pose, comes down to the table."""
    m = model.robot_spec(calibrated=False).compile()
    d = mujoco.MjData(m)
    reset_data(m, d)
    moving = {b for b in range(1, m.nbody) if m.body_weldid[b] != 0}
    return any(_corners(m, d, g)[:, 2].min() < 0.005 for g in _collision_geoms(m, moving))
