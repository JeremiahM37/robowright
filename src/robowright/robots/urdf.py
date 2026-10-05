"""Export a robot's compiled MuJoCo model as URDF + OBJ meshes.

Other simulators load URDF, and several (Drake among them) only accept OBJ
meshes. Exporting from the compiled MJCF rather than shipping a second,
hand-maintained URDF keeps one description of each robot: the kinematics,
inertias and collision shapes every backend sees are the ones MuJoCo uses.

What URDF cannot say is written to a JSON sidecar (``robot.json``): the
arm's servo gains and force limits, joint armature and damping, and each
finger joint's open and closed positions. Equality constraints and tendons
(how MuJoCo couples a gripper's fingers) are dropped; backends reproduce
the coupling by driving every finger joint to its calibrated position.
"""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .model import RobotModel, closing_speeds

BASE = "robowright_base"
VERSION = 19  # bump when the output format changes, to invalidate caches
_HINGE, _SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)


def cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "robowright" / "urdf"


def export(model: RobotModel) -> Path:
    """Directory holding ``robot.urdf``, ``robot.json`` and ``meshes/``; built once and cached."""
    h = hashlib.sha1(f"{VERSION}|{_cache_key(model)}".encode()).hexdigest()[:12]
    out = cache_root() / f"{model.name}-{h}"
    if (out / "robot.json").exists():
        return out
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    (tmp / "meshes").mkdir(parents=True, exist_ok=True)
    _write(model, tmp)
    try:
        tmp.replace(out)
    except OSError:  # another process won the race
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _cache_key(model: RobotModel) -> str:
    """The model's fields with file loaders replaced by the files they load (a repr would hold addresses)."""
    import dataclasses

    parts = []
    for f in dataclasses.fields(model):
        v = getattr(model, f.name)
        if f.name == "mjcf":
            src = Path(v())
            v = f"{src}@{src.stat().st_mtime_ns}"
        elif f.name == "attach" and v is not None:
            src = Path(v.mjcf())
            v = f"{src}@{src.stat().st_mtime_ns}:{v.site}:{v.prefix}:{v.body}"
        parts.append(f"{f.name}={v!r}")
    return "|".join(parts)


def load(model: RobotModel) -> tuple[Path, dict]:
    d = export(model)
    return d / "robot.urdf", json.loads((d / "robot.json").read_text())


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.ravel(v))


def _rpy(quat) -> np.ndarray:
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    R = R.reshape(3, 3)
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if abs(np.cos(pitch)) > 1e-9:
        roll, yaw = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(R[1, 0], R[0, 0])
    else:
        roll, yaw = np.arctan2(-R[1, 2], R[1, 1]), 0.0
    return np.array([roll, pitch, yaw])


def _rot(quat) -> np.ndarray:
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    return R.reshape(3, 3)


def _safe(name: str) -> str:
    return name.replace("/", "__")


def _write(model: RobotModel, out: Path) -> None:
    spec = model.robot_spec()
    m = spec.compile()
    d = mujoco.MjData(m)
    root = ET.Element("robot", name=model.name)
    meta: dict = {"robot": model.name, "links": {}, "joints": {}, "geoms": {}, "colors": {}}
    has_visual_only = any(not (m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom))
    written_meshes: dict[int, str] = {}
    shift: dict[int, np.ndarray] = {0: np.zeros(3)}  # link-frame origin within each body frame

    def mesh_file(mid: int) -> str:
        if mid not in written_meshes:
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            verts, faces = m.mesh_vert[va : va + vn], m.mesh_face[fa : fa + fn]
            name = f"meshes/{mid:03d}_{_safe(m.mesh(mid).name) or 'mesh'}.obj"
            with open(out / name, "w") as f:
                f.write("".join(f"v {x:.7g} {y:.7g} {z:.7g}\n" for x, y, z in verts))
                f.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces))
            written_meshes[mid] = name
        return written_meshes[mid]

    def geometry(parent, g):
        t, size = m.geom_type[g], m.geom_size[g]
        geom = ET.SubElement(parent, "geometry")
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            ET.SubElement(geom, "box", size=_fmt(2 * size))
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            ET.SubElement(geom, "sphere", radius=f"{size[0]:.9g}")
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ET.SubElement(geom, "cylinder", radius=f"{size[0]:.9g}", length=f"{2 * size[1]:.9g}")
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ET.SubElement(geom, "capsule", radius=f"{size[0]:.9g}", length=f"{2 * size[1]:.9g}")
        elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            ET.SubElement(geom, "sphere", radius=f"{float(np.mean(size)):.9g}")
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            ET.SubElement(geom, "mesh", filename=mesh_file(m.geom_dataid[g]))
        else:
            return False
        return True

    floating = model.floating
    if not floating:
        # A massless root at the robot's base frame: the model's first body may sit at an offset.
        _inertial(ET.SubElement(root, "link", name=BASE), 1e-4, np.zeros(3), [1, 0, 0, 0], np.full(3, 1e-8))
    for b in range(1, m.nbody):
        body = m.body(b)
        link = _safe(body.name)
        parent = m.body_parentid[b]
        joints = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] in (_HINGE, _SLIDE)]
        free = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] not in (_HINGE, _SLIDE)]
        if free and not (floating and parent == 0 and body.name == model.base_body):
            raise ValueError(f"{model.name}: body {body.name} has a free or ball joint; only the floating base may")
        # Joint axes in URDF pass through the child link origin, so the link frame
        # sits at the (first) joint's anchor; geoms and children are shifted to match.
        anchor = m.jnt_pos[joints[0]].copy() if joints else np.zeros(3)
        shift[b] = anchor
        R_body = _rot(m.body_quat[b])
        origin_xyz = m.body_pos[b] - shift[parent] + R_body @ anchor
        prev = _safe(m.body(parent).name) if parent > 0 else (None if floating else BASE)
        chain = joints or [None]
        for k, j in enumerate(chain):
            child = link if k == len(chain) - 1 else f"{link}__j{k}"
            if prev is not None:
                if j is None:
                    jt = "fixed"
                elif m.jnt_type[j] == _SLIDE:
                    jt = "prismatic"
                else:
                    jt = "revolute" if m.jnt_limited[j] else "continuous"
                jname = m.joint(j).name if j is not None else f"{link}__fixed"
                je = ET.SubElement(root, "joint", name=_safe(jname), type=jt)
                ET.SubElement(je, "parent", link=prev)
                ET.SubElement(je, "child", link=child)
                if k == 0:
                    ET.SubElement(je, "origin", xyz=_fmt(origin_xyz), rpy=_fmt(_rpy(m.body_quat[b])))
                if j is not None:
                    ET.SubElement(je, "axis", xyz=_fmt(m.jnt_axis[j]))
                    dof = m.jnt_dofadr[j]
                    lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)) if jt != "continuous" else (0, 0)
                    effort = _effort(m, j)
                    if jt != "continuous":
                        ET.SubElement(je, "limit", lower=f"{lo:.9g}", upper=f"{hi:.9g}", effort=f"{effort:.9g}", velocity="10")
                    else:
                        ET.SubElement(je, "limit", effort=f"{effort:.9g}", velocity="10")
                    ET.SubElement(je, "dynamics", damping=f"{m.dof_damping[dof]:.9g}", friction=f"{m.dof_frictionloss[dof]:.9g}")
                    meta["joints"][m.joint(j).name] = _joint_meta(m, j, effort)
            if child != link:
                le = ET.SubElement(root, "link", name=child)
                _inertial(le, 1e-4, np.zeros(3), [1, 0, 0, 0], np.full(3, 1e-8))
            prev = child
        le = ET.SubElement(root, "link", name=link)
        meta["links"][body.name] = link
        _inertial(le, m.body_mass[b], m.body_ipos[b] - anchor, m.body_iquat[b], m.body_inertia[b])
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b:
                continue
            collides = bool(m.geom_contype[g] or m.geom_conaffinity[g])
            pos, rpy = _fmt(m.geom_pos[g] - anchor), _fmt(_rpy(m.geom_quat[g]))
            if collides:
                ce = ET.SubElement(le, "collision", name=f"g{g}")
                ET.SubElement(ce, "origin", xyz=pos, rpy=rpy)
                if geometry(ce, g):
                    meta["geoms"][f"g{g}"] = {"link": link, "friction": float(m.geom_friction[g, 0])}
                else:
                    le.remove(ce)
            # MuJoCo renders groups 0-2 by default; Menagerie hides collision-only geoms in group 3.
            if m.geom_group[g] <= 2 or not has_visual_only:
                ve = ET.SubElement(le, "visual")
                ET.SubElement(ve, "origin", xyz=pos, rpy=rpy)
                if geometry(ve, g):
                    rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
                    mat = ET.SubElement(ve, "material", name=f"m{g}")
                    ET.SubElement(mat, "color", rgba=_fmt(rgba))
                    meta["colors"].setdefault(link, []).append([float(x) for x in rgba])
                else:
                    le.remove(ve)

    # Collision pairs MuJoCo never checks (adjacent bodies, explicit excludes) - other
    # engines need to be told, or a gripper's interleaved links fight each other.
    excluded = set()
    for i in range(m.nexclude):
        sig = int(m.exclude_signature[i])
        excluded.add(tuple(sorted((_safe(m.body(sig >> 16).name), _safe(m.body(sig & 0xFFFF).name)))))
    meta["excluded_pairs"] = sorted(excluded)
    meta["arm_joints"] = list(model.arm_joints)
    if model.has_gripper:
        der = model.derived
        effort, driven = _grip_effort(m, model)
        per_driver = _moving_fingers(m, model) / len(driven)
        meta["gripper"] = {
            "joints": {n: list(v) for n, v in der.gripper_joints.items()},
            "main": der.gripper_joint,
            "effort": effort,
            # For engines that drive only the driven joints and couple the rest (mimic joints,
            # constraints): one driver moving both fingers needs both fingers' force.
            "coupled_effort": {n: effort[n] * per_driver for n in driven},
            # Datasheet grippers only: the closing pace their stiffened servo keeps (the rest close
            # as their engines drive them, as validated).
            "speed": closing_speeds(model) if model.grip_force is not None else {},
            # Joints the gripper actuator pushes directly. The rest are linkage joints that
            # MuJoCo couples with equality constraints: drive them from the measured
            # position of driven[0], or a blocked finger tilts its pad into the object.
            "driven": driven,
        }
    meta["hand"] = _safe(model.hand)
    meta["root"] = _safe(model.base_body) if floating else BASE
    meta["floating"] = floating
    meta["fingers"] = {
        "left_finger": [_safe(n) for n in model.left_finger],
        "right_finger": [_safe(n) for n in model.right_finger],
    }
    mujoco.mj_forward(m, d)
    ET.indent(root)
    (out / "robot.urdf").write_text('<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode"))
    (out / "robot.json").write_text(json.dumps(meta, indent=1))


def _inertial(link, mass, pos, quat, inertia):
    ie = ET.SubElement(link, "inertial")
    ET.SubElement(ie, "origin", xyz=_fmt(pos), rpy=_fmt(_rpy(quat)))
    ET.SubElement(ie, "mass", value=f"{max(float(mass), 1e-4):.9g}")
    ix, iy, iz = np.maximum(inertia, 1e-8)
    ET.SubElement(ie, "inertia", ixx=f"{ix:.9g}", iyy=f"{iy:.9g}", izz=f"{iz:.9g}", ixy="0", ixz="0", iyz="0")


def _actuator_for(m, j):
    for i in range(m.nu):
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[i, 0] == j:
            return i
    return None


def _effort(m, j) -> float:
    a = _actuator_for(m, j)
    if a is not None and m.actuator_forcelimited[a]:
        return float(np.abs(m.actuator_forcerange[a]).max())
    if m.jnt_actfrclimited[j]:
        return float(np.abs(m.jnt_actfrcrange[j]).max())
    return 1000.0


def _joint_meta(m, j, effort) -> dict:
    a = _actuator_for(m, j)
    dof = m.jnt_dofadr[j]
    out = {
        "type": "slide" if m.jnt_type[j] == _SLIDE else "hinge",
        "range": [float(x) for x in m.jnt_range[j]] if m.jnt_limited[j] else None,
        "armature": float(m.dof_armature[dof]),
        "damping": float(m.dof_damping[dof]),
        "frictionloss": float(m.dof_frictionloss[dof]),
        "effort": effort,
    }
    if a is not None and m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE:
        out["kp"] = float(m.actuator_gainprm[a, 0])
        out["kv"] = float(-m.actuator_biasprm[a, 2])
    return out


MAX_PAD_FORCE = 30.0  # N; roughly what real parallel grippers squeeze with


def _grip_effort(m, model: RobotModel) -> tuple[dict, list]:
    """Drive effort for each finger joint, and which joints the actuator drives directly.

    The effort is the joint force (N) or torque (N m) MuJoCo's model applies when
    closing on a mid-sized object, so the squeeze matches - capped so the force at
    the fingertips stays under ``MAX_PAD_FORCE``. Some models squeeze far harder
    than a real gripper (xArm 7: ~400 N at the pads); MuJoCo's soft contacts absorb
    that, stiffer engines push the pads through the object.
    """
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    der = model.derived
    for name, (c, o) in der.gripper_joints.items():
        d.qpos[m.joint(name).qposadr[0]] = c + 0.5 * (o - c)
    d.ctrl[m.actuator(model.gripper_actuator).id] = model.gripper_closed
    mujoco.mj_forward(m, d)
    hand = m.body(model.hand).id
    tcp = d.xpos[hand] + d.xmat[hand].reshape(3, 3) @ der.tcp_offset

    pad = model.grip_force or MAX_PAD_FORCE  # a datasheet's jaw force replaces the blanket cap

    def cap(name):
        j = m.joint(name).id
        if m.jnt_type[j] == _SLIDE:
            return pad
        r = tcp - d.xanchor[j]
        lever = np.linalg.norm(r - (r @ d.xaxis[j]) * d.xaxis[j])
        return pad * max(lever, 0.01)

    own = {name: float(abs(d.qfrc_actuator[m.joint(name).dofadr[0]])) for name in der.gripper_joints}
    driven = sorted((n for n, f in own.items() if f > 1e-6), key=lambda n: (n != der.gripper_joint, n))
    # Joints MuJoCo couples to the driven one (equality, tendon) get no actuator force
    # of their own; driven individually, they get the strongest joint's of the same kind.
    out = {}
    for name, f in own.items():
        kind = m.jnt_type[m.joint(name).id]
        peers = [g for n, g in own.items() if m.jnt_type[m.joint(n).id] == kind]
        out[name] = min(f if f > 1e-6 else max(peers), cap(name))
    return out, driven


def _moving_fingers(m, model: RobotModel) -> int:
    """How many fingers the gripper moves (the SO-101 moves one jaw against a fixed one)."""
    from .model import body_labels

    gripper = {m.joint(n).id for n in model.derived.gripper_joints}
    labels = body_labels(m, model)
    hand = m.body(model.hand).id
    moving = 0
    for side in ("left_finger", "right_finger"):
        bodies = [b for b, lab in labels.items() if lab == side]
        chain = set()
        for b in bodies:
            while b not in (hand, 0):
                chain.add(b)
                b = m.body_parentid[b]
        moving += any(m.jnt_bodyid[j] in chain for j in gripper)
    return max(moving, 1)
