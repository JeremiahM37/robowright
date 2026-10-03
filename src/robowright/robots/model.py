"""What robowright needs to know about a robot, and what it works out itself.

A :class:`RobotModel` names a handful of things a person has to decide - which
joints form the arm, which bodies are the fingers, which actuator drives the
gripper - and points at an MJCF file. Everything else is derived from the
compiled model rather than hand-tuned per robot:

* the tool axis (the hand-frame direction the fingers point along),
* the tool centre point, just behind the fingertips, midway between the fingers,
* how far the fingertips reach past the TCP (so grasps stay off the table),
* the gripper calibration: each finger joint's position when open and closed,
  which is how backends without MuJoCo's tendons and equality constraints
  drive the same fingers.

Kinematics also run on the MJCF model (:class:`Kinematics`), so the same file
is the single description of the robot for every backend.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

PREFIX = "robot/"
_HINGE, _SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
_PRINCIPAL = np.vstack([np.eye(3), -np.eye(3)])


@dataclass(frozen=True)
class Attachment:
    """A second MJCF (typically a gripper) attached at a site of the arm."""

    mjcf: Callable[[], Path]
    site: str
    prefix: str = "gripper/"


@dataclass(frozen=True)
class RobotModel:
    name: str
    title: str
    mjcf: Callable[[], Path]
    arm_joints: tuple[str, ...]  # the controlled joints (for legged robots: every actuated joint)
    hand: str = ""  # body the TCP is rigidly attached to
    left_finger: tuple[str, ...] = ()  # bodies (and their subtrees) that make up each finger
    right_finger: tuple[str, ...] = ()
    gripper_actuator: str | None = None
    gripper_open: float = 1.0  # actuator ctrl for fully open / fully closed
    gripper_closed: float = 0.0
    base_pos: tuple = (0.0, 0.0, 0.0)
    base_yaw: float = 0.0
    attach: Attachment | None = None
    home: tuple = (0.2, 0.0, 0.1)  # TCP position of the home pose (world frame)
    seed: tuple | None = None  # IK seed for the home pose; default: the model's "home" keyframe
    tool_axis: tuple | None = None  # override the derived tool axis (hand frame)
    tcp_inset: float | None = None  # how far behind the fingertips the TCP sits
    tcp: tuple | None = None  # override the derived TCP (hand frame)
    timestep: float = 0.002
    family: str = "arm"  # "arm" or "legged"
    base_body: str | None = None  # the floating base of a mobile robot
    stand: tuple | None = None  # legged: standing joint angles; default: the model's keyframe
    crouch: tuple | None = None  # legged: fully crouched joint angles; default: bend the bent joints further
    servo: tuple | None = None  # (kp, kv): turn torque motors into joint PD servos, as robot firmware does
    maker: str = ""
    dof_note: str = ""
    tags: tuple = ()
    license: str = "Apache-2.0"
    source: str = "MuJoCo Menagerie"
    extra: dict = field(default_factory=dict)

    # -- MJCF composition ------------------------------------------------------
    def robot_spec(self) -> mujoco.MjSpec:
        """The robot alone (arm plus attached gripper), names unprefixed, base at the origin."""
        s = mujoco.MjSpec.from_file(str(self.mjcf()))
        for k in list(s.keys):
            s.delete(k)
        if self.attach is not None:
            g = mujoco.MjSpec.from_file(str(self.attach.mjcf()))
            for k in list(g.keys):
                g.delete(k)
            # The gripper's contact settings matter for grasping; the arm's solver settings win otherwise.
            s.option.impratio, s.option.cone = g.option.impratio, g.option.cone
            _copy_options(s, g)
            s.attach(g, prefix=self.attach.prefix, site=s.site(self.attach.site))
        if self.servo is not None:
            _motors_to_servos(s, *self.servo)
        _exclude_resting_contacts(s)
        return s

    @property
    def has_gripper(self) -> bool:
        return self.gripper_actuator is not None

    @property
    def floating(self) -> bool:
        return self.base_body is not None

    def stand_q(self) -> np.ndarray:
        """Joint angles of the standing (legged) or home (fallback) pose."""
        if self.stand is not None:
            return np.asarray(self.stand, float)
        q = self.keyframe_q()
        return np.zeros(self.n_arm) if q is None else q

    @functools.cached_property
    def total_mass(self) -> float:
        m = self.robot_spec().compile()
        return float(m.body_mass.sum())

    @functools.cached_property
    def stand_height(self) -> float:
        """Base height at which the standing robot's lowest point just touches the floor."""
        s = self.robot_spec()
        m = s.compile()
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        for j, v in zip(self.arm_joints, self.stand_q()):
            d.qpos[m.joint(j).qposadr[0]] = v
        free = m.body(self.base_body).jntadr[0]
        d.qpos[m.jnt_qposadr[free] : m.jnt_qposadr[free] + 7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        low = min(_corners(m, d, g)[:, 2].min() for g in geoms)
        return float(-low + 0.002)

    def add_to(self, world: mujoco.MjSpec) -> None:
        """Attach the robot to ``world`` at its base pose with every name prefixed ``robot/``."""
        frame = world.worldbody.add_frame(pos=list(self.base_pos), quat=[np.cos(self.base_yaw / 2), 0, 0, np.sin(self.base_yaw / 2)])
        robot = self.robot_spec()
        robot.option.timestep = world.option.timestep
        _copy_options(robot, world)
        world.attach(robot, prefix=PREFIX, frame=frame)

    def keyframe_q(self) -> np.ndarray | None:
        s = mujoco.MjSpec.from_file(str(self.mjcf()))
        keys = {k.name: k for k in s.keys}
        k = keys.get("home") or keys.get("stand") or (next(iter(keys.values())) if keys else None)
        if k is None or not len(k.qpos):
            return None
        m = s.compile()
        qpos = np.asarray(k.qpos)
        return np.array([qpos[m.joint(j).qposadr[0]] for j in self.arm_joints])

    @property
    def n_arm(self) -> int:
        return len(self.arm_joints)

    @functools.cached_property
    def derived(self) -> Derived:
        return _derive(self)

    def part_of(self, body: str) -> str | None:
        """Finger label for an unprefixed body name, or None."""
        for part, names in (("left_finger", self.left_finger), ("right_finger", self.right_finger)):
            if body in names:
                return part
        return None


@dataclass
class Derived:
    tool_axis: np.ndarray  # hand frame
    grip_axis: np.ndarray  # hand frame, right finger -> left finger
    tcp_offset: np.ndarray  # hand frame
    finger_reach: float  # fingertips' reach beyond the TCP along the tool axis
    gripper_joints: dict  # joint -> (q_closed, q_open), every non-arm joint
    gripper_joint: str  # the joint whose travel defines "opening"
    max_aperture: float  # metres between the fingers when open


_OPTIONS = ("timestep", "iterations", "ls_iterations", "impratio", "integrator", "cone", "noslip_iterations")


def _copy_options(src: mujoco.MjSpec, dst: mujoco.MjSpec) -> None:
    """Give ``dst`` the solver options of ``src`` so attaching it raises no conflicts."""
    for k in _OPTIONS:
        setattr(dst.option, k, getattr(src.option, k))


def body_labels(m: mujoco.MjModel, model: RobotModel, prefix: str = "") -> dict[int, str]:
    """Map every robot body id to a part label: ``left_finger``, ``right_finger`` or its own name."""
    out = {}
    for b in range(1, m.nbody):
        name = m.body(b).name
        if not name.startswith(prefix):
            continue
        r, label = b, None
        while r > 0 and label is None:
            label = model.part_of(m.body(r).name[len(prefix) :])
            r = m.body_parentid[r]
        out[b] = label or name[len(prefix) :]
    return out


def _motors_to_servos(s: mujoco.MjSpec, kp: float, kv: float) -> None:
    """Replace torque motors with joint PD servos limited to the motors' torque range.

    Robots such as the Unitree Go2 are modelled with raw torque motors; their
    firmware runs a joint PD loop, which is what a position-commanded test expects.
    """
    for a in s.actuators:
        if a.gaintype == mujoco.mjtGain.mjGAIN_FIXED and a.biastype == mujoco.mjtBias.mjBIAS_NONE:
            lo, hi = a.ctrlrange
            a.gainprm[0] = kp
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[0], a.biasprm[1], a.biasprm[2] = 0.0, -kp, -kv
            a.forcerange = [lo, hi]
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE


def _exclude_resting_contacts(s: mujoco.MjSpec) -> None:
    """Exclude contacts between robot bodies that already touch in the rest pose.

    Some models ship links whose collision meshes overlap where they join
    (Kinova Gen3's base and shoulder, for one). Left in, that contact locks
    the joint; anything touching at rest is part of the design, not a collision.
    """
    m = s.compile()
    d = mujoco.MjData(m)
    reset_data(m, d)
    pairs = set()
    for c in d.contact[: d.ncon]:
        b1, b2 = sorted((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]))
        if b1 > 0:
            pairs.add((m.body(b1).name, m.body(b2).name))
    for b1, b2 in sorted(pairs):
        s.add_exclude(bodyname1=b1, bodyname2=b2)


def reset_data(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """Reset, with every limited joint moved inside its range.

    Some models put a joint's zero outside its limits (finger slides that start
    interpenetrating, for one); starting there wedges the fingers together.
    """
    mujoco.mj_resetData(m, d)
    for j in range(m.njnt):
        if m.jnt_limited[j] and m.jnt_type[j] in (_HINGE, _SLIDE):
            a = m.jnt_qposadr[j]
            d.qpos[a] = np.clip(d.qpos[a], *m.jnt_range[j])
    mujoco.mj_forward(m, d)


def _collision_geoms(m, bodies):
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]


def _corners(m, d, g):
    c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
    pts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    return pts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def _settle(m, d, ctrl_index, value, max_seconds=10.0):
    """Drive one actuator to ``value`` and step until everything stops moving."""
    d.ctrl[ctrl_index] = value
    still = 0
    for _ in range(int(max_seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        still = still + 1 if np.abs(d.qvel).max() < 1e-4 else 0
        if still * m.opt.timestep > 0.2:
            break


def _derive(model: RobotModel) -> Derived:
    s = model.robot_spec()
    s.option.gravity = [0, 0, 0]
    # Finger travel is a property of the gripper, not of whatever the fingers
    # happen to bump into with the arm in its zero pose.
    s.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    m = s.compile()
    d = mujoco.MjData(m)
    reset_data(m, d)
    arm = [m.joint(j).id for j in model.arm_joints]
    for i in range(m.nu):  # hold the arm where it is
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[i, 0] in arm:
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    act = m.actuator(model.gripper_actuator).id
    others = [
        j for j in range(m.njnt) if j not in arm and m.jnt_type[j] in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    ]
    _settle(m, d, act, model.gripper_closed)
    q_closed = {m.joint(j).name: float(d.qpos[m.jnt_qposadr[j]]) for j in others}
    _settle(m, d, act, model.gripper_open)
    q_open = {m.joint(j).name: float(d.qpos[m.jnt_qposadr[j]]) for j in others}
    joints = {n: (q_closed[n], q_open[n]) for n in q_closed}
    if not joints:
        raise ValueError(f"{model.name}: no gripper joints found")
    travel = {
        n: abs(o - c) / (1.0 if m.jnt_type[m.joint(n).id] == int(mujoco.mjtJoint.mjJNT_HINGE) else 0.05) for n, (c, o) in joints.items()
    }
    main = max(travel, key=travel.get)

    hand = m.body(model.hand).id
    labels = body_labels(m, model)
    left = [b for b, lab in labels.items() if lab == "left_finger"]
    right = [b for b, lab in labels.items() if lab == "right_finger"]
    if not left or not right:
        raise ValueError(f"{model.name}: finger bodies {model.left_finger}/{model.right_finger} not found")

    def finger_points():
        mujoco.mj_forward(m, d)
        R, p = d.xmat[hand].reshape(3, 3), d.xpos[hand]
        lp = np.vstack([(_corners(m, d, g) - p) @ R for g in _collision_geoms(m, left)])
        rp = np.vstack([(_corners(m, d, g) - p) @ R for g in _collision_geoms(m, right)])
        return lp, rp

    lp, rp = finger_points()  # open
    if model.tool_axis is not None:
        axis = np.asarray(model.tool_axis, float)
    else:
        centroid = np.vstack([lp, rp]).mean(axis=0)
        axis = _PRINCIPAL[int(np.argmax(_PRINCIPAL @ (centroid / np.linalg.norm(centroid))))]

    def tips(lp, rp):
        near = lambda pts: pts[pts @ axis > (pts @ axis).max() - 0.02].mean(axis=0)  # noqa: E731
        return near(lp), near(rp), max((lp @ axis).max(), (rp @ axis).max())

    lc, rc, tip_open = tips(lp, rp)
    grip = (lc - rc) - axis * ((lc - rc) @ axis)
    aperture = float(np.linalg.norm(grip))
    grip /= np.linalg.norm(grip)
    # The TCP is where the fingers meet, so measure it closed: jaws that swing
    # (rather than slide) meet somewhere else than the midpoint of their open tips.
    _settle(m, d, act, model.gripper_closed)
    lc, rc, tip = tips(*finger_points())
    _settle(m, d, act, model.gripper_open)
    inset = model.tcp_inset if model.tcp_inset is not None else 0.012
    mid = (lc + rc) / 2
    tcp = mid - axis * (mid @ axis) + axis * (tip - inset) if model.tcp is None else np.asarray(model.tcp, float)
    # Swinging jaws can reach further open than closed; clearance must cover both.
    inset = max(inset, inset + tip_open - tip)
    return Derived(axis, grip, tcp, inset, joints, main, aperture)


class Kinematics:
    """Forward and inverse kinematics of the arm, computed on the MJCF model.

    The kinematic model is the robot alone, placed at its base pose, so
    positions are in world coordinates.
    """

    def __init__(self, model: RobotModel):
        self.model = model
        world = mujoco.MjSpec()
        model.add_to(world)
        self.m = world.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = np.array([m.joint(PREFIX + j).qposadr[0] for j in model.arm_joints])
        self.dadr = np.array([m.joint(PREFIX + j).dofadr[0] for j in model.arm_joints])
        jid = [m.joint(PREFIX + j).id for j in model.arm_joints]
        lim = m.jnt_limited[jid].astype(bool)
        self.lower = np.where(lim, m.jnt_range[jid, 0], -2 * np.pi)
        self.upper = np.where(lim, m.jnt_range[jid, 1], 2 * np.pi)
        self.hand = m.body(PREFIX + model.hand).id
        der = model.derived
        self.tool_axis, self.grip_axis, self.tcp_offset = der.tool_axis, der.grip_axis, der.tcp_offset
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))

    def _set(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def fk(self, q) -> np.ndarray:
        """World transform of the TCP (rotation is the hand frame's)."""
        self._set(q)
        R = self.d.xmat[self.hand].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.d.xpos[self.hand] + R @ self.tcp_offset
        return T

    def tcp(self, q) -> np.ndarray:
        return self.fk(q)[:3, 3]

    def ik(self, target, q0, approach=None, yaw=None, iters=150, tol=1e-4, damping=1e-4, rest=None):
        """Damped least-squares IK for the TCP position, tool direction and grip yaw.

        ``approach`` is the world direction the fingers should point along
        (``(0, 0, -1)`` for top-down). ``yaw`` sets the world angle of the
        finger-closing axis about z. Redundant joints are pulled gently toward
        ``rest`` (default ``q0``). Returns ``(q, position_error)``.
        """
        m, d = self.m, self.d
        target = np.asarray(target, float)
        q = np.clip(np.array(q0, float), self.lower, self.upper)
        rest = q.copy() if rest is None else np.asarray(rest, float)
        a = None if approach is None else np.asarray(approach, float) / np.linalg.norm(approach)
        for _ in range(iters):
            T = self.fk(q)
            R, p = T[:3, :3], T[:3, 3]
            mujoco.mj_jac(m, d, self._jacp, self._jacr, p, self.hand)
            Jp, Jr = self._jacp[:, self.dadr], self._jacr[:, self.dadr]
            res, rows = [p - target], [Jp]
            if a is not None:
                t = R @ self.tool_axis
                res.append(0.3 * (t - a))
                rows.append(0.3 * -_skew(t) @ Jr)
            if yaw is not None:
                g = R @ self.grip_axis
                n2 = g[0] ** 2 + g[1] ** 2
                if n2 > 1e-6:
                    ang = np.arctan2(g[1], g[0]) - yaw
                    ang = (ang + np.pi / 2) % np.pi - np.pi / 2  # fingers are symmetric: modulo pi
                    dg = -_skew(g) @ Jr
                    res.append([0.3 * ang])
                    rows.append(0.3 * ((g[0] * dg[1] - g[1] * dg[0]) / n2)[None, :])
            r, J = np.concatenate(res), np.vstack(rows)
            if np.linalg.norm(r[:3]) < tol and (r.size == 3 or np.linalg.norm(r[3:]) < 10 * tol):
                break
            JJt = J @ J.T + damping * np.eye(r.size)
            dq = -J.T @ np.linalg.solve(JJt, r)
            # null-space pull toward the rest posture keeps 7-DoF arms out of odd elbows
            N = np.eye(q.size) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (0.1 * (rest - q))
            step = np.max(np.abs(dq))
            if step > 0.4:
                dq *= 0.4 / step
            q = np.clip(q + dq, self.lower, self.upper)
        return q, float(np.linalg.norm(self.tcp(q) - target))


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
