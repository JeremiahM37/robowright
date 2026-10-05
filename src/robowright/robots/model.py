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
    site: str  # a site of the arm; or, with ``body``, the name given to a new site at that body's origin
    prefix: str = "gripper/"
    body: str | None = None  # for arms whose model has no site at the flange

    def at(self, s: mujoco.MjSpec):
        """The site of arm spec ``s`` the attachment goes on (added at ``body`` if given)."""
        if self.body is not None:
            return s.body(self.body).add_site(name=self.site)
        return s.site(self.site)


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
    # (kp, kv): a gripper modelled as a force motor becomes a position servo over its joint's
    # travel, limited to the motor's force, so it can be opened part-way like the others.
    gripper_servo: tuple | None = None
    # (actuator, joint, leader joint, ratio) for each finger with a motor of its own: the motor is
    # removed and the joint follows the leader (the finger the gripper actuator drives).
    gripper_mirrors: tuple = ()
    # MuJoCo integrator to run the robot with (a mjtIntegrator name, e.g. "implicitfast"); None
    # keeps the model's own.
    integrator: str | None = None
    # (joint, armature): a floor of reflected motor inertia for joints a force-limited servo would
    # otherwise shake at the step rate (models that leave a geared motor's inertia out).
    armature: tuple = ()
    maker: str = ""
    dof_note: str = ""
    tags: tuple = ()
    license: str = "Apache-2.0"
    source: str = "MuJoCo Menagerie"
    # The force each jaw presses a held object with (N), from the maker's datasheet. When set, the
    # gripper is force-limited to it, as real grippers are; otherwise it squeezes as modelled.
    grip_force: float | None = None
    grip_force_source: str = ""
    extra: dict = field(default_factory=dict)

    # -- MJCF composition ------------------------------------------------------
    def robot_spec(self, calibrated: bool = True) -> mujoco.MjSpec:
        """The robot alone (arm plus attached gripper), names unprefixed, base at the origin.

        ``calibrated=False`` leaves the gripper as the model has it (``grip_force`` unapplied).
        """
        s = mujoco.MjSpec.from_file(str(self.mjcf()))
        for k in list(s.keys):
            s.delete(k)
        if self.integrator is not None:
            s.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{self.integrator.upper()}")
        if self.attach is not None:
            g = mujoco.MjSpec.from_file(str(self.attach.mjcf()))
            for k in list(g.keys):
                g.delete(k)
            # The gripper's contact settings matter for grasping; the arm's solver settings win otherwise.
            s.option.impratio, s.option.cone = g.option.impratio, g.option.cone
            _copy_options(s, g)
            s.attach(g, prefix=self.attach.prefix, site=self.attach.at(s))
        if self.servo is not None:
            _motors_to_servos(s, *self.servo)
        fix_gripper(s, self.gripper_actuator, self.gripper_servo, self.gripper_mirrors)
        for name, value in self.armature:
            j = s.joint(name)
            j.armature = max(float(j.armature), value)
        _exclude_resting_contacts(s)
        if calibrated and self.grip_force is not None and self.has_gripper:
            _limit_grip(s, self)
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


def fix_gripper(s: mujoco.MjSpec, actuator: str | None, servo: tuple | None, mirrors: tuple) -> None:
    """Make a gripper one position-controlled actuator, as robowright drives it (see ``RobotModel``)."""
    if servo is not None and actuator is not None:
        a = s.actuator(actuator)
        j = s.joint(a.target)
        kp, kv = servo
        a.forcerange = list(a.forcerange) if a.forcelimited == mujoco.mjtLimited.mjLIMITED_TRUE else list(a.ctrlrange)
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[0] = kp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[0], a.biasprm[1], a.biasprm[2] = 0.0, -kp, -kv
        a.ctrlrange = list(j.range)
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    for act, joint, leader, ratio in mirrors:
        s.delete(s.actuator(act))
        e = s.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=joint, name2=leader)
        e.data[:5] = [0.0, ratio, 0.0, 0.0, 0.0]


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


def _surface(m, d, g):
    """Points on geom ``g`` in world coordinates: a mesh's vertices (its bounding box overshoots a
    curved jaw by millimetres), or the corners of a primitive's box."""
    if m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH):
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    return _corners(m, d, g)


def _settle(m, d, ctrl_index, value, max_seconds=10.0):
    """Drive one actuator to ``value`` and step until everything stops moving."""
    d.ctrl[ctrl_index] = value
    still = 0
    for _ in range(int(max_seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        still = still + 1 if np.abs(d.qvel).max() < 1e-4 else 0
        if still * m.opt.timestep > 0.2:
            break


_CALIBRATION: dict[str, tuple[float, float]] = {}


def _squeeze(model: RobotModel, limit: float | None = None) -> tuple[float, float]:
    """Close the modelled gripper on a 25 mm block held at its TCP, the actuator capped at
    ``limit``; returns (mean normal force of the two jaws on the block, actuator force)."""
    der = model.derived
    s = model.robot_spec(calibrated=False)
    _free_gripper(s, model)
    s.option.gravity = [0, 0, 0]
    if limit is not None:
        a = s.actuator(model.gripper_actuator)
        a.forcerange = [-limit, limit]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    m = s.compile()
    d = mujoco.MjData(m)
    reset_data(m, d)
    hand = m.body(model.hand).id
    R = d.xmat[hand].reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, d.xmat[hand])
    s.worldbody.add_geom(
        name="calibration_block",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0125, 0.0125, 0.0125],
        pos=list(d.xpos[hand] + R @ der.tcp_offset),
        quat=list(q),
    )
    m = s.compile()
    d = mujoco.MjData(m)
    reset_data(m, d)
    arm = [m.joint(j).id for j in model.arm_joints]
    for i in range(m.nu):  # hold the arm where it is
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[i, 0] in arm:
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    act = m.actuator(model.gripper_actuator).id
    _settle(m, d, act, model.gripper_closed, max_seconds=3.0)
    block = m.geom("calibration_block").id
    labels = body_labels(m, model)
    side = {"left_finger": 0.0, "right_finger": 0.0}
    f = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if block not in (c.geom1, c.geom2):
            continue
        label = labels.get(int(m.geom_bodyid[c.geom2 if c.geom1 == block else c.geom1]))
        if label in side:
            mujoco.mj_contactForce(m, d, i, f)
            side[label] += abs(f[0])
    clamp = sum(side.values()) / 2 if min(side.values()) > 0 else 0.0
    return clamp, abs(float(d.actuator_force[act]))


def closing_speeds(model: RobotModel) -> dict[str, float]:
    """Each gripper joint's peak speed as the modelled gripper (before ``grip_force``) closes on
    nothing. Engines that drive the fingers with their own gains damp them to it, so a gripper
    stiffened to its datasheet force closes at the model's pace instead of slamming shut."""
    s = model.robot_spec(calibrated=False)
    _free_gripper(s, model)
    s.option.gravity = [0, 0, 0]
    s.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    m = s.compile()
    d = mujoco.MjData(m)
    reset_data(m, d)
    act = m.actuator(model.gripper_actuator).id
    _settle(m, d, act, model.gripper_open, max_seconds=3.0)
    d.ctrl[act] = model.gripper_closed
    peak = dict.fromkeys(model.derived.gripper_joints, 0.0)
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        for n in peak:
            peak[n] = max(peak[n], abs(float(d.qvel[m.joint(n).dofadr[0]])))
    return peak


def grip_calibration(model: RobotModel) -> tuple[float, float]:
    """``(offset, gain)``: the actuator force a held object feels nothing of (springs holding
    the jaws open), and the jaw force each further newton of actuator force adds.

    Measured by closing the jaws on a block with the actuator capped at two forces below
    its own, so a datasheet's jaw force converts to an actuator force limit whatever the
    transmission: a slide, a tendon, or a sprung linkage of swinging jaws.
    """
    if model.name not in _CALIBRATION:
        _, natural = _squeeze(model)
        (c1, f1), (c2, f2) = _squeeze(model, 0.45 * natural), _squeeze(model, 0.9 * natural)
        if c2 <= c1 or f2 <= f1:
            raise ValueError(f"{model.name}: the gripper's squeeze does not grow with its force; cannot calibrate")
        gain = (c2 - c1) / (f2 - f1)
        _CALIBRATION[model.name] = (f1 - c1 / gain, gain)
    return _CALIBRATION[model.name]


_LEVERAGE: dict[str, dict] = {}


def _leverage(model: RobotModel) -> dict:
    """Joint force per newton of gripper actuator force, for each joint it moves directly."""
    if model.name not in _LEVERAGE:
        m = model.robot_spec(calibrated=False).compile()
        d = mujoco.MjData(m)
        reset_data(m, d)
        act = m.actuator(model.gripper_actuator).id
        adr = d.moment_rowadr[act]
        rows = range(adr, adr + d.moment_rownnz[act])
        _LEVERAGE[model.name] = {m.joint(m.dof_jntid[d.moment_colind[i]]).name: abs(float(d.actuator_moment[i])) for i in rows}
    return _LEVERAGE[model.name]


def _actuated_joints(model: RobotModel) -> frozenset:
    """The joints the gripper actuator moves directly (not through equality constraints)."""
    return frozenset(_leverage(model))


def _free_gripper(s: mujoco.MjSpec, model: RobotModel) -> None:
    """A gripper whose motor's force limit is what limits the squeeze: no dry friction in its
    joints, and a rigid coupling between its fingers.

    Some models give the finger linkage the arm's joint defaults: the xArm 7's six gripper
    joints inherit the arm's 1 N m of friction each, more than its 30 N gripper can drive.
    """
    arm = set(model.arm_joints)
    for j in s.joints:
        if j.name not in arm and j.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            j.frictionloss = 0.0
    # A finger the actuator moves only through an equality is coupled mechanically, so make
    # that as stiff as the timestep allows: soft (MuJoCo's default), it lets the driven finger
    # run ahead under full force while the other lags, and the object is pushed off-centre (the
    # PiPER's 40 N held only 15 N that way). Fingers the actuator drives itself (the Panda's
    # tendon) are left alone: a stiff equality on top over-constrains them and they lock.
    driven = _actuated_joints(model)
    for e in s.equalities:
        if e.type == mujoco.mjtEq.mjEQ_JOINT and not {e.name1, e.name2} <= driven:
            e.solref = [min(e.solref[0], 2.5 * s.option.timestep), 1.0]


def _limit_grip(s: mujoco.MjSpec, model: RobotModel) -> None:
    """Make the gripper a stiff servo limited to the force that presses each jaw with ``grip_force``.

    A real gripper closes until its motor's force limit, so it squeezes alike whatever it
    holds; a soft position servo squeezes in proportion to how far the object stops it.
    """
    _free_gripper(s, model)
    offset, gain = grip_calibration(model)
    limit = offset + model.grip_force / gain
    a = s.actuator(model.gripper_actuator)
    clamp, _ = _squeeze(model)
    stiff = max(1.0, 4.0 * model.grip_force / max(clamp, 1e-3))  # reaches the limit on objects well short of closed
    # Stiffer in position only: MuJoCo's implicit integrator counts the actuator's velocity
    # gain even while its force is capped, so a scaled-up one brakes motion that the capped
    # force never drives, and the jaws lock part-way (the Panda's stuck at 0.88 of 0.5).
    a.gainprm[0] *= stiff
    a.biasprm[1] *= stiff
    a.forcerange = [-limit, limit]
    a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    # It closes no faster than the model did: damping on the driven joints (outside the
    # actuator's force limit, and exported to every engine) holds a jaw at full force to the
    # model's own top speed. Stiffened alone, a light jaw slams into the object and knocks it away.
    speed = closing_speeds(model)
    for name, k in _leverage(model).items():
        j = s.joint(name)
        j.damping[0] = max(float(j.damping[0]), k * limit / max(speed[name], 1e-6))


def _derive(model: RobotModel) -> Derived:
    s = model.robot_spec(calibrated=False)
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
    if model.hand in model.left_finger + model.right_finger:
        # One jaw moves against a jaw fixed to the hand (the SO-101): a held object sits against
        # the fixed jaw, so its centre is half its width (a 25 mm cube's) out from that jaw's
        # inner face, not where the jaws meet when shut - plus 5 mm, so the fixed jaw comes down
        # beside the object even if the arm sags, and the moving jaw pushes it across.
        toward = -grip if model.hand in model.left_finger else grip
        bodies = left if model.hand in model.left_finger else right
        R, p = d.xmat[hand].reshape(3, 3), d.xpos[hand]
        fixed = np.vstack([(_surface(m, d, g) - p) @ R for g in _collision_geoms(m, bodies)])
        near = fixed[fixed @ axis > (fixed @ axis).max() - 0.02]
        face = near[np.argmax(near @ toward)]
        mid = mid - toward * ((mid - face) @ toward) + toward * min(aperture / 2, 0.0175)
    tcp = mid - axis * (mid @ axis) + axis * (tip - inset) if model.tcp is None else np.asarray(model.tcp, float)
    # Swinging jaws can reach further open than closed; clearance must cover both.
    inset = max(inset, inset + tip_open - tip)
    return Derived(axis, grip, tcp, inset, joints, main, aperture)


def _without_meshes(spec: mujoco.MjSpec) -> mujoco.MjModel:
    """``spec`` compiled without its meshes, each body keeping the inertia they gave it.

    Kinematics needs body frames, not shapes, and the meshes were most of a process's memory:
    every arm's kinematic model is kept, at up to 72 MB each (PiPER) against 0.06 MB without.
    Frames, centres of mass and Jacobians come out bit for bit the same.
    """
    full = spec.compile()
    for b in spec.bodies[1:]:
        i = full.body(b.name)
        b.explicitinertial = True
        b.mass, b.ipos, b.iquat, b.inertia = float(i.mass[0]), i.ipos.copy(), i.iquat.copy(), i.inertia.copy()
        b.fullinertia = [np.nan] * 6
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH or g.meshname:
            spec.delete(g)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    return spec.compile()


class Kinematics:
    """Forward and inverse kinematics of the arm, computed on the MJCF model.

    The kinematic model is the robot alone, placed at its base pose, so
    positions are in world coordinates.
    """

    def __init__(self, model: RobotModel):
        self.model = model
        world = mujoco.MjSpec()
        model.add_to(world)
        self.m = _without_meshes(world)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = np.array([m.joint(PREFIX + j).qposadr[0] for j in model.arm_joints])
        self.dadr = np.array([m.joint(PREFIX + j).dofadr[0] for j in model.arm_joints])
        jid = [m.joint(PREFIX + j).id for j in model.arm_joints]
        lim = m.jnt_limited[jid].astype(bool)
        self.lower = np.where(lim, m.jnt_range[jid, 0], -2 * np.pi)
        self.upper = np.where(lim, m.jnt_range[jid, 1], 2 * np.pi)
        # A position servo cannot be told to go past its control range, even where the joint could.
        for a in range(m.nu):
            if m.actuator_trntype[a] == int(mujoco.mjtTrn.mjTRN_JOINT) and m.actuator_trnid[a, 0] in jid and m.actuator_ctrllimited[a]:
                if m.actuator_biastype[a] == int(mujoco.mjtBias.mjBIAS_AFFINE) and m.actuator_biasprm[a, 1] < 0:
                    i = jid.index(m.actuator_trnid[a, 0])
                    lo, hi = m.actuator_ctrlrange[a]
                    self.lower[i], self.upper[i] = max(self.lower[i], lo), min(self.upper[i], hi)
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
