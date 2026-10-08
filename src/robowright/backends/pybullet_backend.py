"""PyBullet backend: a second, independent physics engine.

Running the same test on two engines separates "my code is wrong" from
"this simulator happens to let it work". The robot is loaded from the URDF
robowright exports from its MuJoCo model (see ``robowright.robots.urdf``),
so both engines see the same links, inertias and collision shapes. What
the URDF cannot carry is reproduced here: each finger joint is driven to
its calibrated position with MuJoCo's grip effort as the force cap (in
place of MuJoCo's tendons and equality constraints), and the robot's
weight is cancelled with per-link forces, as MuJoCo's gravity compensation
does.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import os
import sys

import numpy as np
import pybullet as p

from ..robots import urdf
from ..scene import SceneSpec
from .base import CONTACTS, DETERMINISTIC, FORCES, GROUND_TRUTH, RENDER, Backend, Contact, TargetRamp, register
from .mujoco_backend import bin_walls

POSITION_GAIN = 0.3  # PyBullet motor ERP: fraction of the position error corrected per step
GRAVITY = 9.81
ARMATURE_FLOOR = 2e-3  # kg m^2, see __init__
GEAR_FORCE = 1000.0  # N or N m: a finger linkage's gear constraint is effectively rigid
GEAR_ERP = 0.8  # fraction of a gear constraint's position drift corrected per step


@register("pybullet")
class PybulletBackend(Backend):
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, RENDER, DETERMINISTIC, FORCES})
    # Not reusable: after p.restoreState the physics replays bit for bit, but the EGL renderer
    # keeps stale link poses (a restored YAM's camera image differs from a new build's), and
    # suites that rendered from restored worlds lost xdist workers.

    def __init__(self, spec: SceneSpec, seed: int = 0):
        super().__init__(spec, seed)
        rm = self.robot_model
        path, meta = urdf.load(rm)
        self.meta = meta
        self._path = path
        # Physics only: camera images come from a second client (see render), so the robot loads
        # without the renderer uploading its meshes (UR10e: 0.07 s, against 0.17 s with EGL), and
        # a test that never takes a picture never pays for one.
        self.cid = p.connect(p.DIRECT)
        self._view = None
        c = self.cid
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, -GRAVITY, physicsClientId=c)
        p.setTimeStep(spec.dt, physicsClientId=c)
        p.setPhysicsEngineParameter(numSolverIterations=50, deterministicOverlappingPairs=1, physicsClientId=c)
        self._substeps = max(1, round((1.0 / spec.control_hz) / spec.dt))
        self._t = 0.0
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 0.001], rgbaColor=[0.8, 0.8, 0.78, 1], physicsClientId=c)
        self.floor = p.createMultiBody(0, plane, vis, physicsClientId=c)
        p.changeDynamics(self.floor, -1, lateralFriction=1.0, physicsClientId=c)
        yaw = rm.base_yaw
        with _quiet():
            self.robot = p.loadURDF(
                str(path),
                list(rm.base_pos),
                [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)],
                useFixedBase=not rm.floating,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=c,
            )
        joints, links = {}, {}
        for j in range(p.getNumJoints(self.robot, physicsClientId=c)):
            info = p.getJointInfo(self.robot, j, physicsClientId=c)
            joints[info[1].decode()] = j
            links[info[12].decode()] = j
        self._links = links
        parent = {j: p.getJointInfo(self.robot, j, physicsClientId=c)[16] for j in links.values()}
        finger_root = {links[n]: part for part, names in meta["fingers"].items() for n in names}
        self._link_label = {-1: f"robot:{meta['root']}" if rm.floating else "robot:base"}
        for name, j in links.items():
            r, part = j, None
            while r >= 0 and part is None:
                part, r = finger_root.get(r), parent[r]
            self._link_label[j] = f"robot:{part or name}"
        # Servo model: arm joints from the MuJoCo actuators; every finger joint follows its calibration.
        self._arm = [joints[urdf._safe(n)] for n in rm.arm_joints]
        self._links_j = []
        self._geared = []
        if self.has_gripper:
            g = meta["gripper"]
            driven = g["driven"]
            slide = lambda n: meta["joints"][n]["type"] == "slide"  # noqa: E731
            pair = []  # fingers geared to the driven one, side by side
            if g.get("nested"):
                # Fingers a joint equality ties to the driven joint, hanging below it (Stretch's
                # swing from its slide): one finger is driven, as the fingers are what meets the
                # object, its twin geared to it, and the slide follows them. Geared to the slide
                # (nested joints) they slipped; following it, a blocked finger let it run on; each
                # on a motor of its own, whichever reached the object first shoved it aside.
                driven, pair = list(g["nested"])[:1], list(g["nested"])[1:]
            linkage = [n for n in g["joints"] if n not in driven]
            self._fingers = [joints[urdf._safe(n)] for n in driven]
            self._finger_q = np.array([g["joints"][n] for n in driven])  # (k, 2): closed, open
            # The model's closing pace, as the motors' velocity limit (0: none recorded).
            self._finger_speed = [float(g.get("speed", {}).get(n, 0.0)) for n in driven]
            # A sliding finger is tied to the driven one by a gear constraint, as MuJoCo's joint
            # equality (and Genesis's and Isaac Sim's mimic joints) ties it. It used to be a second
            # motor told to follow: with both fingers at their force limit nothing held the pair
            # centred, and a sideways load slid fingers and cube along the jaw together until a
            # finger hit its stop (the YAM's policy runs, 0/20). Revolute linkages (Robotiq, xArm)
            # keep following by motor: geared, the Robotiq 2F-85's six-joint linkage jammed open.
            self._ref = self._fingers[0]
            self._ref_q = rc, ro = g["joints"][driven[0]]
            geared = [n for n in linkage if slide(n) and slide(driven[0])] + pair
            follow = [n for n in linkage if n not in geared]
            self._links_j = [joints[urdf._safe(n)] for n in follow]
            self._links_q = np.array([g["joints"][n] for n in follow]).reshape(-1, 2)
            self._links_force = [g["effort"][n] for n in follow]
            # A passive linkage is moved by the drivers: they need both fingers' force.
            effort = g.get("coupled_effort", g["effort"]) if geared and not follow else g["effort"]
            self._finger_force = np.array([effort.get(n, g["effort"][n]) * (1 + len(pair)) for n in driven])
            self._geared = [(joints[urdf._safe(n)], *g["joints"][n]) for n in geared]
            for j, lc, lo in self._geared:
                k = (lo - lc) / (ro - rc)
                p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0, physicsClientId=c)
                gear = p.createConstraint(
                    self.robot, self._ref, self.robot, j, p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0], physicsClientId=c
                )
                # Bullet holds ratio * q_ref + q = target. Without an erp it holds only the
                # velocities equal and never corrects drift: squeezing the OpenManipulator-X's
                # cube, the jaws parted 17 mm (one at its stop, its twin pushed open) and every
                # pick lifted without the cube; at 0.8 they stay within 3 mm and squeeze 20 N.
                p.changeConstraint(
                    gear, gearRatio=-k, relativePositionTarget=lc - k * rc, maxForce=GEAR_FORCE, erp=GEAR_ERP, physicsClientId=c
                )
            self._main = joints[urdf._safe(g["main"])]
            self._main_q = g["joints"][g["main"]]
        self._arm_force = np.array([meta["joints"][n]["effort"] for n in rm.arm_joints])
        # A telescope's other joints move with the arm joint that drives them (Stretch's arm): each
        # gets a motor of its own, sent where the leader is sent, the leader's force shared out
        # among them. (Gear constraints, as the sliding fingers have, let the segments shuffle
        # against each other by a centimetre while the leader held still.)
        self._arm_followers = []
        for name, (leader, offset, ratio) in meta.get("followers", {}).items():
            self._arm_followers.append((joints[urdf._safe(name)], rm.arm_joints.index(leader), offset, ratio))
        if self._arm_followers:
            share = self._arm_force.copy()
            for _, i, _, _ in self._arm_followers:
                share[i] = self._arm_force[i] / (1 + sum(f[1] == i for f in self._arm_followers))
            self._arm_force = share
        self._lo = np.array([(meta["joints"][n]["range"] or (-2 * np.pi, 2 * np.pi))[0] for n in rm.arm_joints])
        self._hi = np.array([(meta["joints"][n]["range"] or (-2 * np.pi, 2 * np.pi))[1] for n in rm.arm_joints])
        self._gain = np.full(self.n_arm, POSITION_GAIN)
        for name, j in joints.items():
            jm = meta["joints"].get(name.replace("__", "/"), meta["joints"].get(name))
            if jm is None:
                continue
            # PyBullet has no joint armature (the motor's reflected inertia). Its constraint-based
            # motors need a little extra link inertia to stay stable on light links, but folding in
            # MuJoCo's full armature makes the arm sluggish enough to fling a held object when a
            # policy stops it. A small floor does both jobs (measured on the SO-101: 10/10 policy
            # runs with it, 1/10 without, 6/10 with the full value).
            info = p.getDynamicsInfo(self.robot, j, physicsClientId=c)
            damping = jm["damping"]
            if self.has_gripper and name.replace("__", "/") in meta["gripper"]["joints"]:
                # Bullet applies joint damping explicitly; past about m/dt it locks a light finger
                # solid (the Panda's, damped to close at the model's pace). The finger motors'
                # velocity limit sets that pace here instead.
                damping = min(damping, 0.5 * (info[0] + min(jm["armature"], ARMATURE_FLOOR)) / spec.dt)
            p.changeDynamics(
                self.robot,
                j,
                jointDamping=damping,
                localInertiaDiagonal=list(np.array(info[2]) + min(jm["armature"], ARMATURE_FLOOR)),
                physicsClientId=c,
            )
        # PyBullet multiplies the two bodies' friction coefficients; MuJoCo takes the larger.
        # With robot links at 1, a contact's friction is the other body's own coefficient,
        # which matches MuJoCo whenever the object is the grippier side (as scene objects are).
        for j in [-1, *links.values()]:
            p.changeDynamics(self.robot, j, lateralFriction=1.0, spinningFriction=0.005, physicsClientId=c)
        # PyBullet ignores URDF material colours on OBJ meshes; apply them per shape.
        for name, colors in meta["colors"].items():
            j = links.get(name, -1)
            for k, rgba in enumerate(colors):
                p.changeVisualShape(self.robot, j, shapeIndex=k, rgbaColor=rgba, physicsClientId=c)
        # Arms cancel their own weight like MuJoCo's gravcomp; legged robots stand on it.
        all_links = [-1, *links.values()]
        self._weights = (
            [(j, p.getDynamicsInfo(self.robot, j, physicsClientId=c)[0] * GRAVITY) for j in all_links] if rm.family == "arm" else []
        )
        lip, lio = p.getDynamicsInfo(self.robot, -1, physicsClientId=c)[3:5]
        self._com_in_base = (lip, lio)  # PyBullet reports a floating base at its centre of mass
        self._ctrl = np.zeros(len(self.joint_names))
        self._ramp = TargetRamp()
        self._bodies: dict[str, int] = {}
        for o in spec.objects:
            self._bodies[o.name] = self._make_object(o)
        self._label = {self.floor: "floor", self.robot: None, **{b: n for n, b in self._bodies.items()}}
        self._pending: dict[str, np.ndarray] = {}
        self._cams = {cs.name: cs for cs in spec.cameras}
        if rm.floating:
            yaw = rm.base_yaw
            self.set_base_pose((*rm.base_pos[:2], rm.stand_height), (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
            self.set_joint_positions(rm.stand_q())
        else:
            self.set_joint_positions(np.append(np.clip(np.zeros(self.n_arm), self._lo, self._hi), 1.0))

    def _make_object(self, o, c: int | None = None, visual_only: bool = False) -> int:
        c = self.cid if c is None else c
        pos = list(o.initial_pos)
        if o.kind == "bin":
            parts = bin_walls(o.size)
            col = p.createCollisionShapeArray(
                [p.GEOM_BOX] * 5,
                halfExtents=[list(h) for _, h in parts],
                collisionFramePositions=[list(q) for q, _ in parts],
                physicsClientId=c,
            )
            vis = p.createVisualShapeArray(
                [p.GEOM_BOX] * 5,
                halfExtents=[list(h) for _, h in parts],
                visualFramePositions=[list(q) for q, _ in parts],
                rgbaColors=[list(o.rgba)] * 5,
                physicsClientId=c,
            )
            return p.createMultiBody(0, -1 if visual_only else col, vis, pos, _xyzw(o.quat), physicsClientId=c)
        if o.kind == "box":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(o.size), physicsClientId=c)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(o.size), rgbaColor=list(o.rgba), physicsClientId=c)
        elif o.kind == "cylinder":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=o.size[0], height=2 * o.size[1], physicsClientId=c)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=o.size[0], length=2 * o.size[1], rgbaColor=list(o.rgba), physicsClientId=c)
        else:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=o.size[0], physicsClientId=c)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=o.size[0], rgbaColor=list(o.rgba), physicsClientId=c)
        if visual_only:
            return p.createMultiBody(0, -1, vis, pos, _xyzw(o.quat), physicsClientId=c)
        b = p.createMultiBody(o.mass, col, vis, pos, _xyzw(o.quat), physicsClientId=c)
        p.changeDynamics(b, -1, lateralFriction=o.friction, spinningFriction=0.005, rollingFriction=0.0005, physicsClientId=c)
        return b

    # robot
    def _opening(self, q):
        c, o = self._main_q
        return (q - c) / (o - c)

    def qpos(self):
        if not self.has_gripper:
            return np.array([s[0] for s in p.getJointStates(self.robot, self._arm, physicsClientId=self.cid)])
        q = [s[0] for s in p.getJointStates(self.robot, [*self._arm, self._main], physicsClientId=self.cid)]
        return np.append(q[:-1], self._opening(q[-1]))

    def qvel(self):
        if not self.has_gripper:
            return np.array([s[1] for s in p.getJointStates(self.robot, self._arm, physicsClientId=self.cid)])
        v = [s[1] for s in p.getJointStates(self.robot, [*self._arm, self._main], physicsClientId=self.cid)]
        c, o = self._main_q
        return np.append(v[:-1], v[-1] / (o - c))

    def _finger_targets(self, s):
        return self._finger_q[:, 0] + float(np.clip(s, 0, 1)) * (self._finger_q[:, 1] - self._finger_q[:, 0])

    def set_ctrl(self, target):
        target = np.asarray(target, float)
        arm = np.clip(target[: self.n_arm], self._lo, self._hi)
        self._ctrl = np.append(arm, np.clip(target[self.n_arm], 0, 1)) if self.has_gripper else arm
        self._ramp.set(self._ctrl)
        self._command(1.0)  # ramped from the last step's in step()

    def _command(self, frac: float) -> None:
        """Command the servos ``frac`` of the way along this control step's move (see :class:`TargetRamp`).

        A PyBullet position motor closes a fixed fraction of its error every substep, so stepped
        targets moved the arm in a staircase: a velocity spike several times the commanded speed
        (27 m/s^2 against MuJoCo's 5 on the same policy), then a coast, and the ARX L5's light grip
        lost its cube on every lift (0/20). The arm's motors also get the ramp's velocity as
        feed-forward, so they move at the commanded speed. (Modelling MuJoCo's kp/kv servo per
        joint instead, without the coupling between joints, overshot on the light SO-101 and
        knocked cubes aside as it came down: 8/12 policy runs, against 12/12 with the ramp.)
        The fingers keep stepped targets: their motors already close at a capped pace, and ramped
        they changed nothing but the ARX L5's second placement (33 mm off, against 31 allowed).
        """
        ramp = self._ramp
        x = ramp.at(frac)
        self._sq, self._sv = x[: self.n_arm], ramp.velocity(self.control_dt)[: self.n_arm]
        self._drive_arm()
        if self.has_gripper and frac == 1.0:
            self._drive_fingers(ramp.end[-1])

    def _drive_fingers(self, s: float) -> None:
        for j, q, f, v in zip(self._fingers, self._finger_targets(s), self._finger_force, self._finger_speed):
            kw = {"maxVelocity": v} if v > 0 else {}
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=float(q),
                force=float(f),
                positionGain=POSITION_GAIN,
                physicsClientId=self.cid,
                **kw,
            )

    def _drive_arm(self) -> None:
        p.setJointMotorControlArray(
            self.robot,
            self._arm,
            p.POSITION_CONTROL,
            targetPositions=list(self._sq),
            targetVelocities=list(self._sv),
            forces=list(self._arm_force),
            positionGains=list(self._gain),
            physicsClientId=self.cid,
        )
        if self._arm_followers:
            f = self._arm_followers
            p.setJointMotorControlArray(
                self.robot,
                [j for j, *_ in f],
                p.POSITION_CONTROL,
                targetPositions=[o + r * self._sq[i] for _, i, o, r in f],
                targetVelocities=[r * self._sv[i] for _, i, _, r in f],
                forces=[self._arm_force[i] for _, i, _, _ in f],
                positionGains=[self._gain[i] for _, i, _, _ in f],
                physicsClientId=self.cid,
            )

    def ctrl(self):
        return self._ctrl.copy()

    def set_gain_scale(self, joint, scale):
        self._gain[self.joint_names.index(joint)] = POSITION_GAIN * scale
        self.set_ctrl(self._ctrl)

    def set_joint_positions(self, q):
        q = np.asarray(q, float)
        self._ramp.reset()  # placed, not moved: no ramp
        for j, v in zip(self._arm, q[: self.n_arm]):
            p.resetJointState(self.robot, j, float(v), 0.0, physicsClientId=self.cid)
        for j, i, offset, ratio in self._arm_followers:
            p.resetJointState(self.robot, j, float(offset + ratio * q[i]), 0.0, physicsClientId=self.cid)
        if not self.has_gripper:
            self.set_ctrl(q)
            return
        for j, v in zip(self._fingers, self._finger_targets(q[-1])):
            p.resetJointState(self.robot, j, float(v), 0.0, physicsClientId=self.cid)
        s = float(np.clip(q[-1], 0, 1))
        for j, (c, o) in [*zip(self._links_j, self._links_q), *((j, (c, o)) for j, c, o in self._geared)]:
            p.resetJointState(self.robot, j, float(c + s * (o - c)), 0.0, physicsClientId=self.cid)
        self.set_ctrl(q)
        self._follow()

    def _follow(self):
        if not self._links_j:
            return
        c, o = self._ref_q
        s = (p.getJointState(self.robot, self._ref, physicsClientId=self.cid)[0] - c) / (o - c)
        p.setJointMotorControlArray(
            self.robot,
            self._links_j,
            p.POSITION_CONTROL,
            targetPositions=list(self._links_q[:, 0] + s * (self._links_q[:, 1] - self._links_q[:, 0])),
            forces=self._links_force,
            positionGains=[POSITION_GAIN] * len(self._links_j),
            physicsClientId=self.cid,
        )

    def base_pose(self):
        c = self.cid
        com_p, com_q = p.getBasePositionAndOrientation(self.robot, physicsClientId=c)
        pos, (x, y, z, w) = p.multiplyTransforms(com_p, com_q, *p.invertTransform(*self._com_in_base), physicsClientId=c)
        return np.array(pos), np.array([w, x, y, z])

    def base_velocity(self):
        lin, ang = p.getBaseVelocity(self.robot, physicsClientId=self.cid)
        com = np.array(p.getBasePositionAndOrientation(self.robot, physicsClientId=self.cid)[0])
        lin = np.array(lin) + np.cross(ang, self.base_pose()[0] - com)
        return np.concatenate([lin, ang])

    def set_base_pose(self, pos, quat):
        w, x, y, z = quat
        com_p, com_q = p.multiplyTransforms(list(map(float, pos)), [x, y, z, w], *self._com_in_base, physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.robot, com_p, com_q, physicsClientId=self.cid)
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

    def hand_pose(self):
        ls = p.getLinkState(self.robot, self._links[self.meta["hand"]], computeForwardKinematics=True, physicsClientId=self.cid)
        x, y, z, w = ls[5]
        return np.array(ls[4]), np.array([w, x, y, z])

    # time
    def step(self):
        c, n = self.cid, self._substeps
        moving = self._ramp.moving
        for k in range(n):
            if moving:
                self._command((k + 1) / n)
            self._follow()
            # Gravity compensation: hold each robot link up at its centre of mass.
            for j, weight in self._weights:
                com = (
                    p.getBasePositionAndOrientation(self.robot, physicsClientId=c)[0]
                    if j < 0
                    else p.getLinkState(self.robot, j, physicsClientId=c)[0]
                )
                p.applyExternalForce(self.robot, j, [0, 0, weight], com, p.WORLD_FRAME, physicsClientId=c)
            for name, f in self._pending.items():
                body = self.robot if name == "robot" else self._bodies[name]
                pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=self.cid)
                p.applyExternalForce(body, -1, f.tolist(), pos, p.WORLD_FRAME, physicsClientId=self.cid)
            p.stepSimulation(physicsClientId=self.cid)
        self._pending = {}
        if moving:
            self._ramp.arrive()
            self._command(1.0)  # holding: no feed-forward velocity
        self._t += self._substeps * self.spec.dt

    @property
    def control_dt(self):
        return self._substeps * self.spec.dt

    @property
    def time(self):
        return self._t

    # world
    def object_pose(self, name):
        pos, q = p.getBasePositionAndOrientation(self._bodies[name], physicsClientId=self.cid)
        x, y, z, w = q
        return np.array(pos), np.array([w, x, y, z])

    def object_velocity(self, name):
        lin, ang = p.getBaseVelocity(self._bodies[name], physicsClientId=self.cid)
        return np.array([*lin, *ang])

    def set_object_pose(self, name, pos, quat=None):
        b = self._bodies[name]
        if quat is None:
            _, q = p.getBasePositionAndOrientation(b, physicsClientId=self.cid)
        else:
            q = _xyzw(quat)
        p.resetBasePositionAndOrientation(b, list(map(float, pos)), q, physicsClientId=self.cid)
        p.resetBaseVelocity(b, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

    def contacts(self):
        out = []
        for c in p.getContactPoints(physicsClientId=self.cid):
            a = self._name(c[1], c[3])
            b = self._name(c[2], c[4])
            if a != b:
                out.append(Contact(a, b, float(c[9])))
        return out

    def _name(self, body, link):
        if body == self.robot:
            return self._link_label.get(link, "robot:?")
        return self._label.get(body, f"body{body}")

    def apply_force(self, name, force):
        self._pending[name] = self._pending.get(name, np.zeros(3)) + np.asarray(force, float)

    def render(self, camera, width, height):
        v = self._view or self._make_view()
        self._sync_view(v)
        cs = self._cams[camera]
        view = p.computeViewMatrix(cs.pos, cs.lookat, [0, 0, 1], physicsClientId=v.cid)
        proj = p.computeProjectionMatrixFOV(cs.fovy, width / height, 0.01, 5.0, physicsClientId=v.cid)
        with _quiet():
            img = p.getCameraImage(width, height, view, proj, renderer=v.renderer, physicsClientId=v.cid)
        return np.asarray(img[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def _make_view(self) -> _View:
        """The scene again, in a client that is never stepped, for the camera.

        Poses are copied in before each picture by resetting them, which is also what makes the
        EGL renderer pick them up (after ``restoreState`` it kept drawing stale ones). Rendering
        cannot change the physics, and a camera's first picture costs the extra load.
        """
        c = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId=c)
        renderer = _load_egl(c)  # the plugin must precede body creation or bodies are invisible to it
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 0.001], rgbaColor=[0.8, 0.8, 0.78, 1], physicsClientId=c)
        p.createMultiBody(0, -1, vis, physicsClientId=c)
        with _quiet():
            robot = p.loadURDF(str(self._path), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=c)
        for name, colors in self.meta["colors"].items():
            j = self._links.get(name, -1)
            for k, rgba in enumerate(colors):
                p.changeVisualShape(robot, j, shapeIndex=k, rgbaColor=rgba, physicsClientId=c)
        bodies = {}
        for o in self.spec.objects:
            bodies[self._bodies[o.name]] = self._make_object(o, c, visual_only=True)
        self._view = _View(c, renderer, robot, bodies, p.getNumJoints(self.robot, physicsClientId=self.cid))
        return self._view

    def _sync_view(self, v: _View) -> None:
        c = self.cid
        p.resetBasePositionAndOrientation(v.robot, *p.getBasePositionAndOrientation(self.robot, physicsClientId=c), physicsClientId=v.cid)
        for j, st in enumerate(p.getJointStates(self.robot, range(v.n_joints), physicsClientId=c)):
            p.resetJointState(v.robot, j, st[0], physicsClientId=v.cid)
        for body, mirror in v.bodies.items():
            p.resetBasePositionAndOrientation(mirror, *p.getBasePositionAndOrientation(body, physicsClientId=c), physicsClientId=v.cid)

    def close(self):
        for cid in (self.cid, self._view.cid if self._view else -1):
            if cid >= 0:
                try:
                    with _quiet():
                        p.disconnect(physicsClientId=cid)
                except Exception:
                    pass
        self.cid, self._view = -1, None


_LIBC = ctypes.CDLL(None)


@contextlib.contextmanager
def _quiet():
    """Silence C-level writes to stdout (the EGL plugin prints GL diagnostics there)."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        _LIBC.fflush(None)  # C stdio buffers flush at exit otherwise - after fd 1 is restored
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


@dataclasses.dataclass
class _View:
    cid: int
    renderer: int
    robot: int
    bodies: dict[int, int]  # physics body -> its copy
    n_joints: int


def _load_egl(cid) -> int:
    """GPU offscreen rendering through PyBullet's EGL plugin (~10x faster), else the CPU renderer."""
    if os.environ.get("ROBOWRIGHT_PYBULLET_RENDERER", "egl") != "egl":
        return p.ER_TINY_RENDERER
    try:
        import importlib.util

        spec = importlib.util.find_spec("eglRenderer")
        if spec is None or spec.origin is None:
            return p.ER_TINY_RENDERER
        with _quiet():
            ok = p.loadPlugin(spec.origin, "_eglRendererPlugin", physicsClientId=cid) >= 0
        return p.ER_BULLET_HARDWARE_OPENGL if ok else p.ER_TINY_RENDERER
    except Exception:
        return p.ER_TINY_RENDERER


def _xyzw(wxyz):
    w, x, y, z = wxyz
    return [x, y, z, w]
