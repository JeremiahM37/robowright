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
import os
import sys

import numpy as np
import pybullet as p

from ..robots import urdf
from ..scene import SceneSpec
from .base import CONTACTS, DETERMINISTIC, FORCES, GROUND_TRUTH, RENDER, Backend, Contact, register
from .mujoco_backend import bin_walls

POSITION_GAIN = 0.3  # PyBullet motor ERP: fraction of the position error corrected per step
GRAVITY = 9.81


@register("pybullet")
class PybulletBackend(Backend):
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, RENDER, DETERMINISTIC, FORCES})

    def __init__(self, spec: SceneSpec, seed: int = 0):
        super().__init__(spec, seed)
        rm = self.robot_model
        path, meta = urdf.load(rm)
        self.meta = meta
        self.cid = p.connect(p.DIRECT)
        c = self.cid
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, -GRAVITY, physicsClientId=c)
        p.setTimeStep(spec.dt, physicsClientId=c)
        p.setPhysicsEngineParameter(numSolverIterations=50, deterministicOverlappingPairs=1, physicsClientId=c)
        self._renderer = _load_egl(c)  # must precede body creation or bodies are invisible to it
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
                useFixedBase=True,
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
        self._link_label = {-1: "robot:base"}
        for name, j in links.items():
            r, part = j, None
            while r >= 0 and part is None:
                part, r = finger_root.get(r), parent[r]
            self._link_label[j] = f"robot:{part or name}"
        # Servo model: arm joints from the MuJoCo actuators; every finger joint follows its calibration.
        self._arm = [joints[urdf._safe(n)] for n in rm.arm_joints]
        g = meta["gripper"]
        driven = g["driven"]
        linkage = [n for n in g["joints"] if n not in driven]
        self._fingers = [joints[urdf._safe(n)] for n in driven]
        self._finger_q = np.array([g["joints"][n] for n in driven])  # (k, 2): closed, open
        self._finger_force = np.array([g["effort"][n] for n in driven])
        # Linkage joints follow the first driven joint's measured travel, as MuJoCo's
        # equality constraints make them; commanding them separately tilts blocked pads.
        self._ref = self._fingers[0]
        self._ref_q = g["joints"][driven[0]]
        self._links_j = [joints[urdf._safe(n)] for n in linkage]
        self._links_q = np.array([g["joints"][n] for n in linkage]).reshape(-1, 2)
        self._links_force = [g["effort"][n] for n in linkage]
        self._main = joints[urdf._safe(g["main"])]
        self._main_q = g["joints"][g["main"]]
        self._arm_force = np.array([meta["joints"][n]["effort"] for n in rm.arm_joints])
        self._lo = np.array([(meta["joints"][n]["range"] or (-2 * np.pi, 2 * np.pi))[0] for n in rm.arm_joints])
        self._hi = np.array([(meta["joints"][n]["range"] or (-2 * np.pi, 2 * np.pi))[1] for n in rm.arm_joints])
        self._gain = np.full(self.n_arm, POSITION_GAIN)
        for name, j in joints.items():
            jm = meta["joints"].get(name.replace("__", "/"), meta["joints"].get(name))
            if jm is None:
                continue
            # PyBullet has no joint armature. Light links under stiff position control go
            # unstable without the motor's reflected inertia, so fold it into the link inertia.
            inertia = p.getDynamicsInfo(self.robot, j, physicsClientId=c)[2]
            p.changeDynamics(
                self.robot,
                j,
                jointDamping=jm["damping"],
                localInertiaDiagonal=[v + jm["armature"] for v in inertia],
                physicsClientId=c,
            )
        for geom in meta["geoms"].values():
            j = links.get(geom["link"], -1)
            p.changeDynamics(self.robot, j, lateralFriction=geom["friction"], spinningFriction=0.005, physicsClientId=c)
        # PyBullet ignores URDF material colours on OBJ meshes; apply them per shape.
        for name, colors in meta["colors"].items():
            j = links.get(name, -1)
            for k, rgba in enumerate(colors):
                p.changeVisualShape(self.robot, j, shapeIndex=k, rgbaColor=rgba, physicsClientId=c)
        self._weights = [(j, p.getDynamicsInfo(self.robot, j, physicsClientId=c)[0] * GRAVITY) for j in [-1, *links.values()]]
        self.joint_names = [*rm.arm_joints, "gripper"]
        self._ctrl = np.zeros(self.n_arm + 1)
        self._bodies: dict[str, int] = {}
        for o in spec.objects:
            self._bodies[o.name] = self._make_object(o)
        self._label = {self.floor: "floor", self.robot: None, **{b: n for n, b in self._bodies.items()}}
        self._pending: dict[str, np.ndarray] = {}
        self._cams = {cs.name: cs for cs in spec.cameras}
        self.set_joint_positions(np.append(np.clip(np.zeros(self.n_arm), self._lo, self._hi), 1.0))

    def _make_object(self, o) -> int:
        c = self.cid
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
            return p.createMultiBody(0, col, vis, pos, _xyzw(o.quat), physicsClientId=c)
        if o.kind == "box":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(o.size), physicsClientId=c)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(o.size), rgbaColor=list(o.rgba), physicsClientId=c)
        elif o.kind == "cylinder":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=o.size[0], height=2 * o.size[1], physicsClientId=c)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=o.size[0], length=2 * o.size[1], rgbaColor=list(o.rgba), physicsClientId=c)
        else:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=o.size[0], physicsClientId=c)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=o.size[0], rgbaColor=list(o.rgba), physicsClientId=c)
        b = p.createMultiBody(o.mass, col, vis, pos, _xyzw(o.quat), physicsClientId=c)
        p.changeDynamics(b, -1, lateralFriction=o.friction, spinningFriction=0.005, rollingFriction=0.0005, physicsClientId=c)
        return b

    # robot
    def _opening(self, q):
        c, o = self._main_q
        return (q - c) / (o - c)

    def qpos(self):
        q = [s[0] for s in p.getJointStates(self.robot, [*self._arm, self._main], physicsClientId=self.cid)]
        return np.append(q[:-1], self._opening(q[-1]))

    def qvel(self):
        v = [s[1] for s in p.getJointStates(self.robot, [*self._arm, self._main], physicsClientId=self.cid)]
        c, o = self._main_q
        return np.append(v[:-1], v[-1] / (o - c))

    def _finger_targets(self, s):
        return self._finger_q[:, 0] + float(np.clip(s, 0, 1)) * (self._finger_q[:, 1] - self._finger_q[:, 0])

    def set_ctrl(self, target):
        target = np.asarray(target, float)
        self._ctrl = np.append(np.clip(target[: self.n_arm], self._lo, self._hi), np.clip(target[self.n_arm], 0, 1))
        p.setJointMotorControlArray(
            self.robot,
            [*self._arm, *self._fingers],
            p.POSITION_CONTROL,
            targetPositions=[*self._ctrl[: self.n_arm], *self._finger_targets(self._ctrl[-1])],
            forces=[*self._arm_force, *self._finger_force],
            positionGains=[*self._gain, *[POSITION_GAIN] * len(self._fingers)],
            physicsClientId=self.cid,
        )

    def ctrl(self):
        return self._ctrl.copy()

    def set_gain_scale(self, joint, scale):
        self._gain[self.joint_names.index(joint)] = POSITION_GAIN * scale
        self.set_ctrl(self._ctrl)

    def set_joint_positions(self, q):
        q = np.asarray(q, float)
        for j, v in zip(self._arm, q[: self.n_arm]):
            p.resetJointState(self.robot, j, float(v), 0.0, physicsClientId=self.cid)
        for j, v in zip(self._fingers, self._finger_targets(q[-1])):
            p.resetJointState(self.robot, j, float(v), 0.0, physicsClientId=self.cid)
        s = float(np.clip(q[-1], 0, 1))
        for j, (c, o) in zip(self._links_j, self._links_q):
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

    def hand_pose(self):
        ls = p.getLinkState(self.robot, self._links[self.meta["hand"]], computeForwardKinematics=True, physicsClientId=self.cid)
        x, y, z, w = ls[5]
        return np.array(ls[4]), np.array([w, x, y, z])

    # time
    def step(self):
        c = self.cid
        for _ in range(self._substeps):
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
                pos, _ = p.getBasePositionAndOrientation(self._bodies[name], physicsClientId=self.cid)
                p.applyExternalForce(self._bodies[name], -1, f.tolist(), pos, p.WORLD_FRAME, physicsClientId=self.cid)
            p.stepSimulation(physicsClientId=self.cid)
        self._pending = {}
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
        cs = self._cams[camera]
        view = p.computeViewMatrix(cs.pos, cs.lookat, [0, 0, 1], physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(cs.fovy, width / height, 0.01, 5.0, physicsClientId=self.cid)
        with _quiet():
            img = p.getCameraImage(width, height, view, proj, renderer=self._renderer, physicsClientId=self.cid)
        return np.asarray(img[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def close(self):
        try:
            with _quiet():
                p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass


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
