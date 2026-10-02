"""PyBullet backend: a second, independent physics engine.

Running the same test on two engines separates "my code is wrong" from
"this simulator happens to let it work". The robot is the same SO-101 URDF
the kinematics use; the gripper's collision geometry is replaced with the
convex pieces and primitives MuJoCo Menagerie uses, because a single convex
hull per jaw would fill the gap between the fingers.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pybullet as p

from .. import assets
from ..scene import SceneSpec
from .base import CONTACTS, DETERMINISTIC, FORCES, GROUND_TRUTH, RENDER, Backend, Contact, register

ROBOT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARMATURE = 2e-3  # kg m^2 added to each joint's link inertia, see __init__
_PART = {"gripper_link": "fixed_jaw", "gripper_frame_link": "fixed_jaw", "moving_jaw_so101_v1_link": "moving_jaw"}

# (link, type, size, xyz, rpy[, mesh]) - the gripper collision set from MuJoCo Menagerie's so101.xml.
_GRIPPER_COLLISION = [
    ("gripper_link", "box", (0.0325, 0.015, 0.015), (-0.0025, 0, -0.022), (0, 0, 0)),
    ("gripper_link", "box", (0.01, 0.015, 0.005), (-0.024, 0, -0.04), (0, 0, 0)),
    ("gripper_link", "box", (0.001, 0.004, 0.004), (-0.009, 0, -0.0982), (0, 0, 0)),
    ("gripper_link", "box", (0.001, 0.005, 0.006), (-0.0108, 0, -0.0905), (0, 0, 0)),
    ("gripper_link", "box", (0.001, 0.009, 0.008), (-0.0125, 0, -0.0727), (0, 0, 0)),
    ("gripper_link", "box", (0.001, 0.01, 0.008), (-0.0143, 0, -0.053), (0, 0, 0)),
    ("gripper_link", "mesh", None, (0, -0.000218, 0.00095), (np.pi, 0, 0), "wrist_roll_follower_so101_gripper_part0_v1.stl"),
    ("moving_jaw_so101_v1_link", "box", (0.01, 0.01, 0.015), (0, -0.013, 0.019), (0, 0, 0)),
    ("moving_jaw_so101_v1_link", "box", (0.001, 0.004, 0.004), (-0.0113, -0.076, 0.01875), (0, 0, 0)),
    ("moving_jaw_so101_v1_link", "box", (0.001, 0.005, 0.006), (-0.0093, -0.067, 0.01875), (0, 0, 0)),
    ("moving_jaw_so101_v1_link", "mesh", None, (0, 0, 0.0189), (0, 0, 0), "moving_jaw_so101_gripper_part0_v1.stl"),
    ("moving_jaw_so101_v1_link", "mesh", None, (0, 0, 0.0189), (0, 0, 0), "moving_jaw_so101_gripper_part1_v1.stl"),
]


def _robot_urdf() -> str:
    """Write (once) a copy of the SO-101 URDF with absolute mesh paths and grasp-ready jaw collisions."""
    src = assets.so101_urdf()
    mesh_dir = src.parent / "assets"
    key = hashlib.sha1(src.read_bytes() + repr(_GRIPPER_COLLISION).encode()).hexdigest()[:12]
    out = Path(tempfile.gettempdir()) / f"robowright-so101-{key}.urdf"
    if out.exists():
        return str(out)
    tree = ET.parse(src)
    root = tree.getroot()
    for link in root.findall("link"):
        if link.get("name") in ("gripper_link", "moving_jaw_so101_v1_link"):
            for c in link.findall("collision"):
                link.remove(c)
        for c in _GRIPPER_COLLISION:
            if c[0] != link.get("name"):
                continue
            col = ET.SubElement(link, "collision")
            ET.SubElement(col, "origin", xyz=" ".join(map(str, c[3])), rpy=" ".join(map(str, c[4])))
            geom = ET.SubElement(col, "geometry")
            if c[1] == "box":
                ET.SubElement(geom, "box", size=" ".join(str(2 * v) for v in c[2]))
            else:
                ET.SubElement(geom, "mesh", filename=str(mesh_dir / c[5]))
    xml = ET.tostring(root, encoding="unicode")
    xml = re.sub(r'filename="assets/', f'filename="{mesh_dir}/', xml)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(xml)
    tmp.replace(out)
    return str(out)


@register("pybullet")
class PybulletBackend(Backend):
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, RENDER, DETERMINISTIC, FORCES})

    def __init__(self, spec: SceneSpec, seed: int = 0):
        super().__init__(spec, seed)
        self.cid = p.connect(p.DIRECT)
        c = self.cid
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, -9.81, physicsClientId=c)
        p.setTimeStep(spec.physics_dt, physicsClientId=c)
        p.setPhysicsEngineParameter(numSolverIterations=50, deterministicOverlappingPairs=1, physicsClientId=c)
        self._renderer = _load_egl(c)  # must precede body creation or bodies are invisible to it
        self._substeps = max(1, round((1.0 / spec.control_hz) / spec.physics_dt))
        self._t = 0.0
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 0.001], rgbaColor=[0.8, 0.8, 0.78, 1], physicsClientId=c)
        self.floor = p.createMultiBody(0, plane, vis, physicsClientId=c)
        p.changeDynamics(self.floor, -1, lateralFriction=1.0, physicsClientId=c)
        self.robot = p.loadURDF(_robot_urdf(), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=c)
        self._jidx, self._link_label = {}, {-1: "robot:base_link"}
        self._lo, self._hi = [], []
        for j in range(p.getNumJoints(self.robot, physicsClientId=c)):
            info = p.getJointInfo(self.robot, j, physicsClientId=c)
            name, link = info[1].decode(), info[12].decode()
            self._link_label[j] = f"robot:{_PART.get(link, link)}"
            if name in ROBOT_JOINTS:
                self._jidx[name] = j
                # PyBullet has no joint armature. Without the servo's reflected rotor
                # inertia (MuJoCo's model uses armature=0.028) the light links go unstable
                # under position control, so fold an approximation into the link inertia.
                inertia = p.getDynamicsInfo(self.robot, j, physicsClientId=c)[2]
                p.changeDynamics(
                    self.robot,
                    j,
                    jointDamping=0.6,
                    lateralFriction=1.0,
                    spinningFriction=0.005,
                    rollingFriction=0.0005,
                    localInertiaDiagonal=[v + ARMATURE for v in inertia],
                    physicsClientId=c,
                )
        self.joint_names = list(ROBOT_JOINTS)
        self._j = [self._jidx[n] for n in ROBOT_JOINTS]
        for n in ROBOT_JOINTS:
            info = p.getJointInfo(self.robot, self._jidx[n], physicsClientId=c)
            self._lo.append(info[8])
            self._hi.append(info[9])
        self._lo, self._hi = np.array(self._lo), np.array(self._hi)
        self._gain = np.full(6, 0.3)
        self._max_force = np.full(6, 2.94)
        self._ctrl = np.zeros(6)
        self._bodies: dict[str, int] = {}
        for o in spec.objects:
            self._bodies[o.name] = self._make_object(o)
        self._label = {self.floor: "floor", self.robot: None, **{b: n for n, b in self._bodies.items()}}
        self._pending: dict[str, np.ndarray] = {}
        self._cams = {cs.name: cs for cs in spec.cameras}
        self.set_ctrl(np.zeros(6))

    def _make_object(self, o) -> int:
        c = self.cid
        pos = list(o.initial_pos)
        if o.kind == "bin":
            sx, sy, sz = o.size
            t = 0.003
            parts = [
                ((0, 0, t), (sx, sy, t)),
                ((sx - t, 0, sz), (t, sy, sz)),
                ((-sx + t, 0, sz), (t, sy, sz)),
                ((0, sy - t, sz), (sx, t, sz)),
                ((0, -sy + t, sz), (sx, t, sz)),
            ]
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
    def qpos(self):
        return np.array([s[0] for s in p.getJointStates(self.robot, self._j, physicsClientId=self.cid)])

    def qvel(self):
        return np.array([s[1] for s in p.getJointStates(self.robot, self._j, physicsClientId=self.cid)])

    def set_ctrl(self, target):
        self._ctrl = np.clip(np.asarray(target, float), self._lo, self._hi)
        p.setJointMotorControlArray(
            self.robot,
            self._j,
            p.POSITION_CONTROL,
            targetPositions=self._ctrl.tolist(),
            forces=self._max_force.tolist(),
            positionGains=self._gain.tolist(),
            physicsClientId=self.cid,
        )

    def ctrl(self):
        return self._ctrl.copy()

    def set_gain_scale(self, joint, scale):
        self._gain[self.joint_names.index(joint)] = 0.3 * scale
        self.set_ctrl(self._ctrl)

    def set_joint_positions(self, q):
        for j, v in zip(self._j, q):
            p.resetJointState(self.robot, j, float(v), 0.0, physicsClientId=self.cid)
        self.set_ctrl(q)

    # time
    def step(self):
        for _ in range(self._substeps):
            for name, f in self._pending.items():
                pos, _ = p.getBasePositionAndOrientation(self._bodies[name], physicsClientId=self.cid)
                p.applyExternalForce(self._bodies[name], -1, f.tolist(), pos, p.WORLD_FRAME, physicsClientId=self.cid)
            p.stepSimulation(physicsClientId=self.cid)
        self._pending = {}
        self._t += self._substeps * self.spec.physics_dt

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
