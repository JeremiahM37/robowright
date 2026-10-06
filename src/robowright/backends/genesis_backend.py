"""Genesis backend: a GPU-oriented engine, run on its CPU backend unless asked otherwise.

One scene steps ~5x faster on Genesis's CPU backend than on a GPU (kernel
launch latency dominates a single small world), and only the CPU backend is
bit-for-bit deterministic; ``ROBOWRIGHT_GENESIS_DEVICE=cpu|gpu`` picks one.

The robot is the URDF exported from its MuJoCo model (see
:mod:`robowright.robots.urdf`), so kinematics, inertias and collision shapes
match the reference backend. URDF cannot couple a gripper's fingers, so the
joints the gripper actuator pushes are servoed to their calibrated position
for the commanded opening, force-capped as MuJoCo models the squeeze, and
every linkage joint is tied to the reference finger by a URDF mimic joint.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Headless offscreen rendering (Genesis renders through pyrender/PyOpenGL).
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter("ignore")
    import genesis as gs

import mujoco

from ..robots import urdf
from ..scene import SceneSpec
from .base import CONTACTS, DETERMINISTIC, FORCES, GROUND_TRUTH, RENDER, STATE, Backend, Contact, TargetRamp, register
from .mujoco_backend import bin_walls

# Driven finger joints without an actuator gain of their own saturate this far off target.
_DRIVE_SAT = 0.25
# A datasheet grip force is the squeeze on any object, so those drivers saturate before the
# jaws meet even a thin one. At 0.25 the xArm's stopped on the test cube 22% off target and
# pressed 1.41 of its 1.57 N m: 25.6 N per jaw against the datasheet's 30.
_DRIVE_SAT_SPEC = 0.05


_DEVICE = "cpu"


def _init() -> str:
    """Genesis can be initialised once per process; every backend instance shares it. Returns the device."""
    global _DEVICE
    if gs._initialized:
        return _DEVICE
    # CPU by default: for one scene it is ~5.6x faster than an RTX 5080 (1.1 vs 6.3 ms per
    # control step, Panda) and the only bit-for-bit deterministic mode. The GPU pays off only
    # with many scenes in parallel. ROBOWRIGHT_GENESIS_DEVICE=gpu opts in.
    device = os.environ.get("ROBOWRIGHT_GENESIS_DEVICE") or "cpu"
    # Double precision: stiff position servos (kp in the thousands) drift in float32.
    gs.init(backend=gs.gpu if device == "gpu" else gs.cpu, precision="64", logging_level="error")
    if device == "cpu" and "OMP_NUM_THREADS" not in os.environ:
        # Genesis's kernels already run on one thread; torch's pool, one thread per core, only
        # spins on its tensors of a few dozen numbers. One process runs as fast without it (0.63 s
        # per policy run either way, bit for bit the same), and six test workers ran each test 2.8x
        # slower with it: the Genesis suite took 113 s on six workers, 84 s with one thread each.
        import torch

        torch.set_num_threads(1)
    _exclude_pairs_hook()
    _DEVICE = device
    return device


def _exclude_pairs_hook() -> None:
    """Teach Genesis's collider the robot's excluded link pairs.

    URDF has no way to say "these two links never collide", and Genesis only
    filters parent-child pairs and pairs overlapping in the zero pose. MuJoCo
    models exclude more (a gripper's interleaved linkage, for one), and left
    in, those contacts lock the joints. The solver carries the pairs as link
    indices; this drops them from the collider's candidate list at build.
    """
    from genesis.engine.solvers.rigid.collider.collider import Collider

    orig = Collider._compute_collision_pair_idx
    if getattr(orig, "_robowright", False):
        return

    def compute(self):
        out = orig(self)
        excluded = getattr(self._solver, "_robowright_excluded", None)
        n, idx, pairs, *flags, large = out
        if not excluded or not n:
            return out
        link = np.array([g.link.idx for g in self._solver.geoms])
        keep = np.array([frozenset((int(link[a]), int(link[b]))) not in excluded for a, b in pairs], dtype=bool)
        pairs, large = pairs[keep], large[keep]
        idx = np.full_like(idx, -1)
        idx[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs), dtype=idx.dtype)
        return (len(pairs), idx, pairs, *flags, large)

    compute._robowright = True
    Collider._compute_collision_pair_idx = compute


def _with_mimic(path, meta) -> Path:
    """A copy of the exported URDF whose linkage finger joints mimic the reference driven joint.

    MuJoCo couples a gripper's linkage with equality constraints. Genesis reads
    URDF ``<mimic>`` as a joint equality constraint, which keeps that coupling
    two-way: when a pad blocks on an object the constraint stops the driver
    too, and both sides of the gripper close together. Servoing linkage joints
    to the driver's measured travel instead lets the driver run on, and the
    half-closed chain jams against itself.
    """
    g = meta.get("gripper")
    followers = meta.get("followers", {})
    if not g and not followers:
        return path
    tree = ET.parse(path)
    joints = {j.get("name"): j for j in tree.getroot().iter("joint")}

    def mimic(name, leader, k, offset):
        ET.SubElement(joints[urdf._safe(name)], "mimic", joint=urdf._safe(leader), multiplier=f"{k:.12g}", offset=f"{offset:.12g}")

    # A telescope's other joints follow the arm joint that drives them (Stretch's arm).
    for name, (leader, offset, ratio) in followers.items():
        mimic(name, leader, ratio, offset)
    if g:
        ref = g["driven"][0]
        rc, ro = g["joints"][ref]
        # Driven joints other than the reference keep their own servo too: MuJoCo's actuator
        # pushes each of them, and its equality constraints keep them in step.
        for name, (c, o) in g["joints"].items():
            if name == ref:
                continue
            k = (o - c) / (ro - rc)
            mimic(name, ref, k, c - k * rc)
    xml = ET.tostring(tree.getroot(), encoding="unicode")
    out = path.with_name(f"robot.genesis-{hashlib.sha1(xml.encode()).hexdigest()[:12]}.urdf")
    if not out.exists():
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        tmp.write_text(xml)
        tmp.replace(out)
    return out


def _hand_anchor(model) -> np.ndarray:
    """Offset of the hand's URDF link frame within its MuJoCo body frame.

    The exporter puts a link's origin on its first joint's axis; the hand
    pose robowright means is the body's, so it is shifted back.
    """
    m = model.robot_spec().compile()
    b = m.body(model.hand).id
    joints = [
        j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    ]
    return m.jnt_pos[joints[0]].copy() if joints else np.zeros(3)


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(float) if hasattr(t, "detach") else np.asarray(t, float)


def _rotate(quat, v) -> np.ndarray:
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    return R.reshape(3, 3) @ np.asarray(v, float)


@register("genesis")
class GenesisBackend(Backend):
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, RENDER, STATE, DETERMINISTIC, FORCES})
    reusable = True

    def __init__(self, spec: SceneSpec, seed: int = 0):
        super().__init__(spec, seed)
        if _init() == "gpu":
            # CUDA reductions sum contact and constraint terms in whatever order threads
            # finish: two identical runs agree to ~1e-7, not bit for bit.
            self.capabilities = self.capabilities - {DETERMINISTIC}
        rm = self.robot_model
        path, meta = urdf.load(rm)
        path = _with_mimic(path, meta)
        self._dt = spec.dt
        self._substeps = max(1, round((1.0 / spec.control_hz) / self._dt))
        self._t = 0.0
        self.scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=self._dt),
            rigid_options=gs.options.RigidOptions(integrator=gs.integrator.implicitfast),
            vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.35, 0.35, 0.35)),
        )
        scene = self.scene
        self._floor = scene.add_entity(gs.morphs.Plane())
        yaw = rm.base_yaw
        # Arm controllers on real robots cancel gravity; bare position servos sag under it.
        # Legged robots stand on their own weight.
        self.robot = scene.add_entity(
            gs.morphs.URDF(
                file=str(path),
                fixed=not rm.floating,
                pos=tuple(rm.base_pos),
                quat=(np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)),
                merge_fixed_links=False,
                default_armature=None,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0 if rm.family == "arm" else 0.0),
        )
        self._objects: dict = {}
        self._walls: dict[str, list] = {}
        for o in spec.objects:
            self._add_object(o)
        self._cams = {}
        for c in spec.cameras:
            self._cams[c.name] = scene.add_camera(res=(c.width, c.height), pos=tuple(c.pos), lookat=tuple(c.lookat), fov=c.fovy, GUI=False)

        robot = self.robot
        link_of = {lk.name: lk for lk in robot.links}
        excluded = {frozenset((link_of[a].idx, link_of[b].idx)) for a, b in meta["excluded_pairs"] if a in link_of and b in link_of}
        scene.sim.rigid_solver._robowright_excluded = excluded
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene.build()

        # joints and dofs
        def dof(name):
            return robot.get_joint(urdf._safe(name)).dofs_idx_local[0]

        jm = meta["joints"]
        self._arm = np.array([dof(j) for j in rm.arm_joints])
        self._arm_followers = [(dof(n), rm.arm_joints.index(lead), o, r) for n, (lead, o, r) in meta.get("followers", {}).items()]
        grip = meta.get("gripper") or {"joints": {}, "driven": [], "effort": {}}
        self._fingers = {n: (dof(n), c, o) for n, (c, o) in grip["joints"].items()}
        driven = grip.get("driven") or grip.get("main", [])
        self._driven = np.array([dof(n) for n in driven])
        self._driven_c = np.array([grip["joints"][n][0] for n in driven])
        self._driven_o = np.array([grip["joints"][n][1] for n in driven])
        if self.has_gripper:
            self._main, self._main_c, self._main_o = self._fingers[grip["main"]]
        all_names = [*rm.arm_joints, *grip["joints"]]
        all_dofs = np.array([dof(n) for n in all_names])
        robot.set_dofs_armature(np.array([jm[n]["armature"] for n in all_names]), all_dofs)
        robot.set_dofs_damping(np.array([jm[n]["damping"] for n in all_names]), all_dofs)
        robot.set_dofs_frictionloss(np.array([jm[n]["frictionloss"] for n in all_names]), all_dofs)
        self._kp = np.array([jm[j].get("kp", 0.0) for j in rm.arm_joints])
        self._kv = np.array([jm[j].get("kv", 0.0) for j in rm.arm_joints])
        eff = np.array([jm[j]["effort"] for j in rm.arm_joints])
        robot.set_dofs_kp(self._kp, self._arm)
        robot.set_dofs_kv(self._kv, self._arm)
        robot.set_dofs_force_range(-eff, eff, self._arm)
        rng = [jm[j]["range"] for j in rm.arm_joints]
        self._lo = np.array([r[0] if r else -np.inf for r in rng])
        self._hi = np.array([r[1] if r else np.inf for r in rng])
        # Finger servos: the actuator's own gains where the model has a position servo on the
        # joint, else a gain that reaches the force cap a fixed fraction of the travel off target.
        inv = _np(robot.get_dofs_invweight())
        # Every finger joint is tied to the driven ones, so the drivers also carry the
        # linkage's dry friction. MuJoCo's grip effort is the net squeeze, measured past
        # that friction; add it back, or a model with stiff linkage friction (xArm 7) stalls.
        ref_c, ref_o = grip["joints"][driven[0]] if driven else (0.0, 1.0)
        friction_load = sum(jm[n]["frictionloss"] * abs((o - c) / (ref_o - ref_c)) for n, (c, o) in grip["joints"].items())

        def finger_gains(names, dofs, sat):
            kp, kv, cap = [], [], []
            for n, d in zip(names, dofs):
                c, o = grip["joints"][n]
                e = grip.get("coupled_effort", grip["effort"])[n] + friction_load / len(names)  # drivers move the linkage
                if "kp" in jm[n]:
                    kp.append(jm[n]["kp"])
                    kv.append(jm[n]["kv"])
                else:
                    k = e / (sat * max(abs(o - c), 1e-6))
                    kp.append(k)
                    # Critically damped, and no faster at full force than the model closes.
                    kv.append(max(2.0 * np.sqrt(k / max(inv[d], 1e-9)), e / max(grip.get("speed", {}).get(n, np.inf), 1e-6)))
                cap.append(e)
            return np.array(kp), np.array(kv), np.array(cap)

        if driven:
            kp, kv, cap = finger_gains(driven, self._driven, _DRIVE_SAT if rm.grip_force is None else _DRIVE_SAT_SPEC)
            robot.set_dofs_kp(kp, self._driven)
            robot.set_dofs_kv(kv, self._driven)
            robot.set_dofs_force_range(-cap, cap, self._driven)

        # Contact friction per link, as the MuJoCo model sets it per geom.
        friction = {}
        for g in meta["geoms"].values():
            friction[g["link"]] = max(friction.get(g["link"], 0.0), g["friction"])
        for lk in robot.links:
            for geom in lk.geoms:
                geom.set_friction(float(np.clip(friction.get(lk.name, 1.0), 0.01, 5.0)))

        # contact labels
        self._label = {lk.idx: "floor" for lk in self._floor.links}
        safe_to_body = {v: k for k, v in meta["links"].items()}
        parts = {n: part for part, names in meta["fingers"].items() for n in names}
        by_idx = {lk.idx: lk for lk in robot.links}
        for lk in robot.links:
            label, cur = None, lk
            while label is None and cur is not None:
                label = parts.get(cur.name)
                cur = by_idx.get(cur.parent_idx)
            body = lk.name.split("__j")[0] if lk.name not in safe_to_body else lk.name
            self._label[lk.idx] = f"robot:{label or safe_to_body.get(body, body)}"
        for name, ent in self._objects.items():
            self._label[ent.links[0].idx] = name
        for name, ents in self._walls.items():
            for e in ents:
                self._label[e.links[0].idx] = name
        self._hand = link_of[meta["hand"]] if rm.hand else None
        self._hand_anchor = _hand_anchor(rm) if rm.hand else None
        self._pending: dict[str, np.ndarray] = {}
        self._ctrl = np.zeros(len(self.joint_names))
        self._servoed = np.concatenate([self._arm, self._driven]) if self.has_gripper else self._arm
        self._ramp = TargetRamp()
        if rm.floating:
            yaw = rm.base_yaw
            self.set_base_pose((*rm.base_pos[:2], rm.stand_height), (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
            self.set_joint_positions(rm.stand_q())
        else:
            q0 = _np(robot.get_dofs_position())
            lo, hi = (_np(x) for x in robot.get_dofs_limit())
            robot.set_dofs_position(np.clip(q0, lo, hi), zero_velocity=True)
        self.set_ctrl(self.qpos())

    def _add_object(self, o) -> None:
        surface = gs.surfaces.Default(color=tuple(o.rgba[:3]))
        if o.kind == "bin":
            ents = []
            for p, hs in bin_walls(o.size):
                pos = np.asarray(o.initial_pos) + _rotate(o.quat, p)
                ents.append(
                    self.scene.add_entity(
                        gs.morphs.Box(size=tuple(2 * np.asarray(hs)), pos=tuple(pos), quat=tuple(o.quat), fixed=True),
                        material=gs.materials.Rigid(friction=max(o.friction, 0.01)),
                        surface=surface,
                    )
                )
            self._walls[o.name] = ents
            return
        if o.kind == "box":
            morph = gs.morphs.Box(size=tuple(2 * np.asarray(o.size)), pos=tuple(o.initial_pos), quat=tuple(o.quat))
            vol = 8 * float(np.prod(o.size))
        elif o.kind == "cylinder":
            r, hh = o.size[0], o.size[1]
            morph = gs.morphs.Cylinder(radius=r, height=2 * hh, pos=tuple(o.initial_pos), quat=tuple(o.quat))
            vol = np.pi * r * r * 2 * hh
        else:
            r = o.size[0]
            morph = gs.morphs.Sphere(radius=r, pos=tuple(o.initial_pos), quat=tuple(o.quat))
            vol = 4 / 3 * np.pi * r**3
        self._objects[o.name] = self.scene.add_entity(
            morph, material=gs.materials.Rigid(rho=o.mass / vol, friction=o.friction), surface=surface
        )

    # robot
    def _opening(self, q):
        return (q - self._main_c) / (self._main_o - self._main_c)

    def qpos(self):
        q = _np(self.robot.get_dofs_position())
        if not self.has_gripper:
            return q[self._arm]
        return np.append(q[self._arm], self._opening(q[self._main]))

    def qvel(self):
        v = _np(self.robot.get_dofs_velocity())
        if not self.has_gripper:
            return v[self._arm]
        return np.append(v[self._arm], v[self._main] / (self._main_o - self._main_c))

    def set_ctrl(self, target):
        target = np.asarray(target, float)
        arm = np.clip(target[: self.n_arm], self._lo, self._hi)
        self._ctrl = np.append(arm, float(np.clip(target[self.n_arm], 0.0, 1.0))) if self.has_gripper else arm
        self._ramp.set(self._ctrl)
        self._command(self._ctrl)  # ramped from the last step's in step()

    def _command(self, x: np.ndarray) -> None:
        if self.has_gripper:
            s = x[self.n_arm]
            x = np.concatenate([x[: self.n_arm], self._driven_c + s * (self._driven_o - self._driven_c)])
        self.robot.control_dofs_position(x, self._servoed)

    def ctrl(self):
        return self._ctrl.copy()

    def set_gain_scale(self, joint, scale):
        i = self.joint_names.index(joint)
        self.robot.set_dofs_kp([self._kp[i] * scale], [self._arm[i]])
        self.robot.set_dofs_kv([self._kv[i] * scale], [self._arm[i]])

    def set_joint_positions(self, q):
        q = np.asarray(q, float)
        s = float(np.clip(q[self.n_arm], 0, 1)) if self.has_gripper else 0.0
        pos = _np(self.robot.get_dofs_position())
        pos[self._arm] = q[: self.n_arm]
        # Every finger joint where it sits at this opening, so the gripper starts at rest.
        for d, c, o in self._fingers.values():
            pos[d] = c + s * (o - c)
        for d, i, offset, ratio in self._arm_followers:
            pos[d] = offset + ratio * q[i]
        self.robot.set_dofs_position(pos, zero_velocity=True)
        self._ramp.reset()  # placed, not moved: no ramp
        self.set_ctrl(q)

    def base_pose(self):
        return _np(self.robot.get_pos()), _np(self.robot.get_quat())

    def base_velocity(self):
        return np.concatenate([_np(self.robot.get_vel()), _np(self.robot.get_ang())])

    def set_base_pose(self, pos, quat):
        self.robot.set_pos(np.asarray(pos, float), zero_velocity=True)
        self.robot.set_quat(np.asarray(quat, float), zero_velocity=True)

    def hand_pose(self):
        pos = _np(self._hand.get_pos())
        quat = _np(self._hand.get_quat())
        return pos - _rotate(quat, self._hand_anchor), quat

    # time
    def step(self):
        solver, ramp, n = self.scene.sim.rigid_solver, self._ramp, self._substeps
        moving = ramp.moving
        for k in range(n):
            if moving:
                self._command(ramp.at((k + 1) / n))
            for name, f in self._pending.items():
                link = (self.robot if name == "robot" else self._objects[name]).links[0]
                solver.apply_links_external_wrench(force=np.asarray(f)[None], links_idx=[link.idx], ref=gs.link_ref_frame.link_COM)
            self.scene.step()
        self._pending = {}
        ramp.arrive()
        self._t += self._substeps * self._dt

    @property
    def control_dt(self):
        return self._substeps * self._dt

    @property
    def time(self):
        return self._t

    # world
    def object_pose(self, name):
        if name in self._walls:
            o = self.spec.object(name)
            return np.asarray(o.initial_pos, float), np.asarray(o.quat, float)
        e = self._objects[name]
        return _np(e.get_pos()), _np(e.get_quat())

    def object_velocity(self, name):
        if name not in self._objects:
            return np.zeros(6)
        e = self._objects[name]
        return np.concatenate([_np(e.get_vel()), _np(e.get_ang())])

    def set_object_pose(self, name, pos, quat=None):
        e = self._objects[name]
        e.set_pos(np.asarray(pos, float), zero_velocity=True)
        if quat is not None:
            e.set_quat(np.asarray(quat, float), zero_velocity=True)

    def contacts(self):
        c = self.scene.sim.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=False)
        la, lb = np.asarray(c["link_a"]), np.asarray(c["link_b"])
        if not len(la):
            return []
        f = np.abs(np.einsum("ij,ij->i", np.asarray(c["force"], float), np.asarray(c["normal"], float)))
        out = []
        for a, b, fn in zip(la, lb, f):
            na, nb = self._label.get(int(a), f"link{a}"), self._label.get(int(b), f"link{b}")
            if na != nb:
                out.append(Contact(na, nb, float(fn)))
        return out

    def apply_force(self, name, force):
        self._pending[name] = self._pending.get(name, np.zeros(3)) + np.asarray(force, float)

    def render(self, camera, width, height):
        cam = self._cams[camera]
        # No garbage collection while Genesis has its GL context current: a MuJoCo renderer
        # collected then (one a dropped scene left in a reference cycle) frees its own EGL
        # context with eglReleaseThread, which un-currents Genesis's too, and the render fails
        # with "Attempt to retrieve context when no valid context".
        collecting = gc.isenabled()
        gc.disable()
        try:
            if tuple(cam.res) != (width, height):
                self._resize(cam, width, height)
            rgb = cam.render(rgb=True)[0]
        finally:
            if collecting:
                gc.enable()
        return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[..., :3])

    def _resize(self, cam, width, height):
        """Cameras are sized at build; re-create the offscreen target at a new size (Genesis has no setter)."""
        r = self.scene.visualizer.rasterizer
        r.remove_camera(cam)
        cam._res = (width, height)
        cam._aspect_ratio = width / height
        r.add_camera(cam)

    def _warmstart(self):
        """The arrays Genesis carries from one step into the next besides the state proper.

        The constraint solver starts from the last step's accelerations, and the
        collider from its last broadphase sort and contact normals. Setting the
        positions clears the first, so without them a state restored mid-contact
        solves its next step from a different start and drifts (~1e-6).
        """
        from genesis.utils.array_class import DataKind

        return [item for item in self.scene.sim.rigid_solver.data if item.kind == DataKind.WARMSTART]

    def get_state(self):
        from genesis.utils.misc import qd_to_numpy

        s = self.scene.sim.rigid_solver
        ws = [qd_to_numpy(item.value, copy=True).astype(float).ravel() for item in self._warmstart()]
        state = np.concatenate([[self._t], _np(s.get_qpos()), _np(s.get_dofs_velocity()), self._ctrl, *ws])
        # A restore recomputes the kinematics from qpos, which can differ in the last bit from what the
        # steps accumulated (~1e-16). Restoring onto this simulation too makes the two runs start alike.
        self.set_state(state)
        return state

    def set_state(self, state):
        from genesis.utils.array_class import fill_data
        from genesis.utils.misc import qd_to_numpy

        s = self.scene.sim.rigid_solver
        state = np.asarray(state, float)
        nq, nv, nc = s.n_qs, s.n_dofs, len(self._ctrl)
        self._t = float(state[0])
        s.set_qpos(state[1 : 1 + nq])
        s.set_dofs_velocity(state[1 + nq : 1 + nq + nv])
        i = 1 + nq + nv
        # States are taken between control steps, when the ramp has arrived: none is pending.
        self._ramp.reset()
        self.set_ctrl(state[i : i + nc])
        i += nc
        if i == len(state):
            return  # a state saved before it carried the warm start: restores approximately
        items, values = self._warmstart(), {}
        for item in items:
            like = qd_to_numpy(item.value, copy=False)
            values[item.name] = state[i : i + like.size].reshape(like.shape).astype(like.dtype)
            i += like.size
        fill_data(items, values)

    def close(self):
        """Destroy the scene and drop every handle into it.

        Genesis frees a scene's device buffers only once nothing references them;
        an entity or camera kept alive here pins ~0.5 GB per world, which runs a
        test session out of memory.
        """
        scene = getattr(self, "scene", None)
        if scene is None:
            return
        with contextlib.suppress(Exception):
            scene.destroy()
        for k in ("scene", "robot", "_floor", "_hand", "_objects", "_walls", "_cams", "_label"):
            setattr(self, k, None)
        gc.collect()
