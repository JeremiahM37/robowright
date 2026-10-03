"""MuJoCo backend: the reference simulator.

The robot is its MuJoCo Menagerie model, attached under the ``robot/`` prefix,
so it runs exactly as its authors tuned it: same actuators, tendons and
equality constraints.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..robots import PREFIX, body_labels
from ..robots.model import reset_data
from ..scene import SceneSpec
from .base import CONTACTS, DETERMINISTIC, FORCES, GROUND_TRUTH, RENDER, STATE, Backend, Contact, register


def _lookat_xyaxes(pos, lookat):
    pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
    fwd = lookat - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.concatenate([right, up])


def build_spec(spec: SceneSpec) -> mujoco.MjSpec:
    s = mujoco.MjSpec()
    s.option.timestep = spec.dt
    spec.robot_model.add_to(s)
    # Arm controllers on real robots cancel gravity; bare position servos sag
    # under it instead, by more than a grasp can tolerate.
    for b in s.bodies:
        if b.name.startswith(PREFIX):
            b.gravcomp = 1.0
    s.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    s.visual.headlight.ambient = [0.35, 0.35, 0.35]
    s.visual.global_.offwidth = 1280
    s.visual.global_.offheight = 960
    s.stat.extent = 1.0
    wb = s.worldbody
    tex = s.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.82, 0.82, 0.8],
        rgb2=[0.74, 0.74, 0.72],
        width=256,
        height=256,
    )
    mat = s.add_material(name="table", texrepeat=[8, 8], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    wb.add_light(
        pos=[0.3, -0.3, 1.5], dir=[-0.2, 0.2, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=1, diffuse=[0.5, 0.5, 0.5]
    )
    s.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.93, 0.95, 0.98],
        rgb2=[0.72, 0.78, 0.86],
        width=256,
        height=1536,
    )
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.05], material="table")
    for o in spec.objects:
        pos = o.initial_pos
        if o.kind == "bin":
            b = wb.add_body(name=o.name, pos=list(pos), quat=list(o.quat))
            for i, (p, hs) in enumerate(bin_walls(o.size)):
                b.add_geom(name=f"{o.name}/wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(p), size=list(hs), rgba=list(o.rgba))
            continue
        b = wb.add_body(name=o.name, pos=list(pos), quat=list(o.quat))
        b.add_freejoint(name=f"{o.name}/free")
        gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}[
            o.kind
        ]
        size = list(o.size) + [0] * (3 - len(o.size))
        b.add_geom(
            name=o.name,
            type=gtype,
            size=size,
            rgba=list(o.rgba),
            mass=o.mass,
            condim=4,
            friction=[o.friction, 0.03, 0.003],
            solref=[0.01, 1],
        )
    for c in spec.cameras:
        wb.add_camera(name=c.name, pos=list(c.pos), xyaxes=list(_lookat_xyaxes(c.pos, c.lookat)), fovy=c.fovy)
    return s


def bin_walls(size, t: float = 0.003):
    """(centre, half-extents) of a bin's floor and four walls, in the bin frame."""
    sx, sy, sz = size
    return [
        ((0, 0, t), (sx, sy, t)),
        ((sx - t, 0, sz), (t, sy, sz)),
        ((-sx + t, 0, sz), (t, sy, sz)),
        ((0, sy - t, sz), (sx, t, sz)),
        ((0, -sy + t, sz), (sx, t, sz)),
    ]


@register("mujoco")
class MujocoBackend(Backend):
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, RENDER, STATE, DETERMINISTIC, FORCES})

    def __init__(self, spec: SceneSpec, seed: int = 0):
        super().__init__(spec, seed)
        rm = self.robot_model
        self.mjspec = build_spec(spec)
        self.model = self.mjspec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        arm = [PREFIX + j for j in rm.arm_joints]
        self._qadr = np.array([m.joint(n).qposadr[0] for n in arm])
        self._dadr = np.array([m.joint(n).dofadr[0] for n in arm])
        by_joint = {m.actuator_trnid[i, 0]: i for i in range(m.nu) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT}
        self._act = np.array([by_joint[m.joint(n).id] for n in arm])
        self._gact = m.actuator(PREFIX + rm.gripper_actuator).id
        self._act_all = np.append(self._act, self._gact)
        self._kp = m.actuator_gainprm[self._act, 0].copy()
        self._bias = m.actuator_biasprm[self._act, 1].copy()
        der = rm.derived
        gj = m.joint(PREFIX + der.gripper_joint)
        self._g_qadr, self._g_dadr = gj.qposadr[0], gj.dofadr[0]
        self._g_closed, self._g_open = der.gripper_joints[der.gripper_joint]
        self._substeps = max(1, round((1.0 / spec.control_hz) / m.opt.timestep))
        self._body = {o.name: m.body(o.name).id for o in spec.objects}
        self._free = {o.name: m.joint(f"{o.name}/free") for o in spec.objects if not o.static}
        self._label = {0: "floor"}
        for b, part in body_labels(m, rm, PREFIX).items():
            self._label[b] = f"robot:{part}"
        for b in range(1, m.nbody):
            self._label.setdefault(b, m.body(b).name)
        self._hand = m.body(PREFIX + rm.hand).id
        self._renderers: dict = {}
        reset_data(m, self.data)
        self.set_ctrl(self.qpos())

    # robot
    def _opening(self, q):
        return (q - self._g_closed) / (self._g_open - self._g_closed)

    def qpos(self):
        return np.append(self.data.qpos[self._qadr], self._opening(self.data.qpos[self._g_qadr]))

    def qvel(self):
        return np.append(self.data.qvel[self._dadr], self.data.qvel[self._g_dadr] / (self._g_open - self._g_closed))

    def set_ctrl(self, target):
        m, rm = self.model, self.robot_model
        target = np.asarray(target, float)
        arm = target[: self.n_arm]
        lim = m.actuator_ctrllimited[self._act].astype(bool)
        lo, hi = m.actuator_ctrlrange[self._act].T
        self.data.ctrl[self._act] = np.where(lim, np.clip(arm, lo, hi), arm)
        g = float(np.clip(target[self.n_arm], 0.0, 1.0))
        self.data.ctrl[self._gact] = rm.gripper_closed + g * (rm.gripper_open - rm.gripper_closed)

    def ctrl(self):
        rm = self.robot_model
        g = (self.data.ctrl[self._gact] - rm.gripper_closed) / (rm.gripper_open - rm.gripper_closed)
        return np.append(self.data.ctrl[self._act], g)

    def set_gain_scale(self, joint, scale):
        i = self.joint_names.index(joint)
        a = self._act[i]
        self.model.actuator_gainprm[a, 0] = self._kp[i] * scale
        self.model.actuator_biasprm[a, 1] = self._bias[i] * scale

    def set_joint_positions(self, q):
        q = np.asarray(q, float)
        self.data.qpos[self._qadr] = q[: self.n_arm]
        self.data.qvel[self._dadr] = 0
        # Put every finger joint where it sits at this opening, so the gripper starts at rest.
        s = float(np.clip(q[self.n_arm], 0, 1))
        for name, (c, o) in self.robot_model.derived.gripper_joints.items():
            j = self.model.joint(PREFIX + name)
            self.data.qpos[j.qposadr[0]] = c + s * (o - c)
            self.data.qvel[j.dofadr[0]] = 0
        self.set_ctrl(q)
        mujoco.mj_forward(self.model, self.data)

    def hand_pose(self):
        return self.data.xpos[self._hand].copy(), self.data.xquat[self._hand].copy()

    # time
    def step(self):
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:] = 0

    @property
    def control_dt(self):
        return self._substeps * self.model.opt.timestep

    @property
    def time(self):
        return float(self.data.time)

    # world
    def object_pose(self, name):
        b = self._body[name]
        return self.data.xpos[b].copy(), self.data.xquat[b].copy()

    def object_velocity(self, name):
        if name not in self._free:
            return np.zeros(6)
        j = self._free[name]
        return self.data.qvel[j.dofadr[0] : j.dofadr[0] + 6].copy()

    def set_object_pose(self, name, pos, quat=None):
        j = self._free[name]
        a = j.qposadr[0]
        self.data.qpos[a : a + 3] = pos
        if quat is not None:
            self.data.qpos[a + 3 : a + 7] = quat
        self.data.qvel[j.dofadr[0] : j.dofadr[0] + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def contacts(self):
        out = []
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a = self._label[self.model.geom_bodyid[c.geom1]]
            b = self._label[self.model.geom_bodyid[c.geom2]]
            if a == b:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            out.append(Contact(a, b, float(abs(f6[0]))))
        return out

    def apply_force(self, name, force):
        self.data.xfrc_applied[self._body[name], :3] += force

    def render(self, camera, width, height):
        r = self._renderers.get((width, height))
        if r is None:
            r = self._renderers[(width, height)] = mujoco.Renderer(self.model, height, width)
        r.update_scene(self.data, camera=camera)
        return r.render()

    def get_state(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        st = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, st, spec)
        return st

    def set_state(self, state):
        mujoco.mj_setState(self.model, self.data, np.asarray(state, float), mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self._renderers.clear()
