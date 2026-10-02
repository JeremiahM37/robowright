"""MuJoCo backend: the reference simulator."""

from __future__ import annotations

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .. import assets  # noqa: E402
from ..scene import SceneSpec  # noqa: E402
from .base import CONTACTS, DETERMINISTIC, FORCES, GROUND_TRUTH, RENDER, STATE, Backend, Contact, register  # noqa: E402

ROBOT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
_PART = {"gripper": "fixed_jaw", "camera_mount": "fixed_jaw", "moving_jaw_so101_v1": "moving_jaw"}


def _lookat_xyaxes(pos, lookat):
    pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
    fwd = lookat - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.concatenate([right, up])


def build_spec(spec: SceneSpec) -> mujoco.MjSpec:
    s = mujoco.MjSpec.from_file(str(assets.so101_mjcf()))
    s.option.timestep = spec.physics_dt
    s.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    s.visual.headlight.ambient = [0.35, 0.35, 0.35]
    s.visual.global_.offwidth = 1280
    s.visual.global_.offheight = 960
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
            sx, sy, sz = o.size
            t = 0.003
            walls = [
                ((0, 0, t), (sx, sy, t)),
                ((sx - t, 0, sz), (t, sy, sz)),
                ((-sx + t, 0, sz), (t, sy, sz)),
                ((0, sy - t, sz), (sx, t, sz)),
                ((0, -sy + t, sz), (sx, t, sz)),
            ]
            for i, (p, hs) in enumerate(walls):
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


@register("mujoco")
class MujocoBackend(Backend):
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, RENDER, STATE, DETERMINISTIC, FORCES})

    def __init__(self, spec: SceneSpec, seed: int = 0):
        super().__init__(spec, seed)
        self.mjspec = build_spec(spec)
        self.model = self.mjspec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joint_names = list(ROBOT_JOINTS)
        self._qadr = np.array([m.joint(n).qposadr[0] for n in ROBOT_JOINTS])
        self._dadr = np.array([m.joint(n).dofadr[0] for n in ROBOT_JOINTS])
        self._act = np.array([m.actuator(n).id for n in ROBOT_JOINTS])
        self._kp = m.actuator_gainprm[self._act, 0].copy()
        self._substeps = max(1, round((1.0 / spec.control_hz) / m.opt.timestep))
        self._body = {o.name: m.body(o.name).id for o in spec.objects}
        self._free = {o.name: m.joint(f"{o.name}/free") for o in spec.objects if not o.static}
        robot_root = m.body("base").id
        self._label = {}
        for b in range(m.nbody):
            name = m.body(b).name
            if name in self._body:
                self._label[b] = name
            elif b == 0:
                self._label[b] = "floor"
            else:
                r = b
                while r not in (0, robot_root):
                    r = m.body_parentid[r]
                self._label[b] = f"robot:{_PART.get(name, name)}" if r == robot_root else name
        self._renderers: dict = {}
        mujoco.mj_resetData(m, self.data)
        mujoco.mj_forward(m, self.data)

    # robot
    def qpos(self):
        return self.data.qpos[self._qadr].copy()

    def qvel(self):
        return self.data.qvel[self._dadr].copy()

    def set_ctrl(self, target):
        lo, hi = self.model.actuator_ctrlrange[self._act].T
        self.data.ctrl[self._act] = np.clip(target, lo, hi)

    def ctrl(self):
        return self.data.ctrl[self._act].copy()

    def set_gain_scale(self, joint, scale):
        a = self._act[self.joint_names.index(joint)]
        kp = self._kp[self.joint_names.index(joint)] * scale
        self.model.actuator_gainprm[a, 0] = kp
        self.model.actuator_biasprm[a, 1] = -kp

    def set_joint_positions(self, q):
        self.data.qpos[self._qadr] = q
        self.data.qvel[self._dadr] = 0
        self.set_ctrl(q)
        mujoco.mj_forward(self.model, self.data)

    # time
    def step(self):
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:] = 0

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
