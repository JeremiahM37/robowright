"""MuJoCo backend: the reference simulator.

The robot is its MuJoCo Menagerie model, attached under the ``robot/`` prefix,
so it runs exactly as its authors tuned it: same actuators, tendons and
equality constraints.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..robots import PREFIX, body_labels
from ..robots.model import joint_followers, reset_data
from ..scene import SceneSpec
from .base import CONTACTS, DETERMINISTIC, FORCES, GROUND_TRUTH, RENDER, STATE, Backend, Contact, TargetRamp, register


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
    # One lighting rig for every robot: Menagerie models bring their own lights, which on top of
    # the scene's saturate the floor and cast a second set of shadows.
    for light in list(s.lights):
        s.delete(light)
    if spec.robot_model.family == "arm":
        # Arm controllers on real robots cancel gravity; bare position servos sag
        # under it instead, by more than a grasp can tolerate.
        for b in s.bodies:
            if b.name.startswith(PREFIX):
                b.gravcomp = 1.0
    # Headlight, ambient and the sun stay below 1 together, so a floor seen from above does not white out.
    s.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    s.visual.headlight.ambient = [0.3, 0.3, 0.3]
    s.visual.global_.offwidth = 1280
    s.visual.global_.offheight = 960
    s.stat.extent = 1.0
    # The sun's shadow map has to cover the floor in view, or beyond its edge the floor speckles.
    s.visual.map.shadowclip = 4.0
    s.visual.quality.shadowsize = 8192
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
    mat = s.add_material(name="table", texrepeat=[8, 8], texuniform=True, specular=0.05, shininess=0.1)
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
    reusable = True

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
        for actuator, leader, _ in rm.arm_couplings:  # a tendon actuator commanded in its leader joint's position
            by_joint[m.joint(PREFIX + leader).id] = m.actuator(PREFIX + actuator).id
        self._act = np.array([by_joint[m.joint(n).id] for n in arm])
        self._followers = joint_followers(m, [m.joint(n).id for n in arm])
        self._kp = m.actuator_gainprm[self._act, 0].copy()
        self._bias = m.actuator_biasprm[self._act, 1].copy()
        # A position servo following a moving target trails it by (kv + joint damping) / kp
        # seconds of its motion. On robowright's own smooth moves (``feedforward``, set while the
        # robot streams one) the targets lead it by as much, as a trajectory controller does:
        # Menagerie's Rizon4 (0.2 s) otherwise ran 0.47 rad behind a planned path and swept a
        # can it had been planned around. A policy's own targets are not led: chunks that start
        # and stop at once overshot with it, and the ARX L5 and iiwa dropped cubes.
        self.feedforward = False
        joint_servo = (m.actuator_trntype[self._act] == mujoco.mjtTrn.mjTRN_JOINT) & (
            m.actuator_biastype[self._act] == mujoco.mjtBias.mjBIAS_AFFINE
        )
        kv = np.maximum(-m.actuator_biasprm[self._act, 2], 0.0) + m.dof_damping[self._dadr]
        self._led = np.zeros(len(self._act))  # the lead last applied
        self._lead = np.where(joint_servo & (self._kp > 0) & (self._bias < 0), kv / np.maximum(self._kp, 1e-9), 0.0)
        if self.has_gripper:
            self._gact = m.actuator(PREFIX + rm.gripper_actuator).id
            der = rm.derived
            gj = m.joint(PREFIX + der.gripper_joint)
            self._g_qadr, self._g_dadr = gj.qposadr[0], gj.dofadr[0]
            self._g_closed, self._g_open = der.gripper_joints[der.gripper_joint]
        if rm.floating:
            self._base = m.body(PREFIX + rm.base_body).id
            free = m.body_jntadr[self._base]
            self._base_q, self._base_v = m.jnt_qposadr[free], m.jnt_dofadr[free]
        self._substeps = max(1, round((1.0 / spec.control_hz) / m.opt.timestep))
        self._body = {o.name: m.body(o.name).id for o in spec.objects}
        self._free = {o.name: m.joint(f"{o.name}/free") for o in spec.objects if not o.static}
        self._label = {0: "floor"}
        for b, part in body_labels(m, rm, PREFIX).items():
            self._label[b] = f"robot:{part}"
        for b in range(1, m.nbody):
            self._label.setdefault(b, m.body(b).name)
        self._hand = m.body(PREFIX + rm.hand).id if rm.hand else None
        self._renderers: dict = {}
        self._acts = np.append(self._act, self._gact) if self.has_gripper else self._act
        self._ramp = TargetRamp()
        reset_data(m, self.data)
        if rm.floating:
            yaw = rm.base_yaw
            self.set_base_pose((*rm.base_pos[:2], rm.stand_height), (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
            self.set_joint_positions(rm.stand_q())
        self.set_ctrl(self.qpos())

    # robot
    def _opening(self, q):
        return (q - self._g_closed) / (self._g_open - self._g_closed)

    def qpos(self):
        q = self.data.qpos[self._qadr]
        return np.append(q, self._opening(self.data.qpos[self._g_qadr])) if self.has_gripper else q.copy()

    def qvel(self):
        v = self.data.qvel[self._dadr]
        return np.append(v, self.data.qvel[self._g_dadr] / (self._g_open - self._g_closed)) if self.has_gripper else v.copy()

    def set_ctrl(self, target):
        m, rm = self.model, self.robot_model
        target = np.asarray(target, float)
        arm = target[: self.n_arm]
        lim = m.actuator_ctrllimited[self._act].astype(bool)
        lo, hi = m.actuator_ctrlrange[self._act].T
        u = np.where(lim, np.clip(arm, lo, hi), arm)
        if self.has_gripper:
            g = float(np.clip(target[self.n_arm], 0.0, 1.0))
            u = np.append(u, rm.gripper_closed + g * (rm.gripper_open - rm.gripper_closed))
        self._ramp.set(u)
        self.data.ctrl[self._acts] = u  # ramped from the last step's in step()

    def ctrl(self):
        rm = self.robot_model
        if not self.has_gripper:
            return self.data.ctrl[self._act].copy()
        g = (self.data.ctrl[self._gact] - rm.gripper_closed) / (rm.gripper_open - rm.gripper_closed)
        return np.append(self.data.ctrl[self._act], g)

    def set_gain_scale(self, joint, scale):
        i = self.joint_names.index(joint)
        a = self._act[i]
        self.model.actuator_gainprm[a, 0] = self._kp[i] * scale
        self.model.actuator_biasprm[a, 1] = self._bias[i] * scale

    def set_joint_positions(self, q):
        q = np.asarray(q, float)
        self._ramp.reset()  # placed, not moved: no ramp
        self._led = np.zeros(len(self._act))
        self.data.qpos[self._qadr] = q[: self.n_arm]
        self.data.qvel[self._dadr] = 0
        if self.has_gripper:
            # Put every finger joint where it sits at this opening, so the gripper starts at rest.
            s = float(np.clip(q[self.n_arm], 0, 1))
            for name, (c, o) in self.robot_model.derived.gripper_joints.items():
                j = self.model.joint(PREFIX + name)
                self.data.qpos[j.qposadr[0]] = c + s * (o - c)
                self.data.qvel[j.dofadr[0]] = 0
        for qa, da, i, offset, ratio in self._followers:  # joints moved with an arm joint (a telescope)
            self.data.qpos[qa] = offset + ratio * q[i]
            self.data.qvel[da] = 0
        self.set_ctrl(q)
        mujoco.mj_forward(self.model, self.data)

    def base_pose(self):
        return self.data.xpos[self._base].copy(), self.data.xquat[self._base].copy()

    def base_velocity(self):
        v = self.data.qvel[self._base_v : self._base_v + 6].copy()
        R = self.data.xmat[self._base].reshape(3, 3)
        return np.concatenate([v[:3], R @ v[3:]])  # MuJoCo's free-joint angular velocity is in the body frame

    def set_base_pose(self, pos, quat):
        a = self._base_q
        self.data.qpos[a : a + 3] = pos
        self.data.qpos[a + 3 : a + 7] = quat
        self.data.qvel[self._base_v : self._base_v + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def hand_pose(self):
        return self.data.xpos[self._hand].copy(), self.data.xquat[self._hand].copy()

    # time
    def step(self):
        m, d, ramp, n = self.model, self.data, self._ramp, self._substeps
        lead = self._lead * ramp.velocity(self.control_dt)[: self.n_arm] if ramp.moving and self.feedforward else np.zeros(self.n_arm)
        if ramp.moving or self._led.any():
            # The lead is ramped too: changing in a step at each control step, it shook the Z1's
            # light wrist into an oscillation at its force limit that never died out.
            lim = m.actuator_ctrllimited[self._act].astype(bool)
            lo, hi = m.actuator_ctrlrange[self._act].T
            for k in range(n):
                u = ramp.at((k + 1) / n) if ramp.moving else ramp.end.copy()
                arm = u[: self.n_arm] + self._led + (k + 1) / n * (lead - self._led)
                u[: self.n_arm] = np.where(lim, np.clip(arm, lo, hi), arm)
                d.ctrl[self._acts] = u
                mujoco.mj_step(m, d)
            d.ctrl[self._acts] = ramp.end  # what was commanded, as ctrl() and a trace read it
            ramp.arrive()
            self._led = lead
        else:
            for _ in range(n):
                mujoco.mj_step(m, d)
        d.xfrc_applied[:] = 0

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
        b = self._base if name == "robot" else self._body[name]
        self.data.xfrc_applied[b, :3] += force

    def open_viewer(self):
        import os
        import sys

        import mujoco.viewer

        from ..errors import CapabilityError

        # MuJoCo's viewer ends the process when GLFW cannot start: check first, and say why.
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise CapabilityError("headed runs need a display (no DISPLAY or WAYLAND_DISPLAY): run without --rw-headed and open the trace")
        import glfw

        if not glfw.init():
            raise CapabilityError("headed runs need a display GLFW can open: run without --rw-headed and open the trace")
        try:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            with self._viewer.lock():  # framed on the task area and the arm, free to orbit
                cam = self._viewer.cam
                cam.lookat[:] = (0.15, 0.03, 0.08)
                cam.distance, cam.azimuth, cam.elevation = 0.9, 150.0, -25.0
        except Exception as e:  # say what to do instead of a GLFW traceback
            raise CapabilityError(f"cannot open MuJoCo's viewer ({type(e).__name__}: {e}): headed runs need a display") from e

    def sync_viewer(self):
        v = getattr(self, "_viewer", None)
        if v is not None and v.is_running():
            v.sync()

    def close_viewer(self):
        v = getattr(self, "_viewer", None)
        if v is not None:
            v.close()
            self._viewer = None

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
        return np.concatenate([st, self._led])  # and the servos' lead, which eases out over the next step

    def set_state(self, state):
        state = np.asarray(state, float)
        k = len(self._led)
        mujoco.mj_setState(self.model, self.data, state[:-k], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)
        self._led = state[-k:].copy()
        # States are taken between control steps, when the ramp has arrived: none is pending.
        self._ramp.reset()
        self._ramp.set(self.data.ctrl[self._acts])

    def close(self):
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self._renderers.clear()
