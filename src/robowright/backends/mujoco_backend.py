"""MuJoCo backend: the reference simulator.

The robot is its MuJoCo Menagerie model, attached under the ``robot/`` prefix,
so it runs exactly as its authors tuned it: same actuators, tendons and
equality constraints.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..errors import CapabilityError
from ..robots import PREFIX, body_labels
from ..robots.model import add_ground_friction, joint_followers, reset_data
from ..scene import SceneSpec
from .base import CONTACTS, DETERMINISTIC, FORCES, GROUND_TRUTH, RENDER, STATE, Backend, Contact, TargetRamp, register

_Renderer = None


def renderer(model, height: int, width: int):
    """A mujoco.Renderer that is freed in its own GL context.

    From MuJoCo 3.17 a renderer frees its MjrContext before its GL context, and so in whatever context is current: a
    renderer closed, or collected, while another's context was current deleted that one's textures and buffers, and it
    drew wrong frames from then on. Its own context is made current first; where it is still current on another thread
    (the live view's), the GL context goes first instead, as before 3.17, which leaves no context current here.
    (The class is made on first use: without OpenGL, mujoco has no Renderer.)
    """
    global _Renderer
    if _Renderer is None:

        class _Own(mujoco.Renderer):
            def close(self):
                gl = getattr(self, "_gl_context", None)
                if gl is not None and getattr(self, "_mjr_context", None):
                    try:
                        gl.make_current()
                    except Exception:  # noqa: BLE001 - current on another thread
                        self._gl_context = None
                        gl.free()
                super().close()

        _Renderer = _Own
    return _Renderer(model, height, width)


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
    # A directional light: `type` from MuJoCo 3.3.3 on, the `directional` flag before it (gymnasium-robotics pins older).
    kind = {"type": mujoco.mjtLightType.mjLIGHT_DIRECTIONAL} if hasattr(mujoco, "mjtLightType") else {"directional": True}
    wb.add_light(pos=[0.3, -0.3, 1.5], dir=[-0.2, 0.2, -1], castshadow=1, diffuse=[0.5, 0.5, 0.5], **kind)
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
    add_ground_friction(s, spec.robot_model)  # needs the floor's name, so after it
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
            if free < 0:
                raise CapabilityError(f"{rm.title}: a wheeled robot here rolls on a free base; this model's base is fixed")
            self._base_q, self._base_v = m.jnt_qposadr[free], m.jnt_dofadr[free]
        self._wheel_u = np.zeros(self.n_arm)  # the wheels' angular velocity setpoints (rad/s)
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
        # A maker's PD loop (pd_gains) is handed each control step's targets whole, as its own
        # loop holds them; ramping them would not run the published controller.
        self._hold_targets = rm.pd_gains is not None
        reset_data(m, self.data)
        if rm.floating:
            yaw = rm.base_yaw
            self.set_base_pose((*rm.base_pos[:2], rm.stand_height), (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
            self.set_joint_positions(rm.stand_q())
        if rm.family == "mobile":
            self.wheels = self._measure_wheels() if self.n_arm == 2 else self._measure_rollers()
            self.set_ctrl(np.zeros(self.n_arm))
        else:
            self.set_ctrl(self.qpos())

    # robot
    def _measure_wheels(self) -> dict:
        """The differential drive's geometry, read from the model at rest: each wheel's radius (from
        its cylinder), its lateral position on the base, and which way its axis turns it forward.
        Nothing here is typed in; a wheel that does not fit the model is refused, not guessed at."""
        m, d = self.model, self.data
        rm = self.robot_model
        if self.n_arm != 2:
            raise CapabilityError(f"{rm.title}: the wheeled drive is differential (two wheels), and this model has {self.n_arm}")
        base = self._base
        R = d.xmat[base].reshape(3, 3)
        wheels = []
        for name in rm.arm_joints:
            j = m.joint(PREFIX + name)
            body = int(m.jnt_bodyid[j.id])
            c = body  # the wheel hangs off the base, through fixed links (a URDF's base_link under base_footprint)
            while c > 0 and c != base:
                c = int(m.body_parentid[c])
            if c != base:
                raise CapabilityError(f"{rm.title}: wheel {name} is not on the base's body, so the drive cannot be measured")
            axis = d.xmat[body].reshape(3, 3) @ m.jnt_axis[j.id]  # the joint's axis, in the world
            # The collision cylinder is the wheel's (a URDF's visual copy of it is the same shape); cylinders that
            # disagree in radius or axle are not one wheel, and are refused.
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
            collide = [g for g in geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
            geoms = collide or geoms
            if not geoms:
                raise CapabilityError(f"{rm.title}: wheel {name} has no cylinder geom, so its radius cannot be measured")
            g = geoms[0]
            radii = {round(float(m.geom_size[h][0]), 9) for h in geoms}
            if len(radii) != 1:
                raise CapabilityError(f"{rm.title}: wheel {name}'s cylinders disagree in radius ({sorted(radii)}), so it is not one wheel")
            if abs(float(d.geom_xmat[g].reshape(3, 3)[:, 2] @ axis)) < 0.99:
                raise CapabilityError(f"{rm.title}: wheel {name}'s cylinder is not along its axle, so its radius is not measurable")
            lateral = float((R.T @ (d.xpos[body] - d.xpos[base]))[1])
            forward = float((R.T @ axis)[1])  # rolling on the floor, a wheel turns the base's +x way when this is positive
            wheels.append({"joint": PREFIX + name, "radius": float(m.geom_size[g][0]), "y": lateral, "sign": 1.0 if forward > 0 else -1.0})
        left, right = sorted(wheels, key=lambda w: -w["y"])
        if abs(left["radius"] - right["radius"]) > 1e-9 or left["y"] <= 0 or right["y"] >= 0:
            raise CapabilityError(f"{rm.title}: the two wheels must be equal and on either side of the base's centre line")
        return {"radius": left["radius"], "track": left["y"] - right["y"], "wheels": (left, right)}

    def _measure_rollers(self) -> dict:
        """A holonomic base's wheels (omni or mecanum: each wheel carries passive rollers, hinges on its
        own bodies), read from the model at rest. Each wheel is a rolling constraint: the floor under the
        roller that touches it takes no sideways slip along the roller's axis, so the wheel's speed is set by
        that roller's direction, its hub's distance to the ground, and where the wheel sits. Returns the
        matrix ``M`` with wheel speeds (rad/s) = M @ (vx, vy, wz) for the base's velocity in its own frame.
        A wheel whose rollers do not fit this (no roller under it, or its roller axis at right angles to
        its rolling direction) is refused, not guessed at."""
        m, d = self.model, self.data
        rm = self.robot_model
        base = self._base
        R = d.xmat[base].reshape(3, 3)
        z = np.array([0.0, 0.0, 1.0])
        rows, radii = [], []
        for name in rm.arm_joints:
            j = m.joint(PREFIX + name)
            body = int(m.jnt_bodyid[j.id])
            axis = d.xmat[body].reshape(3, 3) @ m.jnt_axis[j.id]  # the wheel's axle, in the world
            hub = d.xanchor[j.id]
            rollers = []
            for k in range(m.njnt):
                rb = int(m.jnt_bodyid[k])
                if k == j.id or m.jnt_type[k] != mujoco.mjtJoint.mjJNT_HINGE or rb == body:
                    continue
                c = rb
                while c > 0 and c != body:
                    c = int(m.body_parentid[c])
                if c == body:
                    rollers.append(k)
            if not rollers:
                raise CapabilityError(f"{rm.title}: wheel {name} has no passive rollers, so it is not an omni or mecanum wheel")
            low = min(rollers, key=lambda k: d.xanchor[k][2])  # the roller on the floor
            roller_axis = d.xmat[m.jnt_bodyid[low]].reshape(3, 3) @ m.jnt_axis[low]
            geoms = [
                g for g in range(m.ngeom) if m.geom_bodyid[g] == m.jnt_bodyid[low] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
            ]
            if not geoms:
                raise CapabilityError(f"{rm.title}: roller {m.joint(low).name} has no cylinder geom, so its radius is unknown")
            roller_r = float(m.geom_size[geoms[0]][0])
            axis_w = axis / np.linalg.norm(axis)
            off = d.xanchor[low] - hub
            rho = float(np.linalg.norm(off - np.dot(off, axis_w) * axis_w))  # the roller's centre from the axle
            radius = rho + roller_r  # the axle to the floor under the roller: the wheel's rolling radius
            n = roller_axis - np.dot(roller_axis, z) * z  # the roller's axis on the floor: its drive direction
            if np.linalg.norm(n) < 1e-6:
                raise CapabilityError(f"{rm.title}: roller {m.joint(low).name} stands on its end, so its drive direction is not measurable")
            n = n / np.linalg.norm(n)
            # No slip along n at the contact p: n.(v + w_base z x p) + radius * w_wheel * ((z x axle).n) = 0, in the base's
            # frame. The contact is under the roller on the floor, not under the axle's anchor: a wheel's rollers can sit
            # outboard of where its joint is placed (1 cm on the Robot Soccer Kit, whose turns then came out 18 % short).
            nb, pb, ab = R.T @ n, R.T @ (d.xanchor[low] - d.xpos[base]), R.T @ axis_w
            e = float(np.cross(z, ab) @ nb)  # how the wheel's spin turns its contact along n
            if abs(e) < 0.5:
                raise CapabilityError(f"{rm.title}: wheel {name}'s rollers do not drive it (its axle is along its drive direction)")
            coeff = -1.0 / (radius * e)
            rows.append(coeff * np.array([nb[0], nb[1], -nb[0] * pb[1] + nb[1] * pb[0]]))
            radii.append(radius)
        return {"radius": float(np.mean(radii)), "matrix": np.array(rows), "wheels": tuple(radii)}

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
        if rm.family == "mobile":
            # The wheels' velocity setpoints: what drive_base computes, or what a trace recorded.
            lo, hi = m.actuator_ctrlrange[self._act].T
            self._wheel_u = np.clip(target[: self.n_arm], lo, hi)
            self.data.ctrl[self._act] = self._wheel_u
            return
        arm = target[: self.n_arm]
        lim = m.actuator_ctrllimited[self._act].astype(bool)
        lo, hi = m.actuator_ctrlrange[self._act].T
        u = np.where(lim, np.clip(arm, lo, hi), arm)
        if self.has_gripper:
            g = float(np.clip(target[self.n_arm], 0.0, 1.0))
            u = np.append(u, rm.gripper_closed + g * (rm.gripper_open - rm.gripper_closed))
        if self._hold_targets:
            self._ramp.reset()
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

    def base_estimate(self):
        # A simulator has no localizer: the base's estimate of itself is where it is.
        return self.base_pose()

    def wheel_command(self):
        return self._wheel_u.copy()

    def drive_base(self, linear: float, angular: float, lateral: float = 0.0) -> None:
        """Wheel speeds for a base moving at ``linear`` m/s forward, ``lateral`` m/s sideways and turning
        at ``angular`` rad/s (about +z, counter-clockwise). A differential base has two wheels and no
        sideways motion; an omni or mecanum base takes its wheel speeds from its measured rollers."""
        from .wheels import speeds

        self.set_ctrl(speeds(self.wheels, self.robot_model.title, linear, angular, lateral))

    def stop_base(self) -> None:
        self.set_ctrl(np.zeros(self.n_arm))

    def current_base_quat(self):
        # xquat is computed before the last step integrates, so base_pose lags the joints by a step.
        return self.data.qpos[self._base_q + 3 : self._base_q + 7].copy()

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
            r = self._renderers[(width, height)] = renderer(self.model, height, width)
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
