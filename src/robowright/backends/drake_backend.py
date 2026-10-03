"""Drake backend: Toyota Research Institute's multibody simulator.

The robot is robowright's URDF export of its MuJoCo model (see
:mod:`robowright.robots.urdf`), so the kinematics, inertias and collision
shapes are MuJoCo's. Drake runs it as a discrete ``MultibodyPlant`` with the
SAP contact solver, which treats joint PD servos and coupler constraints
implicitly: the stiff servos of an industrial arm stay stable at the model's
own time step.

What URDF loses is rebuilt from the export's sidecar: servo gains, force
limits and armature become PD-controlled ``JointActuator``\\ s, and the
gripper's tendons and linkages become coupler constraints from each passive
finger joint to the joint the gripper motor drives.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pydrake.geometry import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    AddRigidHydroelasticProperties,
    Box,
    ClippingRange,
    CollisionFilterDeclaration,
    ColorRenderCamera,
    Cylinder,
    GeometrySet,
    HalfSpace,
    MakeRenderEngineVtk,
    ProximityProperties,
    RenderCameraCore,
    RenderEngineVtkParams,
    Sphere,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    ContactModel,
    CoulombFriction,
    DiscreteContactApproximation,
    ExternallyAppliedSpatialForce,
)
from pydrake.multibody.tree import PdControllerGains, SpatialInertia
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import CameraInfo

from ..robots import urdf
from ..scene import SceneSpec
from .base import CONTACTS, DETERMINISTIC, FORCES, GROUND_TRUTH, RENDER, STATE, Backend, Contact, register
from .mujoco_backend import bin_walls

_DRAKE_NS = "http://drake.mit.edu"
_DUMMY = re.compile(r"__j\d+$")  # intermediate links the exporter adds for multi-joint bodies


def _pose(pos, quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> RigidTransform:
    from pydrake.common.eigen_geometry import Quaternion

    q = np.asarray(quat_wxyz, float)
    return RigidTransform(Quaternion(q / np.linalg.norm(q)), np.asarray(pos, float))


def _proximity(friction: float, hydro: str | None = None, size: float = 0.01) -> ProximityProperties:
    """Contact properties: objects and bin walls soft hydroelastic, the floor rigid.

    Point contact between two boxes is a single deepest point, so a cube on a
    bin floor rocks from corner to corner and never comes to rest. A
    hydroelastic patch supports it like a real face. Bin walls are soft too:
    a rigid box against a soft one only yields patches on the box's
    tessellation, and a cube resting across that mesh kept creeping.
    """
    props = ProximityProperties()
    AddContactMaterial(
        dissipation=50.0 if hydro == "soft" else None,  # s/m; less lets a placed cube rock forever
        point_stiffness=POINT_STIFFNESS,
        friction=CoulombFriction(friction, friction),
        properties=props,
    )
    if hydro == "soft":
        AddCompliantHydroelasticProperties(size / 2, HYDRO_MODULUS, props)
    elif hydro == "halfspace":
        AddRigidHydroelasticProperties(props)
    return props


# N/m per geometry (two in contact act in series). Drake's default is derived from a
# 1 mm penetration allowance under the *heaviest* body's weight, which for a small
# cube between two fingers is soft enough that the pads squeeze a centimetre into it.
POINT_STIFFNESS = 2e5
HYDRO_MODULUS = 1e7  # Pa: a hard plastic part; sinks well under a millimetre under its own weight
# SAP softens any contact stiffer than a light body can follow in one step (its
# "near-rigid" regime, threshold 1 by default). For a 30 g cube in a 2 ms step that
# lets the pads sink centimetres into it; a lower threshold keeps grasps rigid.
SAP_NEAR_RIGID = 0.1


def _render_ok() -> bool:
    """Whether VTK can open an offscreen GL context here (EGL on headless machines)."""
    global _RENDER_OK
    if _RENDER_OK is None:
        try:
            from pydrake.geometry import SceneGraph

            sg = SceneGraph()
            sg.AddRenderer("probe", MakeRenderEngineVtk(RenderEngineVtkParams()))
            ctx = sg.CreateDefaultContext()
            core = RenderCameraCore("probe", CameraInfo(8, 8, 1.0), ClippingRange(0.01, 10.0), RigidTransform())
            sg.get_query_output_port().Eval(ctx).RenderColorImage(ColorRenderCamera(core), sg.world_frame_id(), RigidTransform())
            _RENDER_OK = True
        except Exception:
            _RENDER_OK = False
    return _RENDER_OK


_RENDER_OK: bool | None = None


@register("drake")
class DrakeBackend(Backend):
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, STATE, DETERMINISTIC, FORCES})

    def __init__(self, spec: SceneSpec, seed: int = 0):
        super().__init__(spec, seed)
        rm = self.robot_model
        path, meta = urdf.load(rm)
        path = _drake_urdf(path)
        self.meta = meta
        self._render = _render_ok()
        if self._render:
            self.capabilities = self.capabilities | {RENDER}
        builder = DiagramBuilder()
        plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=spec.dt)
        self.plant, self.sg = plant, sg
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
        plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
        plant.set_sap_near_rigid_threshold(SAP_NEAR_RIGID)
        if self._render:
            sg.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams(default_clear_color=[0.85, 0.89, 0.94])))
        parser = Parser(plant)
        parser.SetAutoRenaming(True)
        self.robot = parser.AddModels(str(path))[0]
        X_base = _pose(rm.base_pos, (np.cos(rm.base_yaw / 2), 0, 0, np.sin(rm.base_yaw / 2)))
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(meta["root"], self.robot), X_base)
        self._build_world(spec)
        self._build_actuators(meta)
        plant.Finalize()
        self._filter_collisions(meta)
        self._set_contact_materials(meta)
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.pc = plant.GetMyMutableContextFromRoot(self.context)
        self.sgc = sg.GetMyContextFromRoot(self.context)
        self._index(meta)
        self._substeps = max(1, round((1.0 / spec.control_hz) / spec.dt))
        self.simulator = Simulator(self.diagram, self.context)
        self._k = 0
        self._pending: dict[str, np.ndarray] = {}
        plant.get_applied_spatial_force_input_port().FixValue(self.pc, [])
        self._tau = plant.get_applied_generalized_force_input_port().FixValue(self.pc, np.zeros(plant.num_velocities()))
        self._xd = plant.get_desired_state_input_port(self.robot).FixValue(self.pc, np.zeros(2 * self._nact))
        self._gscale = np.ones(self.n_arm)
        # MuJoCo's reset: every joint at zero, moved inside its limits.
        q = plant.GetPositions(self.pc, self.robot)
        plant.SetPositions(self.pc, self.robot, np.clip(q, self._qlo, self._qhi))
        self._ctrl = np.zeros(self.n_arm + 1)
        self.set_ctrl(self.qpos())

    # --- construction ---------------------------------------------------------
    def _build_world(self, spec: SceneSpec) -> None:
        plant = self.plant
        world = plant.world_body()
        plant.RegisterCollisionGeometry(world, RigidTransform(), HalfSpace(), "floor", _proximity(1.0, "halfspace"))
        plant.RegisterVisualGeometry(world, RigidTransform([0, 0, -0.005]), Box(8, 8, 0.01), "floor_visual", [0.8, 0.8, 0.78, 1])
        self._bodies, self._free = {}, {}
        for o in spec.objects:
            rgba = list(o.rgba)
            if o.kind == "bin":
                body = plant.AddRigidBody(
                    o.name, self.robot_model_instance_for(o.name), SpatialInertia.SolidBoxWithMass(1.0, 0.1, 0.1, 0.1)
                )
                plant.WeldFrames(plant.world_frame(), body.body_frame(), _pose(o.initial_pos, o.quat))
                for i, (p, hs) in enumerate(bin_walls(o.size)):
                    shape, X = Box(*(2 * np.asarray(hs))), RigidTransform(np.asarray(p, float))
                    plant.RegisterCollisionGeometry(body, X, shape, f"{o.name}/wall{i}", _proximity(o.friction, "soft", 0.02))
                    plant.RegisterVisualGeometry(body, X, shape, f"{o.name}/wall{i}/visual", rgba)
                self._bodies[o.name] = body
                continue
            if o.kind == "box":
                shape = Box(*(2 * np.asarray(o.size, float)))
                inertia = SpatialInertia.SolidBoxWithMass(o.mass, *(2 * np.asarray(o.size, float)))
            elif o.kind == "cylinder":
                shape = Cylinder(o.size[0], 2 * o.size[1])
                inertia = SpatialInertia.SolidCylinderWithMass(o.mass, o.size[0], 2 * o.size[1], [0, 0, 1])
            elif o.kind == "sphere":
                shape = Sphere(o.size[0])
                inertia = SpatialInertia.SolidSphereWithMass(o.mass, o.size[0])
            else:
                raise ValueError(f"unknown object kind {o.kind!r}")
            body = plant.AddRigidBody(o.name, self.robot_model_instance_for(o.name), inertia)
            plant.RegisterCollisionGeometry(body, RigidTransform(), shape, o.name, _proximity(o.friction, "soft", min(o.size)))
            plant.RegisterVisualGeometry(body, RigidTransform(), shape, f"{o.name}/visual", rgba)
            plant.SetDefaultFloatingBaseBodyPose(body, _pose(o.initial_pos, o.quat))
            self._bodies[o.name] = self._free[o.name] = body

    def robot_model_instance_for(self, name: str):
        """Each object gets its own model instance, so its name never clashes with a robot link."""
        return self.plant.AddModelInstance(f"object:{name}")

    def _build_actuators(self, meta: dict) -> None:
        """PD servos on the arm and the gripper's driven joints; coupler constraints for the rest of the fingers.

        MuJoCo couples a gripper's joints with tendons and equality constraints,
        which URDF drops. Driving every finger joint to a fixed angle instead
        breaks four-bar grippers: when the pads stop on an object the passive
        links keep turning and tilt the pads off it. Coupling each passive
        joint to its side's driven joint keeps the linkage moving as one.
        """
        plant, rm = self.plant, self.robot_model
        joints, grip = meta["joints"], meta["gripper"]
        js = lambda n: plant.GetJointByName(urdf._safe(n), self.robot)  # noqa: E731
        self._arm_joints = [js(n) for n in rm.arm_joints]
        driven = list(grip["driven"]) or [grip["main"]]
        self._driven = driven
        gains: dict[str, tuple[float, float]] = {}
        for n in rm.arm_joints:
            j = joints[n]
            gains[n] = (j.get("kp", 100.0), j.get("kv", 0.0))
        for n in driven:
            j = joints[n]
            c, o = grip["joints"][n]
            effort = grip["effort"][n]
            if "kp" in j:
                kp = j["kp"]
            else:
                # Saturate at the squeeze force within a fifth of the travel: a firm
                # grip on anything narrower than the open jaws, like MuJoCo's.
                kp = effort / (0.2 * abs(o - c))
            kv = j.get("kv", 0.0) or 2.0 * np.sqrt(kp * max(j["armature"], 1e-3))
            gains[n] = (kp, kv)
        self._kp, self._kv = {}, {}
        for n, (kp, kv) in gains.items():
            j = joints[n]
            effort = grip["effort"][n] if n in grip["joints"] else j["effort"]
            a = plant.AddJointActuator(f"act:{urdf._safe(n)}", js(n), effort)
            a.set_default_rotor_inertia(j["armature"])
            a.set_default_gear_ratio(1.0)
            a.set_controller_gains(PdControllerGains(p=kp, d=kv))
            self._kp[n], self._kv[n] = kp, kv
        self._act_names = list(gains)
        self._nact = len(gains)

        def side(n):
            return max(driven, key=lambda d: (len(_common_prefix(n, d)), -driven.index(d)))

        self._follow = {}
        for n, (c_f, o_f) in grip["joints"].items():
            if n in driven:
                continue
            d = side(n)
            c_d, o_d = grip["joints"][d]
            rho = (o_f - c_f) / (o_d - c_d)
            plant.AddCouplerConstraint(js(n), js(d), rho, c_f - rho * c_d)
            self._follow[n] = d

    def _filter_collisions(self, meta: dict) -> None:
        """Skip the contacts MuJoCo skips: the model's excluded pairs, and parent-child bodies.

        MuJoCo's parent filter works on welded groups: a body welded under a
        finger (a pad, say) is still the finger's part and never collides with
        the finger's parent. Drake only filters bodies a joint connects
        directly, and the exporter's extra links for multi-joint bodies hide
        even those, so the groups are rebuilt here.
        """
        plant = self.plant

        def body(name):
            return plant.world_body() if name == "world" else plant.GetBodyByName(name, self.robot)

        def geoms(bodies):
            return GeometrySet([g for b in bodies for g in plant.GetCollisionGeometriesForBody(b)])

        decl = CollisionFilterDeclaration()
        for a, b in meta["excluded_pairs"]:
            decl.ExcludeBetween(geoms([body(a)]), geoms([body(b)]))
        # Group robot bodies by the body they are welded under (never across the weld to the world:
        # like MuJoCo, the world is nobody's parent for filtering).
        into = self._joint_into()

        def weld_root(b):
            while b.index() in into and into[b.index()].type_name() == "weld" and into[b.index()].parent_body().index() in into:
                b = into[b.index()].parent_body()
            return b

        groups: dict = {}
        for bi in plant.GetBodyIndices(self.robot):
            groups.setdefault(weld_root(plant.get_body(bi)).index(), []).append(plant.get_body(bi))
        for root, members in groups.items():
            parent = self._real_parent(plant.get_body(root))
            if parent is not None and parent.index() in into:
                decl.ExcludeBetween(geoms(members), geoms(groups[weld_root(parent).index()]))
        self.sg.collision_filter_manager().Apply(decl)

    def _joint_into(self) -> dict:
        """Body index -> the robot joint whose child it is."""
        if not hasattr(self, "_into"):
            self._into = {}
            for ji in self.plant.GetJointIndices(self.robot):
                j = self.plant.get_joint(ji)
                self._into[j.child_body().index()] = j
        return self._into

    def _parent_body(self, body):
        j = self._joint_into().get(body.index())
        return None if j is None else j.parent_body()

    def _real_parent(self, body):
        p = self._parent_body(body)
        while p is not None and _DUMMY.search(p.name()):
            p = self._parent_body(p)
        return p

    def _set_contact_materials(self, meta: dict) -> None:
        """The robot's per-geom friction from MuJoCo, and the common point-contact stiffness."""
        plant, sg = self.plant, self.sg
        inspector = sg.model_inspector()
        for bi in plant.GetBodyIndices(self.robot):
            for g in plant.GetCollisionGeometriesForBody(plant.get_body(bi)):
                name = inspector.GetName(g).split("::")[-1]
                mu = meta["geoms"].get(name, {}).get("friction", 1.0)
                props = ProximityProperties(inspector.GetProximityProperties(g))
                props.UpdateProperty("material", "coulomb_friction", CoulombFriction(mu, mu))
                props.UpdateProperty("material", "point_contact_stiffness", POINT_STIFFNESS)
                # Rigid hydroelastic against the soft objects: a grasp becomes two pressure
                # patches instead of one point per pad, which jumps between the pad's
                # faces and edges as it squeezes and makes the grip chatter.
                AddRigidHydroelasticProperties(0.005, props)
                sg.AssignRole(plant.get_source_id(), g, props, _replace_role())

    def _index(self, meta: dict) -> None:
        plant, rm = self.plant, self.robot_model
        robot_q0 = plant.GetPositions(self.pc, self.robot)
        nq = len(robot_q0)
        lo = plant.GetPositionLowerLimits()
        hi = plant.GetPositionUpperLimits()
        start = plant.GetJointByName(urdf._safe(rm.arm_joints[0]), self.robot).position_start()
        del start
        # robot positions are contiguous in the instance's own ordering; map through a selector
        sel_q = np.array(
            [plant.get_joint(j).position_start() for j in plant.GetJointIndices(self.robot) if plant.get_joint(j).num_positions()]
        )
        order = np.argsort(sel_q)
        sel_q = sel_q[order]
        assert len(sel_q) == nq
        self._qlo, self._qhi = lo[sel_q], hi[sel_q]
        self._qa = np.array([j.position_start() for j in self._arm_joints])
        self._va = np.array([j.velocity_start() for j in self._arm_joints])
        js = lambda n: plant.GetJointByName(urdf._safe(n), self.robot)  # noqa: E731
        grip = meta["gripper"]
        self._finger_q = {n: js(n).position_start() for n in grip["joints"]}
        self._finger_v = {n: js(n).velocity_start() for n in grip["joints"]}
        self._g_main = grip["main"]
        self._g_closed, self._g_open = grip["joints"][grip["main"]]
        self._arm_lo = np.array([lo[i] for i in self._qa])
        self._arm_hi = np.array([hi[i] for i in self._qa])
        # Desired-state port: actuated joints of the robot instance, in actuator order.
        acts = [plant.get_joint_actuator(a) for a in plant.GetJointActuatorIndices(self.robot)]
        self._act_order = [a.joint().name() for a in acts]
        self._act_index = {urdf._safe(n): self._act_order.index(urdf._safe(n)) for n in self._act_names}
        self._acts = {n: a for n, a in zip(self._act_order, acts)}
        self._robot_v = np.zeros(plant.num_velocities(), bool)
        for ji in plant.GetJointIndices(self.robot):
            jt = plant.get_joint(ji)
            self._robot_v[jt.velocity_start() : jt.velocity_start() + jt.num_velocities()] = True
        # contact labels
        inv = {v: k for k, v in meta["links"].items()}
        fingers = {link: part for part, links in meta["fingers"].items() for link in links}
        self._label = {plant.world_body().index(): "floor"}
        for bi in plant.GetBodyIndices(self.robot):
            body = plant.get_body(bi)
            b, part = body, None
            while b is not None and part is None:
                part = fingers.get(b.name())
                b = self._parent_body(b)
            self._label[bi] = f"robot:{part or inv.get(body.name(), body.name())}"
        for name, body in self._bodies.items():
            self._label[body.index()] = name
        self._hand = plant.GetBodyByName(meta["hand"], self.robot)
        self._hand_offset = _hand_anchor(rm)
        self._cams = {c.name: c for c in self.spec.cameras}

    # --- robot ------------------------------------------------------------------
    def _opening(self, q):
        return (q - self._g_closed) / (self._g_open - self._g_closed)

    def qpos(self):
        q = self.plant.GetPositions(self.pc)
        return np.append(q[self._qa], self._opening(q[self._finger_q[self._g_main]]))

    def qvel(self):
        v = self.plant.GetVelocities(self.pc)
        return np.append(v[self._va], v[self._finger_v[self._g_main]] / (self._g_open - self._g_closed))

    def set_ctrl(self, target):
        target = np.asarray(target, float)
        arm = np.clip(target[: self.n_arm], self._arm_lo, self._arm_hi)
        g = float(np.clip(target[self.n_arm], 0.0, 1.0))
        self._ctrl = np.append(arm, g)
        xd = np.zeros(2 * self._nact)
        for n, v in zip(self.robot_model.arm_joints, arm):
            xd[self._act_index[urdf._safe(n)]] = v
        for n in self._driven:
            c, o = self.meta["gripper"]["joints"][n]
            xd[self._act_index[urdf._safe(n)]] = c + g * (o - c)
        self._xd.GetMutableData().set_value(xd)

    def ctrl(self):
        return self._ctrl.copy()

    def set_gain_scale(self, joint, scale):
        i = self.joint_names.index(joint)
        n = self.robot_model.arm_joints[i]
        self._gscale[i] = scale
        # PD gains are model data in Drake, read when the step is computed.
        self._acts[urdf._safe(n)].set_controller_gains(PdControllerGains(p=self._kp[n] * scale, d=self._kv[n] * scale))

    def set_joint_positions(self, q):
        q = np.asarray(q, float)
        plant = self.plant
        qq = plant.GetPositions(self.pc)
        vv = plant.GetVelocities(self.pc)
        qq[self._qa] = q[: self.n_arm]
        vv[self._va] = 0
        s = float(np.clip(q[self.n_arm], 0, 1))
        for n, (c, o) in self.meta["gripper"]["joints"].items():
            qq[self._finger_q[n]] = c + s * (o - c)
            vv[self._finger_v[n]] = 0
        plant.SetPositions(self.pc, qq)
        plant.SetVelocities(self.pc, vv)
        self.set_ctrl(q)

    def hand_pose(self):
        X = self.plant.EvalBodyPoseInWorld(self.pc, self._hand)
        R = X.rotation().matrix()
        # The URDF link frame sits at the body's joint anchor; report MuJoCo's body frame.
        return X.translation() - R @ self._hand_offset, X.rotation().ToQuaternion().wxyz()

    # --- time -------------------------------------------------------------------
    def step(self):
        plant, pc = self.plant, self.pc
        forces = []
        for name, f in self._pending.items():
            body = self._bodies[name]
            F = ExternallyAppliedSpatialForce()
            F.body_index = body.index()
            F.p_BoBq_B = np.zeros(3)
            F.F_Bq_W = SpatialForce(np.zeros(3), f)
            forces.append(F)
        plant.get_applied_spatial_force_input_port().FixValue(pc, forces)
        tau, sim, dt = self._tau, self.simulator, self.spec.dt
        for _ in range(self._substeps):
            # Gravity compensation on the robot, like a real arm controller (and MuJoCo's gravcomp).
            g = plant.CalcGravityGeneralizedForces(pc)
            tau.GetMutableData().set_value(np.where(self._robot_v, -g, 0.0))
            # Count steps rather than add up times, so every period has exactly the same substeps.
            self._k += 1
            sim.AdvanceTo(self._k * dt)
        if self._pending:
            self._pending = {}
            plant.get_applied_spatial_force_input_port().FixValue(pc, [])

    @property
    def control_dt(self):
        return self._substeps * self.spec.dt

    @property
    def time(self):
        return float(self.context.get_time())

    # --- world ------------------------------------------------------------------
    def object_pose(self, name):
        X = self.plant.EvalBodyPoseInWorld(self.pc, self._bodies[name])
        return X.translation().copy(), X.rotation().ToQuaternion().wxyz()

    def object_velocity(self, name):
        if name not in self._free:
            return np.zeros(6)
        V = self.plant.EvalBodySpatialVelocityInWorld(self.pc, self._free[name])
        return np.concatenate([V.translational(), V.rotational()])

    def set_object_pose(self, name, pos, quat=None):
        body = self._free[name]
        if quat is None:
            quat = self.object_pose(name)[1]
        self.plant.SetFreeBodyPose(self.pc, body, _pose(pos, quat))
        self.plant.SetFreeBodySpatialVelocity(self.pc, body, _zero_velocity())

    def contacts(self):
        res = self.plant.get_contact_results_output_port().Eval(self.pc)
        out = []
        for i in range(res.num_point_pair_contacts()):
            info = res.point_pair_contact_info(i)
            a, b = self._label.get(info.bodyA_index(), "?"), self._label.get(info.bodyB_index(), "?")
            if a == b:
                continue
            n = info.point_pair().nhat_BA_W
            out.append(Contact(a, b, float(abs(info.contact_force() @ n))))
        for i in range(res.num_hydroelastic_contacts()):
            info = res.hydroelastic_contact_info(i)
            surf = info.contact_surface()
            a, b = self._geom_label(surf.id_M()), self._geom_label(surf.id_N())
            if a == b:
                continue
            # Normal force: the total force along the patch's area-weighted normal.
            n = sum((surf.area(f) * surf.face_normal(f) for f in range(surf.num_faces())), np.zeros(3))
            f = info.F_Ac_W().translational()
            nn = np.linalg.norm(n)
            out.append(Contact(a, b, float(abs(f @ n) / nn) if nn > 0 else float(np.linalg.norm(f))))
        return out

    def _geom_label(self, gid):
        body = self.plant.GetBodyFromFrameId(self.sg.model_inspector().GetFrameId(gid))
        return self._label.get(body.index(), "?")

    def apply_force(self, name, force):
        self._pending[name] = self._pending.get(name, np.zeros(3)) + np.asarray(force, float)

    def render(self, camera, width, height):
        if not self._render:
            raise NotImplementedError("drake cannot render here (no offscreen GL context)")
        c = self._cams[camera]
        core = RenderCameraCore("vtk", CameraInfo(width, height, np.radians(c.fovy)), ClippingRange(0.01, 10.0), RigidTransform())
        img = (
            self.sg.get_query_output_port()
            .Eval(self.sgc)
            .RenderColorImage(ColorRenderCamera(core), self.sg.world_frame_id(), _camera_pose(c))
        )
        return np.array(img.data[:, :, :3], dtype=np.uint8)

    def get_state(self):
        # The servo targets are part of the state, as in MuJoCo's integration state.
        x = self.context.get_discrete_state_vector().CopyToVector()
        return np.concatenate([[self.context.get_time()], self._ctrl, self._gscale, x])

    def set_state(self, state):
        state = np.asarray(state, float)
        k = 1 + self.n_arm + 1
        self.context.SetTime(float(state[0]))
        self._k = round(float(state[0]) / self.spec.dt)
        ctrl, gscale = state[1:k], state[k : k + self.n_arm]
        for j, s in zip(self.joint_names, gscale):
            if s != self._gscale[self.joint_names.index(j)]:
                self.set_gain_scale(j, float(s))
        self.context.SetDiscreteState(state[k + self.n_arm :])
        self.set_ctrl(ctrl)


def _drake_urdf(path: Path) -> Path:
    """The exported URDF, adapted for Drake; built once, next to the export.

    * Collision meshes are declared convex. MuJoCo collides with each mesh's
      convex hull; Drake would otherwise use the hull for point contact but the
      true, concave surface for hydroelastic contact, where a cube can lodge in
      a jaw's hollow and be carried off when the gripper opens.
    * Visual meshes get vertex normals: Drake's VTK renderer refuses OBJ files
      without them, and the export writes only what collision needs.
    """
    out = path.with_name("robot_drake.urdf")
    if out.exists():
        return out
    pid = os.getpid()
    mesh_dir = path.parent / "meshes_vn"
    mesh_dir.mkdir(exist_ok=True)
    for src in sorted((path.parent / "meshes").glob("*.obj")):
        dst = mesh_dir / src.name
        if dst.exists():
            continue
        v, f = [], []
        for line in src.read_text().splitlines():
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
        v, f = np.array(v, float).reshape(-1, 3), np.array(f, int).reshape(-1, 3)
        n = np.zeros_like(v)
        if len(f):
            fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # area-weighted
            for k in range(3):
                np.add.at(n, f[:, k], fn)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        n[np.linalg.norm(n, axis=1) < 0.5] = (0, 0, 1)
        lines = [f"v {a:.7g} {b:.7g} {c:.7g}" for a, b, c in v]
        lines += [f"vn {a:.6g} {b:.6g} {c:.6g}" for a, b, c in n]
        lines += [f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}" for a, b, c in f]
        tmp = dst.with_name(f"{dst.name}.{pid}.tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(dst)
    ET.register_namespace("drake", _DRAKE_NS)
    root = ET.parse(path).getroot()
    for kind in ("collision", "visual"):
        for mesh in root.iter(kind):
            for m in mesh.iter("mesh"):
                if kind == "collision":
                    ET.SubElement(m, f"{{{_DRAKE_NS}}}declare_convex")
                else:
                    m.set("filename", m.get("filename").replace("meshes/", "meshes_vn/", 1))
    tmp = out.with_name(f"{out.name}.{pid}.tmp")
    tmp.write_text('<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode"))
    tmp.replace(out)
    return out


def _common_prefix(a: str, b: str) -> str:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]


def _replace_role():
    from pydrake.geometry import RoleAssign

    return RoleAssign.kReplace


def _zero_velocity():
    from pydrake.multibody.math import SpatialVelocity

    return SpatialVelocity(np.zeros(3), np.zeros(3))


def _camera_pose(c) -> RigidTransform:
    """Drake cameras look along +z with +y down the image."""
    pos, look = np.asarray(c.pos, float), np.asarray(c.lookat, float)
    z = look - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return RigidTransform(RotationMatrix(np.column_stack([x, y, z])), pos)


def _hand_anchor(model) -> np.ndarray:
    """Where the exporter put the hand's URDF link origin within MuJoCo's hand body frame (its first joint's anchor)."""
    import mujoco

    from ..robots.urdf import _HINGE, _SLIDE

    m = model.robot_spec().compile()
    b = m.body(model.hand).id
    for j in range(m.njnt):
        if m.jnt_bodyid[j] == b and m.jnt_type[j] in (_HINGE, _SLIDE):
            return m.jnt_pos[j].copy()
    del mujoco
    return np.zeros(3)
