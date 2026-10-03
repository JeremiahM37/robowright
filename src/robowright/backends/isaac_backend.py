"""Isaac Sim backend: NVIDIA's PhysX 5, driven through Isaac Sim's Kit runtime.

The robot is robowright's URDF export of its MuJoCo model (see
:mod:`robowright.robots.urdf`), converted to USD once by Isaac's own URDF
importer and cached next to the export. Each backend builds a fresh stage
around it and steps PhysX directly, reading and writing state through
PhysX's tensor API rather than Isaac's timeline, so a world costs one stage
and no rendering.

Floating-base (legged) robots import with a free root link, stand on their
own weight (no gravity compensation) and are placed standing at
``stand_height`` in ``stand_q()``; the base pose and velocity come from the
articulation's root link.

What URDF loses is put back as PhysX models it: the arm's servo gains,
force limits, armature and damping become joint drives and DOF properties,
the gripper's linkage becomes PhysX mimic joints (two-way, like MuJoCo's
equality constraints), arm links ignore gravity (MuJoCo's gravcomp) and
every material combines friction by taking the larger coefficient,
MuJoCo's rule.

Isaac Sim needs ``OMNI_KIT_ACCEPT_EULA=YES`` (its licence) and an NVIDIA
GPU; its URDF importer links against ``libxml2.so.2``, which distributions
shipping only libxml2 2.14+ need on ``LD_LIBRARY_PATH``. Kit can start only
once per process, so the first backend starts it and it lives until exit.
It is started without the RTX renderer, which crashes at start-up on some
drivers and is not needed for physics; this backend does not render.
``ROBOWRIGHT_ISAAC_DEVICE=cpu|gpu`` picks the PhysX pipeline: CPU by
default, ~17x faster for one small scene (1.5 vs 25 ms per control step,
Panda, RTX 5080), bit-for-bit deterministic and the only one with contact
reports (the GPU pipeline delivers none, so it also cannot unmount links
sunk into the floor, below).
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..robots import urdf
from ..scene import SceneSpec
from .base import CONTACTS, DETERMINISTIC, FORCES, GROUND_TRUTH, STATE, Backend, Contact, register
from .mujoco_backend import bin_walls

# Driven finger joints without an actuator gain of their own saturate this far off target (as in Genesis).
_DRIVE_SAT = 0.25
# Bump when the USD conversion changes, to invalidate cached robot USDs.
_USD_VERSION = 2

_APP = None
_DEVICE = "cpu"

# Physics only: the extensions robowright uses, without RTX, viewports or UI.
_KIT = """
[package]
title = "robowright physics"
version = "1.0.0"

[dependencies]
"isaacsim.simulation_app" = {{}}
"omni.physics.physx" = {{}}
"omni.physx.tensors" = {{}}
"omni.warp.core" = {{}}
"isaacsim.asset.importer.urdf" = {{}}

[settings.app]
name = "robowright-isaac"
version = "1.0.0"
settings.persistent = false
vulkan = true
enableDeveloperWarnings = false

[settings.app.exts.folders]
'++' = ["{apps}", "{apps}/../exts", "{apps}/../extscache", "{apps}/../extsDeprecated"]
"""


def _cache() -> Path:
    return urdf.cache_root().parent / "isaac"


def _start() -> str:
    """Start Kit once per process; every backend instance shares it. Returns the PhysX device."""
    global _APP, _DEVICE
    if _APP is not None:
        return _DEVICE
    import isaacsim
    from isaacsim import SimulationApp

    apps = Path(isaacsim.__file__).parent / "apps"
    kit = _cache() / "robowright_physics.kit"
    text = _KIT.format(apps=apps)
    if not kit.exists() or kit.read_text() != text:
        kit.parent.mkdir(parents=True, exist_ok=True)
        tmp = kit.with_name(f"{kit.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.replace(kit)
    _APP = SimulationApp({"headless": True, "extra_args": ["--/log/outputStreamLevel=Error"]}, experience=str(kit))
    atexit.register(_APP.close)
    import carb

    s = carb.settings.get_settings()
    # Nothing reads the stage's transforms back: keep PhysX from writing them to USD every step.
    s.set_bool("/physics/updateToUsd", False)
    s.set_bool("/physics/updateVelocitiesToUsd", False)
    s.set_bool("/physics/updateParticlesToUsd", False)
    _DEVICE = os.environ.get("ROBOWRIGHT_ISAAC_DEVICE") or "cpu"
    s.set_bool("/physics/suppressReadback", _DEVICE == "gpu")
    return _DEVICE


def _isaac_urdf(path: Path, meta: dict) -> Path:
    """The exported URDF, adapted for Isaac's importer; built once, next to the export.

    * Mesh files are renamed: the importer names USD prims after them, and
      USD names cannot start with a digit (the export's are ``011_link0.obj``).
    * Every finger joint but the reference driven one mimics it, as in the
      Genesis backend: the importer turns ``<mimic>`` into a PhysX mimic joint.
    """
    root = ET.parse(path).getroot()
    meshes = path.parent / "meshes_isaac"
    meshes.mkdir(exist_ok=True)
    for m in root.iter("mesh"):
        src = m.get("filename")
        name = "m" + Path(src).name
        if not (meshes / name).exists():
            tmp = meshes / f"{name}.{os.getpid()}.tmp"
            shutil.copyfile(path.parent / src, tmp)
            tmp.replace(meshes / name)
        m.set("filename", f"meshes_isaac/{name}")
    g = meta.get("gripper")
    if g:
        joints = {j.get("name"): j for j in root.iter("joint")}
        ref = (g.get("driven") or [g["main"]])[0]
        rc, ro = g["joints"][ref]
        for name, (c, o) in g["joints"].items():
            if name == ref:
                continue
            k = (o - c) / (ro - rc)
            ET.SubElement(joints[urdf._safe(name)], "mimic", joint=urdf._safe(ref), multiplier=f"{k:.12g}", offset=f"{c - k * rc:.12g}")
    xml = ET.tostring(root, encoding="unicode")
    key = f"{_USD_VERSION}|{meta['floating']}|{xml}"
    out = path.with_name(f"robot.isaac-{hashlib.sha1(key.encode()).hexdigest()[:12]}.urdf")
    if not out.exists():
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        tmp.write_text(xml)
        tmp.replace(out)
    return out


def _robot_usd(path: Path, meta: dict) -> Path:
    """The robot as USD, converted by Isaac's URDF importer once and cached (a fraction of a second, but it writes files)."""
    src = _isaac_urdf(path, meta)
    out = src.parent / src.stem.replace("robot.", "usd-")
    if (out / "robot.usd").exists():
        return out / "robot.usd"
    import omni.kit.commands

    _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False  # the hand and finger links keep their names
    cfg.fix_base = not meta["floating"]
    cfg.self_collision = False
    cfg.import_inertia_tensor = True
    cfg.parse_mimic = True
    cfg.convex_decomp = False  # one convex hull per mesh, as MuJoCo collides
    cfg.collision_from_visuals = False
    cfg.make_default_prim = True
    cfg.create_physics_scene = False
    cfg.distance_scale = 1.0
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    ok, _ = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=str(src), import_config=cfg, dest_path=str(tmp / "robot.usd"))
    if not ok or not (tmp / "robot.usd").exists():
        raise RuntimeError(f"Isaac's URDF importer could not convert {src}")
    try:
        tmp.replace(out)
    except OSError:  # another process won the race
        shutil.rmtree(tmp, ignore_errors=True)
    return out / "robot.usd"


def _hand_anchor(model) -> np.ndarray:
    """Where the exporter put the hand's URDF link origin within MuJoCo's hand body frame (its first joint's anchor)."""
    from ..robots.urdf import _HINGE, _SLIDE

    m = model.robot_spec().compile()
    b = m.body(model.hand).id
    for j in range(m.njnt):
        if m.jnt_bodyid[j] == b and m.jnt_type[j] in (_HINGE, _SLIDE):
            return m.jnt_pos[j].copy()
    return np.zeros(3)


def _rotate(quat_wxyz, v) -> np.ndarray:
    w, x, y, z = quat_wxyz
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R @ np.asarray(v, float)


def _wxyz(t) -> np.ndarray:
    x, y, z, w = (float(v) for v in t[3:7])
    return np.array([w, x, y, z])


@register("isaac")
class IsaacBackend(Backend):
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, STATE, DETERMINISTIC, FORCES})

    def __init__(self, spec: SceneSpec, seed: int = 0):
        super().__init__(spec, seed)
        rm = self.robot_model
        gpu = _start() == "gpu"
        if gpu:
            # GPU PhysX sums contact and constraint terms in whatever order threads finish, and the
            # GPU pipeline (which needs readback suppressed) delivers no contact reports.
            self.capabilities = self.capabilities - {DETERMINISTIC, CONTACTS}
        self._gpu = gpu
        import omni.physics.tensors
        import omni.physx
        import omni.usd

        path, meta = urdf.load(rm)
        self.meta = meta
        usd = _robot_usd(path, meta)
        self._dt = spec.dt
        self._substeps = max(1, round((1.0 / spec.control_hz) / self._dt))
        self._k = 0
        self._pending: dict[str, np.ndarray] = {}

        ctx = omni.usd.get_context()
        omni.physx.get_physx_simulation_interface().detach_stage()
        ctx.new_stage()
        self.stage = stage = ctx.get_stage()
        self._build_stage(stage, spec, usd, meta)

        self._physx = omni.physx.get_physx_interface()
        self._sim = omni.physx.get_physx_simulation_interface()
        # Attach PhysX to the new stage directly: Kit would only notice it on its next update.
        self._sim.attach_stage(ctx.get_stage_id())
        self._physx.start_simulation()
        # PhysX adds the articulation to its scene on the first step; the views need it there.
        # The step is undone below (objects back at their initial poses, joints reset).
        self._sim.simulate(self._dt, 0.0)
        self._sim.fetch_results()
        # A legged robot's feet belong on the floor: only an arm's mount is unmounted from it.
        if not rm.floating and self._unmount_from_floor():
            self._sim.simulate(self._dt, 0.0)
            self._sim.fetch_results()
        self._view = omni.physics.tensors.create_simulation_view("warp" if gpu else "numpy", stage_id=ctx.get_stage_id())
        self._view.set_subspace_roots("/")
        self._art = self._view.create_articulation_view(self._art_root)
        self._bodies = {name: self._view.create_rigid_body_view(p) for name, p in self._object_paths.items()}
        if not self._view.check() or self._art.count != 1:
            raise RuntimeError("PhysX did not build the robot articulation")
        self._index(meta)
        self._set_dof_properties(meta)
        for name in self._bodies:
            o = spec.object(name)
            self.set_object_pose(name, o.initial_pos, o.quat)
        q = np.clip(np.zeros(self._ndof), self._qlo, self._qhi)
        self._write_dofs(q, np.zeros(self._ndof))
        self._ctrl = np.zeros(len(self.joint_names))
        self._targets = q.copy()
        if rm.floating:
            yaw = rm.base_yaw
            self.set_base_pose((*rm.base_pos[:2], rm.stand_height), (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
            self.set_joint_positions(rm.stand_q())
        self.set_ctrl(self.qpos())

    # --- construction ----------------------------------------------------------
    def _build_stage(self, stage, spec, usd: Path, meta: dict) -> None:
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        px.CreateTimeStepsPerSecondAttr(int(round(1.0 / self._dt)))
        px.CreateEnableGPUDynamicsAttr(self._gpu)
        px.CreateBroadphaseTypeAttr("GPU" if self._gpu else "MBP")
        # PGS, not PhysX's default TGS: under TGS the xArm7's linkage gripper winds the arm up (joints
        # end radians off target); PGS passes every arm in the conformance suite.
        px.CreateSolverTypeAttr("PGS")
        px.CreateEnableEnhancedDeterminismAttr(True)
        px.CreateEnableStabilizationAttr(False)
        UsdGeom.Xform.Define(stage, "/World")

        materials: dict[float, Sdf.Path] = {}

        def material(mu: float) -> UsdShade.Material:
            """A physics material that, like MuJoCo, takes the larger of two touching friction coefficients."""
            mu = float(np.clip(mu, 0.0, 5.0))
            if mu not in materials:
                p = Sdf.Path(f"/World/Materials/mu_{len(materials)}")
                mat = UsdShade.Material.Define(stage, p)
                api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                api.CreateStaticFrictionAttr(mu)
                api.CreateDynamicFrictionAttr(mu)
                api.CreateRestitutionAttr(0.0)
                pm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
                pm.CreateFrictionCombineModeAttr("max")
                pm.CreateRestitutionCombineModeAttr("min")
                materials[mu] = p
            return UsdShade.Material(stage.GetPrimAtPath(materials[mu]))

        def bind(prim, mu):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material(mu), UsdShade.Tokens.weakerThanDescendants, "physics")

        def report(prim):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)

        def no_sleep(prim):
            # A sleeping body generates no contacts: a cube resting on the floor would stop reporting it.
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateSleepThresholdAttr(0.0)
            rb.CreateStabilizationThresholdAttr(0.0)
            return rb

        def pose(xf, pos, quat=(1, 0, 0, 0), scale=None):
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
            w, x, y, z = (float(v) for v in quat)
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
            if scale is not None:
                xf.AddScaleOp().Set(Gf.Vec3d(*map(float, scale)))

        # Floor: a thick static slab whose top face is z = 0.
        floor = UsdGeom.Cube.Define(stage, "/World/floor")
        floor.CreateSizeAttr(1.0)
        pose(floor, (0, 0, -0.05), scale=(8, 8, 0.1))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        bind(floor.GetPrim(), 1.0)
        report(floor.GetPrim())
        self._label = {"/World/floor": "floor"}

        # Objects.
        self._object_paths: dict[str, str] = {}
        for o in spec.objects:
            p = f"/World/objects/{o.name}"
            if o.kind == "bin":
                xf = UsdGeom.Xform.Define(stage, p)
                pose(xf, o.initial_pos, o.quat)
                for i, (c, hs) in enumerate(bin_walls(o.size)):
                    wall = UsdGeom.Cube.Define(stage, f"{p}/wall{i}")
                    wall.CreateSizeAttr(1.0)
                    pose(wall, c, scale=2 * np.asarray(hs))
                    UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
                    report(wall.GetPrim())
                    self._label[f"{p}/wall{i}"] = o.name
                bind(xf.GetPrim(), max(o.friction, 0.01))
                continue
            if o.kind == "box":
                g = UsdGeom.Cube.Define(stage, p)
                g.CreateSizeAttr(1.0)
                pose(g, o.initial_pos, o.quat, scale=2 * np.asarray(o.size, float))
            elif o.kind == "cylinder":
                g = UsdGeom.Cylinder.Define(stage, p)
                g.CreateRadiusAttr(float(o.size[0]))
                g.CreateHeightAttr(2.0 * float(o.size[1]))
                g.CreateAxisAttr(UsdGeom.Tokens.z)
                pose(g, o.initial_pos, o.quat)
            else:
                g = UsdGeom.Sphere.Define(stage, p)
                g.CreateRadiusAttr(float(o.size[0]))
                pose(g, o.initial_pos, o.quat)
            prim = g.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(o.mass))
            no_sleep(prim)
            bind(prim, o.friction)
            report(prim)
            self._object_paths[o.name] = p
            self._label[p] = o.name

        # Robot: the converted USD, referenced under /World/robot at the model's base pose.
        rm = self.robot_model
        yaw = rm.base_yaw
        base_q = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        robot = UsdGeom.Xform.Define(stage, "/World/robot")
        robot.GetPrim().GetReferences().AddReference(str(usd))
        pose(robot, rm.base_pos, base_q)
        friction: dict[str, float] = {}
        for g in meta["geoms"].values():
            friction[g["link"]] = max(friction.get(g["link"], 0.0), g["friction"])
        self._links: dict[str, str] = {}  # URDF link name -> prim path
        self._joints: dict[str, object] = {}  # URDF joint name -> prim
        parent: dict[str, str] = {}
        welded: set[str] = set()  # links attached to their parent by a fixed joint
        art_root = None
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith("/World/robot/"):
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                name = prim.GetName()
                self._links[name] = str(prim.GetPath())
                # Arm controllers on real robots cancel gravity (MuJoCo's gravcomp): the links ignore it.
                # Legged robots stand on their own weight.
                no_sleep(prim).CreateDisableGravityAttr(rm.family == "arm")
                bind(prim, friction.get(name, 1.0))
                report(prim)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and not prim.IsA(UsdPhysics.Joint):
                art_root = prim  # a floating base: the root link carries the articulation
                self._configure_articulation(PhysxSchema.PhysxArticulationAPI.Apply(prim))
            if prim.IsA(UsdPhysics.Joint):
                j = UsdPhysics.Joint(prim)
                b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    art_root = prim
                    welded.add(b1[0].name)
                    # The importer anchors the fixed base at the world origin; put it at the model's base pose.
                    j.GetLocalPos0Attr().Set(Gf.Vec3f(*map(float, rm.base_pos)))
                    j.GetLocalRot0Attr().Set(Gf.Quatf(*map(float, base_q)))
                    self._configure_articulation(PhysxSchema.PhysxArticulationAPI(prim))
                elif b0 and b1:
                    parent[b1[0].name] = b0[0].name
                    if prim.IsA(UsdPhysics.FixedJoint):
                        welded.add(b1[0].name)
                if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                    self._joints[prim.GetName()] = prim
                    kind = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
                    # Force drives, as MuJoCo's actuators: gains are set in SI through the tensor API.
                    if prim.HasAPI(UsdPhysics.DriveAPI, kind):
                        UsdPhysics.DriveAPI(prim, kind).GetTypeAttr().Set("force")
                    # The importer copies the URDF's velocity limit, a placeholder; MuJoCo's joints have none.
                    PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(1e6)
        if art_root is None:
            raise RuntimeError("the converted robot has no articulation root")
        self._art_root = str(art_root.GetPath())
        if meta.get("gripper"):
            self._couple_fingers(meta["gripper"])

        # MuJoCo never collides bodies welded to the world with each other: the robot's base with the
        # floor or a bin. Those links are the root and whatever hangs off it by fixed joints.
        def on_base(name):
            while name in welded and name in parent:
                name = parent[name]
            return name in welded and name not in parent

        static = [p for p, label in self._label.items() if label == "floor" or label in {o.name for o in spec.objects if o.static}]
        for name, p in self._links.items():
            if on_base(name):
                rel = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(p)).CreateFilteredPairsRel()
                for q in static:
                    rel.AddTarget(Sdf.Path(q))
        # Contact labels: a finger's whole subtree reports as that finger, other links as their MuJoCo body.
        safe_to_body = {v: k for k, v in meta["links"].items()}
        parts = {n: part for part, names in meta["fingers"].items() for n in names}
        for name, p in self._links.items():
            label, cur = None, name
            while label is None and cur is not None:
                label = parts.get(cur)
                cur = parent.get(cur)
            body = name.split("__j")[0] if name not in safe_to_body else name
            self._label[p] = f"robot:{label or safe_to_body.get(body, body)}"

    @staticmethod
    def _configure_articulation(art) -> None:
        art.CreateEnabledSelfCollisionsAttr(False)
        art.CreateSleepThresholdAttr(0.0)
        art.CreateStabilizationThresholdAttr(0.0)
        art.CreateSolverPositionIterationCountAttr(32)
        art.CreateSolverVelocityIterationCountAttr(4)

    def _unmount_from_floor(self) -> bool:
        """Stop the floor colliding with robot links that are sunk into it as the robot stands.

        Some models mount the arm so that a link's collision shape reaches
        through the floor (the UR10e's shoulder sits 2.7 cm into it). MuJoCo's
        soft contacts absorb that; PhysX's rigid ones fight it, and the first
        joint teleport blows the arm apart. Genesis drops pairs overlapping in
        the starting pose for the same reason; this drops them for the floor.
        """
        from pxr import PhysicsSchemaTools, Sdf, UsdPhysics

        headers, data = self._sim.get_contact_report()
        sunk = set()
        for h in headers:
            paths = [str(PhysicsSchemaTools.intToSdfPath(h.actor0)), str(PhysicsSchemaTools.intToSdfPath(h.actor1))]
            if "/World/floor" not in paths:
                continue
            other = paths[1 - paths.index("/World/floor")]
            if not self._label.get(other, "").startswith("robot:"):
                continue
            if any(c.separation < -1e-3 for c in data[h.contact_data_offset : h.contact_data_offset + h.num_contact_data]):
                sunk.add(other)
        if not sunk:
            return False
        rel = UsdPhysics.FilteredPairsAPI.Apply(self.stage.GetPrimAtPath("/World/floor")).CreateFilteredPairsRel()
        for p in sorted(sunk):
            rel.AddTarget(Sdf.Path(p))
        return True

    def _couple_fingers(self, g: dict) -> None:
        """Tie every finger joint to the reference driven joint with a hard PhysX mimic joint.

        The importer reads the URDF's ``<mimic>`` tags but leaves the mimic's
        reference unset in its USD output and makes it compliant (25 Hz), so
        the coupling is rewritten here from the calibration. PhysX's relation
        is ``q + gearing * q_ref + offset = 0`` in USD units (degrees for hinges).
        """
        from pxr import PhysxSchema, Sdf

        ref = (g.get("driven") or [g["main"]])[0]
        rc, ro = g["joints"][ref]
        ref_prim = self._joints[urdf._safe(ref)]
        unit = {True: 180.0 / np.pi, False: 1.0}
        u_ref = unit[self.meta["joints"][ref]["type"] == "hinge"]
        for name, (c, o) in g["joints"].items():
            if name == ref:
                continue
            prim = self._joints[urdf._safe(name)]
            k = (o - c) / (ro - rc)
            u = unit[self.meta["joints"][name]["type"] == "hinge"]
            insts = [i for i in ("rotX", "rotY", "rotZ") if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, i)] or ["rotX"]
            for inst in insts:
                mj = PhysxSchema.PhysxMimicJointAPI.Apply(prim, inst)
                mj.CreateReferenceJointRel().SetTargets([ref_prim.GetPath()])
                mj.CreateGearingAttr().Set(float(-k * u / u_ref))
                mj.CreateOffsetAttr().Set(float(-u * (c - k * rc)))
                # The importer makes the mimic compliant (25 Hz); zero makes it a hard constraint, as MuJoCo's equality is.
                prim.CreateAttribute(f"physxMimicJoint:{inst}:naturalFrequency", Sdf.ValueTypeNames.Float).Set(0.0)
                prim.CreateAttribute(f"physxMimicJoint:{inst}:dampingRatio", Sdf.ValueTypeNames.Float).Set(0.0)

    def _index(self, meta: dict) -> None:
        art, rm = self._art, self.robot_model
        mt = art.shared_metatype
        self._dof_names = list(mt.dof_names)
        self._ndof = len(self._dof_names)
        dof = {n: i for i, n in enumerate(self._dof_names)}
        self._arm = np.array([dof[urdf._safe(j)] for j in rm.arm_joints])
        lim = self._row(art.get_dof_limits()).reshape(self._ndof, 2)
        self._qlo, self._qhi = lim[:, 0], lim[:, 1]
        self._lo, self._hi = self._qlo[self._arm], self._qhi[self._arm]
        link_names = list(mt.link_names)
        self._hand = link_names.index(meta["hand"]) if rm.hand else None
        self._hand_anchor = _hand_anchor(rm) if rm.hand else None
        self._root = link_names.index(meta["root"])
        self._nlinks = len(link_names)
        self._root_com = self._row(art.get_coms()).reshape(-1, 7)[self._root, :3]  # in the root link's frame
        self._idx = np.zeros(1, dtype=np.int32) if not self._gpu else self._gpu_idx()
        self._obj_idx = self._idx
        self._cpu_idx = np.zeros(1, dtype=np.int32) if not self._gpu else self._gpu_idx("cpu")
        if self.has_gripper:
            g = meta["gripper"]
            self._fingers = {n: (dof[urdf._safe(n)], c, o) for n, (c, o) in g["joints"].items()}
            driven = g.get("driven") or [g["main"]]
            self._driven_names = driven
            self._driven = np.array([dof[urdf._safe(n)] for n in driven])
            self._driven_c = np.array([g["joints"][n][0] for n in driven])
            self._driven_o = np.array([g["joints"][n][1] for n in driven])
            self._main, self._main_c, self._main_o = self._fingers[g["main"]]

    @staticmethod
    def _gpu_idx(device="cuda:0"):
        import warp as wp

        return wp.zeros(1, dtype=wp.int32, device=device)

    def _set_dof_properties(self, meta: dict) -> None:
        """Servo gains, force caps, armature and damping, from the MuJoCo model (SI units).

        Joint dry friction (MuJoCo's ``frictionloss``) is left out, as the
        Drake and PyBullet backends leave it out. MuJoCo solves it as a soft
        constraint that lets a servo creep onto its target; PhysX's only joint
        friction is Coulomb stiction (it rejects a dynamic friction above the
        static one), which parks a weak servo wherever its pull drops below
        the friction: 0.03 rad short on the PiPER's wrist.
        """
        rm, jm = self.robot_model, meta["joints"]
        n = self._ndof
        kp, kv, cap, armature = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        by_dof = {}
        for name in jm:
            safe = urdf._safe(name)
            if safe in self._dof_names:
                by_dof[self._dof_names.index(safe)] = name
        for d, name in by_dof.items():
            j = jm[name]
            armature[d] = j["armature"]
            kv[d] = j["damping"]  # passive damping, as drive damping (mimic followers have no drive: tiny, dropped)
        for i, name in enumerate(rm.arm_joints):
            d = self._arm[i]
            j = jm[name]
            kp[d] = j.get("kp", 0.0)
            kv[d] += j.get("kv", 0.0)
            cap[d] = j["effort"]
        if self.has_gripper:
            g = meta["gripper"]
            driven = self._driven_names
            try:
                M = self._row(self._art.get_mass_matrices()).reshape(n, n)  # fixed base: joint space only
            except Exception:
                M = None
            for name, d in zip(driven, self._driven):
                c, o = g["joints"][name]
                e = g["effort"][name]
                j = jm[name]
                if "kp" in j:
                    kp[d], kv[d] = j["kp"], kv[d] + j["kv"]
                else:
                    k = e / (_DRIVE_SAT * max(abs(o - c), 1e-6))
                    inertia = float(M[d, d]) if M is not None else max(j["armature"], 1e-3)
                    kp[d], kv[d] = k, kv[d] + 2.0 * np.sqrt(k * max(inertia, 1e-6))  # critically damped
                cap[d] = e
        self._kp, self._kv, self._cap = kp, kv, cap
        self._gscale = np.ones(self.n_arm)
        art = self._art
        art.set_dof_stiffnesses(self._t32(kp[None], cpu=True), self._cpu_idx)
        art.set_dof_dampings(self._t32(kv[None], cpu=True), self._cpu_idx)
        art.set_dof_max_forces(self._t32(np.where(cap > 0, cap, 1e9)[None], cpu=True), self._cpu_idx)
        art.set_dof_armatures(self._t32(armature[None], cpu=True), self._cpu_idx)

    # --- tensor helpers ----------------------------------------------------------
    def _t32(self, a, cpu: bool = False):
        """A float32 array for the tensor API: on the GPU for state, on the host for model properties."""
        a = np.ascontiguousarray(a, dtype=np.float32)
        if not self._gpu:
            return a
        import warp as wp

        # Warp, not torch: the torch Isaac Sim 5.1 pins has no kernels for Blackwell GPUs (RTX 50xx).
        return wp.array(a, dtype=wp.float32, device="cpu" if cpu else "cuda:0")

    @staticmethod
    def _row(t) -> np.ndarray:
        """The one articulation's (or body's) row of a tensor-API array, flat, as float64."""
        if not isinstance(t, np.ndarray):
            t = t.numpy()
        return np.asarray(t, dtype=float).reshape(-1)

    def _write_dofs(self, q, v) -> None:
        self._vel = None
        self._art.set_dof_positions(self._t32(q[None]), self._idx)
        self._art.set_dof_velocities(self._t32(v[None]), self._idx)
        self._view.update_articulations_kinematic()

    # --- robot -------------------------------------------------------------------
    def _opening(self, q):
        return (q - self._main_c) / (self._main_o - self._main_c)

    def qpos(self):
        q = self._row(self._art.get_dof_positions())
        if not self.has_gripper:
            return q[self._arm]
        return np.append(q[self._arm], self._opening(q[self._main]))

    def _dof_vel(self) -> np.ndarray:
        """Joint velocities as the joints moved over the last physics step.

        PhysX's own joint velocities include the solver's correction velocity:
        a jaw clamped on an object reports a steady few cm/s while it does not
        move at all, and a gripper never looks stalled. Displacement over the
        step is what an encoder measures, and equals PhysX's velocity whenever
        nothing was corrected (positions integrate implicitly).
        """
        if self._vel is None:
            return self._row(self._art.get_dof_velocities())
        return self._vel

    def qvel(self):
        v = self._dof_vel()
        if not self.has_gripper:
            return v[self._arm]
        return np.append(v[self._arm], v[self._main] / (self._main_o - self._main_c))

    def set_ctrl(self, target):
        target = np.asarray(target, float)
        arm = np.clip(target[: self.n_arm], self._lo, self._hi)
        t = self._targets
        t[self._arm] = arm
        if self.has_gripper:
            s = float(np.clip(target[self.n_arm], 0.0, 1.0))
            t[self._driven] = self._driven_c + s * (self._driven_o - self._driven_c)
            self._ctrl = np.append(arm, s)
        else:
            self._ctrl = arm
        self._art.set_dof_position_targets(self._t32(t[None]), self._idx)

    def ctrl(self):
        return self._ctrl.copy()

    def set_gain_scale(self, joint, scale):
        i = self.joint_names.index(joint)
        self._gscale[i] = scale
        self._apply_gains()

    def _apply_gains(self):
        kp, kv = self._kp.copy(), self._kv.copy()
        kp[self._arm] *= self._gscale
        kv[self._arm] *= self._gscale
        self._art.set_dof_stiffnesses(self._t32(kp[None], cpu=True), self._cpu_idx)
        self._art.set_dof_dampings(self._t32(kv[None], cpu=True), self._cpu_idx)

    def set_joint_positions(self, q):
        q = np.asarray(q, float)
        pos = self._row(self._art.get_dof_positions())
        pos[self._arm] = q[: self.n_arm]
        if self.has_gripper:
            # Every finger joint where it sits at this opening, so the gripper starts at rest.
            s = float(np.clip(q[self.n_arm], 0, 1))
            for d, c, o in self._fingers.values():
                pos[d] = c + s * (o - c)
        self._write_dofs(pos, np.zeros(self._ndof))
        self.set_ctrl(q)

    def hand_pose(self):
        t = self._row(self._art.get_link_transforms()).reshape(-1, 7)[self._hand]
        quat = _wxyz(t)
        return t[:3] - _rotate(quat, self._hand_anchor), quat

    # --- time --------------------------------------------------------------------
    def step(self):
        sim, dt = self._sim, self._dt
        for i in range(self._substeps):
            for name, f in self._pending.items():
                if name == "robot":
                    self._push_base(f)
                else:
                    self._bodies[name].apply_forces(self._t32(np.asarray(f)[None]), self._obj_idx, True)
            if i == self._substeps - 1:
                q0 = self._row(self._art.get_dof_positions())
            # Count steps rather than add up times, so every period has exactly the same substeps.
            sim.simulate(dt, self._k * dt)
            sim.fetch_results()
            self._k += 1
        self._vel = (self._row(self._art.get_dof_positions()) - q0) / dt
        self._pending = {}

    @property
    def control_dt(self):
        return self._substeps * self._dt

    @property
    def time(self):
        return self._k * self._dt

    # --- world -------------------------------------------------------------------
    def object_pose(self, name):
        if name not in self._bodies:
            o = self.spec.object(name)
            return np.asarray(o.initial_pos, float), np.asarray(o.quat, float)
        t = self._row(self._bodies[name].get_transforms())
        return t[:3].copy(), _wxyz(t)

    def object_velocity(self, name):
        if name not in self._bodies:
            return np.zeros(6)
        return self._row(self._bodies[name].get_velocities()).copy()

    def set_object_pose(self, name, pos, quat=None):
        b = self._bodies[name]
        t = self._row(b.get_transforms())
        t[:3] = pos
        if quat is not None:
            w, x, y, z = quat
            t[3:7] = (x, y, z, w)
        b.set_transforms(self._t32(t[None]), self._obj_idx)
        b.set_velocities(self._t32(np.zeros((1, 6))), self._obj_idx)

    def contacts(self):
        from pxr import PhysicsSchemaTools

        headers, data = self._sim.get_contact_report()
        out = []
        for h in headers:
            if not h.num_contact_data:
                continue
            a = self._label.get(str(PhysicsSchemaTools.intToSdfPath(h.actor0)))
            b = self._label.get(str(PhysicsSchemaTools.intToSdfPath(h.actor1)))
            if a is None or b is None:
                a = a or str(PhysicsSchemaTools.intToSdfPath(h.collider0))
                b = b or str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            if a == b:
                continue
            impulse = 0.0
            for c in data[h.contact_data_offset : h.contact_data_offset + h.num_contact_data]:
                impulse += float(np.dot(c.impulse, c.normal))
            out.append(Contact(a, b, abs(impulse) / self._dt))
        return out

    def apply_force(self, name, force):
        self._pending[name] = self._pending.get(name, np.zeros(3)) + np.asarray(force, float)

    def _push_base(self, force) -> None:
        """A world-frame force at the base's centre of mass, as MuJoCo's ``xfrc_applied`` acts."""
        pos, quat = self.base_pose()
        f = np.zeros((1, self._nlinks, 3))
        at = np.zeros((1, self._nlinks, 3))
        f[0, self._root] = force
        at[0, self._root] = pos + _rotate(quat, self._root_com)
        self._art.apply_forces_and_torques_at_position(self._t32(f), None, self._t32(at), self._idx, True)

    # --- floating base -------------------------------------------------------------
    def base_pose(self):
        t = self._row(self._art.get_root_transforms())
        return t[:3].copy(), _wxyz(t)

    def base_velocity(self):
        """World-frame linear (of the base frame's origin, not its centre of mass) then angular velocity."""
        v = self._row(self._art.get_root_velocities())
        _, quat = self.base_pose()
        lin, ang = v[:3], v[3:6]
        return np.concatenate([lin - np.cross(ang, _rotate(quat, self._root_com)), ang])

    def set_base_pose(self, pos, quat):
        w, x, y, z = (float(v) for v in quat)
        t = np.array([*map(float, pos), x, y, z, w])
        self._art.set_root_transforms(self._t32(t[None]), self._idx)
        self._art.set_root_velocities(self._t32(np.zeros((1, 6))), self._idx)
        self._view.update_articulations_kinematic()
        self._vel = None

    # --- state -------------------------------------------------------------------
    def get_state(self):
        """The full state; for a floating base, also snaps the live simulation onto it.

        PhysX's root pose does not survive a read and write unchanged: writing
        back the pose just read moves the base by ~1e-8 m (the tensor API
        converts it to and from the root's centre-of-mass frame in float32),
        and the walk diverges by microradians within a few steps. Restoring
        the captured state onto this simulation too makes it the exact state
        any later ``set_state`` reproduces, so a replay is bit-identical.

        One thing it cannot carry: PhysX keeps each touching pair's contact
        manifold, friction anchors and warm-start impulses, which the tensor
        API neither reads nor writes. A state captured before any contact has
        been stepped (a trace's start) restores exactly; one captured mid-grasp or
        mid-stance resumes ~3e-5 off after the next step (measured: Panda
        holding the cube, G1 standing).
        """
        parts = [[self._k], self._ctrl, self._gscale]
        parts += [self._row(self._art.get_dof_positions()), self._row(self._art.get_dof_velocities())]
        if self.robot_model.floating:
            parts += [self._row(self._art.get_root_transforms()), self._row(self._art.get_root_velocities())]
        for b in self._bodies.values():
            parts += [self._row(b.get_transforms()), self._row(b.get_velocities())]
        state = np.concatenate([np.asarray(p, float).ravel() for p in parts])
        if self.robot_model.floating:
            vel = self._vel  # keep the measured joint velocities: nothing moved
            self.set_state(state)
            self._vel = vel
        return state

    def set_state(self, state):
        s = np.asarray(state, float)
        nc, na, nd = len(self.joint_names), self.n_arm, self._ndof
        self._k = int(round(s[0]))
        i = 1
        ctrl = s[i : i + nc]
        i += nc
        self._gscale = s[i : i + na].copy()
        i += na
        q, v = s[i : i + nd], s[i + nd : i + 2 * nd]
        i += 2 * nd
        if self.robot_model.floating:
            self._art.set_root_transforms(self._t32(s[i : i + 7][None]), self._idx)
            self._art.set_root_velocities(self._t32(s[i + 7 : i + 13][None]), self._idx)
            i += 13
        self._write_dofs(q, v)
        for b in self._bodies.values():
            b.set_transforms(self._t32(s[i : i + 7][None]), self._obj_idx)
            b.set_velocities(self._t32(s[i + 7 : i + 13][None]), self._obj_idx)
            i += 13
        self._apply_gains()
        self.set_ctrl(ctrl)

    def close(self):
        """Release the PhysX scene and the stage; Kit itself stays up for the next world."""
        view = getattr(self, "_view", None)
        if view is None:
            return
        try:
            view.invalidate()
            self._physx.release_physics_objects()
            import omni.usd

            omni.usd.get_context().close_stage()
        except Exception:
            pass
        for k in ("_view", "_art", "_bodies", "stage"):
            setattr(self, k, None)
