"""A Drake simulation that robowright did not build: a MultibodyPlant in a Simulator, or in a Diagram with a context.

The environment owns the diagram and its Simulator and steps it with ``simulator.AdvanceTo``; this adapter
reads the plant's state from the simulator's context. Bodies and joints are named by their SDF or URDF
names. A name that two model instances share is qualified as ``model::body`` (``block_a::base_link``), and
the bare name still works where it is unique.

What Drake gives, and what this adapter says it cannot give:

* Names. Bodies are the links the plant holds (the world body is not one). Joints are the revolute and
  prismatic ones, by joint name; a free or welded joint has no single value and is not named. Drake has
  frames and sites, but this adapter does not name them, so ``names("site")`` is empty.
* Contacts. The plant's point-pair contacts are read, each with its normal force: the component of the
  contact force along the contact normal, which is what holding is judged by. Hydroelastic contacts (a
  compliant model in the plant) are refused with the reason rather than read as zero, since this adapter
  has not been checked against them.
* Forces. ``apply_force`` sets the plant's applied-spatial-force input for the physics steps that follow:
  it is held over the whole control step, as Drake holds an input, and cleared once the step is taken. The
  force acts at the body's origin, which for a body whose origin is its centre (a block) is its centre.
* The clock. Time is the simulator's context time, not a count of steps.
* Rendering and saved state are not read here. Drake draws through its own visualizer, and a trace of a
  Drake run is recorded without a state to redraw it from.
* Replay. A trace keeps the steps, the forces robowright applied and the calls the test made, so
  ``replay(trace, make_env=...)`` rebuilds the simulation and re-runs them. The factory must build the same
  diagram with the same initial state and the same controllers: the inputs the run's own code set between steps
  are not in the trace, and replay reports where a rebuilt run first differs.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..errors import CapabilityError
from .attached_engine import AttachedEngine
from .base import CONTACTS, FORCES, GROUND_TRUTH


def _qualified(items: list[tuple[str, str]]) -> list[str]:
    """Labels for ``(model instance, name)`` pairs: the bare name where it is unique, else ``instance::name``."""
    counts = Counter(name for _, name in items)
    return [name if counts[name] == 1 else f"{inst}::{name}" for inst, name in items]


def _drake_classes():
    from pydrake.multibody.tree import PrismaticJoint, QuaternionFloatingJoint, RevoluteJoint

    return RevoluteJoint, PrismaticJoint, QuaternionFloatingJoint


class DrakeEngine(AttachedEngine):
    """Handles are ``("body", index)``, with the index of the body in the plant."""

    engine = "drake"
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, FORCES})

    def __init__(self, system, context, *, plant=None, seed: int = 0, dt: float | None = None):
        from pydrake.multibody.plant import MultibodyPlant

        self.system = system  # the Diagram (or a bare plant) that holds the plant
        self.context = context  # the root context the simulator steps: the one its clock is read from
        self.plant = plant if plant is not None else self._find_plant(system, MultibodyPlant)
        super().__init__(seed, dt=dt)
        self._forces: dict[int, np.ndarray] = {}
        self._scan()

    @staticmethod
    def _find_plant(system, cls):
        if isinstance(system, cls):
            return system
        found = [s for s in (system.GetSystems() if hasattr(system, "GetSystems") else []) if isinstance(s, cls)]
        if len(found) != 1:
            raise ValueError(f"found {len(found)} MultibodyPlant(s) in the diagram; pass plant= to attach(simulator=..., plant=...)")
        return found[0]

    def _scan(self) -> None:
        """Bodies, joints and free bodies of the plant, named once: the plant's structure does not change after Finalize."""
        plant = self.plant
        Revolute, Prismatic, Floating = _drake_classes()
        from pydrake.multibody.tree import BodyIndex, JointIndex

        bodies = []
        for i in range(1, plant.num_bodies()):  # body 0 is the world
            body = plant.get_body(BodyIndex(i))
            bodies.append((plant.GetModelInstanceName(body.model_instance()), body.name(), i))
        labels = _qualified([(inst, name) for inst, name, _ in bodies])
        self._body_label = {i: label for label, (_, _, i) in zip(labels, bodies)}
        self._body_index = {label: i for label, (_, _, i) in zip(labels, bodies)}

        joints, free = [], []
        for j in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(j))
            if isinstance(joint, (Revolute, Prismatic)) and joint.num_positions() == 1:
                joints.append((plant.GetModelInstanceName(joint.model_instance()), joint))
            elif isinstance(joint, Floating):
                free.append(joint.child_body().index())
        jlabels = _qualified([(inst, joint.name()) for inst, joint in joints])
        self._joints = {label: joint for label, (_, joint) in zip(jlabels, joints)}
        self.joint_names = list(self._joints)
        self.free_bodies = [self._body_label[int(i)] for i in free]

    # --- the plant's state in the simulator's context ---------------------------------
    def _pc(self):
        return self.plant.GetMyMutableContextFromRoot(self.context)

    def _body(self, index: int):
        from pydrake.multibody.tree import BodyIndex

        return self.plant.get_body(BodyIndex(index))

    # --- naming -----------------------------------------------------------------------
    def names(self, kind: str = "body") -> list[str]:
        if kind == "body":
            return list(self._body_index)
        if kind == "geom" or kind == "site":
            return []  # frames and sites are not named here: see the module docstring
        raise ValueError(f"unknown kind {kind!r}")

    def resolve(self, name: str):
        if name in self._body_index:
            return ("body", self._body_index[name])
        return None

    def label(self, handle) -> str:
        return self._body_label[handle[1]]

    # --- ground truth -----------------------------------------------------------------
    def pose(self, handle) -> tuple[np.ndarray, np.ndarray]:
        X = self.plant.EvalBodyPoseInWorld(self._pc(), self._body(handle[1]))
        return np.asarray(X.translation(), dtype=float), np.asarray(X.rotation().ToQuaternion().wxyz(), dtype=float)

    def velocity(self, handle) -> np.ndarray:
        v = self.plant.EvalBodySpatialVelocityInWorld(self._pc(), self._body(handle[1]))
        return np.concatenate([np.asarray(v.translational(), float), np.asarray(v.rotational(), float)])

    # --- contacts ---------------------------------------------------------------------
    def _raw_contacts(self):
        """``(label a, label b, normal force)`` for every point-pair contact between two different bodies."""
        cr = self.plant.get_contact_results_output_port().Eval(self._pc())
        if cr.num_hydroelastic_contacts():
            raise CapabilityError(
                "this plant has hydroelastic contacts, which the Drake adapter does not read yet: its contact forces would be missing"
            )
        out = []
        for i in range(cr.num_point_pair_contacts()):
            c = cr.point_pair_contact_info(i)
            a, b = int(c.bodyA_index()), int(c.bodyB_index())
            if a == b:
                continue
            normal = np.asarray(c.point_pair().nhat_BA_W, float)
            force = np.asarray(c.contact_force(), float)
            out.append((self._body_label[a], self._body_label[b], abs(float(force @ normal))))
        return out

    def contact_pairs(self) -> list[tuple]:
        return self._raw_contacts()

    def contacts_of(self, handle) -> list[str]:
        me = self.label(handle)
        out = set()
        for a, b, _ in self._raw_contacts():
            if a == me:
                out.add(b)
            elif b == me:
                out.add(a)
        return sorted(out)

    # --- joints and the free bodies -----------------------------------------------------
    def named_joints(self) -> list[str]:
        return list(self._joints)

    def _joint(self, name: str):
        if name not in self._joints:
            raise KeyError(f"no joint named {name!r}; joints: {', '.join(self.joint_names) or 'none'}")
        return self._joints[name]

    def joint_qadr(self, name: str) -> int:
        """The index of the joint's value in the plant's positions (the core's ``joint()`` only checks the name through this)."""
        return int(self._joint(name).position_start())

    def joint_value(self, name: str) -> float:
        return float(self.plant.GetPositions(self._pc())[self._joint(name).position_start()])

    def set_joint(self, name: str, value: float) -> None:
        joint = self._joint(name)
        pc = self._pc()
        q = np.array(self.plant.GetPositions(pc), dtype=float)
        v = np.array(self.plant.GetVelocities(pc), dtype=float)
        q[joint.position_start()] = float(value)
        v[joint.velocity_start()] = 0.0
        self.plant.SetPositions(pc, q)
        self.plant.SetVelocities(pc, v)

    def qpos(self) -> np.ndarray:
        """The named joints' values, in ``joint_names`` order (a free body's pose is not part of it)."""
        return np.array([self.joint_value(n) for n in self.joint_names], dtype=float)

    def qvel(self) -> np.ndarray:
        pc = self._pc()
        v = self.plant.GetVelocities(pc)
        return np.array([float(v[self._joint(n).velocity_start()]) for n in self.joint_names], dtype=float)

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        from pydrake.common.eigen_geometry import Quaternion
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.multibody.math import SpatialVelocity

        if name not in self.free_bodies:
            raise ValueError(f"{name!r} is not a free body, so its pose cannot be set (free bodies: {', '.join(self.free_bodies)})")
        body = self._body(self._body_index[name])
        if quat is None:
            X = self.plant.EvalBodyPoseInWorld(self._pc(), body)
            R = X.rotation()
        else:
            w, x, y, z = (float(c) for c in quat)
            R = RotationMatrix(Quaternion(w, x, y, z))
        pc = self._pc()
        self.plant.SetFreeBodyPose(pc, body, RigidTransform(R, np.asarray(pos, dtype=float)))
        self.plant.SetFreeBodySpatialVelocity(pc, body, SpatialVelocity(np.zeros(3), np.zeros(3)))

    def apply_force(self, name: str, force) -> None:
        """A world-frame force (N) on the body for the physics steps until ``note_step``."""
        if name not in self._body_index:
            raise KeyError(f"no body named {name!r} in this simulation")
        self._forces[self._body_index[name]] = np.asarray(force, dtype=float)
        self._set_forces()

    def _set_forces(self) -> None:
        from pydrake.multibody.math import SpatialForce
        from pydrake.multibody.plant import ExternallyAppliedSpatialForce

        forces = []
        for index, f in self._forces.items():
            ext = ExternallyAppliedSpatialForce()
            ext.body_index = self._body(index).index()
            ext.p_BoBq_B = np.zeros(3)
            ext.F_Bq_W = SpatialForce(np.zeros(3), f)
            forces.append(ext)
        port = self.plant.get_applied_spatial_force_input_port()
        try:
            port.FixValue(self._pc(), forces)
        except RuntimeError as e:
            raise CapabilityError(f"the plant's applied-force input is connected in this diagram, so robowright cannot push: {e}") from e

    def note_step(self) -> None:
        """A force lasts for the control step it was applied to, as a push does: clear them once that step is taken."""
        if self._forces:
            self._forces.clear()
            self._set_forces()

    # --- clock ------------------------------------------------------------------------
    @property
    def time(self) -> float:
        return float(self.context.get_time())

    def trace_meta(self) -> dict:
        # Replay re-runs the steps and the forces robowright applied, but not the inputs the user set between them
        # (actuation ports, controllers wired outside the diagram): the factory must rebuild those itself.
        return {"engine": self.engine, "inputs_recorded": False}
