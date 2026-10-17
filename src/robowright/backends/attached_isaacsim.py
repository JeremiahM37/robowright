"""An Isaac Sim world that robowright did not build: a script's own ``isaacsim.core.api.World``, with the robots and
objects it added to ``world.scene``, stepped by its own ``world.step()``. (Isaac Lab environments: attached_isaaclab.)

Everything is read from the world's scene registry, through Isaac Sim's own API:

* Names. Each articulation's links are bodies named ``<articulation>/<link>`` (``franka/panda_hand``); a link name
  only one articulation has also answers alone (``panda_hand``). Rigid objects are bodies by their scene name
  (``cube``), and are the free bodies a test can move and push. Joints are ``<articulation>/<joint>``, and a joint
  name only one articulation has also answers alone.
* Where things are: world frame, quaternions (w, x, y, z), as Isaac Sim's ``get_world_pose`` gives them.
* How fast: from the change in pose over the last step. Isaac Sim's own velocity for a body that has come to rest is
  the one it had as it settled (a cube still on the ground read 0.046 m/s upward for 250 steps, unchanged), so it is
  used only before two steps have been seen.
* Contacts are not read: Isaac Sim reports them only for prims set up to (contact sensors, contact filters), and this
  adapter does not change the user's scene to get them. Asking for them says so.
* The clock is the world's (``world.current_time``); a control step is one ``world.step()``.
* Rendering and saved state are not read here: a run is recorded without a state to redraw or replay it from.
"""

from __future__ import annotations

import numpy as np

from ..errors import CapabilityError
from .attached_engine import AttachedEngine
from .attached_isaaclab import _np, _unique_aliases
from .base import FORCES, GROUND_TRUTH


class IsaacSimEngine(AttachedEngine):
    """Handles are ``("link", articulation, link index)`` or ``("object", name)``."""

    engine = "isaacsim"

    def __init__(self, world, *, seed: int = 0, dt: float | None = None):
        self.world = world
        super().__init__(seed, dt=dt if dt is not None else float(world.get_physics_dt()))
        self.capabilities = frozenset({GROUND_TRUTH, FORCES})
        self._scan()
        self._poses: list = []  # the last two (time, {handle: (pos, quat)}) seen after a step: velocities come from them

    def _scan(self) -> None:
        reg = self.world.scene._scene_registry
        self._arts = {**reg._articulated_systems, **reg._robots}
        self._objects = dict(reg._rigid_objects)
        self._links: dict[str, tuple[str, int]] = {}
        self._joints: dict[str, tuple[str, int]] = {}
        for name, art in self._arts.items():
            for i, link in enumerate(art._articulation_view.body_names or []):
                self._links[f"{name}/{link}"] = (name, i)
            for j, joint in enumerate(art.dof_names or []):
                self._joints[f"{name}/{joint}"] = (name, j)
        self._link_alias = _unique_aliases(list(self._links))
        self._joint_alias = _unique_aliases(list(self._joints))
        self.joint_names = list(self._joints)
        self.free_bodies = list(self._objects)

    # --- naming -----------------------------------------------------------------------
    def names(self, kind: str = "body") -> list[str]:
        if kind == "body":
            return [*self._links, *self._objects]
        if kind in ("geom", "site"):
            return []  # Isaac Sim's scene names links and objects; a test names those
        raise ValueError(f"unknown kind {kind!r}")

    def resolve(self, name: str):
        if name in self._objects:
            return ("object", name)
        full = name if name in self._links else self._link_alias.get(name)
        if full is not None:
            art, i = self._links[full]
            return ("link", art, i)
        if name in self._arts:  # an articulation by its own name: its root link
            return ("link", name, 0)
        return None

    def label(self, handle) -> str:
        if handle[0] == "object":
            return handle[1]
        return f"{handle[1]}/{self._arts[handle[1]]._articulation_view.body_names[handle[2]]}"

    # --- ground truth -----------------------------------------------------------------
    def _physics(self, art_name: str):
        return self._arts[art_name]._articulation_view._physics_view

    def pose(self, handle) -> tuple[np.ndarray, np.ndarray]:
        if handle[0] == "object":
            pos, quat = self._objects[handle[1]].get_world_pose()
            return _np(pos).reshape(3), _np(quat).reshape(4)
        t = _np(self._physics(handle[1]).get_link_transforms())[0, handle[2]]  # x y z, then qx qy qz qw
        return t[:3].copy(), np.array([t[6], t[3], t[4], t[5]])

    def velocity(self, handle) -> np.ndarray:
        if len(self._poses) == 2 and handle in self._poses[0][1] and self._poses[1][0] > self._poses[0][0]:
            (t0, a), (t1, b) = self._poses
            (p0, q0), (p1, q1) = a[handle], b[handle]
            dt = t1 - t0
            return np.concatenate([(p1 - p0) / dt, _angular(q0, q1) / dt])
        return self._reported_velocity(handle)

    def note_step(self) -> None:
        """After each step: every body's pose, for the velocities (see the module)."""
        handles = [("object", n) for n in self._objects] + [("link", a, i) for a, i in self._links.values()]
        if self._poses and self.time < self._poses[-1][0]:
            self._poses = []  # the clock went back (a reset): the last pose is not where the body came from
        self._poses = (self._poses + [(self.time, {h: self.pose(h) for h in handles})])[-2:]

    def _reported_velocity(self, handle) -> np.ndarray:
        if handle[0] == "object":
            o = self._objects[handle[1]]
            return np.concatenate([_np(o.get_linear_velocity()).reshape(3), _np(o.get_angular_velocity()).reshape(3)])
        return _np(self._physics(handle[1]).get_link_velocities())[0, handle[2]].reshape(6).copy()

    # --- contacts ---------------------------------------------------------------------
    def _no_contacts(self):
        raise CapabilityError(
            "contacts are not read from an Isaac Sim world: Isaac Sim reports them only for prims set up to (a contact "
            "sensor or contact filter), and robowright does not change your scene to get them"
        )

    def contact_pairs(self) -> list[tuple]:
        self._no_contacts()

    def contacts_of(self, handle) -> list[str]:
        self._no_contacts()

    # --- joints and the free bodies -----------------------------------------------------
    def named_joints(self) -> list[str]:
        return list(self._joints)

    def _joint(self, name: str):
        full = name if name in self._joints else self._joint_alias.get(name)
        if full is None:
            raise KeyError(f"no joint named {name!r}; joints: {', '.join(self._joints) or 'none'}")
        return self._joints[full]

    def joint_qadr(self, name: str) -> int:
        return self._joint(name)[1]

    def joint_value(self, name: str) -> float:
        art, j = self._joint(name)
        return float(_np(self._arts[art].get_joint_positions()).reshape(-1)[j])

    def set_joint(self, name: str, value: float) -> None:
        art, j = self._joint(name)
        self._arts[art].set_joint_positions(np.array([float(value)]), joint_indices=np.array([j]))
        self._arts[art].set_joint_velocities(np.array([0.0]), joint_indices=np.array([j]))
        self._poses = []  # a jump, not a motion

    def qpos(self) -> np.ndarray:
        return np.array([self.joint_value(n) for n in self.joint_names], dtype=float)

    def qvel(self) -> np.ndarray:
        out = []
        for n in self.joint_names:
            art, j = self._joint(n)
            out.append(float(_np(self._arts[art].get_joint_velocities()).reshape(-1)[j]))
        return np.array(out, dtype=float)

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        if name not in self._objects:
            raise ValueError(f"{name!r} is not a free body, so its pose cannot be set (free bodies: {', '.join(self._objects)})")
        o = self._objects[name]
        q = o.get_world_pose()[1] if quat is None else np.asarray(quat, dtype=float)
        o.set_world_pose(position=np.asarray(pos, dtype=float), orientation=q)
        o.set_linear_velocity(np.zeros(3))
        o.set_angular_velocity(np.zeros(3))
        self._poses = []  # a jump, not a motion

    def apply_force(self, name: str, force) -> None:
        """A world-frame force (N) on a free body, for the physics step being taken."""
        if name not in self._objects:
            free = ", ".join(self._objects) or "none"
            raise CapabilityError(f"pushes reach the free bodies of an Isaac Sim world ({free}), not {name!r}")
        view = self._objects[name]._rigid_prim_view
        view.apply_forces(np.asarray(force, dtype=float).reshape(1, 3), is_global=True)

    # --- clock ------------------------------------------------------------------------
    @property
    def time(self) -> float:
        return float(self.world.current_time)

    def trace_meta(self) -> dict:
        return {"engine": self.engine, "inputs_recorded": False}


def _angular(q0, q1) -> np.ndarray:
    """The rotation from q0 to q1 (w, x, y, z), as an axis times its angle, in the world frame."""
    w0, v0 = q0[0], -np.asarray(q0[1:])  # q0's conjugate
    w1, v1 = q1[0], np.asarray(q1[1:])
    w = w1 * w0 - v1 @ v0
    v = w1 * v0 + w0 * v1 + np.cross(v1, v0)
    if w < 0:  # the shorter way round
        w, v = -w, -v
    n = float(np.linalg.norm(v))
    return np.zeros(3) if n < 1e-12 else v / n * 2.0 * np.arctan2(n, w)


def is_isaacsim_world(obj) -> bool:
    """An Isaac Sim ``World`` (or a subclass): isaacsim.core.api's, with its scene."""
    return hasattr(obj, "scene") and any(c.__module__.startswith("isaacsim.core.api") for c in type(obj).__mro__)


def isaacsim_engine_of(world, *, seed: int = 0, dt: float | None = None) -> IsaacSimEngine | None:
    if not is_isaacsim_world(world):
        return None
    return IsaacSimEngine(world, seed=seed, dt=dt)
