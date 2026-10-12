"""A SAPIEN scene that robowright did not build (ManiSkill 3 environments and bare SAPIEN scenes).

The environment owns a SAPIEN scene and steps it; this adapter reads that scene through the names its
actors and articulations have. ManiSkill keeps every value batched: a pose is a ``(num_envs, 7)`` torch
tensor even with one environment. The adapter takes row 0 and refuses an environment with more than one,
so the test never sees a batch dimension and never picks the wrong world out of a batch.

What SAPIEN gives, and what this adapter says it cannot give:

* Names. Actors are named by their ManiSkill key (``env.scene.actors``). Articulation links are named by
  their URDF link name (``panda_hand_tcp``), and a joint by its URDF joint name.
* The clock. SAPIEN has no simulation time to read. Time is the number of control steps the environment
  took times its control period, ``1 / control_freq``. It restarts at each reset, as ManiSkill's own
  ``elapsed_steps`` does.
* Forces. PhysX clears a force after each physics substep, so one ``apply_force`` acts for one substep.
  A control step is ``control_freq`` substeps of ``sim_freq`` each: the force is applied once, scaled by the
  substep count, the impulse the control step would give. Forces within a step are not reproduced exactly.
* Controls. The environment's own actuators are not read back, so ``ctrl()`` is empty.
"""

from __future__ import annotations

import re

import numpy as np

from ..errors import CapabilityError
from .attached_engine import AttachedEngine
from .base import CONTACTS, FORCES, GROUND_TRUTH

_SCENE_PREFIX = re.compile(r"^scene-\d+_")


def _label(entity_name: str) -> str:
    """The name the environment gave a body. ManiSkill prefixes an actor's SAPIEN entity with its scene (``scene-0_cube``)
    and leaves an articulation link's entity as the link name, so a contact must drop the prefix to match a name."""
    return _SCENE_PREFIX.sub("", entity_name)


class _Pose:
    """ManiSkill's pose holder: the raw (x, y, z, w, x, y, z) row the adapter reads, as a batched tensor would be."""

    def __init__(self, raw):
        self.raw_pose = raw


def _raw_of(pose) -> np.ndarray:
    """A SAPIEN 3 pose (position, then wxyz quaternion) as the raw row ManiSkill's pose holds."""
    return np.concatenate([np.asarray(pose.p, dtype=float).reshape(-1), np.asarray(pose.q, dtype=float).reshape(-1)])


def _is_sapien3_scene(obj) -> bool:
    """A plain SAPIEN 3 ``Scene``, as SAPIEN's own examples build it: actors and articulations are listed, not held as dicts."""
    return all(hasattr(obj, a) for a in ("get_all_actors", "get_all_articulations", "physx_system"))


class _Sapien3Actor:
    """A rigid body of a SAPIEN 3 scene, with the reads and writes the adapter makes of a ManiSkill actor."""

    def __init__(self, entity):
        import sapien.physx as px

        self._px = px
        self.entity = entity

    @property
    def name(self) -> str:
        return self.entity.name

    def _dynamic(self):
        return self.entity.find_component_by_type(self._px.PhysxRigidDynamicComponent)

    @property
    def px_body_type(self) -> str:
        dyn = self._dynamic()
        if dyn is None:
            return "static"
        return "kinematic" if dyn.kinematic else "dynamic"

    @property
    def pose(self) -> _Pose:
        return _Pose(_raw_of(self.entity.pose))

    @property
    def linear_velocity(self) -> np.ndarray:
        dyn = self._dynamic()
        return np.zeros(3) if dyn is None else np.asarray(dyn.linear_velocity, dtype=float)

    @property
    def angular_velocity(self) -> np.ndarray:
        dyn = self._dynamic()
        return np.zeros(3) if dyn is None else np.asarray(dyn.angular_velocity, dtype=float)

    def apply_force(self, force) -> None:
        dyn = self._dynamic()
        dyn.add_force_at_point(np.asarray(force, dtype=np.float32), np.asarray(self.entity.pose.p, dtype=np.float32))

    def set_pq(self, pos, quat) -> None:
        from sapien import Pose

        self.entity.set_pose(Pose(p=np.asarray(pos, dtype=float), q=np.asarray(quat, dtype=float)))

    def set_zero_velocity(self) -> None:
        dyn = self._dynamic()
        if dyn is not None:
            dyn.set_linear_velocity(np.zeros(3, dtype=np.float32))
            dyn.set_angular_velocity(np.zeros(3, dtype=np.float32))


class _Sapien3Link:
    """A link of a SAPIEN 3 articulation (its component carries the pose and velocity)."""

    def __init__(self, comp):
        self.comp = comp

    @property
    def name(self) -> str:
        return self.comp.name

    @property
    def pose(self) -> _Pose:
        return _Pose(_raw_of(self.comp.get_entity_pose()))

    @property
    def linear_velocity(self) -> np.ndarray:
        return np.asarray(self.comp.linear_velocity, dtype=float)

    @property
    def angular_velocity(self) -> np.ndarray:
        return np.asarray(self.comp.angular_velocity, dtype=float)


class _Sapien3Articulation:
    """An articulation of a SAPIEN 3 scene. Its joint values are one row, as ManiSkill's batched ones are, read with ``_row``."""

    def __init__(self, art, name: str):
        self.art = art
        self.name = name

    @property
    def links(self) -> list[_Sapien3Link]:
        return [_Sapien3Link(c) for c in self.art.get_links()]

    @property
    def active_joints(self) -> list:
        return list(self.art.get_active_joints())

    @property
    def qpos(self) -> np.ndarray:
        return np.asarray(self.art.get_qpos(), dtype=float).reshape(-1)

    @property
    def qvel(self) -> np.ndarray:
        return np.asarray(self.art.get_qvel(), dtype=float).reshape(-1)

    def set_joint_value(self, i: int, value: float) -> None:
        q = self.qpos.copy()
        q[i] = float(value)
        self.art.set_qpos(q)
        qv = self.qvel.copy()
        qv[i] = 0.0
        self.art.set_qvel(qv)


class _Sapien3View:
    """A plain SAPIEN 3 scene read through the three names ManiSkill's scene has: ``px``, ``actors`` and ``articulations``.

    SAPIEN 3 lists its actors and articulations rather than keeping them by name, so they are named here: an actor by its
    entity name, an articulation by its own name or its root link, and a repeated name by a suffix. Taken once, when the
    engine binds, so a handle keeps meaning the same thing.
    """

    def __init__(self, scene):
        self.px = scene.physx_system
        self.actors: dict[str, _Sapien3Actor] = {}
        for i, entity in enumerate(scene.get_all_actors()):
            self.actors[_unique(entity.name or f"actor{i}", self.actors)] = _Sapien3Actor(entity)
        self.articulations: dict[str, _Sapien3Articulation] = {}
        for i, art in enumerate(scene.get_all_articulations()):
            base = art.name or (art.get_root().name if art.get_links() else "") or f"articulation{i}"
            self.articulations[_unique(base, self.articulations)] = _Sapien3Articulation(art, base)


def _unique(name: str, taken) -> str:
    if name not in taken:
        return name
    k = 2
    while f"{name}#{k}" in taken:
        k += 1
    return f"{name}#{k}"


def _row(t) -> np.ndarray:
    """A batched torch tensor (or array) of one environment, as a float64 numpy row."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t, dtype=float).reshape(-1)


class SapienEngine(AttachedEngine):
    """Handles are ``("actor", name)`` or ``("link", articulation name, link name)``.

    Handles are names, not SAPIEN objects, so they still resolve after the environment rebuilds its actors
    on a reset (ManiSkill reconfigures some tasks that way).
    """

    engine = "sapien"
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, FORCES})

    def __init__(self, env, *, seed: int = 0, dt: float | None = None):
        self.core = getattr(env, "unwrapped", env)
        self._bare = _is_sapien3_scene(self.core)  # a plain SAPIEN 3 scene, which is stepped by its own loop
        self._bare_view = _Sapien3View(self.core) if self._bare else None
        num_envs = 1 if self._bare else int(getattr(self.core, "num_envs", 1))
        if num_envs != 1:
            raise CapabilityError(
                f"attach reads one SAPIEN scene, and this environment runs {num_envs} in parallel: build it with num_envs=1"
            )
        if self._bare:
            self._physics_dt = float(self.core.timestep)
            control_dt = self._physics_dt  # one control step is one scene step unless the caller says otherwise
        else:
            self._physics_dt = 1.0 / float(self.core.sim_freq)
            control_dt = 1.0 / float(self.core.control_freq)
        super().__init__(seed, dt=dt if dt is not None else control_dt)
        self._steps = 0
        self._refresh()

    def _refresh(self) -> None:
        """The free bodies and joints of the scene as it is now."""
        scene = self._scene
        self.free_bodies = [name for name, a in scene.actors.items() if a.px_body_type == "dynamic"]
        self.joint_names = [j.name for art in scene.articulations.values() for j in art.active_joints]

    @property
    def _scene(self):
        return self._bare_view if self._bare else self.core.scene

    # --- naming ---------------------------------------------------------------------
    def names(self, kind: str = "body") -> list[str]:
        if kind == "body":
            links = [link.name for art in self._scene.articulations.values() for link in art.links]
            return list(self._scene.actors) + [n for n in links if n not in self._scene.actors]
        if kind == "geom" or kind == "site":
            return []  # SAPIEN has collision shapes without names of their own; a site is an actor here
        raise ValueError(f"unknown kind {kind!r}")

    def resolve(self, name: str):
        if name in self._scene.actors:
            return ("actor", name)
        for art_name, art in self._scene.articulations.items():
            for link in art.links:
                if link.name == name:
                    return ("link", art_name, name)
        return None

    def label(self, handle) -> str:
        return handle[1] if handle[0] == "actor" else handle[2]

    def _obj(self, handle):
        if handle[0] == "actor":
            return self._scene.actors[handle[1]]
        for link in self._scene.articulations[handle[1]].links:
            if link.name == handle[2]:
                return link
        raise KeyError(f"no link named {handle[2]!r} in articulation {handle[1]!r}")

    # --- ground truth -----------------------------------------------------------------
    def pose(self, handle) -> tuple[np.ndarray, np.ndarray]:
        raw = _row(self._obj(handle).pose.raw_pose)  # ManiSkill's raw pose is (x, y, z, w, x, y, z)
        return raw[:3].copy(), raw[3:7].copy()

    def velocity(self, handle) -> np.ndarray:
        obj = self._obj(handle)
        return np.concatenate([_row(obj.linear_velocity), _row(obj.angular_velocity)])

    # --- contacts ---------------------------------------------------------------------
    def _raw_contacts(self):
        """``(label a, label b, normal force)`` for every PhysX contact record between two different bodies."""
        out = []
        for c in self._scene.px.get_contacts():
            a, b = _label(c.bodies[0].entity.name), _label(c.bodies[1].entity.name)
            if a == b:
                continue
            normal_impulse = sum(float(np.dot(_row(p.impulse), _row(p.normal))) for p in c.points)
            out.append((a, b, abs(normal_impulse) / self._physics_dt))
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
    def _joint(self, name: str):
        for art in self._scene.articulations.values():
            for i, j in enumerate(art.active_joints):
                if j.name == name:
                    return art, i
        raise KeyError(f"no joint named {name!r}; joints: {', '.join(self.joint_names) or 'none'}")

    def joint_qadr(self, name: str) -> int:
        """The index of a joint's value in ``qpos()``: the articulations' active joints, in order."""
        offset = 0
        for art in self._scene.articulations.values():
            for i, j in enumerate(art.active_joints):
                if j.name == name:
                    return offset + i
            offset += len(art.active_joints)
        raise KeyError(f"no joint named {name!r}; joints: {', '.join(self.joint_names) or 'none'}")

    def joint_value(self, name: str) -> float:
        art, i = self._joint(name)
        return float(_row(art.qpos)[i])

    def set_joint(self, name: str, value: float) -> None:
        art, i = self._joint(name)
        if isinstance(art, _Sapien3Articulation):
            art.set_joint_value(i, value)
            return
        q = art.qpos.clone()
        q[0, i] = float(value)
        art.set_qpos(q)
        qv = art.qvel.clone()
        qv[0, i] = 0.0
        art.set_qvel(qv)

    def qpos(self) -> np.ndarray:
        return np.concatenate([_row(art.qpos) for art in self._scene.articulations.values()] or [np.zeros(0)])

    def qvel(self) -> np.ndarray:
        return np.concatenate([_row(art.qvel) for art in self._scene.articulations.values()] or [np.zeros(0)])

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        if name not in self.free_bodies:
            raise ValueError(f"{name!r} is not a free body, so its pose cannot be set (free bodies: {', '.join(self.free_bodies)})")
        actor = self._scene.actors[name]
        if quat is None:
            quat = _row(actor.pose.raw_pose)[3:7]
        if isinstance(actor, _Sapien3Actor):
            actor.set_pq(pos, quat)
            actor.set_zero_velocity()
            return
        from mani_skill.utils.structs.pose import Pose

        p = np.asarray(pos, dtype=np.float32)[None]
        q = np.asarray(quat, dtype=np.float32)[None]
        actor.set_pose(Pose.create_from_pq(p=p, q=q))
        zero = np.zeros((1, 3), dtype=np.float32)
        actor.set_linear_velocity(zero)
        actor.set_angular_velocity(zero)

    def apply_force(self, name: str, force) -> None:
        """The force for one control step, delivered as the impulse that step would give.

        PhysX clears a force after each substep (measured: a 1 N force on a 0.064 kg cube changed its velocity by
        F * sim_dt / m and no more), so the force goes in once, scaled by the substeps in one control step.
        """
        actor = self._scene.actors.get(name)
        if actor is None:
            raise KeyError(f"no body named {name!r} in this simulation")
        n = max(1, int(round(self.control_dt / self._physics_dt)))
        actor.apply_force(np.asarray(force, dtype=np.float32) * n)

    # --- clock ----------------------------------------------------------------------
    @property
    def time(self) -> float:
        return self._steps * self.control_dt

    def note_step(self) -> None:
        self._steps += 1

    def rebind(self, env) -> None:
        """Called at every reset: the clock restarts as ManiSkill's elapsed steps do, and names are read again."""
        self.core = getattr(env, "unwrapped", env)
        self._steps = 0
        self._refresh()


def sapien_engine_of(env, *, seed: int = 0, dt: float | None = None) -> SapienEngine | None:
    """The SAPIEN adapter for an environment with a SAPIEN scene (ManiSkill's ``env.unwrapped.scene``), or None."""
    core = getattr(env, "unwrapped", env)
    scene = getattr(core, "scene", None)
    if scene is not None and all(hasattr(scene, attr) for attr in ("px", "actors", "articulations")):
        return SapienEngine(env, seed=seed, dt=dt)
    if _is_sapien3_scene(core):  # a plain SAPIEN 3 scene, as its own examples build one
        return SapienEngine(core, seed=seed, dt=dt)
    return None
