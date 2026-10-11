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
        num_envs = int(getattr(self.core, "num_envs", 1))
        if num_envs != 1:
            raise CapabilityError(
                f"attach reads one SAPIEN scene, and this environment runs {num_envs} in parallel: build it with num_envs=1"
            )
        self._physics_dt = 1.0 / float(self.core.sim_freq)
        control_dt = 1.0 / float(self.core.control_freq)
        super().__init__(seed, dt=dt if dt is not None else control_dt)
        self._steps = 0
        self._refresh()

    def _refresh(self) -> None:
        """The free bodies and joints of the scene as it is now."""
        scene = self.core.scene
        self.free_bodies = [name for name, a in scene.actors.items() if a.px_body_type == "dynamic"]
        self.joint_names = [j.name for art in scene.articulations.values() for j in art.active_joints]

    @property
    def _scene(self):
        return self.core.scene

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
        from mani_skill.utils.structs.pose import Pose

        actor = self._scene.actors[name]
        if quat is None:
            quat = _row(actor.pose.raw_pose)[3:7]
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
    if scene is None or not all(hasattr(scene, attr) for attr in ("px", "actors", "articulations")):
        return None
    return SapienEngine(env, seed=seed, dt=dt)
