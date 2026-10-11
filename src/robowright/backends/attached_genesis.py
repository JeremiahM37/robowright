"""A Genesis scene that robowright did not build: a ``genesis.Scene`` that its own code built and steps.

The environment owns the scene and calls ``scene.step()``; this adapter reads it through its entities and links.
Every link is a body, named by its link name (``hand``, ``box_baselink``). A name two entities share is qualified
as ``entity::link``. An entity's name also works as the name of its base link, so ``cube`` is ``box_baselink``.

What Genesis gives, and what this adapter says it cannot give:

* Names. Entities keep the name they were added with (``add_entity(..., name=...)``). A joint is named by its
  URDF or MJCF name (``joint1``, ``finger_joint1``); only hinge and slide joints, which have one value, are named.
* Contacts. ``get_contacts`` gives each contact's normal and the force on geom A, and this adapter reads the normal
  component of that force, which is what holding is judged by. It is read from the most recent ``scene.step()``.
  A contact is seen by both entities that touch, so each one is counted once.
* Forces. ``apply_links_external_wrench`` acts for one simulation step and is then cleared (measured: a force
  applied before one step changed the velocity by F * dt / m, and nothing after it). A push is therefore applied
  before each step it covers, as the core does.
* The clock. ``scene.get_time()``, which Genesis keeps and advances.
* Rendering and saved state are not read here. Genesis draws through its own viewer and cameras; a trace of a
  Genesis run is recorded without a state to redraw it from.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .attached_engine import AttachedEngine
from .base import CONTACTS, FORCES, GROUND_TRUTH


def _np(x) -> np.ndarray:
    """A Genesis tensor (or array) as a float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _qualified(items: list[tuple[str, str]]) -> list[str]:
    """Labels for ``(entity, name)`` pairs: the bare name where it is unique, else ``entity::name``."""
    counts = Counter(name for _, name in items)
    return [name if counts[name] == 1 else f"{ent}::{name}" for ent, name in items]


def _is_free(entity) -> bool:
    """A free entity (a box, a cup) moves with six degrees of freedom; a fixed one (the robot's base, the ground) does not."""
    from genesis import JOINT_TYPE

    return any(int(j.type) == int(JOINT_TYPE.FREE) for j in entity.joints)


class GenesisEngine(AttachedEngine):
    """Handles are ``("link", entity index, link index)``: a link of an entity, in the order the scene holds them."""

    engine = "genesis"
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, FORCES})

    def __init__(self, scene, *, seed: int = 0, dt: float | None = None):
        self.scene = scene
        self._entities = list(scene.entities)
        super().__init__(seed, dt=dt if dt is not None else float(scene.dt))
        self._restore = lambda: None  # set by attach: puts the scene's own step and reset back on close
        self._scan()

    def _scan(self) -> None:
        ents = self._entities
        link_items = [(e.name, link.name) for e in ents for link in e.links]
        labels = iter(_qualified(link_items))
        self._link_label: dict[tuple[int, int], str] = {}
        self._by_label: dict[str, tuple[int, int]] = {}
        self._label_of_global: dict[int, str] = {}
        for ei, e in enumerate(ents):
            for li, link in enumerate(e.links):
                label = next(labels)
                self._link_label[(ei, li)] = label
                self._by_label[label] = (ei, li)
                self._label_of_global[int(link.idx)] = label
        self._alias = {e.name: (ei, _base_index(e)) for ei, e in enumerate(ents) if e.links}

        joints = [(e.name, j, ei) for ei, e in enumerate(ents) for j in e.joints if self._one_value(j)]
        jlabels = _qualified([(ent, j.name) for ent, j, _ in joints])
        self._joints = {label: (ei, j) for label, (_, j, ei) in zip(jlabels, joints)}
        self.joint_names = list(self._joints)
        self.free_bodies = [e.name for e in ents if e.links and _is_free(e)]

    @staticmethod
    def _one_value(joint) -> bool:
        from genesis import JOINT_TYPE

        return int(joint.type) in (int(JOINT_TYPE.REVOLUTE), int(JOINT_TYPE.PRISMATIC)) and len(joint.dofs_idx_local) == 1

    # --- naming -----------------------------------------------------------------------
    def names(self, kind: str = "body") -> list[str]:
        if kind == "body":
            return list(self._by_label)
        if kind == "geom" or kind == "site":
            return []  # Genesis geoms are not named by the scene: a link is the body a test names
        raise ValueError(f"unknown kind {kind!r}")

    def resolve(self, name: str):
        if name in self._by_label:
            return ("link", *self._by_label[name])
        if name in self._alias:
            return ("link", *self._alias[name])
        return None

    def label(self, handle) -> str:
        return self._link_label[(handle[1], handle[2])]

    def _link(self, handle):
        return self._entities[handle[1]].links[handle[2]]

    # --- ground truth -----------------------------------------------------------------
    def pose(self, handle) -> tuple[np.ndarray, np.ndarray]:
        link = self._link(handle)
        return _np(link.get_pos()).reshape(3), _np(link.get_quat()).reshape(4)  # Genesis quaternions are (w, x, y, z)

    def velocity(self, handle) -> np.ndarray:
        link = self._link(handle)
        return np.concatenate([_np(link.get_vel()).reshape(3), _np(link.get_ang()).reshape(3)])

    # --- contacts ---------------------------------------------------------------------
    def _raw_contacts(self):
        """``(label a, label b, normal force)`` for every contact between two different links, each counted once."""
        seen: set = set()
        out = []
        for e in self._entities:
            c = e.get_contacts()
            if not len(c["geom_a"]):
                continue
            pos, normal, force_a = _np(c["position"]), _np(c["normal"]), _np(c["force_a"])
            ga, gb = _np(c["geom_a"]).astype(int), _np(c["geom_b"]).astype(int)
            la, lb = _np(c["link_a"]).astype(int), _np(c["link_b"]).astype(int)
            for k in range(len(ga)):
                key = (int(ga[k]), int(gb[k]), tuple(np.round(pos[k], 9)))
                if key in seen:
                    continue
                seen.add(key)
                a, b = self._label_of_global[int(la[k])], self._label_of_global[int(lb[k])]
                if a == b:
                    continue
                out.append((a, b, abs(float(force_a[k] @ normal[k]))))
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
        """The entity-local dof index of the joint (the core's ``joint()`` only checks the name through this)."""
        _, j = self._joint(name)
        return int(j.dofs_idx_local[0])

    def joint_value(self, name: str) -> float:
        ei, j = self._joint(name)
        return float(_np(self._entities[ei].get_dofs_position(dofs_idx_local=j.dofs_idx_local)).reshape(-1)[0])

    def set_joint(self, name: str, value: float) -> None:
        ei, j = self._joint(name)
        self._entities[ei].set_dofs_position(np.array([float(value)]), dofs_idx_local=j.dofs_idx_local)

    def qpos(self) -> np.ndarray:
        return np.array([self.joint_value(n) for n in self.joint_names], dtype=float)

    def qvel(self) -> np.ndarray:
        out = []
        for n in self.joint_names:
            ei, j = self._joint(n)
            out.append(float(_np(self._entities[ei].get_dofs_velocity(dofs_idx_local=j.dofs_idx_local)).reshape(-1)[0]))
        return np.array(out, dtype=float)

    def _free_entity(self, name: str):
        if name not in self.free_bodies:
            raise ValueError(f"{name!r} is not a free body, so its pose cannot be set (free bodies: {', '.join(self.free_bodies)})")
        return self._entities[next(i for i, e in enumerate(self._entities) if e.name == name)]

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        e = self._free_entity(name)
        e.set_pos(np.asarray(pos, dtype=float))
        if quat is not None:
            e.set_quat(np.asarray(quat, dtype=float))
        e.zero_all_dofs_velocity()

    def apply_force(self, name: str, force) -> None:
        """A world-frame force (N) at the base link's origin, for the next simulation step only (Genesis clears it)."""
        found = self.resolve(name)
        if found is None:
            raise KeyError(f"no body named {name!r} in this simulation")
        _, ei, li = found
        entity = self._entities[ei]
        entity.apply_links_external_wrench(force=np.asarray(force, dtype=float), links_idx_local=[entity.links[li].idx_local])

    # --- clock ------------------------------------------------------------------------
    @property
    def time(self) -> float:
        return float(_np(self.scene.get_time()))

    def close(self) -> None:
        self._restore()
        super().close()


def _base_index(entity) -> int:
    """The index, among its links, of the entity's base link: the one an entity's pose and push refer to."""
    base = int(entity.base_link.idx)
    return next(i for i, link in enumerate(entity.links) if int(link.idx) == base)


def genesis_engine_of(scene, *, seed: int = 0, dt: float | None = None) -> GenesisEngine | None:
    """The Genesis adapter for a ``genesis.Scene`` that is built, or None."""
    if not type(scene).__module__.startswith("genesis."):
        return None
    return GenesisEngine(scene, seed=seed, dt=dt)
