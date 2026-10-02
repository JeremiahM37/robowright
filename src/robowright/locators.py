"""Locators: stable handles to things in the scene.

Like a Playwright locator, a handle is a query, not a snapshot - reading
``cube.position`` always returns the current value, so it is safe to hold a
handle across actions and pass it to ``expect``.
"""

from __future__ import annotations

import numpy as np

from .errors import CapabilityError


class Subject:
    """Anything with a position that ``expect`` can reason about."""

    name: str

    @property
    def position(self) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"


class Point(Subject):
    def __init__(self, xyz, name: str | None = None):
        self._p = np.asarray(xyz, dtype=float)
        self.name = name or f"({', '.join(f'{v:.3f}' for v in self._p)})"

    @property
    def position(self):
        return self._p.copy()


class ObjectHandle(Subject):
    def __init__(self, world, name: str):
        self.world = world
        self.name = name
        self.spec = world.spec.object(name)

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        w = self.world
        if w.has_ground_truth:
            return w.backend.object_pose(self.name)
        if self.name in w.perception:
            return w.perception[self.name]()
        raise CapabilityError(
            f"the {w.backend.name} backend has no ground-truth pose for {self.name!r}; "
            f"register one with world.perception[{self.name!r}] = lambda: (pos, quat)"
        )

    @property
    def position(self) -> np.ndarray:
        return self.pose()[0]

    @property
    def quaternion(self) -> np.ndarray:
        return self.pose()[1]

    @property
    def yaw(self) -> float:
        w, x, y, z = self.quaternion
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @property
    def up_axis(self) -> np.ndarray:
        w, x, y, z = self.quaternion
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    @property
    def velocity(self) -> np.ndarray:
        return self.world.backend.object_velocity(self.name)

    @property
    def top(self) -> float:
        """Height of the object's top surface."""
        if self.spec.kind == "bin":
            return float(self.position[2] + 2 * self.spec.size[2])
        return float(self.position[2] + self.spec.half_height)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds in world frame (bins: the interior volume)."""
        p = self.position
        if self.spec.kind == "bin":
            sx, sy, sz = self.spec.size
            c, s = abs(np.cos(self.yaw)), abs(np.sin(self.yaw))
            hx, hy = c * sx + s * sy, s * sx + c * sy
            return np.array([p[0] - hx, p[1] - hy, p[2]]), np.array([p[0] + hx, p[1] + hy, p[2] + 2 * sz])
        h = np.array(self.spec.size[:3] if self.spec.kind == "box" else [self.spec.size[0]] * 2 + [self.spec.half_height])
        return p - h, p + h

    def contacts(self) -> list[str]:
        """Names of everything currently touching this object."""
        self.world.require("contacts", "contact queries")
        out = set()
        for c in self.world.backend.contacts():
            if c.a == self.name:
                out.add(c.b)
            elif c.b == self.name:
                out.add(c.a)
        return sorted(out)


class SceneLocator:
    def __init__(self, world):
        self.world = world

    def __getitem__(self, name: str) -> ObjectHandle:
        if name not in self.world.object_names:
            raise KeyError(f"no object named {name!r}; scene has {self.world.object_names}")
        return ObjectHandle(self.world, name)

    def __iter__(self):
        return iter(self.all())

    def all(self, **filters) -> list[ObjectHandle]:
        return [ObjectHandle(self.world, o.name) for o in self.world.spec.objects if _match(o, filters)]

    def get(self, **filters) -> ObjectHandle:
        """Exactly one object matching ``color=``, ``kind=``, ``tag=`` filters (strict, like Playwright)."""
        found = self.all(**filters)
        if len(found) != 1:
            raise LookupError(f"expected exactly one object matching {filters}, found {[h.name for h in found]}")
        return found[0]

    def nearest(self, to, **filters) -> ObjectHandle:
        p = to.position if isinstance(to, Subject) else np.asarray(to, float)
        cands = [h for h in self.all(**filters) if h.spec.kind != "bin"]
        if not cands:
            raise LookupError(f"no movable object matches {filters}")
        return min(cands, key=lambda h: float(np.linalg.norm(h.position - p)))


def _match(o, f) -> bool:
    if "color" in f and o.color != f["color"]:
        return False
    if "kind" in f and o.kind != f["kind"]:
        return False
    if "tag" in f and f["tag"] not in o.tags:
        return False
    if "name" in f and o.name != f["name"]:
        return False
    return True


def as_subject(world, x) -> Subject:
    if isinstance(x, Subject):
        return x
    if isinstance(x, str):
        return world.scene[x]
    return Point(x)
