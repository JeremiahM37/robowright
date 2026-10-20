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
        """Linear then angular velocity: the engine's, or on hardware the change in perceived pose
        since the last read at an earlier time (infinite until there are two)."""
        w = self.world
        if w.has_ground_truth:
            return w.backend.object_velocity(self.name)
        t = w.time
        p, q = self.pose()
        last = w._perceived.get(self.name)
        if last is not None and t - last[0] < 1e-9:
            return last[3]
        if last is None:
            v = np.full(6, np.inf)
        else:
            dt = t - last[0]
            dq = q if np.dot(q, last[2]) >= 0 else -q  # the same turn, either sign
            ang = 2 * np.arccos(np.clip(abs(float(np.dot(dq, last[2]))), 0.0, 1.0)) / dt
            v = np.concatenate([(p - last[1]) / dt, [ang, 0.0, 0.0]])
        w._perceived[self.name] = (t, p, q, v)
        return v

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
        """Every object matching ``filters``, checked against the world as it is now (see :meth:`get`)."""
        unknown = set(filters) - _FILTERS
        if unknown:
            raise TypeError(f"unknown scene filter(s) {sorted(unknown)}; the filters are {sorted(_FILTERS)}")
        out = []
        for o in self.world.spec.objects:
            if _match(o, filters):
                h = ObjectHandle(self.world, o.name)
                if _match_state(self.world, h, filters):
                    out.append(h)
        return out

    def count(self, **filters) -> int:
        """How many objects match ``filters`` now: ``scene.count(inside="bin")``."""
        return len(self.all(**filters))

    def get(self, **filters) -> ObjectHandle:
        """Exactly one object matching ``filters`` (strict, like Playwright's locators). What it is: ``name=``,
        ``color=``, ``kind=``, ``tag=``. Where it is now, as Playwright filters a locator by what it has: ``inside="bin"``
        (its centre within the bin), ``on="tray"`` (resting on top of it), ``touching="floor"``, ``near=(point or
        object, radius)``, ``held=True`` (between the gripper's fingers). What this robot can do with it, as
        Playwright finds an element by its role: ``graspable=True`` (not a bin, and narrower than its gripper opens).
        And ``where=fn`` for anything else (``fn(handle) -> bool``). ``scene.get(color="red", inside="bin")`` is the red
        object in the bin."""
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


_FILTERS = {"color", "kind", "tag", "name", "inside", "on", "touching", "near", "held", "graspable", "where"}
_ON_GAP = 0.01  # m between an object's bottom and the top it rests on


def _match_state(world, h, f) -> bool:
    """The filters on where an object is now; the others are checked by :func:`_match`."""
    if "inside" in f:
        c = as_subject(world, f["inside"])
        if c.name == h.name:
            return False
        lo, hi = c.bounds()
        if not (np.all(h.position >= lo) and np.all(h.position <= hi)):
            return False
    if "on" in f:
        base = as_subject(world, f["on"])
        if base.name == h.name:
            return False
        lo, hi = base.bounds()
        p, bottom = h.position, h.bounds()[0][2]
        if not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and abs(bottom - base.top) <= _ON_GAP):
            return False
    if "touching" in f:
        if f["touching"] not in h.contacts():
            return False
    if "near" in f:
        target, radius = f["near"]
        q = target.position if isinstance(target, Subject) else as_subject(world, target).position
        if float(np.linalg.norm(h.position - np.asarray(q, float))) > radius:
            return False
    if "held" in f:
        gripper = getattr(world.robot, "gripper", None)
        held = gripper is not None and gripper.holding() == h.name
        if held != bool(f["held"]):
            return False
    if "graspable" in f:
        from .robot import grasp_width, widest_gap

        robot = world.robot
        fits = h.spec.kind != "bin" and getattr(robot, "gripper", None) is not None and grasp_width(h.spec) <= widest_gap(robot.model)
        if fits != bool(f["graspable"]):
            return False
    if "where" in f and not f["where"](h):
        return False
    return True


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
