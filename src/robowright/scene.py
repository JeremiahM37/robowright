"""Backend-neutral scene description.

A ``SceneSpec`` is plain data: every backend builds the same world from it,
and traces store it so a run can be rebuilt exactly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

COLORS = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.7, 0.25, 1.0),
    "blue": (0.15, 0.35, 0.85, 1.0),
    "yellow": (0.95, 0.8, 0.1, 1.0),
    "orange": (0.95, 0.5, 0.1, 1.0),
    "purple": (0.55, 0.25, 0.75, 1.0),
    "white": (0.92, 0.92, 0.92, 1.0),
    "gray": (0.5, 0.5, 0.5, 1.0),
}


@dataclass
class ObjectSpec:
    """A rigid object. ``kind`` is ``box``, ``cylinder``, ``sphere`` or ``bin``.

    ``size`` is half-extents for boxes and bins (x, y, z), ``(radius, half_height)``
    for cylinders and ``(radius,)`` for spheres. ``yaw`` is about world z.
    Bins are static open-topped containers; everything else is free to move.
    """

    name: str
    kind: str = "box"
    size: tuple = (0.0125, 0.0125, 0.0125)
    pos: tuple = (0.2, 0.0, None)
    yaw: float = 0.0
    color: str = "red"
    mass: float = 0.03
    friction: float = 1.0
    tags: tuple = ()

    @property
    def rgba(self) -> tuple:
        return COLORS.get(self.color, COLORS["gray"])

    @property
    def half_height(self) -> float:
        if self.kind in ("box", "bin"):
            return self.size[2]
        if self.kind == "cylinder":
            return self.size[1]
        return self.size[0]

    @property
    def initial_pos(self) -> tuple:
        x, y, z = self.pos
        return (x, y, self.half_height if z is None else z)

    @property
    def static(self) -> bool:
        return self.kind == "bin"

    @property
    def quat(self) -> tuple:
        return (math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2))


@dataclass
class CameraSpec:
    name: str
    pos: tuple = (0.6, -0.42, 0.42)
    lookat: tuple = (0.17, 0.02, 0.07)
    fovy: float = 45.0
    width: int = 320
    height: int = 240


@dataclass
class SceneSpec:
    robot: str = "so101"
    objects: list[ObjectSpec] = field(default_factory=list)
    cameras: list[CameraSpec] = field(default_factory=lambda: [CameraSpec("front")])
    control_hz: float = 50.0
    physics_dt: float = 0.005

    def object(self, name: str) -> ObjectSpec:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SceneSpec:
        d = dict(d)
        d["objects"] = [
            ObjectSpec(**{**o, "size": tuple(o["size"]), "pos": tuple(o["pos"]), "tags": tuple(o.get("tags", ()))}) for o in d["objects"]
        ]
        d["cameras"] = [CameraSpec(**{**c, "pos": tuple(c["pos"]), "lookat": tuple(c["lookat"])}) for c in d["cameras"]]
        return cls(**d)


def tabletop(*objects: ObjectSpec, cameras=None, **kw) -> SceneSpec:
    """The default SO-101 tabletop. With no arguments: a red cube and a blue bin."""
    if not objects:
        objects = (
            ObjectSpec("cube", "box", (0.0125, 0.0125, 0.0125), (0.22, -0.06, None), color="red"),
            ObjectSpec("bin", "bin", (0.05, 0.05, 0.02), (0.2, 0.12, 0.0), color="blue", mass=0.0),
        )
    spec = SceneSpec(objects=list(objects), **kw)
    if cameras is not None:
        spec.cameras = list(cameras)
    return spec
