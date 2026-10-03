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
    robot: str = "so101"  # a name from robowright.robots
    objects: list[ObjectSpec] = field(default_factory=list)
    cameras: list[CameraSpec] = field(default_factory=lambda: [CameraSpec("front")])
    control_hz: float = 50.0
    physics_dt: float | None = None  # None: the robot model's own timestep

    @property
    def robot_model(self):
        from . import robots

        return robots.get(self.robot)

    @property
    def dt(self) -> float:
        return self.physics_dt or self.robot_model.timestep

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


def tabletop(*objects: ObjectSpec, cameras=None, robot: str = "so101", **kw) -> SceneSpec:
    """The default tabletop. With no objects: a red cube and a blue bin.

    The layout is the same for every robot; each robot is mounted where its
    top-down workspace covers it (see ``robowright.robots``), and the default
    camera is pulled back to frame the whole arm.
    """
    if not objects:
        objects = (
            ObjectSpec("cube", "box", (0.0125, 0.0125, 0.0125), (0.22, -0.06, None), color="red"),
            ObjectSpec("bin", "bin", (0.05, 0.05, 0.02), (0.2, 0.12, 0.0), color="blue", mass=0.0),
        )
    spec = SceneSpec(robot=robot, objects=list(objects), **kw)
    if cameras is not None:
        spec.cameras = list(cameras)
    else:
        spec.cameras = [default_camera(robot)]
    return spec


def open_floor(*objects: ObjectSpec, cameras=None, robot: str = "go2", **kw) -> SceneSpec:
    """An empty floor for mobile robots, with the camera framing the robot."""
    spec = SceneSpec(robot=robot, objects=list(objects), **kw)
    spec.cameras = list(cameras) if cameras is not None else [default_camera(robot)]
    return spec


def default_scene(robot: str = "so101") -> SceneSpec:
    from . import robots

    return open_floor(robot=robot) if robots.get(robot).family == "legged" else tabletop(robot=robot)


def default_camera(robot: str = "so101") -> CameraSpec:
    """A front-three-quarter view of the task area that keeps the whole arm in frame."""
    from . import robots

    m = robots.get(robot)
    if m.family == "legged":
        h = m.stand_height
        if "humanoid" in m.tags:  # tall and narrow: a level view of the whole height
            k = h / 0.38
            return CameraSpec("front", pos=(0.55 * k, -0.75 * k, 0.9 * h), lookat=(0.0, 0.0, 0.7 * h), fovy=45.0)
        k = max(h, 0.3) / 0.3
        return CameraSpec("front", pos=(0.55 * k, -0.75 * k, 0.45 * k), lookat=(0.0, 0.0, 0.6 * h), fovy=45.0)
    if robot == "so101":
        return CameraSpec("front")
    # Fit the robot (in its home pose) and the task area into the same three-quarter view.
    import numpy as np

    from .robot import _kinematics, home_q

    kin = _kinematics(robot)
    kin.fk(home_q(robot))
    pts = np.vstack([kin.d.xpos[1:], [[0.1, -0.12, 0.0], [0.32, 0.2, 0.06]]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) / 2
    view = np.array([0.43, -0.44, 0.35])
    view /= np.linalg.norm(view)
    dist = 1.08 * radius / np.tan(np.radians(45.0) / 2)
    pos = centre + view * dist
    return CameraSpec("front", pos=tuple(float(x) for x in pos), lookat=tuple(float(x) for x in centre), fovy=45.0)
