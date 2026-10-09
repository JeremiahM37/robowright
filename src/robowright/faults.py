"""Fault injection: the things that go wrong on real robots, on demand.

Faults are seeded from the world's RNG, so a failing run with a given seed
fails the same way again, and they are written to the trace so replay and
codegen reproduce them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# How far ``randomize_scene`` (and so every ``trials`` run) moves each free object: up to 1.5 cm
# along each table axis, and any turn about z up to 45 degrees, which for a cube is every yaw
# it can have.
SCENE_XY = 0.015
SCENE_YAW = np.pi / 4


@dataclass
class Fault:
    def describe(self) -> dict:
        return {"type": type(self).__name__, **self.__dict__}


@dataclass
class JointNoise(Fault):
    """Gaussian noise (radians; a third of it in metres on a sliding joint) on every joint reading the robot sees."""

    std: float = 0.01


@dataclass
class ActionDelay(Fault):
    """Commands reach the motors ``steps`` control periods late."""

    steps: int = 2


@dataclass
class Push(Fault):
    """A world-frame force (N) on an object from ``at`` for ``duration`` seconds."""

    object: str = ""
    force: tuple = (0.0, 0.0, 0.0)
    at: float = 0.0
    duration: float = 0.1


@dataclass
class WeakJoint(Fault):
    """Scale a joint's position gain, like a tired or misconfigured servo."""

    joint: str = "shoulder_lift"
    scale: float = 0.5


@dataclass
class CameraDropout(Fault):
    """Each policy camera frame is blacked out with probability ``p``."""

    p: float = 0.1


class Faults:
    def __init__(self, world):
        self.world = world
        self.active: list[Fault] = []
        self._delay: deque | None = None
        # One seeded stream per source of randomness. Drawn from one shared stream, a jitter's
        # draws shifted every later sensor-noise sample, and a test regenerated from a trace
        # (which places the jittered object rather than re-jittering it) read different noise.
        seed = world.seed
        self._rng = {k: np.random.default_rng([seed, i]) for i, k in enumerate(("jitter", "joint_noise", "camera_dropout", "scene"), 1)}
        self._fresh = {k: r.bit_generator.state for k, r in self._rng.items()}
        self._fresh_world = world.rng.bit_generator.state
        # name -> (the pose the scene gave it, the pose randomize_scene moved it to)
        self._randomized: dict = {}

    def _add(self, f: Fault) -> Fault:
        self.active.append(f)
        if self.world.trace:
            self.world.trace.event("fault", type(f).__name__, f.describe())
        return f

    # public API --------------------------------------------------------------
    def joint_noise(self, std: float = 0.01):
        return self._add(JointNoise(std))

    def action_delay(self, steps: int = 2):
        # Pre-fill with the command already in flight, so the first new target
        # arrives ``steps`` periods late like every later one.
        self._delay = deque(self.world.backend.ctrl().copy() for _ in range(steps))
        return self._add(ActionDelay(steps))

    def push(self, object: str, force, at: float | None = None, duration: float = 0.1):
        at = self.world.time if at is None else at
        return self._add(Push(object, tuple(map(float, force)), float(at), float(duration)))

    def weak_joint(self, joint: str, scale: float = 0.5):
        """Scale an arm joint's servo stiffness (a worn gearbox, a motor overheating)."""
        names = list(self.world.backend.robot_model.arm_joints)
        if joint not in names:
            raise ValueError(f"weak_joint takes an arm joint; {self.world.backend.robot_model.title}'s are: {', '.join(names)}")
        self.world.backend.set_gain_scale(joint, scale)
        return self._add(WeakJoint(joint, scale))

    def camera_dropout(self, p: float = 0.1):
        return self._add(CameraDropout(p))

    def randomize_scene(self, xy: float = SCENE_XY, yaw: float = SCENE_YAW) -> dict:
        """Move every free object resting on the table to a seeded random pose near where the
        scene put it: up to ``xy`` metres along each table axis, turned up to ``yaw`` radians.

        ``@pytest.mark.trials`` calls this before every trial, so its trials are different runs
        rather than copies of one. Bins stay put, objects stacked on, under or inside another are
        left alone, and no object is moved closer to another than it already was or than their
        footprints allow. Returns ``{name: new position}``.
        """
        rng = self._rng["scene"]
        objs = self.world.spec.objects
        start = {o.name: self.world.backend.object_pose(o.name) for o in objs}
        xy0 = {n: np.asarray(p[:2], float) for n, (p, _) in start.items()}
        touching = {  # stacked, or one inside the other: moving either would upset both
            n
            for a in objs
            for b in objs
            if a is not b and np.linalg.norm(xy0[a.name] - xy0[b.name]) < _footprint(a) + _footprint(b)
            for n in (a.name, b.name)
        }
        movable = [o for o in objs if not o.static and o.mass > 0 and o.name not in touching]
        placed = {n: v for n, v in xy0.items()}
        moved = {}
        for o in movable:
            for _ in range(50):
                cand = xy0[o.name] + rng.uniform(-xy, xy, 2)
                if all(
                    np.linalg.norm(cand - placed[b.name]) >= min(np.linalg.norm(xy0[o.name] - xy0[b.name]), _footprint(o) + _footprint(b))
                    for b in objs
                    if b is not o
                ):
                    break
            else:
                continue  # crowded: leave it where the scene put it
            pos, quat = start[o.name]
            turn = rng.uniform(-yaw, yaw)
            q = np.asarray(quat, float)
            new_quat = _yaw_quat(2 * np.arctan2(q[3], q[0]) + turn)
            new_pos = np.array([cand[0], cand[1], pos[2]])
            self.world.move_object(o.name, new_pos, new_quat)
            placed[o.name] = cand
            self._randomized[o.name] = ((np.asarray(pos, float), q), new_pos)
            moved[o.name] = new_pos
        return moved

    def used_randomness(self) -> bool:
        """Whether anything random happened in this world: a seeded fault drew a number, or
        the test drew from ``world.rng``. Without it every seed runs the same."""
        return self.world.rng.bit_generator.state != self._fresh_world or any(
            r.bit_generator.state != self._fresh[k] for k, r in self._rng.items()
        )

    def jitter(self, object: str, xy_std: float = 0.01, yaw_std: float = 0.0):
        """Randomise an object's starting pose (domain randomisation), seeded.

        Under ``trials`` the object has already been moved by ``randomize_scene``; a jitter
        replaces that move rather than adding to it, so the spread is the one asked for."""
        rng = self._rng["jitter"]
        pos, quat = self.world.backend.object_pose(object)
        if object in self._randomized:
            (pos0, quat0), moved_to = self._randomized.pop(object)
            if np.linalg.norm(np.asarray(pos[:2]) - moved_to[:2]) < 1e-3:
                pos, quat = pos0, quat0
        pos = pos + np.array([*rng.normal(0, xy_std, 2), 0.0])
        if yaw_std:
            yaw = 2 * np.arctan2(quat[3], quat[0]) + rng.normal(0, yaw_std)
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.world.move_object(object, pos, quat)
        return pos

    # hooks used by World / Robot -------------------------------------------------
    def filter_ctrl(self, target: np.ndarray) -> np.ndarray:
        d = next((f for f in self.active if isinstance(f, ActionDelay)), None)
        if d is None:
            return target
        self._delay.append(target.copy())
        return self._delay.popleft()

    def filter_qpos(self, q: np.ndarray, units: np.ndarray | None = None) -> np.ndarray:
        """``q`` as the robot reads it. ``units`` scales the noise of each reading (a slide's
        metres against a hinge's radians)."""
        for f in self.active:
            if isinstance(f, JointNoise):
                noise = self._rng["joint_noise"].normal(0, f.std, q.shape)
                q = q + (noise if units is None else noise * units)
        return q

    def filter_image(self, img: np.ndarray) -> np.ndarray:
        for f in self.active:
            if isinstance(f, CameraDropout) and self._rng["camera_dropout"].random() < f.p:
                return np.zeros_like(img)
        return img

    def forces_for_step(self, step: int) -> dict:
        t = self.world.time
        out = {}
        for f in self.active:
            if isinstance(f, Push) and f.at <= t < f.at + f.duration:
                out[f.object] = out.get(f.object, np.zeros(3)) + np.array(f.force)
        return out


def _footprint(o) -> float:
    """The radius of the circle an object covers on the table."""
    if o.kind in ("box", "bin"):
        return float(np.hypot(o.size[0], o.size[1]))
    return float(o.size[0])


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
