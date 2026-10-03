"""Fault injection: the things that go wrong on real robots, on demand.

Faults are seeded from the world's RNG, so a failing run with a given seed
fails the same way again, and they are written to the trace so replay and
codegen reproduce them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Fault:
    def describe(self) -> dict:
        return {"type": type(self).__name__, **self.__dict__}


@dataclass
class JointNoise(Fault):
    """Gaussian noise (radians) on every joint reading the robot sees."""

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
        names = self.world.backend.joint_names
        if joint not in names:
            raise ValueError(f"unknown joint {joint!r}; {self.world.backend.robot_model.title} has: {', '.join(names)}")
        self.world.backend.set_gain_scale(joint, scale)
        return self._add(WeakJoint(joint, scale))

    def camera_dropout(self, p: float = 0.1):
        return self._add(CameraDropout(p))

    def jitter(self, object: str, xy_std: float = 0.01, yaw_std: float = 0.0):
        """Randomise an object's starting pose (domain randomisation), seeded."""
        rng = self.world.rng
        pos, quat = self.world.backend.object_pose(object)
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

    def filter_qpos(self, q: np.ndarray) -> np.ndarray:
        for f in self.active:
            if isinstance(f, JointNoise):
                q = q + self.world.rng.normal(0, f.std, q.shape)
        return q

    def filter_image(self, img: np.ndarray) -> np.ndarray:
        for f in self.active:
            if isinstance(f, CameraDropout) and self.world.rng.random() < f.p:
                return np.zeros_like(img)
        return img

    def forces_for_step(self, step: int) -> dict:
        t = self.world.time
        out = {}
        for f in self.active:
            if isinstance(f, Push) and f.at <= t < f.at + f.duration:
                out[f.object] = out.get(f.object, np.zeros(3)) + np.array(f.force)
        return out
