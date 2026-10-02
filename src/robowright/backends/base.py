"""The contract every backend implements.

A backend owns physics (or hardware) and nothing else: no IK, no waiting,
no assertions. Those live in the backend-independent core so a test means
the same thing in every backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..scene import SceneSpec

# Capabilities a backend may advertise. Matchers that need one fail with a
# clear message on backends without it, instead of silently passing.
GROUND_TRUTH = "ground_truth"  # exact object poses
CONTACTS = "contacts"  # contact pairs and forces
RENDER = "render"  # camera images
STATE = "state"  # save/restore full simulator state
DETERMINISTIC = "deterministic"  # same inputs -> bit-identical outputs
FORCES = "forces"  # apply external wrenches (fault injection)


@dataclass(frozen=True)
class Contact:
    a: str  # object name, or "robot:<part>", or "floor"
    b: str
    force: float  # normal force magnitude, N


class Backend(ABC):
    name: str = "base"
    capabilities: frozenset = frozenset()

    def __init__(self, spec: SceneSpec, seed: int = 0):
        self.spec = spec
        self.seed = seed

    # --- robot -----------------------------------------------------------
    joint_names: list[str]

    @abstractmethod
    def qpos(self) -> np.ndarray:
        """Measured robot joint positions (radians), in ``joint_names`` order."""

    @abstractmethod
    def qvel(self) -> np.ndarray: ...

    @abstractmethod
    def set_ctrl(self, target: np.ndarray) -> None:
        """Position targets for every robot joint."""

    @abstractmethod
    def ctrl(self) -> np.ndarray: ...

    def set_gain_scale(self, joint: str, scale: float) -> None:
        raise NotImplementedError(f"{self.name} cannot scale actuator gains")

    # --- time ------------------------------------------------------------
    @property
    def control_dt(self) -> float:
        return 1.0 / self.spec.control_hz

    @abstractmethod
    def step(self) -> None:
        """Advance one control period."""

    @property
    @abstractmethod
    def time(self) -> float: ...

    # --- world -----------------------------------------------------------
    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(f"{self.name} has no ground-truth object poses")

    def object_velocity(self, name: str) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no ground-truth object velocities")

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        raise NotImplementedError(f"{self.name} cannot move objects")

    def contacts(self) -> list[Contact]:
        raise NotImplementedError(f"{self.name} does not report contacts")

    def apply_force(self, name: str, force) -> None:
        """Apply a world-frame force (N) to an object's centre for the next step."""
        raise NotImplementedError(f"{self.name} cannot apply external forces")

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        raise NotImplementedError(f"{self.name} cannot render")

    def get_state(self) -> np.ndarray:
        raise NotImplementedError(f"{self.name} cannot save state")

    def set_state(self, state: np.ndarray) -> None:
        raise NotImplementedError(f"{self.name} cannot restore state")

    def close(self) -> None:
        pass


_REGISTRY: dict[str, type[Backend]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def create(name: str, spec: SceneSpec, seed: int = 0, **kw) -> Backend:
    if name not in _REGISTRY:
        # Import lazily so an optional backend's dependency is only needed when used.
        import importlib

        mod = {"mujoco": "mujoco_backend", "pybullet": "pybullet_backend", "lerobot": "lerobot_backend"}.get(name)
        if mod:
            importlib.import_module(f"robowright.backends.{mod}")
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](spec, seed=seed, **kw)


def available() -> list[str]:
    import importlib

    out = []
    for name, mod in (("mujoco", "mujoco"), ("pybullet", "pybullet")):
        try:
            importlib.import_module(mod)
            out.append(name)
        except ImportError:
            pass
    return out
