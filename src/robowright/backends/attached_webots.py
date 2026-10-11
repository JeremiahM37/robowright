"""A Webots world that robowright did not build, read through the Supervisor API of an extern controller.

The world's own controllers run inside Webots. A process of the test's own attaches as a Supervisor
(``controller.Supervisor``, connected to a robot the world declares with ``controller "<extern>"``),
and the simulation advances when that process calls ``step``. This adapter reads the nodes that
process names, by their DEF names, and steps the world one ``basicTimeStep`` at a time.

What Webots gives, and what this adapter says it cannot give:

* Names. Webots has no name lookup for an arbitrary node other than its DEF name, so the caller
  maps each name robowright should use to a DEF name (``bodies={"tcp": "PROBE_TCP"}``).
* Ground truth. A node's world pose (position, orientation as a wxyz quaternion) and, for a Solid,
  its linear and angular velocity.
* The clock. ``getTime()`` of the Supervisor, in seconds.
* Not given: contacts, joint values, setting poses or applying forces through the Supervisor API
  of an extern controller in this adapter. Each raises :class:`CapabilityError` rather than returning
  something invented. The world's controllers are not readable here either, so ``ctrl()`` is empty.
"""

from __future__ import annotations

import numpy as np

from ..errors import CapabilityError
from .attached_engine import AttachedEngine
from .base import GROUND_TRUTH


def _wxyz_of_matrix(m) -> np.ndarray:
    """A quaternion (w, x, y, z) from the rotation part of a row-major 4x4 homogeneous transform."""
    r = np.array([[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]], dtype=float)
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=float)
    return q if q[0] >= 0 else -q


class WebotsEngine(AttachedEngine):
    """Handles are the Webots nodes, found by DEF name on first use and kept after that."""

    engine = "webots"
    capabilities = frozenset({GROUND_TRUTH})

    def __init__(self, supervisor, bodies: dict[str, str], *, seed: int = 0, dt: float | None = None):
        super().__init__(seed, dt=dt)
        self.supervisor = supervisor
        self._def_of = dict(bodies)
        self._nodes: dict[str, object] = {}
        self.free_bodies = []

    # --- naming -----------------------------------------------------------------------
    def names(self, kind: str = "body") -> list[str]:
        return list(self._def_of) if kind == "body" else []

    def resolve(self, name: str):
        if name not in self._def_of:
            return None
        if name not in self._nodes:
            node = self.supervisor.getFromDef(self._def_of[name])
            if node is None:
                raise KeyError(f"no node with DEF name {self._def_of[name]!r} in this Webots world")
            self._nodes[name] = node
        return self._nodes[name]

    def label(self, handle) -> str:
        for name, node in self._nodes.items():
            if node is handle:
                return name
        return "?"

    # --- ground truth -----------------------------------------------------------------
    def pose(self, handle) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array(handle.getPosition(), dtype=float)
        return pos, _wxyz_of_matrix(handle.getPose())

    def velocity(self, handle) -> np.ndarray:
        return np.array(handle.getVelocity(), dtype=float)

    @property
    def time(self) -> float:
        return float(self.supervisor.getTime())

    # --- not given by this adapter ----------------------------------------------------
    def contacts_of(self, handle) -> list[str]:
        raise CapabilityError("contacts are not read through the Webots Supervisor API in this adapter")

    def contact_pairs(self) -> list[tuple]:
        raise CapabilityError("contacts are not read through the Webots Supervisor API in this adapter")

    def joint_value(self, name: str) -> float:
        raise CapabilityError("joint values are not read through the Webots Supervisor API in this adapter")

    def set_joint(self, name: str, value: float) -> None:
        raise CapabilityError("the world's own controllers move the robot; this adapter does not set joints")

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        raise CapabilityError("this adapter does not move bodies: the world's own controllers and physics do")

    def apply_force(self, name: str, force) -> None:
        raise CapabilityError("forces are not applied through the Webots Supervisor API in this adapter")


class WebotsEnv:
    """The stepping side of a Webots world, shaped as the environment ``attach`` wraps.

    ``step`` is one ``basicTimeStep`` of the simulation, which is also the control period of a world
    whose own controllers run every step. Their loop is whatever calls ``step``.
    """

    def __init__(self, supervisor, bodies: dict[str, str]):
        self.supervisor = supervisor
        self.bodies = dict(bodies)
        self.timestep_ms = int(supervisor.getBasicTimeStep())

    @property
    def dt(self) -> float:
        return self.timestep_ms / 1000.0

    def step(self, action=None):
        if self.supervisor.step(self.timestep_ms) == -1:
            raise RuntimeError("Webots ended the simulation (the world was quit or closed)")

    def reset(self, seed=None, options=None):
        raise CapabilityError("this adapter does not reset a Webots world: start a new one")


def webots_engine_of(env, *, seed: int = 0, dt: float | None = None) -> WebotsEngine:
    """The adapter for a :class:`WebotsEnv`."""
    if not isinstance(env, WebotsEnv):
        raise ValueError(f"engine='webots' needs a WebotsEnv, not {type(env).__name__}")
    return WebotsEngine(env.supervisor, env.bodies, seed=seed, dt=dt if dt is not None else env.dt)
