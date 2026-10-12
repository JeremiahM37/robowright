"""A Gazebo (gz sim) world that robowright did not build, read over Gazebo's own transport.

The world runs in its own ``gz sim`` process, with whatever plugins it loads; no ROS is involved. This adapter
reads it through :class:`robowright.gazebo.Gazebo`, which subscribes to the world's topics and calls its
control service. What Gazebo gives, and what this adapter says it cannot give:

* Names. Every model and link that Gazebo publishes a pose for, by its name (``world.body(name)``).
* Ground truth. A body's world pose (position, orientation as a wxyz quaternion) and its linear and angular
  velocity over the last ~0.02 s of sim time, from the pose stream.
* Joints. A hinge or slide joint's position by name, from the ``joint_state`` topic of the JointStatePublisher
  system. A world that does not load that system has no such topic, and the error says so.
* Time. The world's clock (``/clock``, one message per physics step).
* Stepping, in one of two modes, decided when the world is attached. A paused world is stepped by robowright
  through Gazebo's world control service, one physics iteration per ``world.step()``, and robowright moves it
  nowhere else. A running world is observed: ``world.step()`` waits for one physics step of its own clock, so
  it runs in real time, as Gazebo runs it.
* Not given: contacts, forces, setting poses, and saving or restoring state. Each raises
  :class:`CapabilityError` rather than returning something invented. The world's own plugins move things.
"""

from __future__ import annotations

import numpy as np

from ..errors import CapabilityError
from ..gazebo import Gazebo
from .attached_engine import AttachedEngine
from .base import GROUND_TRUTH


class GazeboEngine(AttachedEngine):
    """Handles are the names Gazebo publishes poses for; a joint is named by its JointStatePublisher's joint."""

    engine = "gazebo"
    capabilities = frozenset({GROUND_TRUTH})

    def __init__(self, gz: Gazebo, *, paused: bool, seed: int = 0, dt: float | None = None):
        super().__init__(seed, dt=dt)
        self.gz = gz
        self.paused = paused  # the mode the world was in when attached: stepped (paused) or observed (running)

    # --- naming -----------------------------------------------------------------------
    def names(self, kind: str = "body") -> list[str]:
        return self.gz.models() if kind == "body" else []

    def resolve(self, name: str):
        return name if self.gz.has(name) else None

    def label(self, handle) -> str:
        return str(handle)

    # --- ground truth -----------------------------------------------------------------
    def pose(self, handle) -> tuple[np.ndarray, np.ndarray]:
        return self.gz.pose(handle)

    def velocity(self, handle) -> np.ndarray:
        return self.gz.velocity(handle)

    @property
    def time(self) -> float:
        return float(self.gz.clock_time)

    @property
    def control_dt(self) -> float:
        return self.gz.physics_step or self._dt

    # --- joints -----------------------------------------------------------------------
    def named_joints(self) -> list[str]:
        return sorted(self._joints())

    def joint_qadr(self, name: str) -> int:
        """The index of a joint in :meth:`named_joints` (the core's ``joint()`` only checks the name through this)."""
        names = self.named_joints()
        if name not in names:
            raise KeyError(f"no joint named {name!r} in this Gazebo world; joints: {', '.join(names) or 'none'}")
        return names.index(name)

    def joint_value(self, name: str) -> float:
        return float(self._joints()[name])

    def _joints(self) -> dict[str, float]:
        joints = self.gz.joints()
        if not joints:
            raise CapabilityError(
                f"Gazebo world {self.gz.world!r} publishes no joint states: it does not load gz::sim::systems::JointStatePublisher "
                "(add it to the model whose joints you want, as the shipped lift_drag.sdf does)"
            )
        return joints

    # --- not given by this adapter ----------------------------------------------------
    def contacts_of(self, handle) -> list[str]:
        raise CapabilityError("contacts are not read through Gazebo's transport in this adapter")

    def contact_pairs(self) -> list[tuple]:
        raise CapabilityError("contacts are not read through Gazebo's transport in this adapter")

    def set_joint(self, name: str, value: float) -> None:
        raise CapabilityError("the world's own plugins move the joints; this adapter does not set them")

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        raise CapabilityError("this adapter does not move bodies: the world's own plugins and physics do")

    def apply_force(self, name: str, force) -> None:
        raise CapabilityError("forces are not applied through Gazebo's transport in this adapter")

    # --- stepping ---------------------------------------------------------------------
    def advance(self) -> None:
        """One physics step: the world's own clock, in observed mode; a request to Gazebo, in stepped mode."""
        if self.paused:
            self.gz.advance(1)
            return
        dt = self.gz.physics_step
        if dt is None:  # a running world's first clock messages measure its step within milliseconds
            self.gz.wait_until_clock(self.gz.clock_time + 1e-9)
            dt = self.gz.physics_step
        self.gz.wait_until_clock(self.gz.clock_time + dt / 2)

    def trace_meta(self) -> dict:
        return {"engine": self.engine, "world": self.gz.world, "mode": "stepped" if self.paused else "observed"}

    def close(self) -> None:
        self.gz.close()
        super().close()


def gazebo_engine_of(world_name: str | None, *, seed: int = 0, timeout: float = 30.0) -> GazeboEngine:
    """The adapter for the Gazebo world ``world_name`` (default: the only one running in this GZ_PARTITION)."""
    gz = Gazebo(world=world_name, timeout=timeout)
    return GazeboEngine(gz, paused=gz.paused, seed=seed)
