"""The contract an engine implements so robowright can be attached to a simulation it did not build.

:mod:`robowright.attached` is engine-free: it wraps the environment's ``step`` and ``reset``, checks
invariants, records the trace and runs the waits. Everything that touches an engine goes through
the methods below, so supporting another engine means one adapter subclass of
:class:`AttachedEngine`, not a change to the core.

Required of every engine: the naming (``names``, ``resolve``, ``label``), ground truth for a named
thing (``pose``, ``velocity``, ``contacts_of``, ``contact_pairs``), joint values (``joint_names``,
``joint_value``), the free bodies a test can move (``free_bodies``, ``set_object_pose``), forces
(``apply_force``), and the clock (``time``, ``control_dt``). Optional, and raising
:class:`CapabilityError` with the reason when an engine has none: ``render``, ``get_state`` and
``set_state``. A handle is whatever ``resolve`` returns; the core passes it back unchanged.
"""

from __future__ import annotations

import numpy as np

from ..errors import CapabilityError
from ..scene import SceneSpec
from .base import CONTACTS, FORCES, GROUND_TRUTH, Backend, Contact


class _UnknownRobot:
    """The fields of a robot model the core reads. The robot here belongs to the environment,
    so robowright describes none of it: no arm, no gripper, no floating base."""

    floating = False
    arm_joints: tuple = ()
    extra: dict = {}  # (not loaded from a file by robowright: nothing to record for a trace to reload)
    title = "attached"
    family = "attached"
    n_arm = 0
    has_gripper = False


class AttachedEngine(Backend):
    """Base for an engine adapter. Subclasses set ``engine`` and implement the methods below."""

    name = "attached"
    engine = "?"  # the name a caller passes as ``attach(engine=...)``, and records in a trace
    builds = False  # the environment built it: robowright changes nothing in it
    capabilities = frozenset({GROUND_TRUTH, CONTACTS, FORCES})

    def __init__(self, seed: int = 0, dt: float | None = None):
        # Backend.__init__ reads a robot model from a SceneSpec; there is none, so set what it sets.
        self.seed = seed
        self.spec = SceneSpec(robot="attached", objects=[], cameras=[])
        self.robot_model = _UnknownRobot()
        self.n_arm, self.has_gripper = 0, False
        self._dt = float(dt) if dt is not None else 0.0
        self._renderers: dict = {}
        self.free_bodies: list[str] = []
        self.joint_names: list[str] = []

    # --- required of an engine ----------------------------------------------------------
    def names(self, kind: str = "body") -> list[str]:
        raise NotImplementedError

    def resolve(self, name: str):
        """A handle for the body, site or link called ``name``, or None."""
        raise NotImplementedError

    def label(self, handle) -> str:
        raise NotImplementedError

    def pose(self, handle) -> tuple[np.ndarray, np.ndarray]:
        """World position and orientation as a wxyz quaternion."""
        raise NotImplementedError

    def velocity(self, handle) -> np.ndarray:
        """World-frame linear then angular velocity."""
        raise NotImplementedError

    def contacts_of(self, handle) -> list[str]:
        raise NotImplementedError

    def contact_pairs(self) -> list[tuple]:
        """Every contact as ``(label1, label2, normal force)``."""
        raise NotImplementedError

    def joint_value(self, name: str) -> float:
        raise NotImplementedError

    def named_joints(self) -> list[str]:
        """The joints an agent can name with ``world.joint``: the engine's own joint list, or [] for an engine that names none."""
        return list(self.joint_names)

    def ensure_bound(self) -> bool:
        """Bind to the simulation if it was not reachable when attached. True when this call bound it."""
        return False

    @property
    def pending(self) -> bool:
        """True while the simulation has not been found yet (an engine that binds lazily, see ``ensure_bound``)."""
        return False

    def set_joint(self, name: str, value: float) -> None:
        raise NotImplementedError

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        raise NotImplementedError

    def apply_force(self, name: str, force) -> None:
        """Add ``force`` (world frame, N) to the free body ``name`` for the physics step being taken."""
        raise NotImplementedError

    @property
    def time(self) -> float:
        raise NotImplementedError

    def note_step(self) -> None:
        """Called once per control step the environment took, for engines whose clock is not read from the engine."""

    # --- optional: a default says what the engine cannot do --------------------------------
    def qpos(self) -> np.ndarray:
        return np.zeros(len(self.joint_names))

    def qvel(self) -> np.ndarray:
        return np.zeros(len(self.joint_names))

    def ctrl(self) -> np.ndarray:
        return np.full(0, np.nan)

    def set_ctrl(self, target) -> None:
        raise CapabilityError("the environment's own loop sets its controls; robowright only observes them")

    def step(self) -> None:
        raise CapabilityError("time advances when the environment's loop steps it; robowright observes those steps")

    @property
    def control_dt(self) -> float:
        return self._dt

    def set_control_dt(self, dt: float) -> None:
        """Measured from the first environment step, which is what the environment's control rate is."""
        self._dt = float(dt)

    def trace_meta(self) -> dict:
        """What a trace records about this engine, beyond the common fields."""
        return {"engine": self.engine}

    def trace_files(self) -> dict[str, bytes]:
        """Files a saved trace keeps next to its arrays, by name."""
        return {}

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        raise CapabilityError(f"frame redraw is not available for the {self.engine} engine: its frames come from its own renderer")

    def get_state(self) -> np.ndarray:
        raise CapabilityError(f"saving and restoring state is not available for the {self.engine} engine")

    def set_state(self, state) -> None:
        raise CapabilityError(f"saving and restoring state is not available for the {self.engine} engine")

    def rebind(self, env) -> None:
        """Follow the environment if it replaced its simulation (a reset that rebuilds it). Most engines do not."""

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    # --- shared -------------------------------------------------------------------------
    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.pose(self._handle(name))

    def object_velocity(self, name: str) -> np.ndarray:
        return self.velocity(self._handle(name))

    def contacts(self) -> list[Contact]:
        return [Contact(a, b, f) for a, b, f in self.contact_pairs()]

    def _handle(self, name: str):
        h = self.resolve(name)
        if h is None:
            raise KeyError(f"no body named {name!r} in this simulation")
        return h
