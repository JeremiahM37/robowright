"""The contract every backend implements.

A backend owns physics (or hardware) and nothing else: no IK, no waiting,
no assertions. Those live in the backend-independent core so a test means
the same thing in every backend.
"""

from __future__ import annotations

import atexit
import gc
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
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


class TargetRamp:
    """Spreads each control step's change of servo targets across that step's physics substeps.

    Targets arrive once per control step. Handed to a servo as a step, they move a stiff one in
    a staircase: a velocity spike at every control step, then a coast. That shook light grips
    loose on PyBullet (ARX L5 0/20) and Genesis (PiPER 14/20, iiwa 15/20), while MuJoCo's and
    Drake's softer servos smoothed it. Every backend now ramps, as a real servo's interpolator
    does, so every engine is handed the same commands: substep ``k`` of ``n`` gets the targets
    ``(k + 1) / n`` of the way from the last step's to the new ones.
    """

    def __init__(self):
        self.start: np.ndarray | None = None
        self.end: np.ndarray | None = None

    def set(self, target) -> None:
        self.end = np.array(target, float)
        if self.start is None:  # placed, restored or just built: nothing to ramp from
            self.start = self.end.copy()

    def reset(self) -> None:
        """Forget where the targets were: the robot was placed, not moved, so the next ones apply at once."""
        self.start = None

    @property
    def moving(self) -> bool:
        return self.start is not None and not np.array_equal(self.start, self.end)

    def at(self, frac: float) -> np.ndarray:
        if frac >= 1.0:  # exactly the targets, not start + (end - start): a trace records them
            return self.end.copy()
        return self.start + frac * (self.end - self.start)

    def velocity(self, dt: float) -> np.ndarray:
        return (self.end - self.start) / dt

    def arrive(self) -> None:
        self.start = self.end.copy()


class Backend(ABC):
    """Physics for one scene.

    The robot's state is the arm joints (radians or metres, in the model's
    ``arm_joints`` order) followed by one gripper value: its opening, from
    0 (closed) to 1 (open). How a backend moves the fingers to a commanded
    opening is its own business; ``robot_model.derived.gripper_joints`` gives
    every finger joint's closed and open position for backends that drive
    the joints individually.
    """

    model_changes: list = []  # how this differs from the robot's model file (set by create; see robowright.fidelity)

    name: str = "base"
    capabilities: frozenset = frozenset()
    # Whether a closed world's backend may be kept and restored for the next world with the same
    # scene (see :func:`create`). Needs a restore that leaves nothing behind: every
    # reusing backend is checked bit-for-bit against a fresh build in tests/test_reuse.py.
    reusable: bool = False
    # One scene per process (Isaac Sim has one stage): building another closes the kept ones.
    exclusive: bool = False
    # Whether robowright builds the robot it runs (from its model file, recording what it changes:
    # robowright.fidelity) or reaches one that is already there (over ROS 2: a real arm, or a
    # robot someone else's simulator runs).
    builds: bool = True

    def __init__(self, spec: SceneSpec, seed: int = 0):
        self.spec = spec
        self.seed = seed
        self.robot_model = spec.robot_model
        self.n_arm = self.robot_model.n_arm
        self.has_gripper = self.robot_model.has_gripper
        self.joint_names = [*self.robot_model.arm_joints, *(["gripper"] if self.has_gripper else [])]

    # --- robot -----------------------------------------------------------
    joint_names: list[str]

    @abstractmethod
    def qpos(self) -> np.ndarray:
        """Measured arm joint positions, then the gripper opening (0..1)."""

    @abstractmethod
    def qvel(self) -> np.ndarray: ...

    @abstractmethod
    def set_ctrl(self, target: np.ndarray) -> None:
        """Position targets for the arm joints, then the commanded gripper opening (0..1)."""

    @abstractmethod
    def ctrl(self) -> np.ndarray: ...

    def set_gain_scale(self, joint: str, scale: float) -> None:
        raise NotImplementedError(f"{self.name} cannot scale actuator gains")

    def set_joint_positions(self, q: np.ndarray) -> None:
        """Teleport the arm joints and gripper opening (setup only), at rest, with matching targets."""
        raise NotImplementedError(f"{self.name} cannot teleport joints")

    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and (w, x, y, z) quaternion of the robot's hand body."""
        raise NotImplementedError(f"{self.name} does not report link poses")

    # --- mobile robots (``robot_model.floating``) ----------------------------
    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and (w, x, y, z) quaternion of the floating base."""
        raise NotImplementedError(f"{self.name} does not support floating-base robots")

    def base_velocity(self) -> np.ndarray:
        """World-frame linear then angular velocity of the floating base."""
        raise NotImplementedError(f"{self.name} does not support floating-base robots")

    def set_base_pose(self, pos, quat) -> None:
        """Teleport the floating base (setup only), at rest."""
        raise NotImplementedError(f"{self.name} does not support floating-base robots")

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
        """Apply a world-frame force (N) to an object's centre (or ``"robot"``: the floating base) for the next step."""
        raise NotImplementedError(f"{self.name} cannot apply external forces")

    def perception(self) -> dict:
        """Object pose sources this backend has without ground truth (``{name: () -> (pos, quat)}``),
        which become ``world.perception``: what hardware sees through its own sensors."""
        return {}

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        raise NotImplementedError(f"{self.name} cannot render")

    def get_state(self) -> np.ndarray:
        raise NotImplementedError(f"{self.name} cannot save state")

    def set_state(self, state: np.ndarray) -> None:
        raise NotImplementedError(f"{self.name} cannot restore state")

    def close(self) -> None:
        pass

    # Watching a run (pytest --rw-headed). Engines with a window of their own implement these.
    def open_viewer(self) -> None:
        from ..errors import CapabilityError

        raise CapabilityError(f"{self.name} has no live viewer: run headed with --rw-backend mujoco, or open the trace afterwards")

    def sync_viewer(self) -> None:
        pass

    def close_viewer(self) -> None:
        pass

    def _snapshot_built(self) -> None:
        """Remember the as-built state, for :meth:`_reuse`."""
        if STATE in self.capabilities:
            self._built = self.get_state()

    def _reuse(self, seed: int) -> None:
        """Put a kept backend back exactly as it was built: gains, pending forces, state."""
        self.seed = seed
        for j in self.robot_model.arm_joints:
            self.set_gain_scale(j, 1.0)  # gains are model data, not state, on some engines
        if hasattr(self, "_pending"):
            self._pending = {}
        self.set_state(self._built)


_REGISTRY: dict[str, type[Backend]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def engine(name: str) -> type[Backend]:
    """The backend class registered as ``name``, importing its module on first use."""
    if name not in _REGISTRY:
        # Import lazily so an optional backend's dependency is only needed when used.
        import importlib

        mod = _MODULES.get(name)
        if mod:
            try:
                importlib.import_module(f"robowright.backends.{mod}")
            except ModuleNotFoundError as e:
                hint = _INSTALL.get(name)
                if hint and e.name == _REQUIRES.get(name):
                    raise ModuleNotFoundError(f"the {name!r} engine is not installed: {hint}", name=e.name) from e
                raise
        else:
            from ..plugins import load_backend

            load_backend(name)  # an engine another package provides (see robowright.plugins)
    if name not in _REGISTRY:
        from ..plugins import backend_names

        known = sorted(set(_MODULES) | set(_REGISTRY) | set(backend_names()))
        raise ValueError(f"unknown backend {name!r}; known backends: {', '.join(known)} (installed: {', '.join(available()) or 'none'})")
    return _REGISTRY[name]


def create(name: str, spec: SceneSpec, seed: int = 0, **kw) -> Backend:
    """A backend for ``spec``: a kept one restored to as-built if there is one, else a new build.

    Building is most of what a short test costs (MuJoCo: 0.05-0.14 s against 0.05 s for a pick;
    Drake: 0.3-0.4 s against 0.5 s), and restoring a state takes a tenth of a millisecond.
    ``ROBOWRIGHT_REUSE=0`` builds every world afresh.
    """
    from .. import fidelity

    cls = engine(name)
    key = (name, fidelity.mode(), repr(spec), repr(sorted(kw.items()))) if cls.reusable and _keep() else None
    b = _KEPT.pop(key, None) if key else None
    if b is not None:
        try:
            b._reuse(seed)
            b._key = key
            return b
        except Exception:
            b.close()
    if cls.exclusive:
        for k in [k for k in _KEPT if k[0] == name]:
            _KEPT.pop(k).close()
    _reclaim()
    with fidelity.recording() as log:
        b = cls(spec, seed=seed, **kw)
        if b.builds:  # a simulator robowright built (a robot behind ROS 2 is what it is)
            if spec.robot_model.family == "arm":
                fidelity.record(
                    "interface",
                    "gravity compensated",
                    "the arm's links carry no weight, as industrial arm controllers cancel it; hobby-servo arms "
                    "(the SO-101, Koch) do not do this, so their tests are kinder than the real arm",
                    engine=name,
                )
            if name in ("drake", "genesis", "isaac"):  # (PyBullet says how it translates, itself)
                fidelity.record(
                    "interface",
                    "model translated",
                    "the MuJoCo model exported to URDF (geometry, inertias, joint limits, servo gains and force limits), "
                    "finger couplings as the engine's mimic joints or couplers",
                    engine=name,
                )
    # What a simulator ran differs from the model file by these; a robot behind ROS 2 (real, or
    # another simulator) runs as it is, and robowright changes nothing in it.
    b.model_changes = list(dict.fromkeys([*spec.robot_model.changes(), *log])) if b.builds else []
    start = b.robot_model.start
    if start is not None and not b.robot_model.floating:
        # An arm whose zero pose is not a place to start (a URDF's arm folded into the table).
        b.set_joint_positions(np.append(start, 1.0) if b.has_gripper else np.asarray(start, float))
    # Captured on every build, kept or not: on Genesis and Isaac a capture snaps the simulation
    # onto the captured state, so a kept scene and a new one must both have taken it.
    b._snapshot_built()
    if key:
        b._key = key
    return b


def release(b: Backend) -> None:
    """Done with ``b``: keep it for the next world with the same scene, or close it."""
    key, n = getattr(b, "_key", None), _keep()
    if not key or not n:
        b.close()
        return
    b._key = None  # a backend released twice is kept once
    old = _KEPT.pop(key, None)
    if old is not None:
        old.close()
    _KEPT[key] = b
    while len(_KEPT) > n:
        _KEPT.popitem(last=False)[1].close()
    # Exit handlers run last-registered first. MuJoCo's EGL module registers one that tears the
    # display down when it is first imported, often after this module; re-registering here closes
    # kept scenes (and their GL contexts) before that, instead of after it with a traceback.
    atexit.unregister(close_kept)
    atexit.register(close_kept)


_last_reclaim = 0.0


def _reclaim() -> None:
    """Give the memory of closed worlds back before building another.

    A closed world sits in a reference cycle until Python's collector next runs, which native
    allocations (a MuJoCo model with its meshes) do not prompt, and a build's transient compile
    fragments the heap. A MuJoCo test worker that ran every robot grew to 4.4 GB; collected and
    trimmed before each build (and with mesh-free kinematics models) it peaks near 2.7 GB,
    which leaves room for more workers.
    At most once a second: a collection takes ~10 ms, and PyBullet builds a scene per world.
    """
    global _last_reclaim
    now = time.monotonic()
    if now - _last_reclaim < 1.0:
        return
    _last_reclaim = now
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def _libc():
    try:
        import ctypes

        lib = ctypes.CDLL("libc.so.6")
        return lib if hasattr(lib, "malloc_trim") else None  # glibc only
    except (OSError, AttributeError):
        return None


_LIBC = _libc()


def close_kept() -> None:
    while _KEPT:
        _KEPT.popitem()[1].close()


def _keep() -> int:
    """How many built scenes a process keeps (``ROBOWRIGHT_REUSE``, default 2; 0 turns reuse off)."""
    return int(os.environ.get("ROBOWRIGHT_REUSE", "2"))


_KEPT: OrderedDict[tuple, Backend] = OrderedDict()


_MODULES = {
    "mujoco": "mujoco_backend",
    "pybullet": "pybullet_backend",
    "genesis": "genesis_backend",
    "drake": "drake_backend",
    "isaac": "isaac_backend",
    "ros2": "ros2_backend",
}
_REQUIRES = {"mujoco": "mujoco", "pybullet": "pybullet", "genesis": "genesis", "drake": "pydrake", "isaac": "isaacsim", "ros2": "rclpy"}


_INSTALL = {
    "pybullet": 'pip install "robowright[pybullet]"',
    "genesis": 'install PyTorch first (https://pytorch.org), then pip install "robowright[genesis]"',
    "drake": 'pip install "robowright[drake]" (needs Python 3.12 or newer)',
    "isaac": "Isaac Sim is not a pip extra; see the Install section of the README",
}


def available() -> list[str]:
    import importlib.util

    from ..plugins import backend_names

    # find_spec, not import: importing genesis or drake takes seconds.
    built_in = [name for name, mod in _REQUIRES.items() if importlib.util.find_spec(mod) is not None]
    return built_in + [n for n in backend_names() if n not in built_in]
