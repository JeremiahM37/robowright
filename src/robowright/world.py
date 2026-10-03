"""The World: one running scene, its robot, its clock and its trace.

Everything that advances time goes through :meth:`World.step`, so waiting,
fault injection, invariants and tracing behave identically for actions,
expectations and policy rollouts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backends import base as backends
from .errors import CapabilityError, ExpectationError, InvariantViolation
from .faults import Faults
from .legged import LeggedRobot
from .locators import SceneLocator
from .robot import Robot
from .scene import SceneSpec, default_scene
from .trace import Recorder

# Keep robowright internals out of pytest failure tracebacks (--full-trace shows them).
__tracebackhide__ = True


@dataclass
class Settings:
    expect_timeout: float = 2.0  # simulated seconds an expect() retries for
    action_timeout: float = 10.0  # simulated seconds an action may take
    max_joint_speed: float = 2.0  # rad/s for planned joint moves
    max_tcp_speed: float = 0.4  # m/s average tool speed for planned joint moves
    trace: str = "on"  # on | off | retain-on-failure
    trace_dir: str = "robowright-traces"
    frame_every: int = 5  # control steps between trace frames
    trace_cameras: list | None = None
    image_size: tuple = (320, 240)
    realtime: bool = False  # pace stepping to wall clock (hardware-style)


class World:
    def __init__(
        self, spec: SceneSpec, backend: str = "mujoco", seed: int = 0, name: str = "world", settings: Settings | None = None, **backend_kw
    ):
        self.spec = spec
        self.seed = seed
        self.name = name
        self.settings = settings or Settings()
        self.rng = np.random.default_rng(seed)
        self.backend = backends.create(backend, spec, seed=seed, **backend_kw)
        self.object_names = [o.name for o in spec.objects]
        self.step_count = 0
        self.status = "running"
        self.perception: dict[str, Callable] = {}
        self._invariants: list = []
        self._step_hooks: list[Callable] = []
        self._soft_failures: list[str] = []
        self.faults = Faults(self)
        self.robot = (LeggedRobot if self.backend.robot_model.family == "legged" else Robot)(self)
        self.scene = SceneLocator(self)
        self.trace: Recorder | None = None
        if self.settings.trace != "off":
            s = self.settings
            self.trace = Recorder(self, s.frame_every, s.trace_cameras, s.image_size)
        self._began = False
        self.trace_path: Path | None = None

    # --- capabilities ------------------------------------------------------
    @property
    def has_ground_truth(self) -> bool:
        return backends.GROUND_TRUTH in self.backend.capabilities

    @property
    def has_contacts(self) -> bool:
        return backends.CONTACTS in self.backend.capabilities

    def require(self, cap: str, what: str):
        if cap not in self.backend.capabilities:
            raise CapabilityError(f"{what} needs the {cap!r} capability, which the {self.backend.name} backend does not provide")

    # --- time ----------------------------------------------------------------
    @property
    def time(self) -> float:
        return self.backend.time

    @property
    def dt(self) -> float:
        return self.backend.control_dt

    def _begin(self):
        if not self._began:
            self._began = True
            if self.trace:
                self.trace.begin()

    def step(self, n: int = 1):
        """Advance ``n`` control periods with the robot's current targets."""
        self._begin()
        for _ in range(n):
            target = self.robot._target.copy()
            applied = self.faults.filter_ctrl(target)
            self.backend.set_ctrl(applied)
            forces = self.faults.forces_for_step(self.step_count)
            for name, f in forces.items():
                self.backend.apply_force(name, f)
            if self.settings.realtime:
                import time as _t

                _t.sleep(self.dt)
            self.backend.step()
            self.step_count += 1
            if self.trace:
                self.trace.record_step(self.step_count, self.backend.ctrl(), forces)
            for hook in self._step_hooks:
                hook(self)
            self._check_invariants()

    def wait(self, seconds: float):
        """Let time pass while holding the current targets."""
        if self.trace:
            # Recorded so codegen reproduces timing: what a push or a fault does depends on it.
            self.trace.event("wait", "world.wait", {"seconds": float(seconds)})
        self.step(max(1, round(seconds / self.dt)))

    def run_until(self, predicate: Callable[[], bool], timeout: float, hold: float = 0.0) -> bool:
        """Step until ``predicate()`` has been true for ``hold`` seconds, or ``timeout`` passes.

        The predicate is checked before the first step, so a condition that is
        already true costs no simulated time.
        """
        self._begin()
        deadline = self.time + timeout
        held_since = None
        while True:
            if predicate():
                held_since = self.time if held_since is None else held_since
                if self.time - held_since >= hold - 1e-9:
                    return True
            else:
                held_since = None
            if self.time >= deadline - 1e-9:
                return False
            self.step()

    # --- invariants ------------------------------------------------------------
    def add_invariant(self, check, description: str, record: dict | None = None):
        self._invariants.append((check, description))
        if self.trace:
            self.trace.event("invariant", description, record or {}, status="ok", detail="registered")

    def _check_invariants(self):
        for check, desc in self._invariants:
            ok, detail = check()
            if not ok:
                self._invariants.clear()
                msg = f"invariant violated at t={self.time:.3f}s: {desc}\n  {detail}"
                if self.trace:
                    self.trace.event("violation", desc, status="failed", detail=detail)
                raise InvariantViolation(msg)

    # --- world edits (recorded so replay can reproduce them) ---------------------
    def move_object(self, name: str, pos, quat=None):
        self.backend.set_object_pose(name, pos, quat)
        if self.trace:
            self.trace.event(
                "edit",
                "move_object",
                {"object": name, "pos": list(map(float, pos)), "quat": None if quat is None else list(map(float, quat))},
            )

    def log(self, message: str):
        if self.trace:
            self.trace.event("log", message)

    def soft_fail(self, message: str):
        self._soft_failures.append(message)

    # --- lifecycle ---------------------------------------------------------------
    def close(self, failed: bool | None = None, trace_path: str | Path | None = None) -> Path | None:
        if failed is None:
            failed = self.status == "failed"
        if self._soft_failures and not failed:
            failed = True
        self.status = "failed" if failed else "passed"
        saved = None
        if self.trace and (self.settings.trace == "on" or (self.settings.trace == "retain-on-failure" and failed)):
            path = trace_path or Path(self.settings.trace_dir) / f"{_safe(self.name)}.zip"
            saved = self.trace_path = self.trace.save(path)
        self.backend.close()
        if self._soft_failures:
            failures, self._soft_failures = self._soft_failures, []
            raise ExpectationError(f"{len(failures)} soft expectation(s) failed:\n\n" + "\n\n".join(failures))
        return saved

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close(failed=exc_type is not None)
        except ExpectationError:
            if exc_type is None:
                raise
        return False


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.[]=" else "_" for c in name)[:180]


def launch(
    scene: SceneSpec | None = None,
    backend: str = "mujoco",
    seed: int = 0,
    name: str = "world",
    settings: Settings | None = None,
    robot: str | None = None,
    **kw,
) -> World:
    """Build a world. ``scene`` defaults to the robot's default scene (for arms,
    the tabletop with a cube and a bin; for legged robots, an open floor).

    ``robot`` picks the robot for the default scene, or overrides ``scene.robot``.
    """
    if scene is None:
        scene = default_scene(robot or "so101")
    elif robot is not None and robot != scene.robot:
        import dataclasses

        scene = dataclasses.replace(scene, robot=robot)
    return World(scene, backend=backend, seed=seed, name=name, settings=settings, **kw)
