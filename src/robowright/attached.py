"""Attach robowright to a MuJoCo simulation someone else built, and test it as it is.

    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    world = rw.attach(env, advance=lambda: advance_one_step())   # advance is optional
    obs, info = env.reset(seed=0)
    cube = world.body("box")
    for _ in range(400):
        obs, reward, terminated, truncated, info = env.step(policy(obs))  # their loop
    expect(cube).to_be_above(0.1)
    world.close()

How robowright sees their steps. It wraps the environment's ``step`` and ``reset`` (the outermost
object you hold, gymnasium's wrapper or the dm_control env itself): every call to ``step`` is one
physics step that robowright observes, checks invariants on, and records. Wrapping the method is
the one hook that sees a *control* step, the unit a test author thinks in; a MuJoCo callback fires
once per substep and cannot tell an environment's frame skip from a physics step. Steps taken
below that object (``env.unwrapped.step``, ``physics.step``) are not seen; ``world.observe()``
records a step taken that way.

How time moves. Nothing steps the simulation on its own in the basic mode: the test's own loop
steps it. With ``advance=`` (or ``step=`` for a bare model), robowright may drive it, so
``world.step(n)``, ``world.run(...)`` and ``expect(x).to_...(timeout=...)`` auto-wait exactly as
they do on a built world. Without ``advance`` an expectation checks the state once, now, and
never waits.
"""

from __future__ import annotations

import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np

from . import _current
from .backends.attached_backend import AttachedBackend
from .errors import CapabilityError
from .faults import Faults
from .locators import Subject
from .scene import ObjectSpec, SceneSpec
from .trace import Recorder, Trace
from .world import Settings, World, _safe


class AttachedSubject(Subject):
    """A MuJoCo body, site or geom by name: position, orientation, velocity and contacts from ground truth."""

    def __init__(self, world: AttachedWorld, kind: str, index: int, name: str):
        self.world = world
        self.kind = kind
        self.index = index
        self.name = name

    @property
    def _b(self) -> AttachedBackend:
        return self.world.backend

    @property
    def position(self) -> np.ndarray:
        return self._b.pose(self.kind, self.index)[0]

    @property
    def quaternion(self) -> np.ndarray:
        return self._b.pose(self.kind, self.index)[1]

    @property
    def yaw(self) -> float:
        w, x, y, z = self.quaternion
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @property
    def up_axis(self) -> np.ndarray:
        w, x, y, z = self.quaternion
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    @property
    def velocity(self) -> np.ndarray:
        """Linear then angular velocity, world frame."""
        return self._b.velocity(self.kind, self.index)

    def bounds(self):
        raise CapabilityError(f"{self.name!r} has no bounds: a MuJoCo body is not a box; test its position instead")

    def contacts(self) -> list[str]:
        """What this touches, by body name and by geom name: a geom is named as the environment names it.

        Environments often test contacts between geoms (gym-aloha's reward does), so both names are
        listed, and ``to_be_touching`` accepts either.
        """
        self.world.require("contacts", "contact queries")
        b = self._b
        body = b.body_of(self.kind, self.index)
        out = set()
        for b1, b2, _, g1, g2 in b.contact_pairs():
            if self.kind == "geom":
                if self.index not in (g1, g2):
                    continue
                other_body, other_geom = (b2, g2) if g1 == self.index else (b1, g1)
            else:
                if body not in (b1, b2):
                    continue
                other_body, other_geom = (b2, g2) if b1 == body else (b1, g1)
            if other_body == body:
                continue
            out.add(b.label(other_body))
            if b.geom_name(other_geom):
                out.add(b.geom_name(other_geom))
        return sorted(out)


class AttachedJoint(Subject):
    """A MuJoCo joint by name: its scalar value (a hinge angle in rad, a slide in m), read from the state."""

    def __init__(self, world: AttachedWorld, name: str, qadr: int):
        self.world = world
        self.name = name
        self.qadr = qadr

    @property
    def value(self) -> float:
        return float(self.world.data.qpos[self.qadr])


class AttachedScene:
    """``world.scene[name]``: the same lookup as a built world's, over the simulation's named things."""

    def __init__(self, world: AttachedWorld):
        self.world = world

    def __getitem__(self, name: str) -> AttachedSubject:
        return self.world.body(name)

    def __iter__(self):
        return iter(self.all())

    def all(self) -> list[AttachedSubject]:
        return [self.world.body(n) for n in self.world.object_names]


class AttachedRecorder(Recorder):
    """The standard trace, plus the simulation's model, so the trace can be redrawn from its state."""

    def meta(self) -> dict:
        return {**super().meta(), "attached": {"model": "model.mjb", "nq": int(self.world.backend.model.nq)}}

    def save(self, path: str | Path) -> Path:
        path = super().save(path)
        with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as z:
            z.writestr("model.mjb", model_bytes(self.world.backend.model))
        return path


class AttachedWorld(World):
    """A World over a simulation robowright did not build. Its ``backend`` is an :class:`AttachedBackend`."""

    def __init__(self, backend: AttachedBackend, advance: Callable[[], object] | None, *, name: str, seed: int, settings: Settings | None):
        self.backend = backend
        self.name = name
        self.seed = seed
        self.settings = settings or Settings()
        self.rng = np.random.default_rng(seed)
        self.fidelity = None  # there is no robowright fidelity mode: the model is theirs, as it is
        self.model_changes: list = []  # robowright changes nothing in the simulation unless asked
        self.object_names = list(backend.free_bodies)
        self.spec = SceneSpec(robot="attached", objects=[ObjectSpec(name=n, kind="body") for n in self.object_names], cameras=[])
        self.step_count = 0
        self.status = "running"
        self.perception: dict = {}
        self._perceived: dict = {}
        self._invariants: list = []
        self._step_hooks: list[Callable] = []
        self._soft_failures: list[str] = []
        self.faults = Faults(self)
        self.robot = None  # the environment's robot is not robowright's to drive
        self.scene = AttachedScene(self)
        s = self.settings
        self.trace = AttachedRecorder(self, s.frame_every, s.trace_cameras, s.image_size) if s.trace != "off" else None
        self._began = False
        self.trace_path: Path | None = None
        self._advance = advance
        self._dt_known = False

    # --- what the simulation is -------------------------------------------------
    @property
    def model(self) -> mujoco.MjModel:
        return self.backend.model

    @property
    def data(self) -> mujoco.MjData:
        return self.backend.data

    def bodies(self) -> list[str]:
        """Every named body: the names a test can look up with ``world.body(name)``."""
        return self.backend.names("body")

    def body(self, name: str) -> AttachedSubject:
        """A body, site or geom by name, as a subject for ``expect``."""
        found = self.backend.resolve(name)
        if found is None:
            raise KeyError(f"no body, site or geom named {name!r}; bodies: {', '.join(self.bodies()) or 'none'}")
        kind, i = found
        return AttachedSubject(self, kind, i, name)

    def joint(self, name: str) -> AttachedJoint:
        """A hinge or slide joint by name, for ``expect(...).to_be_between``: its value, not a position."""
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            named = [n for n in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)) if n]
            raise KeyError(f"no joint named {name!r}; joints: {', '.join(named) or 'none'}")
        # int(): numpy's int32 does not compare equal to MuJoCo's enum members
        if int(self.model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise CapabilityError(f"joint {name!r} is not a hinge or slide, so it has one scalar value; use world.body() for its pose")
        return AttachedJoint(self, name, int(self.model.jnt_qposadr[j]))

    # --- time ---------------------------------------------------------------------
    def step(self, n: int = 1):
        """Advance ``n`` physics steps through the advance callable (``advance=``, or ``step=``)."""
        if self._advance is None:
            raise CapabilityError(
                "time advances when your loop steps the simulation; pass advance=... to attach() to let robowright step it, "
                "or call world.observe() after each of your steps"
            )
        self._begin()
        for _ in range(n):
            self._advance()

    def run_until(self, predicate: Callable[[], bool], timeout: float, hold: float = 0.0) -> bool:
        """World.run_until, except that without an advance callable nothing can pass: the state is checked once."""
        if self._advance is None:
            return bool(predicate()) and hold <= 1e-9
        return super().run_until(predicate, timeout, hold)

    def run(self, step_fn: Callable[[], object], until: Callable[[], bool], timeout: float) -> bool:
        """Drive ``step_fn`` (one control step of their loop) until ``until()`` holds or ``timeout`` passes."""
        saved, self._advance = self._advance, step_fn
        try:
            return self.run_until(until, timeout)
        finally:
            self._advance = saved

    def push(self, body: str, force, duration: float = 0.1, at: float | None = None):
        """A world-frame force (N) on a free body from ``at`` for ``duration`` s, applied to the physics steps it covers."""
        if body not in self.object_names:
            raise ValueError(f"push moves free bodies; {body!r} is not one (free bodies: {', '.join(self.object_names) or 'none'})")
        return self.faults.push(body, force, at=at, duration=duration)

    def observe(self) -> None:
        """Record one physics step that the environment's own loop took below ``env.step``."""
        self._physics_step(None, pushes=False)

    def _begin(self):
        if not self._began:
            self._began = True
            state0 = self.backend.get_state()
            if self.trace:
                self.trace.begin(state0)

    def _on_reset(self, seed):
        """The trace starts when the environment is first reset, so its first sample is a real start state."""
        self._begin()
        if self.trace:
            self.trace.event("log", "env.reset", {"seed": seed})

    def _physics_step(self, fn, pushes: bool = True):
        """One physics step: apply pushes for it, run the environment's own step, then observe the result."""
        self._begin()
        forces = self.faults.forces_for_step(self.step_count) if pushes else {}
        saved = self.backend.data.xfrc_applied.copy() if forces else None
        for name, f in forces.items():
            self.backend.apply_force(name, f)
        t0 = self.backend.time
        try:
            out = fn() if fn is not None else None
        finally:
            if saved is not None:  # what was there before the push, not zero: nothing of ours is left behind
                self.backend.data.xfrc_applied[:] = saved
        t1 = self.backend.time
        if not self._dt_known and t1 > t0:
            self.backend.set_control_dt(t1 - t0)  # the environment's control period, as it steps
            self._dt_known = True
        self.step_count += 1
        if self.trace:
            self.trace.record_step(self.step_count, self.backend.ctrl(), forces)
        for hook in self._step_hooks:
            hook(self)
        self._check_invariants()
        return out


def model_bytes(model: mujoco.MjModel) -> bytes:
    """The model as a self-contained MuJoCo binary (``.mjb``), which the trace keeps for redrawing."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.mjb"
        mujoco.mj_saveModel(model, str(p), None)
        return p.read_bytes()


def _physics_of(env):
    """The MjModel and MjData of an environment: gymnasium's MuJoCo envs, or dm_control's physics."""
    core = getattr(env, "unwrapped", env)
    model, data = getattr(core, "model", None), getattr(core, "data", None)
    if isinstance(model, mujoco.MjModel) and isinstance(data, mujoco.MjData):
        return model, data
    physics = getattr(core, "physics", None) or getattr(getattr(core, "_env", None), "physics", None)
    if physics is not None:  # dm_control: the raw structs are behind .ptr
        return physics.model.ptr, physics.data.ptr
    raise ValueError(f"cannot find a MuJoCo model in {type(core).__name__}; pass rw.attach(model=..., data=...) for a bare simulation")


def attach(
    env=None,
    *,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    step: Callable[[], object] | None = None,
    advance: Callable[[], object] | None = None,
    name: str = "attached",
    seed: int = 0,
    settings: Settings | None = None,
) -> AttachedWorld:
    """Attach to an environment (``env``), or to a bare ``model``/``data`` pair.

    ``advance`` is one control step of their loop, for robowright to call when a test waits.
    With ``model``/``data`` and ``step``, ``step`` is the bare stepper: ``world.step()`` calls it and
    observes the result, and their own loop should step through ``world.step()`` too.
    """
    if env is not None and (model is not None or data is not None):
        raise ValueError("attach takes an environment, or a model and data, not both")
    if env is None and (model is None or data is None):
        raise ValueError("attach needs an environment, or model= and data=")
    if step is not None and advance is not None:
        raise ValueError("pass step= (a bare model's stepper) or advance= (a loop to wait through), not both")
    if env is not None:
        model, data = _physics_of(env)
    backend = AttachedBackend(model, data, seed=seed)
    world = AttachedWorld(backend, advance, name=_safe(name), seed=seed, settings=settings)
    if env is not None:
        _wrap_env(env, world)
    elif step is not None:
        world._advance = lambda: world._physics_step(step)
    _current.set(world)
    return world


def _wrap_env(env, world: AttachedWorld) -> None:
    """Route the environment's own ``step`` and ``reset`` through the world. See the module docstring."""
    orig_step, orig_reset = env.step, env.reset

    def step(*args, **kwargs):
        return world._physics_step(lambda: orig_step(*args, **kwargs))

    def reset(*args, **kwargs):
        out = orig_reset(*args, **kwargs)
        world._on_reset(kwargs.get("seed"))
        return out

    env.step = step
    env.reset = reset


class AttachedRenderer:
    """Redraws an attached trace from its recorded joint state, with the model it was recorded with."""

    def __init__(self, trace: str | Path | Trace):
        self.trace = tr = trace if isinstance(trace, Trace) else Trace(trace)
        with zipfile.ZipFile(tr.path) as z:
            raw = z.read("model.mjb")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.mjb"
            p.write_bytes(raw)
            self.model = mujoco.MjModel.from_binary_path(str(p))
        self.data = mujoco.MjData(self.model)
        self.steps = len(tr.arrays["t"])
        self._renderer, self._renderer_size = None, None

    @property
    def cameras(self) -> list[str]:
        return [n for n in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)) if n]

    def pose(self, step: int) -> None:
        self.data.qpos[:] = self.trace.arrays["qpos"][step]
        mujoco.mj_forward(self.model, self.data)

    def frame(self, step: int, camera: str | None = None, size=(640, 480)) -> np.ndarray:
        self.pose(step)
        if self._renderer is None or self._renderer_size != tuple(size):
            self._renderer = mujoco.Renderer(self.model, int(size[1]), int(size[0]))
            self._renderer_size = tuple(size)
        if camera is None:  # a model with no cameras is drawn from the free camera (id -1)
            camera = self.cameras[0] if self.cameras else -1
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def frames(self, camera: str | None = None, size=(640, 480), every: int = 1):
        for step in range(0, self.steps, every):
            yield step, self.frame(step, camera, size)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
