"""Attach robowright to a simulation someone else built, and test it as it is.

The engine is found from the environment: MuJoCo (gymnasium, dm_control, robosuite), SAPIEN (ManiSkill 3),
or PyBullet (panda-gym, gym-pybullet-drones, or a bare client with ``engine="pybullet", client=...``). Each engine is one
adapter on the contract in :mod:`robowright.backends.attached_engine`; nothing here reads an engine directly.


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

import io
import json
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np

from . import _current
from .backends import base as backends
from .backends.attached_backend import AttachedBackend
from .backends.attached_engine import AttachedEngine
from .errors import CapabilityError
from .faults import Faults
from .locators import Subject
from .scene import ObjectSpec, SceneSpec
from .trace import Recorder, Trace
from .world import Settings, World, _safe


class AttachedSubject(Subject):
    """A named thing in the environment's simulation: position, orientation, velocity and contacts from ground truth."""

    def __init__(self, world: AttachedWorld, handle, name: str):
        self.world = world
        self.handle = handle  # what the engine's resolve() returned for the name
        self.name = name

    @property
    def _b(self) -> AttachedEngine:
        return self.world.backend

    @property
    def position(self) -> np.ndarray:
        return self._b.pose(self.handle)[0]

    @property
    def quaternion(self) -> np.ndarray:
        return self._b.pose(self.handle)[1]

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
        return self._b.velocity(self.handle)

    def bounds(self):
        raise CapabilityError(f"{self.name!r} has no bounds: a body is not a box in this simulation; test its position instead")

    def contacts(self) -> list[str]:
        """What this touches, by the names the engine gives it (MuJoCo: body and geom names).

        Environments often test contacts between geoms (gym-aloha's reward does), so both names are
        listed where the engine has them, and ``to_be_touching`` accepts either.
        """
        self.world.require("contacts", "contact queries")
        return self._b.contacts_of(self.handle)


class AttachedJoint(Subject):
    """A hinge or slide joint by name: its scalar value (a hinge angle in rad, a slide in m), read from the state."""

    def __init__(self, world: AttachedWorld, name: str):
        self.world = world
        self.name = name

    @property
    def value(self) -> float:
        return self.world.backend.joint_value(self.name)


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


# Marks a physics step that the environment's loop took below env.step (world.observe()): no action to record.
_OBSERVE = object()


class AttachedRecorder(Recorder):
    """The standard trace, plus what the engine keeps so a trace can be redrawn from its state (a MuJoCo model),
    and what the environment needs to be rebuilt and stepped again (its spec, its actions)."""

    def meta(self) -> dict:
        return {**super().meta(), "attached": self.world.backend.trace_meta(), "env": self.world.env_record()}

    def save(self, path: str | Path) -> Path:
        path = super().save(path)
        files = {**self.world.backend.trace_files(), **self.world.controls_files()}
        if files:
            with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as z:
                for name, data in files.items():
                    z.writestr(name, data)
        return path


class AttachedWorld(World):
    """A World over a simulation robowright did not build. Its ``backend`` is an :class:`AttachedEngine`."""

    def __init__(self, backend: AttachedEngine, advance: Callable[[], object] | None, *, name: str, seed: int, settings: Settings | None):
        self.backend = backend
        self.name = name
        self.seed = seed
        self.settings = settings or Settings()
        self.rng = np.random.default_rng(seed)
        self.backend_options: dict = {}
        self.source: str | None = None
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
        # How the environment was made, and what its loop did each step, so codegen and replay can do it again.
        self.env_spec: dict | None = None
        self.controls: list[tuple[int, str, np.ndarray | None]] = []  # (step, "action" | "observe", action)
        self.controls_note: str | None = None  # why the run cannot be stepped again, if it cannot
        self._action_shape: tuple | None = None

    # --- what the simulation is -------------------------------------------------
    @property
    def model(self) -> mujoco.MjModel:
        if not isinstance(self.backend, AttachedBackend):
            raise CapabilityError(f"the {self.backend.engine} engine has no MuJoCo model; use world.backend for its state")
        return self.backend.model

    @property
    def data(self) -> mujoco.MjData:
        if not isinstance(self.backend, AttachedBackend):
            raise CapabilityError(f"the {self.backend.engine} engine has no MuJoCo data; use world.backend for its state")
        return self.backend.data

    def bodies(self) -> list[str]:
        """Every named body: the names a test can look up with ``world.body(name)``."""
        return self.backend.names("body")

    def body(self, name: str) -> AttachedSubject:
        """A body, site or geom by name, as a subject for ``expect``."""
        found = self.backend.resolve(name)
        if found is None:
            raise KeyError(f"no body, site or geom named {name!r}; bodies: {', '.join(self.bodies()) or 'none'}")
        return AttachedSubject(self, found, name)

    def joint(self, name: str) -> AttachedJoint:
        """A hinge or slide joint by name, for ``expect(...).to_be_between``: its value, not a position."""
        self.backend.joint_qadr(name)  # raises the engine's own KeyError or CapabilityError for a bad name
        return AttachedJoint(self, name)

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

    def set_joint(self, name: str, value: float) -> None:
        """Move a hinge or slide joint to ``value`` now, as a recorded fault: for bodies that no free joint moves."""
        self.backend.set_joint(name, value)
        if self.trace:
            self.trace.event("edit", "set_joint", {"joint": name, "value": float(value)})

    def push(self, body: str, force, duration: float = 0.1, at: float | None = None):
        """A world-frame force (N) on a free body from ``at`` for ``duration`` s, applied to the physics steps it covers."""
        if body not in self.object_names:
            raise ValueError(f"push moves free bodies; {body!r} is not one (free bodies: {', '.join(self.object_names) or 'none'})")
        return self.faults.push(body, force, at=at, duration=duration)

    def observe(self) -> None:
        """Record one physics step that the environment's own loop took below ``env.step``."""
        self._physics_step(None, pushes=False)

    # --- what a trace needs to run the environment again ----------------------------
    def _note_unreplayable(self, reason: str) -> None:
        self.controls_note = self.controls_note or reason

    def _note_control(self, action) -> None:
        if action is _OBSERVE:
            self.controls.append((self.step_count, "observe", None))
            return
        arr = np.array(action, copy=True)
        if arr.dtype.kind not in "biuf":
            self._note_unreplayable("the environment's actions are not numeric arrays")
        elif self._action_shape is None:
            self._action_shape = arr.shape
        elif arr.shape != self._action_shape:
            self._note_unreplayable("the environment's actions change shape during the run")
        self.controls.append((self.step_count, "action", arr))

    def controls_files(self) -> dict[str, bytes]:
        """``actions.npz``: every physics step as (step number, observed?, action row). Nothing if a step cannot be kept."""
        if not self.controls or self.controls_note:
            return {}
        rows = [a for _, kind, a in self.controls if kind == "action"]
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            steps=np.array([s for s, _, _ in self.controls], dtype=np.int64),
            observe=np.array([kind == "observe" for _, kind, _ in self.controls], dtype=bool),
            action=np.stack(rows) if rows else np.zeros((0,)),
        )
        return {"actions.npz": buf.getvalue()}

    def env_record(self) -> dict:
        """What the trace says about the environment: its gymnasium spec, and whether its actions were kept."""
        return {"spec": self.env_spec, "actions_recorded": self.controls_note is None, "note": self.controls_note}

    def _begin(self):
        if not self._began:
            self._began = True
            # An engine with no state save (PyBullet) starts its trace with no state0, not an error.
            state0 = self.backend.get_state() if backends.STATE in self.backend.capabilities else None
            if self.trace:
                self.trace.begin(state0)

    def _on_reset(self, seed, options=None):
        """The trace starts when the environment is first reset, so its first sample is a real start state."""
        self._begin()
        if self.trace:
            args = {"seed": seed}
            if options is not None:
                try:
                    json.dumps(options)
                    args["options"] = options
                except (TypeError, ValueError):
                    self._note_unreplayable("the reset options are not plain values")
            self.trace.event("log", "env.reset", args)

    def _rebind(self, env) -> None:
        """Point at the environment's current simulation when it has replaced it (a reset that rebuilds a MuJoCo model)."""
        if self.backend.engine != "mujoco":
            self.backend.rebind(env)
            return
        model, data = _physics_of(env)
        if model is self.backend.model and data is self.backend.data:
            return
        old = self.backend
        self.backend = AttachedBackend(model, data, seed=self.seed)
        self.backend.set_control_dt(old.control_dt)

    def _physics_step(self, fn, pushes: bool = True, action=_OBSERVE):
        """One physics step: apply pushes for it, run the environment's own step, then observe the result."""
        self._begin()
        forces = self.faults.forces_for_step(self.step_count) if pushes else {}
        saved = self.backend.data.xfrc_applied.copy() if forces and self.backend.engine == "mujoco" else None
        for name, f in forces.items():
            self.backend.apply_force(name, f)  # MuJoCo: undone in the finally below; other engines: one step only
        t0 = self.backend.time
        try:
            out = fn() if fn is not None else None
        finally:
            if saved is not None:  # what was there before the push, not zero: nothing of ours is left behind
                self.backend.data.xfrc_applied[:] = saved
        self.backend.note_step()
        t1 = self.backend.time
        if not self._dt_known and t1 > t0:
            self.backend.set_control_dt(t1 - t0)  # the environment's control period, as it steps
            self._dt_known = True
        self.step_count += 1
        if self.trace:
            self.trace.record_step(self.step_count, self.backend.ctrl(), forces)
            self._note_control(action)
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


def _raw(wrapper, attr: str):
    """The MuJoCo struct a binding wrapper holds, or the object itself when it already is one."""
    if isinstance(wrapper, (mujoco.MjModel, mujoco.MjData)):
        return wrapper
    return getattr(wrapper, attr, None)


def _physics_of(env):
    """The MjModel and MjData of an environment: gymnasium's MuJoCo envs, or dm_control's physics."""
    core = getattr(env, "unwrapped", env)
    model, data = getattr(core, "model", None), getattr(core, "data", None)
    if isinstance(model, mujoco.MjModel) and isinstance(data, mujoco.MjData):
        return model, data
    sim = getattr(core, "sim", None)  # robosuite: sim.model._model and sim.data._data are the raw MuJoCo structs
    if sim is not None:
        model, data = _raw(getattr(sim, "model", None), "_model"), _raw(getattr(sim, "data", None), "_data")
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
    engine: str | None = None,
    client: int | None = None,
    dt: float | None = None,
    name: str = "attached",
    seed: int = 0,
    settings: Settings | None = None,
) -> AttachedWorld:
    """Attach to an environment (``env``), a bare ``model``/``data`` pair, or a PyBullet client.

    The engine is found from the environment: a MuJoCo model and data it holds, else a SAPIEN scene
    (ManiSkill 3), else a PyBullet client (panda-gym). ``engine="pybullet", client=<physics client id>``
    names one directly, and ``engine="sapien"`` requires a SAPIEN scene.

    ``advance`` is one control step of their loop, for robowright to call when a test waits.
    With ``model``/``data`` or ``client`` and ``step``, ``step`` is the bare stepper: ``world.step()``
    calls it and observes the result, and their own loop should step through ``world.step()`` too.
    ``dt`` is the control period in seconds: needed for a bare PyBullet client, where nothing in
    PyBullet says how many physics steps make one control step.
    """
    if engine not in (None, "mujoco", "pybullet", "sapien"):
        raise ValueError(f"engine must be 'mujoco', 'pybullet' or 'sapien', not {engine!r}")
    if env is not None and (model is not None or data is not None or client is not None):
        raise ValueError("attach takes an environment, or a model and data, or a client, not more than one")
    if env is None and client is None and (model is None or data is None):
        raise ValueError("attach needs an environment, or model= and data=, or engine='pybullet' with client=")
    if client is not None and (engine == "mujoco" or model is not None or data is not None):
        raise ValueError("a PyBullet client is not a MuJoCo model: pass engine='pybullet' and client=, or model= and data=")
    if step is not None and advance is not None:
        raise ValueError("pass step= (a bare model's stepper) or advance= (a loop to wait through), not both")
    if env is not None:
        backend = _engine_of_env(env, seed=seed, dt=dt, engine=engine)
    elif client is not None:
        from .backends.attached_pybullet import PyBulletEngine

        backend = PyBulletEngine(client, seed=seed, dt=dt)
    else:
        backend = AttachedBackend(model, data, seed=seed)
    world = AttachedWorld(backend, advance, name=_safe(name), seed=seed, settings=settings)
    if env is not None:
        world.env_spec = _env_spec(env)
        _wrap_env(env, world)
    elif step is not None:
        world._advance = lambda: world._physics_step(step)
        world._note_unreplayable("the steps were taken by a bare stepper (step=), not by an environment's step")
    _current.set(world)
    return world


def _engine_of_env(env, *, seed: int, dt: float | None, engine: str | None) -> AttachedEngine:
    """The adapter for an environment: its MuJoCo model and data if it has them, else its SAPIEN scene, else its PyBullet client."""
    from .backends.attached_sapien import sapien_engine_of

    if engine == "sapien":
        found = sapien_engine_of(env, seed=seed, dt=dt)
        if found is None:
            name = type(getattr(env, "unwrapped", env)).__name__
            raise ValueError(f"cannot find a SAPIEN scene in {name}: it has no scene with actors, articulations and px")
        return found
    if engine != "pybullet":
        try:
            model, data = _physics_of(env)
            return AttachedBackend(model, data, seed=seed)
        except ValueError as mujoco_error:
            if engine == "mujoco":
                raise
            found = sapien_engine_of(env, seed=seed, dt=dt)
            if found is not None:
                return found
            from .backends.attached_pybullet import pybullet_engine_of

            found = pybullet_engine_of(env, seed=seed, dt=dt)
            if found is None:
                raise mujoco_error from None
            return found
    from .backends.attached_pybullet import pybullet_engine_of

    found = pybullet_engine_of(env, seed=seed, dt=dt)
    if found is None:
        name = type(getattr(env, "unwrapped", env)).__name__
        raise ValueError(f"cannot find a PyBullet client in {name}; pass attach(engine='pybullet', client=...)")
    return found


def _env_spec(env) -> dict | None:
    """The gymnasium spec an environment was made from (``gymnasium.make`` sets ``env.spec``), or None.

    ``entry_point`` names the module the environment's class lives in, so a rebuild in a fresh process can import
    the package that registered it. ``max_episode_steps`` is what this make used, and ``registered_max_episode_steps``
    the registry's default: a rebuild passes the first only when it differs from the second.
    """
    spec = getattr(env, "spec", None) or getattr(getattr(env, "unwrapped", None), "spec", None)
    env_id = getattr(spec, "id", None)
    if not isinstance(env_id, str):
        return None
    entry = getattr(spec, "entry_point", None)
    if not isinstance(entry, str) and callable(entry):  # a class object: the module and name it can be imported by
        entry = f"{entry.__module__}:{entry.__qualname__}"
    record = {
        "id": env_id,
        "kwargs": dict(getattr(spec, "kwargs", None) or {}),
        "entry_point": entry if isinstance(entry, str) else None,
        "max_episode_steps": getattr(spec, "max_episode_steps", None),
        "registered_max_episode_steps": _registered_max_episode_steps(env_id),
        "reproducible": True,
    }
    try:
        json.dumps(record)
    except (TypeError, ValueError):
        return {**record, "kwargs": {}, "reproducible": False, "reason": "the make() kwargs are not plain values"}
    return record


def _registered_max_episode_steps(env_id: str):
    try:
        import gymnasium

        return gymnasium.spec(env_id).max_episode_steps
    except Exception:  # noqa: BLE001 - not registered in this process: the rebuild passes the recorded limit as it is
        return None


def _wrap_env(env, world: AttachedWorld) -> None:
    """Route the environment's own ``step`` and ``reset`` through the world. See the module docstring."""
    orig_step, orig_reset = env.step, env.reset

    def step(*args, **kwargs):
        action = args[0] if args else kwargs.get("action")
        return world._physics_step(lambda: orig_step(*args, **kwargs), action=action)

    def reset(*args, **kwargs):
        out = orig_reset(*args, **kwargs)
        world._rebind(env)  # robosuite builds a new MjModel and MjData on every reset
        seed = kwargs.get("seed", args[0] if args else None)
        options = kwargs.get("options", args[1] if len(args) > 1 else None)
        world._on_reset(seed, options)
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
