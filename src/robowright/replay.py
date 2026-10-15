"""Deterministic replay of a trace.

Replay rebuilds the scene from the trace, restores the exact starting state,
and feeds the recorded motor commands and external forces back in step by
step. No IK, no policy, no test code runs - so if the replayed states match
the recorded ones, the run is reproducible, and if they diverge you learn
exactly at which step.
"""

from __future__ import annotations

import importlib
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np

from .backends import base as backends
from .trace import Trace


@dataclass
class ReplayResult:
    steps: int
    max_qpos_error: float  # radians
    max_object_error: float  # metres
    first_divergent_step: int | None  # first step above tolerance, if any
    qpos_error: np.ndarray = field(repr=False)
    object_error: np.ndarray = field(repr=False)
    notes: list[str] = field(default_factory=list)  # what was not replayed, or differs, besides the states
    max_body_error: float = 0.0  # metres, over every body's position, for an attached run whose engine cannot save its state
    max_body_angle: float = 0.0  # radians, over every body's orientation, the same way
    engine: str | None = None  # the engine an attached run was replayed on; None for a robowright-built run

    @property
    def identical(self) -> bool:
        return self.max_qpos_error == 0.0 and self.max_object_error == 0.0 and self.max_body_error == 0.0 and self.max_body_angle == 0.0

    @property
    def matches(self) -> bool:
        return self.first_divergent_step is None

    def summary(self) -> str:
        if self.engine is not None and self.engine != "mujoco":
            # Such an engine is rebuilt and stepped again, not restored: its replay is stated as "within", never "identical".
            worst = (
                f"joints {self.max_qpos_error:.2e} rad, bodies {self.max_body_error:.2e} m, "
                f"orientations {self.max_body_angle:.2e} rad, objects {self.max_object_error:.2e} m"
            )
            if self.first_divergent_step is None:
                text = f"replayed {self.steps} steps on {self.engine}: the run agrees with the recorded one within tolerance ({worst})"
            else:
                text = (
                    f"replayed {self.steps} steps on {self.engine}: differs from the recorded run, first above tolerance at "
                    f"step {self.first_divergent_step} ({worst})"
                )
        elif self.identical:
            text = f"replayed {self.steps} steps: bit-identical"
        else:
            div = "" if self.first_divergent_step is None else f", first divergence above tolerance at step {self.first_divergent_step}"
            text = (
                f"replayed {self.steps} steps: max joint error {self.max_qpos_error:.2e} rad, "
                f"max object error {self.max_object_error:.2e} m{div}"
            )
        verdict = "final state matches the recorded one" if self.matches else "MISMATCH: the replayed run diverged from the recorded one"
        return "\n".join([text, verdict, *(f"note: {n}" for n in self.notes)])


def _is_edit(e) -> bool:
    return e["type"] == "edit" or (e["type"] == "fault" and e["name"] == "WeakJoint")


def replay(trace: str | Path | Trace, backend: str | None = None, tol: float = 1e-6, make_env=None) -> ReplayResult:
    """Run a trace again and compare every step with the recorded one.

    An attached run is rebuilt by calling ``make_env()`` (default: ``gymnasium.make`` with the recorded id and kwargs),
    then its recorded resets, actions and the calls the test made are repeated on it. ``backend`` does not apply.

    Two engines take their steps in their own code, so ``make_env`` returns the simulation for them: a Drake
    ``Simulator`` (or a ``(Simulator, plant)`` pair), or a built ``genesis.Scene``. Replay steps it with the recorded
    clock (Drake: ``AdvanceTo`` at each recorded time; Genesis: ``scene.step()``), resets, moves, joint edits and the
    forces each step had. What it cannot rebuild is whatever the run's own code set on the simulation between steps
    (Drake actuation inputs, controllers wired outside the diagram, Genesis joint targets): those are not in the trace,
    so ``make_env`` must set the same ones, or the replay diverges and says where.
    """
    tr = trace if isinstance(trace, Trace) else Trace(trace)
    meta = tr.meta
    if "attached" in meta:
        return _replay_attached(tr, make_env, tol)
    from . import fidelity

    with fidelity.using(meta.get("fidelity")):  # rebuilt as it ran
        # The options the run's backend was created with (a ROS 2 robot's settings); another engine gets none.
        opts = {} if backend and backend != meta["backend"] else meta.get("backend_options") or {}
        b = backends.create(backend or meta["backend"], tr.scene(), seed=meta["seed"], **opts)
    try:
        restored = tr.state0 is not None and backends.STATE in b.capabilities
        if restored:
            b.set_state(tr.state0)
        begin = meta.get("begin_event_index", 0)
        edits: dict[int, list] = {}
        for idx, e in enumerate(tr.events):
            if not _is_edit(e):
                continue
            # Edits made before recording began are already in state0 - except
            # gain changes, which live in the model rather than the state.
            if restored and idx < begin and e["type"] == "edit":
                continue
            edits.setdefault(e["step"], []).append(e)
        a = tr.arrays
        names = meta["object_names"]
        n = len(a["t"])
        qerr = np.zeros(n)
        oerr = np.zeros(n)
        for i in range(1, n):
            _apply(b, edits.get(i - 1, []))
            b.set_ctrl(a["ctrl"][i])
            if "led" in a:  # the servos were led along this step's move, as the run's backend did
                b.feedforward = bool(a["led"][i])
            for k, name in enumerate(meta.get("force_names", names)):
                f = a["forces"][i][k]
                if np.any(f):
                    b.apply_force(name, f)
            b.step()
            qerr[i] = float(np.max(np.abs(b.qpos() - a["qpos"][i])))
            if names and backends.GROUND_TRUTH in b.capabilities:
                pos = np.array([b.object_pose(nm)[0] for nm in names])
                oerr[i] = float(np.max(np.linalg.norm(pos - a["obj_pos"][i], axis=1)))
        bad = np.nonzero((qerr > tol) | (oerr > tol))[0]
        return ReplayResult(n - 1, float(qerr.max()), float(oerr.max()), int(bad[0]) if bad.size else None, qerr, oerr)
    finally:
        b.close()


def _apply(b, events):
    for e in events:
        if e["type"] == "fault":
            b.set_gain_scale(e["args"]["joint"], e["args"]["scale"])
        elif e["name"] == "move_object":
            b.set_object_pose(e["args"]["object"], e["args"]["pos"], e["args"]["quat"])
        elif e["name"] == "reset_to":
            if "base_pos" in e["args"]:
                yaw = e["args"].get("yaw", 0.0)
                b.set_base_pose(e["args"]["base_pos"], (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
            b.set_joint_positions(np.array(e["args"]["q"]))


def schedule(tr: Trace) -> list[dict]:
    """An attached run as the calls that made it, in order: resets, the environment's steps, and the test's own calls.

    Physics step ``s`` happens after step ``s-1`` and before ``s+1``; an event recorded at step ``c`` happens after
    step ``c``. Ties keep the order they were recorded in. Expectations and invariants are kept at the step where
    they were recorded, so an expectation is checked against the same state it passed in.
    """
    items: list[tuple[tuple, dict]] = []
    for i, e in enumerate(tr.events):
        if e["type"] == "log" and e["name"] == "env.reset":
            items.append(((e["step"], 0, i), {"kind": "reset", "args": e.get("args") or {}}))
        elif e["type"] in ("expect", "invariant", "edit", "fault"):
            items.append(((e["step"], 0, i), {"kind": "event", "event": e}))
    for s, kind, row in tr.controls:
        items.append(((s - 1, 1, s), {"kind": "control", "step": s, "observe": kind == "observe", "action": row}))
    return [item for _, item in sorted(items, key=lambda kv: kv[0])]


# Engines whose own code takes the steps (a Drake Simulator, a Genesis scene, a Webots world stepped by its Supervisor):
# replayed by stepping the rebuilt simulation, with the forces the trace kept, instead of env.step(action) for a
# gymnasium-style environment.
STEPPED = ("drake", "genesis", "webots")


def _replay_attached(tr: Trace, make_env, tol: float) -> ReplayResult:
    import mujoco

    from .attached import _is_drake_simulator, _physics_of, attach, model_bytes, restore_rng
    from .world import Settings

    if not (tr.meta.get("env") or {}).get("actions_recorded", True):
        raise ValueError(f"this run cannot be replayed: {tr.meta['env']['note']}")
    bare = bool((tr.meta.get("env") or {}).get("bare"))
    if make_env is None:
        make_env = _bare_from_trace(tr) if bare else _make_from_spec(tr)
    recorded = tr.meta["attached"].get("engine")
    built = make_env()
    if built is None:
        raise ValueError("make_env returned None: a factory must return the built simulation or environment")
    plant = None
    if isinstance(built, tuple):  # a Drake simulation may come with its plant, when its diagram holds more than one
        built, plant = built
    step_fn = None  # a Webots world's step, taken before attach wraps the environment's step
    if bare:
        # A bare model: the caller's own mj_step moved it, so the replay takes that step itself, with the control the trace kept.
        if isinstance(built, mujoco.MjModel):
            built = (built, mujoco.MjData(built))
        if isinstance(built, tuple):
            model, data = built
        else:
            model, data = _physics_of(built)
        world = attach(model=model, data=data, seed=tr.meta["seed"], settings=Settings(trace="off"))
        env = None
        if tr.state0 is not None:  # the start state the recorded run began from, before its first step
            world.backend.set_state(tr.state0)
    elif recorded == "drake":
        if not _is_drake_simulator(built):
            raise ValueError("this run is a Drake simulation: make_env must return its Simulator, or (Simulator, plant)")
        world = attach(simulator=built, plant=plant, seed=tr.meta["seed"], settings=Settings(trace="off"))
    elif recorded == "webots":
        step_fn = built.step
        world = attach(built, engine="webots", seed=tr.meta["seed"], settings=Settings(trace="off"))
    else:
        world = attach(built, seed=tr.meta["seed"], settings=Settings(trace="off"))
    if not bare:
        env = built
    stepped = recorded in STEPPED or bare
    if stepped and not bare and world.backend.engine != recorded:
        raise ValueError(f"this run was recorded on {recorded}, but make_env built a {world.backend.engine} simulation")
    notes: list[str] = []
    if not tr.meta["attached"].get("inputs_recorded", True):
        notes.append("the inputs the simulation's own code sets between steps are not in the trace: make_env must rebuild the same ones")
    skipped: set[str] = set()
    names = tr.meta["object_names"]
    a = tr.arrays
    n = len(a["t"])
    qerr, oerr = np.zeros(n), np.zeros(n)
    bodies = _stored_bodies(tr)  # recorded for an engine that cannot be restored: every body, every step
    berr, bang = np.zeros(n), np.zeros(n)
    engine = world.backend.engine

    def compare(i: int) -> None:
        if i >= n:
            return
        q = world.backend.qpos()
        if q.size:  # a simulation with no joints (a table and a block) has no joint values to compare
            qerr[i] = float(np.max(np.abs(q - a["qpos"][i])))
        if names and backends.GROUND_TRUTH in world.backend.capabilities:
            pos = np.array([world.backend.object_pose(nm)[0] for nm in names])
            oerr[i] = float(np.max(np.linalg.norm(pos - a["obj_pos"][i], axis=1)))
        if bodies is not None:
            b = world.backend
            poses = [b.pose(b.resolve(nm)) for nm in bodies["names"]]
            pos = np.array([p for p, _ in poses])
            quat = np.array([q for _, q in poses])
            berr[i] = float(np.max(np.linalg.norm(pos - bodies["pos"][i], axis=1)))
            bang[i] = float(np.max(_quat_angle(quat, bodies["quat"][i])))

    try:
        stored = None if bare else _stored_model(tr)
        if stored is not None and model_bytes(world.backend.model) != stored:
            notes.append(
                "the environment as rebuilt has a different MuJoCo model from the recorded one (another version or options, "
                "or a model drawn from random numbers when it is built, as dm_control's lqr is)"
            )
        rows = iter([row for _, kind, row in tr.controls if kind == "action"])
        items = schedule(tr)
        compared_start = stepped  # a simulation with its own steps starts from the state make_env built, before any reset
        if stepped:
            compare(world.step_count)
        for item in items:
            if item["kind"] == "reset":
                if recorded == "genesis":
                    env.reset()  # a Genesis scene resets to its built state; it takes no seed
                else:
                    if "rng" in item["args"]:  # the random state the reset drew from when it was recorded
                        restore_rng(env, item["args"]["rng"])
                    kw = _reset_kwargs(env, item["args"], skipped)
                    env.reset(**kw)
                if not compared_start:
                    compare(world.step_count)  # the start state, before any step
                    compared_start = True
            elif item["kind"] == "control":
                if stepped:
                    s = item["step"]
                    if bare:
                        data.ctrl[:] = a["ctrl"][s]  # the control the recorded step was taken with
                        world.replay_step(lambda: mujoco.mj_step(model, data), _recorded_forces(tr, s))
                    elif recorded == "drake":
                        t = float(a["t"][s])  # the recorded clock, so each AdvanceTo lands on the recorded time
                        world.replay_step(lambda t=t: env.AdvanceTo(t), _recorded_forces(tr, s))
                    elif recorded == "webots":
                        world.replay_step(step_fn, _recorded_forces(tr, s))
                    else:
                        world.replay_step(None, _recorded_forces(tr, s))  # the scene's own step
                elif item["observe"]:
                    world.observe()
                else:
                    env.step(next(rows))
                compare(world.step_count)
            else:
                e = item["event"]
                if e["type"] == "edit" and e["name"] == "move_object":
                    world.move_object(e["args"]["object"], e["args"]["pos"], e["args"]["quat"])
                elif e["type"] == "edit" and e["name"] == "set_joint":
                    world.set_joint(e["args"]["joint"], e["args"]["value"])
                elif stepped and e["type"] == "fault" and e["name"] == "Push":
                    pass  # a push was applied to its steps and those steps' forces are in the trace
                elif e["type"] == "fault" and e["name"] == "Push":
                    world.faults.push(e["args"]["object"], e["args"]["force"], at=e["args"]["at"], duration=e["args"]["duration"])
                elif e["type"] == "fault" or (e["type"] == "edit"):
                    skipped.add(f"{e['type']} {e['name']}")
        if skipped:
            notes.append("not replayed: " + ", ".join(sorted(skipped)))
        notes.append("expectations and invariants are not checked again: only the states are compared")
        bad = np.nonzero((qerr > tol) | (oerr > tol) | (berr > tol) | (bang > tol))[0]
        if bad.size:
            notes.append(
                f"diverged at step {int(bad[0])}: if the environment draws random numbers from a generator the trace does not "
                "capture, seed that generator in a --factory module:func that builds it, the way the recorded run was seeded"
            )
        return ReplayResult(
            n - 1,
            float(qerr.max()),
            float(oerr.max()),
            int(bad[0]) if bad.size else None,
            qerr,
            oerr,
            notes,
            max_body_error=float(berr.max()),
            max_body_angle=float(bang.max()),
            engine=engine,
        )
    finally:
        world.close(failed=False)


def load_factory(spec: str):
    """The function a ``module:function`` or ``path/to/file.py:function`` names, for use as ``make_env``.

    A module is imported as any import would be; a file is loaded under a private name, so nothing it defines can
    shadow a real module. Either way the module's own top-level code runs first, as it would for the test.
    """
    import importlib
    import importlib.util
    import inspect

    target, sep, func = spec.rpartition(":")
    if not sep or not target or not func:
        raise ValueError(f"a factory is given as module:function or path/to/file.py:function, not {spec!r}")
    try:
        if target.endswith(".py") or "/" in target or "\\" in target:
            path = Path(target)
            if not path.is_file():
                raise ValueError(f"factory file {target!r} does not exist")
            file_spec = importlib.util.spec_from_file_location(f"robowright_factory_{path.stem}", path)
            module = importlib.util.module_from_spec(file_spec)
            file_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(target)
    except ValueError:
        raise
    except Exception as e:  # the module's own code may fail however it likes: say which module it was
        raise ValueError(f"cannot import the factory's module {target!r}: {type(e).__name__}: {e}") from e
    if not hasattr(module, func):
        raise ValueError(f"module {target!r} has no {func!r}: the factory is a function that builds the simulation")
    fn = getattr(module, func)
    if not callable(fn):
        raise ValueError(f"{target}:{func} is not callable: the factory is a function that builds the simulation")
    try:
        inspect.signature(fn).bind()
    except TypeError as e:
        raise ValueError(f"{target}:{func} must take no arguments and return the simulation: {e}") from None
    return fn


def _recorded_forces(tr: Trace, step: int) -> dict:
    """The external forces the trace kept for one step, by the body they were applied to; zero forces are left out."""
    names = tr.meta.get("force_names", tr.meta["object_names"])
    f = tr.arrays["forces"][step]
    return {name: f[k] for k, name in enumerate(names) if np.any(f[k])}


def _quat_angle(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """The rotation angle between each wxyz quaternion in ``q`` and its partner in ``r`` (rows), in radians.

    Taken from the chord |q - r| rather than from arccos of the dot product: near identity arccos turns a
    rounding error of 1e-16 in the dot product into an angle of about 4e-8 rad, which would read as a difference.
    """
    sign = np.where(np.sum(q * r, axis=1) < 0.0, -1.0, 1.0)  # q and -q are the same rotation
    chord = np.linalg.norm(q - sign[:, None] * r, axis=1)
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def _stored_bodies(tr: Trace) -> dict | None:
    with zipfile.ZipFile(tr.path) as z:
        if "bodies.npz" not in z.namelist():
            return None
        with np.load(BytesIO(z.read("bodies.npz"))) as f:
            return {"names": [str(x) for x in f["names"]], "pos": f["pos"], "quat": f["quat"]}


def make_spec_call(spec: dict) -> tuple[str, dict]:
    """The id and keyword arguments ``gymnasium.make`` needs to rebuild a recorded environment in a fresh process.

    A third-party environment is named ``"<module>:<id>"``: gymnasium imports that module first, which registers the
    environment. Without it a fresh process only knows ids that something else has already imported.
    """
    env_id = spec["id"]
    entry = spec.get("entry_point")
    if isinstance(entry, str) and ":" in entry:
        module = entry.split(":", 1)[0].split(".", 1)[0]
        if module != "gymnasium":
            env_id = f"{module}:{env_id}"
    kwargs = dict(spec.get("kwargs") or {})
    limit = spec.get("max_episode_steps")
    if limit is not None and limit != spec.get("registered_max_episode_steps"):
        kwargs["max_episode_steps"] = limit  # the make() passed a limit other than the default (None: no limit was recorded)
    return env_id, kwargs


def _bare_from_trace(tr: Trace):
    """The model a bare run was recorded with, rebuilt from the copy the trace keeps (its start state is set by the replay)."""

    def make():
        import tempfile

        import mujoco

        raw = _stored_model(tr)
        if raw is None:
            raise ValueError("this trace keeps no model: pass --factory module:func that builds the model and data")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.mjb"
            p.write_bytes(raw)
            model = mujoco.MjModel.from_binary_path(str(p))
        return model, mujoco.MjData(model)

    return make


def _reset_params(env) -> set | None:
    """The argument names the environment's reset takes, or None when it takes any (``**kwargs``)."""
    import inspect

    try:
        params = inspect.signature(inspect.unwrap(env.reset)).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


def _reset_kwargs(env, args: dict, skipped: set) -> dict:
    """The arguments a recorded reset was made with, limited to those this environment's reset takes: a dm_control
    reset takes no seed, and passing one would fail the replay before it started."""
    takes = _reset_params(env)
    kw = {}
    for key in ("seed", "options"):
        if key not in args:
            continue
        if takes is None or key in takes:
            kw[key] = args[key]
        elif args[key] is not None:
            skipped.add(f"reset {key}={args[key]!r} (the environment's reset takes no {key})")
    return kw


def _make_from_spec(tr: Trace):
    env_rec = tr.meta.get("env") or {}
    spec = env_rec.get("spec")
    if not spec or not spec.get("reproducible") or not env_rec.get("actions_recorded", True):
        why = env_rec.get("note") or (spec or {}).get("reason") or "this run's environment was not made by gymnasium.make"
        raise ValueError(
            f"cannot rebuild this run's environment ({why}): "
            "pass make_env=... (or --factory module:func) that builds it the way the run did"
        )
    if spec.get("library") == "dm_control.suite":
        domain, task = dm_control_task(spec)

        def load():
            from dm_control import suite

            return suite.load(domain, task)

        return load
    # An environment registered with the old gym 0.x package is rebuilt with gym.make, not gymnasium.make.
    library = spec.get("library", "gymnasium")
    try:
        make = importlib.import_module(library).make
    except ImportError as e:
        raise ValueError(
            f"this run's environment is {spec['id']!r}, rebuilt with {library}, which is not installed: install it or pass make_env=..."
        ) from e
    env_id, kwargs = make_spec_call(spec)

    def build():
        try:
            return make(env_id, **kwargs)
        except Exception as e:  # a package that registers the id may fail to import here: say which, and the way out
            raise ValueError(
                f"cannot rebuild this run's environment ({spec['id']!r} with {library}.make: "
                f"{type(e).__name__}: {e}); "
                "pass make_env=... (or --factory module:func) that builds it the way the run did"
            ) from e

    return build


def dm_control_task(spec: dict) -> tuple[str, str]:
    """The dm_control suite (domain, task) a recorded run's environment was loaded as: the first of the domain's
    tasks whose task class and model shape are the run's (tasks that differ only in their reward share both, and
    step alike)."""
    try:
        from dm_control import suite
    except ImportError as e:
        raise ValueError("this run's environment is a dm_control suite task, and dm_control is not installed") from e
    rec = spec["dm_control"]
    for domain, task in suite.ALL_TASKS:
        if domain != rec["domain"]:
            continue
        env = suite.load(domain, task)
        m = env.physics.model
        if type(env.task).__name__ == rec["task_class"] and [m.nq, m.nv, m.nu, m.nbody, m.ngeom] == list(rec["shape"]):
            return domain, task
    raise ValueError(
        f"no dm_control suite task of domain {rec['domain']!r} has this run's task ({rec['task_class']}) and model: "
        "pass make_env=... (or --factory module:func) that loads it"
    )


def _stored_model(tr: Trace) -> bytes | None:
    with zipfile.ZipFile(tr.path) as z:
        return z.read("model.mjb") if "model.mjb" in z.namelist() else None
