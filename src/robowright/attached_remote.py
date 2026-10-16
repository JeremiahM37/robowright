"""``rw.attach`` on a vector environment whose sub-environments run in worker processes: gymnasium's ``AsyncVectorEnv``
and Stable-Baselines3's ``SubprocVecEnv``.

A worker's simulation is in its own process, so robowright cannot read it in place as it reads a ``SyncVectorEnv``'s.
Each sub-environment is wrapped, inside its worker, by :func:`attachable`; robowright then reaches it through the
vector environment's own pipes (``vec.call``, ``vec.env_method``), as Playwright reaches a browser it did not start
in-process::

    vec = gymnasium.make_vec("HalfCheetah-v5", 4, vectorization_mode="async", wrappers=[rw.attachable])
    worlds = rw.attach(vec)

In the test process each index is an ordinary attached world: its subjects, expectations, invariants, faults and trace
are the same as a ``SyncVectorEnv``'s. Only the engine is remote: :class:`RemoteEngine` asks the worker's own engine
adapter (the one ``rw.attach(env)`` would use there). Reads are taken once per vector step: what one step read is
fetched for every index in one round trip after the next, and a read is served from that until the simulation moves.

The worker side never raises: an exception in a gymnasium worker ends it and closes its environment, so errors come
back as values and are raised in the test process.
"""

from __future__ import annotations

import functools
import json

import numpy as np

from .backends.attached_engine import AttachedEngine
from .errors import CapabilityError

MARK = "RobowrightAttachable"  # the worker wrapper's class name, as gymnasium prints a wrapped environment


def attachable(env):
    """Wrap a sub-environment so ``rw.attach`` can reach it from another process: ``gymnasium.make_vec(...,
    vectorization_mode="async", wrappers=[rw.attachable])``, or ``AsyncVectorEnv([lambda: rw.attachable(make()), ...])``.
    Without a ``rw.attach``, it passes everything through unchanged."""
    return _wrapper_class()(env)


@functools.cache
def _wrapper_class():
    import gymnasium

    class RobowrightAttachable(gymnasium.Wrapper):
        """robowright's side of an ``AsyncVectorEnv`` worker (see the module)."""

        _rw_token = None  # given out once, when rw.attach first asks: requests for this worker are filed under it

        def __init__(self, env):
            super().__init__(env)
            self._rw_engine = None
            self._rw_handles: dict = {}
            self._rw_pushes: dict = {}
            self._rw_events: list = []
            self._rw_watch: list = []  # the reads robowright takes after a step (see reset)

        # --- the environment's own calls -----------------------------------------------
        def step(self, action):
            if self._rw_engine is not None:  # attached (until then nothing is kept)
                self._rw_events.append(("step",))
            e, pushes, self._rw_pushes = self._rw_engine, self._rw_pushes, {}
            saved = None
            if pushes and e is not None and e.engine == "mujoco" and not e.pending:
                saved = e.data.xfrc_applied.copy()  # put back after the step: nothing of robowright's is left behind
            for name, force in pushes.items():
                e.apply_force(name, force)
            try:
                return self.env.step(action)
            finally:
                if saved is not None:
                    e.data.xfrc_applied[:] = saved

        def reset(self, *, seed=None, options=None):
            from .attached import _rng_seeded, capture_rng

            if self._rw_events and self._rw_events[-1] == ("step",):
                # A reset in the same vector step as a physics step (gymnasium's SAME_STEP autoreset): the state that
                # step ended in is gone by the time robowright reads, so what it reads is kept for that step now.
                self._rw_events[-1] = ("step", [(key, self._rw_try(key)) for key in self._rw_watch])
            seeded = _rng_seeded(self.env)
            rng = capture_rng(self.env) if self._rw_engine is not None else None
            self._rw_pushes = {}
            out = self.env.reset(seed=seed, options=options)
            if self._rw_engine is not None:  # attached: the reset goes on the trace, and the simulation may be a new one
                try:
                    json.dumps(options)
                except (TypeError, ValueError):
                    options = "<not plain values>"
                try:
                    rebound, error = self._rw_bind(after_reset=True), None
                except Exception as ex:  # noqa: BLE001 - raised in the test process, which reads the event
                    rebound, error = False, str(ex)
                self._rw_events.append(("reset", seed, options, rng, seeded, rebound, error))
            return out

        # --- robowright's calls: vec.call("robowright", {token: [(method, args), ...]}) to every worker -----
        def robowright(self, requests):
            if requests == "hello":  # who is this worker: one call to all of them, answered in index order
                if self._rw_token is None:
                    import os

                    self._rw_token = os.urandom(8).hex()
                return self._rw_token
            mine = requests.get(self._rw_token)
            if mine is None:
                return None
            return [self._rw_try(key) for key in mine]

        def _rw_try(self, key):
            try:
                return True, self._rw_serve(*key)
            except BaseException as ex:  # noqa: BLE001 - never raised here: it would end the worker
                return False, (type(ex).__name__, str(ex))

        def _rw_bind(self, after_reset: bool = False) -> bool:
            """Bind to the environment's simulation, or follow it to a new one. True when the simulation is another."""
            from .attached import _engine_of_env, _physics_of
            from .backends.attached_backend import AttachedBackend, PendingBackend

            e = self._rw_engine
            if e is None or isinstance(e, PendingBackend):
                try:
                    self._rw_engine = _engine_of_env(self.env, seed=0, dt=None, engine=None)
                except ValueError:
                    if after_reset:
                        raise
                    self._rw_engine = PendingBackend()  # built by the first reset (safety-gymnasium)
                    return False
            elif e.engine == "mujoco":
                model, data = _physics_of(self.env)
                if model is e.model and data is e.data:
                    return False
                self._rw_engine = AttachedBackend(model, data)  # robosuite builds a new MjModel and MjData each reset
            else:
                e.rebind(self.env)
                if not e.ensure_bound():
                    return False
            self._rw_handles = {}
            return True

        def _rw_handle(self, name: str):
            h = self._rw_handles.get(name)
            if h is None:
                h = self._rw_engine._handle(name)  # KeyError for a name the simulation does not have
                self._rw_handles[name] = h
            return h

        def _rw_serve(self, method: str, args: tuple):
            from .attached import _env_spec, _is_gym_env, _is_gymnasium_env

            if method == "attach":
                self._rw_events = []
                self._rw_bind()
                return {
                    **self._rw_info(),
                    "spec": _env_spec(self.env),
                    "resets_first": _is_gymnasium_env(self.env) or _is_gym_env(self.env),
                }
            if method == "info":
                return self._rw_info()
            if method == "watch":
                self._rw_watch = list(args[0])
                return None
            if method == "events":
                events, self._rw_events = self._rw_events, []
                return events
            e = self._rw_engine
            if method == "push":
                name, force = args
                self._rw_pushes[name] = self._rw_pushes.get(name, np.zeros(3)) + np.asarray(force, dtype=float)
                return None
            if method in ("pose", "velocity", "contacts_of"):
                return getattr(e, method)(self._rw_handle(args[0]))
            if method == "resolve":
                return args[0] if e.resolve(args[0]) is not None else None
            if method == "time":
                return e.time
            if method in _ENGINE_CALLS:
                out = getattr(e, method)(*args)
                if method in _MUTATIONS:
                    self._rw_handles = {}
                return out
            raise ValueError(f"robowright's worker has no {method!r}")

        def _rw_info(self) -> dict:
            e = self._rw_engine
            return {
                "engine": e.engine,
                "pending": e.pending,
                "capabilities": sorted(e.capabilities),
                "free_bodies": list(e.free_bodies),
                "joint_names": list(e.joint_names),
                "trace_meta": None if e.pending else e.trace_meta(),
                "trace_files": None if e.pending else e.trace_files(),
            }

    return RobowrightAttachable


# What a worker's engine answers by name (handles stay in the worker: a name stands for one here).
_ENGINE_CALLS = frozenset(
    {
        "names", "joint_kinds", "contact_pairs", "joint_value", "named_joints", "qpos", "qvel", "ctrl",
        "render", "get_state", "set_state", "set_joint", "set_object_pose", "contacts", "joint_qadr",
    }
)  # fmt: skip
_MUTATIONS = frozenset({"set_state", "set_joint", "set_object_pose"})
_ERRORS = {"KeyError": KeyError, "ValueError": ValueError, "IndexError": IndexError, "CapabilityError": CapabilityError}


class _Gymnasium:
    """gymnasium's AsyncVectorEnv: ``vec.call`` runs a method in every worker."""

    kind = "AsyncVectorEnv"
    how = (
        "gymnasium.make_vec(..., vectorization_mode='async', wrappers=[rw.attachable]), "
        "or AsyncVectorEnv([lambda: rw.attachable(make_env()), ...])"
    )
    steps, resets, user_calls = ("step_async", "step_wait"), "reset_wait", ("call", "set_attr")

    def __init__(self, vec):
        self.vec, self.n, self._call = vec, vec.num_envs, vec.call

    def call(self, name: str, *args) -> list:
        return list(self._call(name, *args))

    def split(self, actions) -> list:
        from gymnasium.vector.utils import iterate

        return list(iterate(self.vec.action_space, actions))


class _StableBaselines3(_Gymnasium):
    """Stable-Baselines3's SubprocVecEnv: ``vec.env_method`` runs a method in every worker."""

    kind = "SubprocVecEnv"
    how = (
        "SubprocVecEnv([lambda: rw.attachable(make_env()), ...]), "
        "or make_vec_env(..., vec_env_cls=SubprocVecEnv, wrapper_class=rw.attachable)"
    )
    steps, resets, user_calls = ("step_async", "step_wait"), "reset", ("env_method", "set_attr")

    def __init__(self, vec):
        self.vec, self.n, self._call = vec, vec.num_envs, vec.env_method

    def split(self, actions) -> list:
        if isinstance(actions, dict):
            return [{k: v[i] for k, v in actions.items()} for i in range(self.n)]
        return [actions[i] for i in range(self.n)]


def is_attachable(link) -> bool:
    """Whether every sub-environment was wrapped by :func:`attachable`. Asked through ``__str__``, which every gymnasium
    environment has (and prints its wrappers in): a call to a missing name would end the worker."""
    return all(MARK in str(s) for s in link.call("__str__"))


class _Hub:
    """The test process's link to the workers of one vector environment: one round trip per batch of calls, and per index
    a cache of what was read since the simulation last moved."""

    def __init__(self, link):
        self.link = link
        self.n = link.n
        self.tokens = link.call("robowright", "hello")
        self.cache: list[dict] = [{} for _ in range(self.n)]
        self.used: list[set] = [set() for _ in range(self.n)]  # what each index read since its last step
        self.closed = False
        self.watched: list = [None] * self.n  # the reads each worker keeps for a step a reset follows at once

    def call(self, requests: dict) -> dict:
        if self.closed:
            raise CapabilityError("the AsyncVectorEnv this world was attached to is closed")
        results = self.link.call("robowright", {self.tokens[i]: r for i, r in requests.items()})
        return {i: results[i] for i in requests}

    @staticmethod
    def value(result):
        ok, v = result
        if ok:
            return v
        kind, message = v
        raise _ERRORS.get(kind, RuntimeError)(message if kind in _ERRORS else f"in the worker: {kind}: {message}")

    def read(self, i: int, method: str, *args):
        key = (method, args)
        self.used[i].add(key)
        if key not in self.cache[i]:
            self.cache[i][key] = self.call({i: [key]})[i][0]
        return self.value(self.cache[i][key])

    def mutate(self, i: int, method: str, *args):
        out = self.value(self.call({i: [(method, args)]})[i][0])
        self.cache[i].clear()
        return out

    def moved(self) -> list[list]:
        """After a vector step or reset: each index's events, and what it read last time, read again in one round trip."""
        keys = [sorted(self.used[i], key=repr) for i in range(self.n)]
        results = self.call({i: [("events", ()), *keys[i]] for i in range(self.n)})
        events = []
        for i in range(self.n):
            self.cache[i] = dict(zip(keys[i], results[i][1:], strict=True))
            self.used[i] = set()
            events.append(self.value(results[i][0]))
        return events


class RemoteEngine(AttachedEngine):
    """A sub-environment's engine, in its worker process (see the module)."""

    engine = "remote"

    def __init__(self, hub: _Hub, index: int, info: dict, seed: int = 0, dt: float | None = None):
        super().__init__(seed=seed, dt=dt)
        self.hub, self.index = hub, index
        self._rebound = False
        self.adopt(info)

    def adopt(self, info: dict) -> None:
        self.info = info
        self.remote_engine = info["engine"]  # the engine in the worker
        self.capabilities = frozenset(info["capabilities"])
        self.free_bodies = list(info["free_bodies"])
        self.joint_names = list(info["joint_names"])

    def _read(self, method: str, *args):
        return self.hub.read(self.index, method, *args)

    def names(self, kind: str = "body") -> list[str]:
        return self._read("names", kind)

    def joint_kinds(self) -> dict[str, str]:
        return self._read("joint_kinds")

    def resolve(self, name: str):
        return self._read("resolve", name)

    def label(self, handle) -> str:
        return handle

    def pose(self, handle):
        return self._read("pose", handle)

    def velocity(self, handle) -> np.ndarray:
        return self._read("velocity", handle)

    def contacts_of(self, handle) -> list[str]:
        return self._read("contacts_of", handle)

    def contact_pairs(self) -> list[tuple]:
        return self._read("contact_pairs")

    def joint_value(self, name: str) -> float:
        return self._read("joint_value", name)

    def contacts(self) -> list:
        return self._read("contacts")

    def joint_qadr(self, name: str) -> int:
        return self._read("joint_qadr", name)

    def named_joints(self) -> list[str]:
        return self._read("named_joints")

    def ensure_bound(self) -> bool:
        rebound, self._rebound = self._rebound, False
        return rebound

    @property
    def pending(self) -> bool:
        return self.info["pending"]

    def set_joint(self, name: str, value: float) -> None:
        self.hub.mutate(self.index, "set_joint", name, float(value))

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        q = None if quat is None else np.asarray(quat, dtype=float)
        self.hub.mutate(self.index, "set_object_pose", name, np.asarray(pos, dtype=float), q)

    def apply_force(self, name: str, force) -> None:
        self.hub.call({self.index: [("push", (name, np.asarray(force, dtype=float)))]})

    @property
    def time(self) -> float:
        return 0.0 if self.pending else self._read("time")

    def qpos(self) -> np.ndarray:
        return self._read("qpos")

    def qvel(self) -> np.ndarray:
        return self._read("qvel")

    def ctrl(self) -> np.ndarray:
        return self._read("ctrl")

    def trace_meta(self) -> dict:
        return dict(self.info["trace_meta"] or {"engine": self.remote_engine})

    def trace_files(self) -> dict[str, bytes]:
        return dict(self.info["trace_files"] or {})

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        return self._read("render", camera, int(width), int(height))

    def get_state(self) -> np.ndarray:
        return self._read("get_state")

    def set_state(self, state) -> None:
        self.hub.mutate(self.index, "set_state", np.asarray(state))


def attach_remote(vec, *, advance, name: str, seed: int, settings):
    """One world per sub-environment of a vector environment whose sub-environments run in worker processes (gymnasium's
    AsyncVectorEnv, Stable-Baselines3's SubprocVecEnv), each wrapped there by :func:`attachable`."""
    from .attached import AttachedWorld, _current

    link = _Gymnasium(vec) if hasattr(vec, "call") else _StableBaselines3(vec)
    if getattr(vec, "_robowright_hub", None) is not None:
        raise CapabilityError(f"this {link.kind} is attached already: close those worlds first")
    if not is_attachable(link):
        raise CapabilityError(
            f"cannot attach to this {link.kind}: its sub-environments run in worker processes, and robowright reaches one "
            f"only through robowright's wrapper inside it. Make it with {link.how}"
        )
    hub = _Hub(link)
    infos = hub.call({i: [("attach", ())] for i in range(hub.n)})
    worlds = []
    for i in range(hub.n):
        info = hub.value(infos[i][0])
        w = AttachedWorld(RemoteEngine(hub, i, info, seed=seed + i), advance, name=f"{name}[{i}]", seed=seed + i, settings=settings)
        w.index = i
        w.env_spec = info["spec"]
        w._resets_first = info["resets_first"]
        worlds.append(w)
    _hook(vec, hub, worlds)
    _current.set(worlds[0])
    return _remote_worlds(worlds, vec, hub)


def _hook(vec, hub: _Hub, worlds: list) -> None:
    """Route the vector environment's steps and resets through the worlds (instance attributes: its class is untouched)."""
    link = hub.link
    step_async_name, step_wait_name = link.steps
    orig = {k: getattr(vec, k) for k in (*link.steps, link.resets, *link.user_calls)}
    pending: dict = {}

    def step_async(actions):
        pushes, forces, times = {}, [], []
        for i, w in enumerate(worlds):
            f = w.faults.forces_for_step(w.step_count) if not w.backend.pending else {}
            forces.append(f)
            times.append(w.backend.time)
            if f:
                pushes[i] = [("push", (k, np.asarray(v, dtype=float))) for k, v in f.items()]
        for i in range(hub.n):
            keys = sorted(hub.used[i] | set(hub.cache[i]), key=repr)
            if keys != hub.watched[i]:
                hub.watched[i] = keys
                pushes.setdefault(i, []).append(("watch", (keys,)))
        if pushes:
            for rs in hub.call(pushes).values():
                for r in rs:
                    hub.value(r)
        pending["step"] = (link.split(actions), forces, times)
        return orig[step_async_name](actions)

    def step_wait(*args, **kwargs):
        out = orig[step_wait_name](*args, **kwargs)
        actions, forces, times = pending.pop("step", (None, None, None))
        _moved(hub, worlds, actions, forces, times)
        return out

    def reset(*args, **kwargs):
        out = orig[link.resets](*args, **kwargs)
        _moved(hub, worlds, None, None, None)
        return out

    def user_call(name):
        def call(*args, **kwargs):  # the user's own calls can change the simulations: reads are taken again
            for c in hub.cache:
                c.clear()
            return orig[name](*args, **kwargs)

        return call

    setattr(vec, step_async_name, step_async)
    setattr(vec, step_wait_name, step_wait)
    setattr(vec, link.resets, reset)
    for name in link.user_calls:
        setattr(vec, name, user_call(name))
    link._call = orig[link.user_calls[0]]  # robowright's own calls do not clear the reads it keeps

    def unhook():
        for k in orig:
            vec.__dict__.pop(k, None)  # back to the class's own methods

    vec._robowright_hub = hub
    vec._robowright_unhook = unhook


def _moved(hub: _Hub, worlds: list, actions, forces, times) -> None:
    """Record what each index did in the vector step (or reset) that just ended: a physics step, a reset, or both."""
    events = hub.moved()
    for i, w in enumerate(worlds):
        for ev in events[i]:
            if ev[0] == "step":
                if actions is None:
                    continue  # a step taken outside step_async (a call of the user's own): there is no action to keep
                after = hub.cache[i]
                if len(ev) > 1:  # a reset followed in the same vector step: the step is read from what the worker kept
                    hub.cache[i] = dict(ev[1])
                try:
                    w._physics_step(None, action=actions[i], forces=forces[i], t0=times[i])
                finally:
                    hub.cache[i] = after
            else:
                _, seed, options, rng, seeded, rebound, error = ev
                if error is not None:
                    raise ValueError(f"sub-environment {i}: after reset, {error}")
                if rebound:
                    w.backend.adopt(hub.value(hub.call({i: [("info", ())]})[i][0]))
                    w._adopt()
                if seed is None and w._resets_first and not seeded:
                    w._note_unreplayable(
                        "the environment was first reset without a seed, so its start is random and cannot be repeated: "
                        "call vec.reset(seed=[...]) before the first step"
                    )
                if options == "<not plain values>":
                    w._note_unreplayable("the reset options are not plain values")
                    options = None
                w._on_reset(seed, options, rng=rng if w.trace else None)


def _remote_worlds(worlds, vec, hub):
    from .attached import AttachedVectorWorlds

    class RemoteVectorWorlds(AttachedVectorWorlds):
        """The worlds of a vector environment's workers. Closing them also releases the vector environment: it can be
        attached again."""

        def close(self, failed: bool | None = None) -> list:
            try:
                return super().close(failed=failed)
            finally:
                unhook = vec.__dict__.pop("_robowright_unhook", None)
                if unhook is not None:
                    unhook()
                vec.__dict__.pop("_robowright_hub", None)
                hub.closed = True

    return RemoteVectorWorlds(worlds)
