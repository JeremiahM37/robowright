"""Codegen: turn a trace back into a pytest test.

Every action, expectation, world edit and fault a test makes is recorded
with its arguments, so a trace can be rewritten as plain robowright code.
Point it at the trace of a failure and you get a regression test that
rebuilds the exact situation - same scene, seed, start pose, randomised
object positions and injected faults - and then makes the same calls.
"""

from __future__ import annotations

import io
import keyword
import pprint
import re
from pathlib import Path

import numpy as np

from .trace import Trace, _call

# Action steps up to this many are written into the test; a longer run is saved beside it as an .npz.
INLINE_ACTIONS = 50


def _ident(name: str) -> str:
    s = re.sub(r"\W+", "_", name).strip("_").lower() or "case"
    if s[0].isdigit() or keyword.iskeyword(s):
        s = "t_" + s
    return s


def _subject(name: str) -> str:
    return {"robot": "robot", "gripper": "robot.gripper", "tcp": "robot.tcp", "base": "robot.base"}.get(name, f"scene[{name!r}]")


def _value(v) -> str:
    if isinstance(v, dict):
        if "$ref" in v:
            return _subject(v["$ref"])
        if "$point" in v:
            return "(" + ", ".join(repr(float(x)) for x in v["$point"]) + ")"
        if "$callable" in v:
            return f"policy  # {v['$callable']}: recreate this policy"
        return repr(v)
    if isinstance(v, float):
        return repr(v)  # shortest exact round-trip: regenerated runs must match bit for bit
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    return repr(v)


def _kwargs(d: dict, skip=()) -> str:
    return ", ".join(f"{k}={_value(v)}" for k, v in d.items() if k not in skip and v is not None)


def _policy_lines(args: dict, imports: set) -> list[str]:
    pol = args.get("policy") or {}
    cfg = pol.get("$policy") if isinstance(pol, dict) else None
    path = pol.get("$callable") if isinstance(pol, dict) else None
    if cfg and cfg.get("class"):
        mod, _, cls = cfg["class"].rpartition(".")
        imports.add(f"from {mod} import {cls}")
        return [f"policy = {cls}({_kwargs(cfg.get('kwargs', {}))})"]
    return [f"policy = ...  # TODO: recreate {path or 'the policy'}"]


def _robot_value(v) -> str:
    """An option of a robot loaded from a file, as code (an attachment is rebuilt, the rest as values)."""
    from .robots import Attachment

    if isinstance(v, Attachment):
        body = f"mjcf=lambda: Path({str(v.mjcf())!r}), site={v.site!r}, prefix={v.prefix!r}, body={v.body!r}, quat={v.quat!r}"
        return f"rw.robots.Attachment({body})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_robot_value(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(_robot_value(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    return _value(v)


def _robot_line(rec: dict, imports: set) -> str:
    """``ROBOT = rw.robots.load(...)``: registers the robot the trace was run with, from its model file."""
    from .robots.detect import restore_options

    if "stack" in rec:  # the description the robot's own stack publishes, read again when the test runs
        args = [f"name={rec['name']!r}", *(f"{k}={v!r}" for k, v in rec["stack"].items())]
        return f"ROBOT = rw.ros2_robot({', '.join(args)})"
    opts = restore_options(rec.get("overrides", {}))
    if _has_attachment(opts):
        imports.add("from pathlib import Path")
    args = [repr(rec["file"]), f"name={rec['name']!r}"] + [f"{k}={_robot_value(v)}" for k, v in opts.items()]
    return f"ROBOT = rw.robots.load({', '.join(args)})"


def _has_attachment(opts: dict) -> bool:
    from .robots import Attachment

    return any(isinstance(v, Attachment) for v in opts.values())


def _launch_lines(m: dict, backend: str | None, name: str) -> list[str]:
    """The ``with rw.launch(...) as world:`` line. A built-in robot on a built-in engine keeps its one-line call;
    the backend options a model file's robot needs (a ROS 2 robot's settings) make it one argument per line when long."""
    head = f"    with rw.launch(SCENE, backend={backend or m['backend']!r}, seed={m['seed']}, name={name!r}"
    opts = (m.get("backend_options") or {}) if backend in (None, m["backend"]) else {}
    if not opts:
        return [f"{head}{_settings(m)}) as world:"]
    settings = _settings(m).removeprefix(", ")
    parts = ([settings] if settings else []) + [f"{k}={_value(v)}" for k, v in opts.items() if v is not None]
    one = f"{head}{''.join(', ' + p for p in parts)}) as world:"
    if len(one) <= 100:
        return [one]
    return [
        "    with rw.launch(",
        "        SCENE,",
        *[f"        {p}," for p in [f"backend={backend or m['backend']!r}", f"seed={m['seed']}", f"name={name!r}", *parts]],
        "    ) as world:",
    ]


def _settings(meta: dict) -> str:
    """``settings=...`` for the motion and timing settings the run changed from the defaults."""
    from .world import Settings

    default = Settings()
    changed = {k: v for k, v in meta.get("settings", {}).items() if getattr(default, k) != v}
    if meta.get("fidelity") == "adjusted":  # rebuilt as it ran (published is the default)
        changed["fidelity"] = "adjusted"
    return f", settings=rw.Settings({_kwargs(changed)})" if changed else ""


def generate(
    trace: str | Path | Trace,
    test_name: str | None = None,
    stop_at_failure: bool = True,
    backend: str | None = None,
    keep_failed: bool = True,
    factory: str | None = None,
) -> str:
    """The trace as a pytest test. ``backend`` runs it on another engine than the one recorded.

    ``keep_failed``: a step that raised, but went on from, becomes ``pytest.raises`` (a failing test's trace);
    False leaves it out as a comment, for an agent's exploratory session whose wrong attempts should not be asserted.
    ``factory`` (attached runs only) is ``module:function`` or ``path/to/file.py:function``: the test imports it as
    ``make_env``. Without one, an attached test gets a stub to fill in.
    """
    tr = trace if isinstance(trace, Trace) else Trace(trace)
    m = tr.meta
    if "attached" in m:
        if backend is not None:
            raise ValueError("an attached run is generated for the environment it ran in: backend does not apply")
        return _generate_attached(tr, test_name, stop_at_failure, factory)
    name = _test_name(m, test_name)
    imports = {"import robowright as rw", "from robowright import expect"}
    body: list[str] = []
    failed_at = None
    left_out = 0
    for e in tr.events:
        t, a = e["type"], e.get("args", {})
        n_before = len(body)
        if t == "edit" and e["name"] == "reset_to":
            q = a["q"]
            if a.get("default"):
                body.append("robot.reset_to()")
            elif "base_pos" in a:  # legged: joints, then where the base stands
                body.append(f"robot.reset_to({_value(q)}, base_pos={_value(a['base_pos'])}, yaw={_value(a.get('yaw', 0.0))})")
            else:
                body.append(f"robot.reset_to({_value(q[:-1])}, gripper={_value(q[-1])})")
        elif t == "wait":
            body.append(f"world.wait({_value(a['seconds'])})")
        elif t == "edit" and e["name"] == "move_object":
            body.append(f"world.move_object({a['object']!r}, {_value(a['pos'])}, {_value(a['quat'])})")
        elif t == "fault":
            f = {k: v for k, v in a.items() if k != "type"}
            fn = {
                "JointNoise": "joint_noise",
                "ActionDelay": "action_delay",
                "Push": "push",
                "WeakJoint": "weak_joint",
                "CameraDropout": "camera_dropout",
            }[e["name"]]
            body.append(f"world.faults.{fn}({_kwargs(f)})")
        elif t == "action":
            target = {"robot": "robot", "arm": "robot.arm", "gripper": "robot.gripper", "base": "robot.base"}[e["name"].split(".")[0]]
            method = e["name"].split(".", 1)[1]
            if method == "run_policy":
                body.extend(_policy_lines(a, imports))
                kw = dict(a)
                kw.pop("policy", None)
                until = kw.pop("until", None)
                u = ""
                if isinstance(until, dict) and "$condition" in until:
                    c = until["$condition"]
                    parts = [_subject(c["subject"]), repr(c["matcher"]), *(_value(x) for x in c["args"])]
                    parts += [f"{k}={_value(v)}" for k, v in c["kwargs"].items()]
                    u = f", until=rw.condition({', '.join(parts)})"
                body.append(f"rollout = robot.run_policy(policy{u}{', ' + _kwargs(kw) if _kwargs(kw) else ''})")
                body.append("assert rollout.success, rollout")
            else:
                body.append(f"{target}.{method}({_kwargs(a)})")
        elif t in ("expect", "invariant") and "matcher" in a:
            neg = ".not_" if a.get("negate") else ""
            mode = ".always" if t == "invariant" else ""
            body.append(f"expect({_subject(a['subject'])}){mode}{neg}.{a['matcher']}({_kwargs(a['kwargs'])})")
        if e.get("status") == "raised" and (body or not keep_failed):
            # A call that failed but was carried on from (an agent's attempt that moved the world):
            # the test makes it too, for the same time to pass, and expects it to fail again.
            err = "ExpectationError" if t in ("expect", "invariant") else e.get("detail", "").split(":", 1)[0] or "RobowrightError"
            if not keep_failed:
                first = (e.get("detail") or "").splitlines()[0:1]
                first = (first[0] if first else "").removeprefix(f"{err}: ")[:160]
                del body[n_before:]
                body.append(f"# left out: {_call(e)} raised {err}: {first}".rstrip(": "))
                left_out += 1
                continue
            imports.add("import pytest")
            imports.add(f"from robowright.errors import {err}")
            body[-1:] = [f"with pytest.raises({err}):", "    " + body[-1]]
            continue
        if e.get("status") == "failed" and t != "violation":
            failed_at = failed_at or e
            if stop_at_failure:
                break
    scene = pprint.pformat(m["scene"], sort_dicts=False, width=100)
    header = [
        f'"""Regression test generated by `robowright codegen` from {Path(tr.path).name}.',
        "",
        f"Original test: {m['name']} ({m['backend']}, seed {m['seed']}, status {m['status']}).",
    ]
    if failed_at:
        header.append(f"It failed at t={failed_at['t']:.2f}s in {failed_at['name']}: {failed_at.get('detail', '').splitlines()[0][:120]}")
    header += ['"""', ""]
    if m.get("source") == "session":  # the header names the session, not a trace file that was never written
        header[0] = '"""Regression test generated from an interactive session (robowright mcp, Session.generate_test).'
        header[2] = f"Original session: {m['name']} ({m['backend']}, seed {m['seed']}, status {m['status']})."
    summary = []
    if left_out:
        summary = [
            f"# Left out {left_out} failed step(s), commented in the body: a failed step can still have moved the world part-way,"
            " so without them this test may not reproduce the session exactly."
        ]
    robot = [_robot_line(m["robot_file"], imports), ""] if m.get("robot_file") else []
    lines = (
        summary
        + header
        + sorted(imports, key=lambda s: (s.startswith("from "), s))  # isort order: plain imports first
        + [""]
        + robot
        + [
            f"SCENE = rw.SceneSpec.from_dict({scene})",
            "",
            "",
            f"def {name}():",
        ]
        + _launch_lines(m, backend, name)
        + [
            # Bind only what the body uses, so the generated file passes a linter as is.
            "        robot, scene = world.robot, world.scene" if any("scene[" in ln for ln in body) else "        robot = world.robot",
        ]
    )
    lines += ["        " + ln for ln in body] or ["        pass"]
    return "\n".join(lines) + "\n"


def _test_name(meta: dict, test_name: str | None) -> str:
    return test_name or f"test_{_ident(re.sub(r'^.*::', '', meta['name'])).removeprefix('test_')}_regression"


def companion(trace: str | Path | Trace, test_name: str | None = None) -> tuple[str, bytes] | None:
    """The ``.npz`` an attached test reads its data from, as ``(file name, bytes)``, beside the test; None if it inlines it.

    A bare model's test reads its model, control and start state from here (a model can be large, so it is not inlined).
    """
    from .replay import _stored_model

    tr = trace if isinstance(trace, Trace) else Trace(trace)
    if "attached" not in tr.meta:
        return None
    if (tr.meta.get("env") or {}).get("bare"):
        raw = _stored_model(tr)
        if raw is None:
            return None
        buf = io.BytesIO()
        state0 = tr.state0
        np.savez_compressed(
            buf,
            model=np.frombuffer(raw, dtype=np.uint8),
            ctrl=np.asarray(tr.arrays["ctrl"], dtype=float),
            state0=np.asarray(state0 if state0 is not None else [], dtype=float),
            has_state0=np.array(state0 is not None),
        )
        return f"{_test_name(tr.meta, test_name)}_bare.npz", buf.getvalue()
    rows = [row for _, kind, row in tr.controls if kind == "action"]
    if len(rows) <= INLINE_ACTIONS:
        return None
    buf = io.BytesIO()
    np.savez_compressed(buf, action=np.stack(rows))
    return f"{_test_name(tr.meta, test_name)}_controls.npz", buf.getvalue()


def _avalue(v) -> str:
    """An expectation's argument as code: a subject is looked up by name in the attached simulation."""
    if isinstance(v, dict):
        if "$ref" in v:
            return f"world.body({v['$ref']!r})"
        if "$point" in v:
            return "(" + ", ".join(repr(float(x)) for x in v["$point"]) + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_avalue(x) for x in v) + "]"
    return repr(v)


def _push_applied(tr: Trace, a: dict) -> bool:
    """Whether a recorded push moved anything: a push made in a run whose own loop stepped (observed, not stepped
    through the world) was never applied, and the test must not apply it either."""
    ts = tr.arrays["t"]
    window = np.nonzero((ts >= float(a["at"])) & (ts < float(a["at"]) + float(a["duration"])))[0]
    return bool(np.any(tr.arrays["forces"][window])) if window.size else False


def _attached_event(e: dict, tr: Trace | None = None, stepped: bool = False) -> str | None:
    """One recorded call of the test as a line. None for events that change nothing and need no line."""
    t, name, a = e["type"], e["name"], e.get("args") or {}
    if t == "edit" and name == "move_object":
        return f"world.move_object({a['object']!r}, {_avalue(a['pos'])}, {_avalue(a['quat'])})"
    if t == "edit" and name == "set_joint":
        return f"world.set_joint({a['joint']!r}, {_avalue(a['value'])})"
    if t == "fault" and name == "Push":
        line = f"world.faults.push({a['object']!r}, {_avalue(a['force'])}, at={_avalue(a['at'])}, duration={_avalue(a['duration'])})"
        if stepped and tr is not None and not _push_applied(tr, a):
            return f"# not replayed: the run's own loop stepped this simulation and applied no force for this push: {_call(e)}"
        return line
    if t in ("fault", "edit"):
        return f"# not replayed, because an attached run has no robot to apply it to: {_call(e)}"
    if t in ("expect", "invariant") and "matcher" in a:
        # The time the expectation waited is already in the schedule as steps: it checks the state once, where it was
        # recorded, rather than stepping on for its own timeout.
        kw = ", ".join(
            [f"{k}={_avalue(v)}" for k, v in a.get("kwargs", {}).items() if k not in ("timeout", "hold") and v is not None]
            + ["timeout=0.0"]
        )
        neg = ".not_" if a.get("negate") else ""
        mode = ".always" if t == "invariant" else ""
        return f"expect(world.body({a['subject']!r})){mode}{neg}.{a['matcher']}({kw})"
    return None


# The factory's stub, for each engine whose simulation the trace cannot rebuild: what make_env must return.
_FACTORY_WHAT = {
    "drake": "a Drake Simulator, or (Simulator, plant) when the diagram holds more than one plant",
    "genesis": "a built genesis.Scene",
}


def _factory_lines(factory: str) -> tuple[list[str], list[str]]:
    """The imports and the ``make_env`` definition for a ``--factory`` given on the command line."""
    target, _, func = factory.rpartition(":")
    if target.endswith(".py") or "/" in target or "\\" in target:
        path = str(Path(target).resolve())
        return (
            ["import importlib.util"],
            [
                "",
                "",
                "def _load_factory():",
                f"    spec = importlib.util.spec_from_file_location('robowright_factory', {path!r})",
                "    module = importlib.util.module_from_spec(spec)",
                "    spec.loader.exec_module(module)",
                f"    return module.{func}",
                "",
                "",
                "make_env = _load_factory()",
            ],
        )
    return ([f"from {target} import {func} as make_env"], [])


def _factory_stub(engine: str | None, why: str) -> list[str]:
    """The stub a test is written with when no factory was given: the one function the user fills in."""
    if engine is None:
        return [
            "",
            "",
            "def make_env():",
            f"    # Stub: the environment of this run cannot be rebuilt from the trace ({why}).",
            "    # Build it the way the recorded run did (the same constructor, options and model), then delete this line.",
            "    raise NotImplementedError('make_env() is a stub: fill in the environment the recorded run used')",
        ]
    what = _FACTORY_WHAT[engine]
    return [
        "",
        "",
        "def make_env():",
        f"    # Stub: the simulation of this run cannot be rebuilt from the trace ({why}).",
        f"    # Return {what}, built the way the recorded run did, then delete this line.",
        "    raise NotImplementedError('make_env() is a stub: fill in the simulation the recorded run used')",
    ]


def _generate_bare(tr: Trace, test_name: str | None, stop_at_failure: bool, factory: str | None) -> str:
    """A test for a bare model/data run (the caller's own ``mj_step`` loop): it rebuilds the model from the trace's copy,
    applies the start state and the control each step was taken with, and steps the same way."""
    from .replay import _stored_model, schedule

    m = tr.meta
    name = _test_name(m, test_name)
    if factory is None and _stored_model(tr) is None:
        raise ValueError("this run's trace keeps no model to rebuild it from: pass --factory module:func that builds the model")
    npz = f"{name}_bare.npz"  # written beside the test by companion(): the model, the control and the start state
    body: list[str] = []
    failed_at = None
    items = schedule(tr)
    i = 0
    while i < len(items):
        it = items[i]
        if it["kind"] == "control":
            j = i
            while j + 1 < len(items) and items[j + 1]["kind"] == "control":
                j += 1
            first, count = items[i]["step"], j - i + 1
            body += [
                f"for k in range({first}, {first + count}):",
                "    data.ctrl[:] = CTRL[k]  # the control the recorded step was taken with",
                "    mujoco.mj_step(model, data)",
                "    world.observe()",
            ]
            i = j + 1
            continue
        if it["kind"] == "event":
            e = it["event"]
            line = _attached_event(e, tr, stepped=False)
            if line:
                body.append(line)
            if e.get("status") == "failed":
                failed_at = failed_at or e
                if stop_at_failure:
                    break
        i += 1

    header = [
        '"""Regression test generated by `robowright codegen` from ' + Path(tr.path).name + ".",
        "",
        f"Original run: {m['name']} (attached, a bare model and data), {m['steps']} steps, status {m['status']}.",
        "The model is rebuilt from the trace's copy. Each step is the test's own mj_step with the recorded control,",
        "then world.observe(), as the run's loop did; expectations are checked where they were recorded.",
    ]
    if failed_at:
        header.append(f"It failed at t={failed_at['t']:.2f}s in {failed_at['name']}: {failed_at.get('detail', '').splitlines()[0][:120]}")
    header += ['"""']

    stdlib: list[str] = ["from pathlib import Path"]
    lines: list[str] = []
    if factory is not None:
        from .replay import load_factory

        load_factory(factory)  # fail here, with the reason, rather than in the generated test
        extra, stub = _factory_lines(factory)
        stdlib += [ln for ln in extra if not ln.startswith("from ")]
        lines += [
            "",
            "",
            f"DATA = np.load(Path(__file__).with_name({npz!r}))  # the control and start state of the recorded run",
        ] + stub
    else:
        stdlib += ["import tempfile"]
        lines += [
            "",
            "",
            f"DATA = np.load(Path(__file__).with_name({npz!r}))  # the model, control and start state of the recorded run",
            "",
            "",
            "def make_env():",
            "    with tempfile.TemporaryDirectory() as d:",
            "        path = Path(d) / 'model.mjb'",
            "        path.write_bytes(DATA['model'].tobytes())",
            "        model = mujoco.MjModel.from_binary_path(str(path))",
            "    return model, mujoco.MjData(model)",
        ]
    third = ["import mujoco", "import numpy as np"]
    local = ["import robowright as rw", "from robowright import expect"]
    if factory is not None:
        local += [ln for ln in _factory_lines(factory)[0] if ln.startswith("from ")]
    import_block = [ln for g in (stdlib, third, local) for ln in ([*g, ""] if g else [])][:-1]
    lines += ["", "", "CTRL = DATA['ctrl']", "STATE0 = DATA['state0'] if bool(DATA['has_state0']) else None"]
    lines += [
        "",
        "",
        f"def {name}():",
        "    model, data = make_env()",
        "    if STATE0 is not None:  # the start state the recorded run began from",
        "        mujoco.mj_setState(model, data, STATE0, mujoco.mjtState.mjSTATE_FULLPHYSICS)",
        "        mujoco.mj_forward(model, data)",
        f"    world = rw.attach(model=model, data=data, seed={m['seed']}, settings=rw.Settings(trace='retain-on-failure'))",
        "    try:",
    ]
    lines += ["        " + ln for ln in body] or ["        pass"]
    lines += ["    finally:", "        world.close()"]
    return "\n".join(header + [""] + import_block + lines) + "\n"


def _generate_attached(tr: Trace, test_name: str | None, stop_at_failure: bool, factory: str | None = None) -> str:
    from .replay import STEPPED, schedule

    m = tr.meta
    env_rec = m.get("env") or {}
    if not env_rec.get("actions_recorded", True):
        raise ValueError(f"this run cannot be turned into a test: {env_rec.get('note')}")
    if env_rec.get("bare"):
        return _generate_bare(tr, test_name, stop_at_failure, factory)
    engine = (m.get("attached") or {}).get("engine")
    if engine == "webots":  # the test would need engine="webots" and its own Supervisor process: not written yet
        raise ValueError("a test cannot be generated for a webots run yet: replay it with make_env=... (robowright replay --factory)")
    stepped = engine in STEPPED  # a Drake or Genesis simulation: the test steps it through world.step, as the run did
    name = _test_name(m, test_name)
    spec = env_rec.get("spec")
    rebuilt = bool(spec and spec.get("reproducible")) and factory is None and not stepped
    rows = [row for _, kind, row in tr.controls if kind == "action"]
    inline = len(rows) <= INLINE_ACTIONS
    body: list[str] = []
    row_no = 0
    failed_at = None
    items = schedule(tr)
    i = 0
    while i < len(items):
        it = items[i]
        if it["kind"] == "control":
            j = i
            # A stepped simulation's steps are one kind: each is a physics step the world takes.
            while j + 1 < len(items) and items[j + 1]["kind"] == "control" and (stepped or items[j + 1]["observe"] == it["observe"]):
                j += 1
            count = j - i + 1
            if stepped:
                body.append("world.step()" if count == 1 else f"world.step({count})")
            elif it["observe"]:
                body += ["world.observe()"] if count == 1 else [f"for _ in range({count}):", "    world.observe()"]
            elif inline:
                body += [f"env.step(np.array({rows[row_no + k].tolist()!r}))" for k in range(count)]
            elif count == 1:
                body.append(f"env.step(actions[{row_no}])")
            else:
                body += [f"for k in range({row_no}, {row_no + count}):", "    env.step(actions[k])"]
            if not it["observe"] and not stepped:
                row_no += count
            i = j + 1
            continue
        if it["kind"] == "reset":
            if stepped:
                if engine == "genesis":
                    body.append("env.reset()  # a Genesis scene resets to its built state; it takes no seed")
                else:
                    body.append(f"# not replayed: {engine} has no reset that the run's log recorded")
            else:
                seed = it["args"].get("seed")
                if seed is None:
                    body.append("# the recorded reset had no seed: this reproduces its start only if the environment is deterministic")
                opts = f", options={it['args']['options']!r}" if "options" in it["args"] else ""
                body.append(f"env.reset(seed={seed!r}{opts})")
        else:
            e = it["event"]
            line = _attached_event(e, tr, stepped)
            if line:
                body.append(line)
            if e.get("status") == "failed":
                failed_at = failed_at or e
                if stop_at_failure:
                    break
        i += 1

    header = [
        '"""Regression test generated by `robowright codegen` from ' + Path(tr.path).name + ".",
        "",
        f"Original run: {m['name']} (attached), {m['steps']} steps, status {m['status']}.",
        "The recorded resets, actions and the calls made during the run are repeated in order; expectations are",
        "checked at the step where they were recorded to pass or fail.",
    ]
    if failed_at:
        header.append(f"It failed at t={failed_at['t']:.2f}s in {failed_at['name']}: {failed_at.get('detail', '').splitlines()[0][:120]}")
    if stepped and not (m.get("attached") or {}).get("inputs_recorded", True):
        header.append("The trace does not hold what the run's own code set on the simulation between steps; make_env must set the same.")
    header += ['"""']

    imports_extra: list[str] = []
    stub: list[str] = []
    library = (spec or {}).get("library", "gymnasium")  # an environment of the old gym 0.x package is rebuilt with gym
    if factory is not None:
        from .replay import load_factory

        load_factory(factory)  # fail here, with the reason, rather than in the generated test
        imports_extra, stub = _factory_lines(factory)
        make = "make_env()"
    elif rebuilt:
        from .replay import make_spec_call

        env_id, kwargs = make_spec_call(spec)
        kw = "".join(f", {k}={v!r}" for k, v in kwargs.items())
        make = f"{library}.make({env_id!r}{kw})"
    else:
        why = (spec or {}).get("reason") or "it was built by hand, not by gymnasium.make"
        if spec:
            why = f"it was made with {library}.make({spec['id']!r}), but {why}"
        if not stepped:
            # No stub: a test that raises NotImplementedError is not a test. The environment is named, and the way to
            # give one is named, so the refusal is the one sentence the user needs.
            raise ValueError(
                f"cannot write a test for this run's environment ({why}): pass --factory module:func that builds it the way the run did"
            )
        # A stepped run (Drake, Genesis, Webots) keeps its model in the trace but not its simulation: the stub is filled in.
        stub = _factory_stub(engine, "its model and parameters are not in the trace")

    npz = name + "_controls.npz"
    lines = _imports(rebuilt, rows, inline, imports_extra, library) + stub
    if stepped:
        dt = float(tr.arrays["t"][1] - tr.arrays["t"][0]) if len(tr.arrays["t"]) > 1 else 0.0
        lines += ["", "", f"DT = {dt!r}  # the recorded control period: one AdvanceTo per physics step"]
    lines += ["", "", f"def {name}():"]
    if stepped:
        if engine == "drake":
            lines += [
                "    built = make_env()",
                "    env, plant = built if isinstance(built, tuple) else (built, None)",
                "    world = rw.attach(",
                "        simulator=env,",
                "        plant=plant,",
                "        advance=lambda: env.AdvanceTo(env.get_context().get_time() + DT),",
                f"        seed={m['seed']},",
                "        settings=rw.Settings(trace='retain-on-failure'),",
                "    )",
            ]
        else:
            lines += [
                "    env = make_env()",
                "    world = rw.attach(",
                "        env,",
                "        advance=lambda: env.step(),",
                f"        seed={m['seed']},",
                "        settings=rw.Settings(trace='retain-on-failure'),",
                "    )",
            ]
        lines += ["    try:"]
    else:
        if not inline and rows:
            lines.append(f"    actions = np.load(Path(__file__).with_name({npz!r}))['action']")
        lines += [
            f"    env = {make}",
            f"    world = rw.attach(env, seed={m['seed']}, settings=rw.Settings(trace='retain-on-failure'))",
            "    try:",
        ]
    lines += ["        " + ln for ln in body] or ["        pass"]
    lines += ["    finally:", "        world.close()"]
    return "\n".join(header + [""] + lines) + "\n"


def _imports(rebuilt: bool, rows: list, inline: bool, extra: list[str] | None = None, library: str = "gymnasium") -> list[str]:
    groups = [
        ["from pathlib import Path"] if (rows and not inline) else [],
        ([f"import {library}"] if rebuilt else []) + (["import numpy as np"] if rows else []),
        ["import robowright as rw", "from robowright import expect"],
        sorted(extra or [], key=lambda s: (s.startswith("from "), s)),
    ]
    out: list[str] = []
    for g in groups:
        if g:
            out += g + [""]
    return out[:-1]
