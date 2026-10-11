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
) -> str:
    """The trace as a pytest test. ``backend`` runs it on another engine than the one recorded.

    ``keep_failed``: a step that raised, but went on from, becomes ``pytest.raises`` (a failing test's trace);
    False leaves it out as a comment, for an agent's exploratory session whose wrong attempts should not be asserted.
    """
    tr = trace if isinstance(trace, Trace) else Trace(trace)
    m = tr.meta
    if "attached" in m:
        if backend is not None:
            raise ValueError("an attached run is generated for the environment it ran in: backend does not apply")
        return _generate_attached(tr, test_name, stop_at_failure)
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
    """The ``.npz`` an attached test reads its actions from, as ``(file name, bytes)``, beside the test; None if it inlines them."""
    tr = trace if isinstance(trace, Trace) else Trace(trace)
    if "attached" not in tr.meta:
        return None
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


def _attached_event(e: dict) -> str | None:
    """One recorded call of the test as a line. None for events that change nothing and need no line."""
    t, name, a = e["type"], e["name"], e.get("args") or {}
    if t == "edit" and name == "move_object":
        return f"world.move_object({a['object']!r}, {_avalue(a['pos'])}, {_avalue(a['quat'])})"
    if t == "edit" and name == "set_joint":
        return f"world.set_joint({a['joint']!r}, {_avalue(a['value'])})"
    if t == "fault" and name == "Push":
        return f"world.faults.push({a['object']!r}, {_avalue(a['force'])}, at={_avalue(a['at'])}, duration={_avalue(a['duration'])})"
    if t in ("fault", "edit"):
        return f"# not replayed, because an attached run has no robot to apply it to: {_call(e)}"
    if t in ("expect", "invariant") and "matcher" in a:
        kw = ", ".join(f"{k}={_avalue(v)}" for k, v in a.get("kwargs", {}).items() if k not in ("timeout", "hold") and v is not None)
        neg = ".not_" if a.get("negate") else ""
        mode = ".always" if t == "invariant" else ""
        return f"expect(world.body({a['subject']!r})){mode}{neg}.{a['matcher']}({kw})"
    return None


def _generate_attached(tr: Trace, test_name: str | None, stop_at_failure: bool) -> str:
    from .replay import schedule

    m = tr.meta
    env_rec = m.get("env") or {}
    if not env_rec.get("actions_recorded", True):
        raise ValueError(f"this run cannot be turned into a test: {env_rec.get('note')}")
    name = _test_name(m, test_name)
    spec = env_rec.get("spec")
    rebuilt = bool(spec and spec.get("reproducible"))
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
            while j + 1 < len(items) and items[j + 1]["kind"] == "control" and items[j + 1]["observe"] == it["observe"]:
                j += 1
            count = j - i + 1
            if it["observe"]:
                body += ["world.observe()"] if count == 1 else [f"for _ in range({count}):", "    world.observe()"]
            elif inline:
                body += [f"env.step(np.array({rows[row_no + k].tolist()!r}))" for k in range(count)]
            elif count == 1:
                body.append(f"env.step(actions[{row_no}])")
            else:
                body += [f"for k in range({row_no}, {row_no + count}):", "    env.step(actions[k])"]
            if not it["observe"]:
                row_no += count
            i = j + 1
            continue
        if it["kind"] == "reset":
            seed = it["args"].get("seed")
            if seed is None:
                body.append("# the recorded reset had no seed: this reproduces its start only if the environment is deterministic")
            opts = f", options={it['args']['options']!r}" if "options" in it["args"] else ""
            body.append(f"env.reset(seed={seed!r}{opts})")
        else:
            e = it["event"]
            line = _attached_event(e)
            if line:
                body.append(line)
            if e.get("status") == "failed":
                failed_at = failed_at or e
                if stop_at_failure:
                    break
        i += 1

    if rebuilt:
        kw = "".join(f", {k}={v!r}" for k, v in spec["kwargs"].items())
        make = f"gymnasium.make({spec['id']!r}{kw})"
        stub: list[str] = []
    else:
        make = "make_env()"
        why = (spec or {}).get("reason") or "it was built by hand, not by gymnasium.make"
        if spec:
            why = f"it was made with gymnasium.make({spec['id']!r}), but {why}"
        stub = [
            "",
            "",
            "def make_env():",
            f"    # Stub: the environment of this run cannot be rebuilt from the trace ({why}).",
            "    # Build it the way the recorded run did (the same constructor, options and model), then delete this line.",
            "    raise NotImplementedError('make_env() is a stub: fill in the environment the recorded run used')",
        ]
    header = [
        '"""Regression test generated by `robowright codegen` from ' + Path(tr.path).name + ".",
        "",
        f"Original run: {m['name']} (attached, {m['backend']}), {m['steps']} steps, status {m['status']}.",
        "The recorded resets, actions and the calls made during the run are repeated in order; expectations are",
        "checked at the step where they were recorded to pass or fail.",
    ]
    if failed_at:
        header.append(f"It failed at t={failed_at['t']:.2f}s in {failed_at['name']}: {failed_at.get('detail', '').splitlines()[0][:120]}")
    header += ['"""']
    npz = name + "_controls.npz"
    lines = _imports(rebuilt, rows, inline) + stub + ["", "", f"def {name}():"]
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


def _imports(rebuilt: bool, rows: list, inline: bool) -> list[str]:
    groups = [
        ["from pathlib import Path"] if (rows and not inline) else [],
        (["import gymnasium"] if rebuilt else []) + (["import numpy as np"] if rows else []),
        ["import robowright as rw", "from robowright import expect"],
    ]
    out: list[str] = []
    for g in groups:
        if g:
            out += g + [""]
    return out[:-1]
