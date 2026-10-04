"""Codegen: turn a trace back into a pytest test.

Every action, expectation, world edit and fault a test makes is recorded
with its arguments, so a trace can be rewritten as plain robowright code.
Point it at the trace of a failure and you get a regression test that
rebuilds the exact situation - same scene, seed, start pose, randomised
object positions and injected faults - and then makes the same calls.
"""

from __future__ import annotations

import keyword
import pprint
import re
from pathlib import Path

from .trace import Trace


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


def _settings(meta: dict) -> str:
    """``settings=...`` for the motion and timing settings the run changed from the defaults."""
    from .world import Settings

    default = Settings()
    changed = {k: v for k, v in meta.get("settings", {}).items() if getattr(default, k) != v}
    return f", settings=rw.Settings({_kwargs(changed)})" if changed else ""


def generate(trace: str | Path | Trace, test_name: str | None = None, stop_at_failure: bool = True, backend: str | None = None) -> str:
    """The trace as a pytest test. ``backend`` runs it on another engine than the one recorded."""
    tr = trace if isinstance(trace, Trace) else Trace(trace)
    m = tr.meta
    name = test_name or f"test_{_ident(re.sub(r'^.*::', '', m['name'])).removeprefix('test_')}_regression"
    imports = {"import robowright as rw", "from robowright import expect"}
    body: list[str] = []
    failed_at = None
    for e in tr.events:
        t, a = e["type"], e.get("args", {})
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
            target = {"robot": "robot", "arm": "robot.arm", "gripper": "robot.gripper"}[e["name"].split(".")[0]]
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
        if e.get("status") == "raised" and body:
            # A call that failed but was carried on from (an agent's attempt that moved the world):
            # the test makes it too, for the same time to pass, and expects it to fail again.
            err = "ExpectationError" if t in ("expect", "invariant") else e.get("detail", "").split(":", 1)[0] or "RobowrightError"
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
    lines = (
        header
        + sorted(imports, key=lambda s: (s.startswith("from "), s))  # isort order: plain imports first
        + [
            "",
            f"SCENE = rw.SceneSpec.from_dict({scene})",
            "",
            "",
            f"def {name}():",
            f"    with rw.launch(SCENE, backend={backend or m['backend']!r}, seed={m['seed']}, name={name!r}{_settings(m)}) as world:",
            # Bind only what the body uses, so the generated file passes a linter as is.
            "        robot, scene = world.robot, world.scene" if any("scene[" in ln for ln in body) else "        robot = world.robot",
        ]
    )
    lines += ["        " + ln for ln in body] or ["        pass"]
    return "\n".join(lines) + "\n"
