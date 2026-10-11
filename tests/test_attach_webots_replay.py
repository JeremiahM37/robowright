"""Replay of a recorded Webots run, in fresh processes: the probe world of test_attach_webots, stepped by robowright.

Recording and replay each start their own Webots, because the controller library connects to Webots once per process.
The probe world and its helpers are test_attach_webots'. Skipped without WEBOTS_HOME and an X display.
"""

import atexit
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from test_attach_webots import BODIES, WEBOTS_HOME, _connect_supervisor, _start_world, _stop

import robowright as rw
from robowright import expect

pytestmark = pytest.mark.skipif(
    not WEBOTS_HOME or not Path(WEBOTS_HOME, "webots").exists(),
    reason="needs an installed Webots R2025a: set WEBOTS_HOME to its directory",
)


# --- replay: each run in a process of its own, because the controller connects to Webots once per process ----------
REPLAY_STEPS = 300  # 2.4 s of simulated time at the world's 8 ms basicTimeStep
_WORLD: subprocess.Popen | None = None


def make_env():
    """The factory a replay names: starts the probe world in this process and returns its WebotsEnv."""
    global _WORLD
    from robowright.backends.attached_webots import WebotsEnv

    _WORLD = _start_world(Path(tempfile.mkdtemp(prefix="webots-replay-")))
    atexit.register(_stop, _WORLD)
    return WebotsEnv(_connect_supervisor(), BODIES)


def record_child(out_dir: str, result: str) -> None:
    """A fresh process: robowright steps the probe world with the world's own controller running, and records it."""
    from robowright.backends.attached_webots import WebotsEnv

    proc = _start_world(Path(tempfile.mkdtemp(prefix="webots-record-")))
    try:
        env = WebotsEnv(_connect_supervisor(), BODIES)
        w = rw.attach(
            env, engine="webots", advance=lambda: env.step(), settings=rw.Settings(trace="on", trace_dir=out_dir), name="webots-run"
        )
        w.step(REPLAY_STEPS)
        expect(w.body("can")).to_be_above(0.0)
        Path(result).write_text(str(w.close()))
    finally:
        _stop(proc)


def replay_child(trace: str, result: str) -> None:
    """A fresh process: the recorded run replayed through make_env, with the numbers written as JSON."""
    from robowright.replay import replay

    r = replay(trace, make_env=make_env)
    Path(result).write_text(
        json.dumps(
            {
                "steps": r.steps,
                "matches": r.matches,
                "max_body_error": r.max_body_error,
                "max_body_angle": r.max_body_angle,
                "summary": r.summary(),
            }
        )
    )


def _child(fn: str, *args: str) -> None:
    env = dict(os.environ)
    src = str(Path(rw.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [src, str(Path(__file__).parent), env.get("PYTHONPATH", "")] if p)
    code = f"import test_attach_webots_replay as t; t.{fn}(*{list(args)!r})"
    r = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_webots_run_replays_in_a_fresh_process_and_agrees_with_the_recording(tmp_path):
    if not os.environ.get("DISPLAY"):
        pytest.skip("needs an X display: set DISPLAY (for example to an Xvfb started with -nolisten tcp)")
    _child("record_child", str(tmp_path), str(tmp_path / "trace.txt"))
    trace = (tmp_path / "trace.txt").read_text()
    _child("replay_child", trace, str(tmp_path / "replay.json"))
    r = json.loads((tmp_path / "replay.json").read_text())
    assert r["matches"], r["summary"]
    assert r["steps"] == REPLAY_STEPS
    assert max(r["max_body_error"], r["max_body_angle"]) <= 1e-6, r["summary"]
    assert "within tolerance" in r["summary"]


def test_codegen_refuses_a_webots_run_with_the_reason(tmp_path):
    """The recorded run is needed for the refusal: a generated test would start a Supervisor the trace does not describe."""
    from robowright.codegen import generate

    if not os.environ.get("DISPLAY"):
        pytest.skip("needs an X display: set DISPLAY (for example to an Xvfb started with -nolisten tcp)")
    _child("record_child", str(tmp_path), str(tmp_path / "trace.txt"))
    trace = (tmp_path / "trace.txt").read_text()
    with pytest.raises(ValueError, match="a test cannot be generated for a webots run yet"):
        generate(trace)
