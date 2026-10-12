"""Replay of a run recorded on an attached Genesis scene: the factory builds the scene again, and replay steps it.

The scene is a ground plane and Genesis's blue box (``urdf/blue_box``), built by ``build()`` for the recording and
again for each replay, so a replay is a fresh scene in the same process, not the recorded one stepped again. It is
built on the CPU backend, headless. Genesis compiles its kernels when a scene is built, and that cache makes each
later build in the process quick.
"""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

gs = pytest.importorskip("genesis")

import robowright as rw  # noqa: E402
from robowright import expect  # noqa: E402
from robowright.codegen import generate  # noqa: E402
from robowright.replay import replay  # noqa: E402
from robowright.trace import Trace  # noqa: E402

TOLERANCE = 1e-6  # the replay default
FAILING_FLOOR = 5.0  # the box cannot be above this: the failing variant's last expectation
START_Z = 0.3  # the box starts this high and falls onto the plane


def build(z: float = START_Z):
    """A built scene, as a Genesis example would build it: the plane and the box, nothing else."""
    if not gs._initialized:  # gs.init may run once per process, and another test module may have run it already
        gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.URDF(file="urdf/blue_box/model.urdf", pos=(-0.45, 0.0, z)), name="cube")
    scene.build()
    return scene


def _record(out_dir, *, own_loop: bool = False, name: str = "genesis-run", failing: bool = False):
    """A run with a push, a move and a reset of the scene, stepped by world.step or, with ``own_loop``, by the scene.
    ``failing`` makes the run's last expectation one the box cannot pass, as a failed test's would be."""
    scene = build()
    w = rw.attach(
        scene,
        advance=None if own_loop else (lambda: scene.step()),
        settings=rw.Settings(trace="on", trace_dir=str(out_dir)),
        name=name,
    )
    steps = (lambda n: [scene.step() for _ in range(n)]) if own_loop else w.step
    steps(100)
    w.push("cube", (-6.0, 0.0, 0.0), duration=0.1)  # 6 N moves the 4.7 N box, as test_attach_genesis measured
    steps(60)
    w.move_object("cube", [-0.4, 0.05, START_Z])
    steps(40)
    scene.reset()  # the scene goes back to its built state: a reset is a recorded event
    steps(30)
    try:
        expect(w.body("cube")).to_be_above(FAILING_FLOOR if failing else 0.0)
    except rw.ExpectationError:
        if not failing:
            raise
    path = w.close()
    assert path is not None
    return path


def _assert_agrees(r, steps: int):
    assert r.engine == "genesis"
    assert r.matches, r.summary()
    assert r.steps == steps
    assert max(r.max_qpos_error, r.max_object_error, r.max_body_error, r.max_body_angle) <= TOLERANCE, r.summary()
    assert "within tolerance" in r.summary()


def test_a_run_driven_by_robowright_replays_with_its_push_move_and_reset(tmp_path):
    path = _record(tmp_path)
    tr = Trace(path)
    assert tr.meta["attached"]["engine"] == "genesis" and tr.meta["attached"]["inputs_recorded"] is False
    assert any(e["type"] == "log" and e["name"] == "env.reset" for e in tr.events)
    assert np.any(tr.arrays["forces"]), "the push is not in the trace's forces"
    r = replay(path, make_env=build)
    _assert_agrees(r, steps=230)
    assert any("make_env must rebuild the same ones" in n for n in r.notes)
    # Measured on this machine: the replay agreed with the recording exactly (every error 0.0) in each of three runs.


def test_a_run_that_the_scene_stepped_itself_replays_too(tmp_path):
    path = _record(tmp_path, own_loop=True)
    _assert_agrees(replay(path, make_env=build), steps=230)


def test_a_changed_initial_state_is_reported_as_diverged_from_the_first_step(tmp_path):
    path = _record(tmp_path)
    r = replay(path, make_env=lambda: build(z=START_Z + 0.05))
    assert not r.matches
    assert r.first_divergent_step == 0, r.summary()
    assert "differs from the recorded run" in r.summary()


def test_a_genesis_trace_without_a_factory_is_refused_with_the_reason(tmp_path):
    path = _record(tmp_path)
    with pytest.raises(ValueError, match="cannot rebuild this run's environment"):
        replay(path)


FACTORY_HEADER = """import genesis as gs

START_Z = %r


"""


def _factory_module(tmp_path: Path) -> Path:
    """A factory module in the test's directory: the same ``build`` the recording used, as a file of its own."""
    path = tmp_path / "make_genesis_factory.py"
    path.write_text(FACTORY_HEADER % START_Z + inspect.getsource(build))
    return path


def _run_in_fresh_process(test_file: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    src = str(Path(rw.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [src, env.get("PYTHONPATH", "")] if p)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
        cwd=test_file.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_codegen_writes_a_stub_for_a_genesis_run_without_a_factory(tmp_path):
    code = generate(_record(tmp_path))
    assert "def make_env():" in code and "a built genesis.Scene" in code
    out = tmp_path / "test_stub_regression.py"
    out.write_text(code)
    r = _run_in_fresh_process(out)
    assert r.returncode != 0 and "NotImplementedError" in r.stdout + r.stderr


def test_generated_test_of_a_passing_genesis_run_passes_in_a_fresh_process(tmp_path):
    path = _record(tmp_path)
    out = tmp_path / "test_genesis_regression.py"
    out.write_text(generate(path, test_name="test_genesis_regression", factory=f"{_factory_module(tmp_path)}:build"))
    r = _run_in_fresh_process(out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 passed" in r.stdout


def test_generated_test_of_a_passing_own_loop_genesis_run_passes_in_a_fresh_process(tmp_path):
    path = _record(tmp_path, own_loop=True)
    out = tmp_path / "test_genesis_own_loop_regression.py"
    out.write_text(generate(path, test_name="test_genesis_own_loop_regression", factory=f"{_factory_module(tmp_path)}:build"))
    r = _run_in_fresh_process(out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 passed" in r.stdout


def test_generated_test_of_a_failing_genesis_run_fails_at_the_same_expectation(tmp_path):
    path = _record(tmp_path, failing=True)
    failed = [e for e in Trace(path).events if e.get("status") == "failed"]
    assert len(failed) == 1 and failed[0]["type"] == "expect", failed
    detail = failed[0]["detail"].splitlines()[0]
    out = tmp_path / "test_genesis_failing_regression.py"
    out.write_text(generate(path, test_name="test_genesis_failing_regression", factory=f"{_factory_module(tmp_path)}:build"))
    r = _run_in_fresh_process(out)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "1 failed" in r.stdout and "ExpectationError" in r.stdout
    assert detail in r.stdout, (detail, r.stdout)
