"""Replay and codegen for attached PyBullet runs (panda-gym): the run is rebuilt from its recorded spec and
stepped again, and its recorded states are compared at every step.

PyBullet cannot save its full state, so replay starts the environment from its recorded reset and compares
every body's pose at each step; the replayed run is stated as "within" a tolerance, never as identical.
Skipped where panda-gym is not installed; the environment needs it and nothing else. The generated tests run in
a fresh process, as a user's project would run them.
"""

import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("pybullet")
pytest.importorskip("panda_gym")

import robowright as rw  # noqa: E402
from robowright import expect  # noqa: E402
from robowright.codegen import companion, generate  # noqa: E402
from robowright.replay import replay  # noqa: E402
from robowright.trace import Trace  # noqa: E402

SRC = Path(rw.__file__).resolve().parent.parent
ENV_ID = "PandaPickAndPlace-v3"
SEED = 1
STEPS = 120


def _record(out_dir: Path, *, changed_step: int | None = None, steps: int = STEPS) -> Path:
    """A panda-gym run with seeded random actions and the calls a test would make on it."""
    env = gym.make(ENV_ID, render_mode="rgb_array")
    world = rw.attach(env, seed=SEED, name="panda-pick-and-place", settings=rw.Settings(trace="on", trace_dir=str(out_dir)))
    try:
        env.reset(seed=SEED)
        expect(world.body("object")).always.to_be_above(0.0)
        rng = np.random.default_rng(SEED)
        actions = rng.uniform(-1, 1, size=(steps, env.action_space.shape[0])).astype(np.float32) * 0.5
        for i in range(steps):
            a = actions[i].copy()
            if i == changed_step:
                a[0] += 0.5
            env.step(a)
        expect(world.body("object")).to_be_above(0.0)
    finally:
        path = world.close()
    assert path is not None
    return path


def _with_action_changed(path: Path, row: int, delta: float = 0.5) -> Path:
    """The trace with one recorded action row moved by ``delta`` on joint 0; everything else kept as recorded."""
    out = path.with_name(path.stem + "-changed.zip")
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "actions.npz":
                arrays = dict(np.load(io.BytesIO(data)))
                arrays["action"] = arrays["action"].copy()
                arrays["action"][row, 0] += delta
                buf = io.BytesIO()
                np.savez_compressed(buf, **arrays)
                data = buf.getvalue()
            dst.writestr(info.filename, data)
    return out


def _run_fresh(test_file: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(SRC), env.get("PYTHONPATH", "")] if p)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
        cwd=test_file.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_the_trace_keeps_every_body_pose_at_every_step(tmp_path):
    tr = Trace(_record(tmp_path))
    with zipfile.ZipFile(tr.path) as z:
        assert "bodies.npz" in z.namelist()
        with np.load(io.BytesIO(z.read("bodies.npz"))) as f:
            names = [str(n) for n in f["names"]]
            assert f["pos"].shape == (len(tr.arrays["t"]), len(names), 3)
            assert f["quat"].shape == (len(tr.arrays["t"]), len(names), 4)
    assert {"panda", "table", "object", "target"} <= set(names)


def test_replay_matches_every_body_at_every_step(tmp_path):
    r = replay(_record(tmp_path))
    assert r.engine == "pybullet"
    assert r.matches, r.summary()
    assert r.steps == STEPS  # the states after each of the STEPS control steps, plus the start
    # Measured: three replays in fresh processes and two recordings of the same run all agree exactly (see the PR);
    # the tolerance is the default and sits far above that.
    assert max(r.max_qpos_error, r.max_object_error, r.max_body_error, r.max_body_angle) <= 1e-6, r.summary()
    assert "within tolerance" in r.summary()


def test_replay_detects_a_changed_action(tmp_path):
    changed_row = 60
    path = _with_action_changed(_record(tmp_path), changed_row)
    r = replay(path)
    assert not r.matches
    # Row k is the action of control step k+1, so the first state that can differ is state k+1.
    assert r.first_divergent_step == changed_row + 1, r.summary()
    assert "differs from the recorded run" in r.summary()


def test_generated_test_passes_in_a_fresh_process_three_times(tmp_path):
    path = _record(tmp_path)
    out = tmp_path / "test_pick_and_place_regression.py"
    out.write_text(generate(path, test_name="test_pick_and_place_regression"))
    if (companion_file := companion(path, "test_pick_and_place_regression")) is not None:  # a long run's actions, beside it
        out.with_name(companion_file[0]).write_bytes(companion_file[1])
    for _ in range(3):
        r = _run_fresh(out)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "1 passed" in r.stdout
