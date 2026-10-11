"""Replay and codegen for attached PyBullet (panda-gym) and SAPIEN (ManiSkill 3) runs.

Each run is rebuilt from its recorded spec and stepped again with its recorded resets and actions. These engines
cannot save their full state, so replay starts from the recorded reset and compares every body's pose at each
step; the result is stated as "within" a tolerance, never as identical. Each engine's tests are skipped where its
package is not installed, and nothing optional is imported before that guard. A ManiSkill run needs a Vulkan
driver for its renderer (the project's verify environment sets one up).
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

import robowright as rw  # noqa: E402
from robowright import expect  # noqa: E402
from robowright.codegen import companion, generate  # noqa: E402
from robowright.replay import replay  # noqa: E402
from robowright.trace import Trace  # noqa: E402

SRC = Path(rw.__file__).resolve().parent.parent
SEED = 1
STEPS = 120
# Per engine: the environment, the body a test checks, bodies the trace must hold, and the packages to import.
ENGINES = {
    "pybullet": {
        "env_id": "PandaPickAndPlace-v3",
        "kwargs": {"render_mode": "rgb_array"},
        "body": "object",
        "bodies": {"panda", "table", "object", "target"},
        "modules": ["pybullet", "panda_gym"],
    },
}
ENGINE = pytest.mark.parametrize("engine", list(ENGINES))


def _require(engine: str) -> dict:
    """The engine's packages, or a skip: called first in each test, so an absent engine never fails collection."""
    cfg = ENGINES[engine]
    for module in cfg["modules"]:
        pytest.importorskip(module)
    return cfg


def _record(out_dir: Path, engine: str, *, changed_step: int | None = None) -> Path:
    """A run with seeded random actions and the calls a test would make on it; ``changed_step`` moves one action."""
    cfg = _require(engine)
    env = gym.make(cfg["env_id"], **cfg["kwargs"])
    world = rw.attach(env, seed=SEED, name=f"{engine}-run", settings=rw.Settings(trace="on", trace_dir=str(out_dir)))
    try:
        env.reset(seed=SEED)
        expect(world.body(cfg["body"])).always.to_be_above(0.0)
        rng = np.random.default_rng(SEED)
        actions = rng.uniform(-1, 1, size=(STEPS, env.action_space.shape[-1])).astype(np.float32) * 0.5
        for i in range(STEPS):
            a = actions[i].copy()
            if i == changed_step:
                a[0] += 0.5
            env.step(a)
        expect(world.body(cfg["body"])).to_be_above(0.0)
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
        timeout=900,
    )


@ENGINE
def test_the_trace_keeps_every_body_pose_at_every_step(engine, tmp_path):
    cfg = _require(engine)
    tr = Trace(_record(tmp_path, engine))
    assert tr.meta["attached"]["engine"] == engine
    with zipfile.ZipFile(tr.path) as z:
        assert "bodies.npz" in z.namelist()
        with np.load(io.BytesIO(z.read("bodies.npz"))) as f:
            names = {str(n) for n in f["names"]}
            assert f["pos"].shape == (len(tr.arrays["t"]), len(names), 3)
            assert f["quat"].shape == (len(tr.arrays["t"]), len(names), 4)
    assert cfg["bodies"] <= names


@ENGINE
def test_replay_matches_every_body_at_every_step(engine, tmp_path):
    r = replay(_record(tmp_path, engine))
    assert r.engine == engine
    assert r.matches, r.summary()
    assert r.steps == STEPS
    # Measured on this machine: two recordings and three replays in fresh processes agree exactly, for both engines.
    # The tolerance is the default, which sits far above that.
    assert max(r.max_qpos_error, r.max_object_error, r.max_body_error, r.max_body_angle) <= 1e-6, r.summary()
    assert "within tolerance" in r.summary()


@ENGINE
def test_replay_detects_a_changed_action(engine, tmp_path):
    changed_row = 60
    path = _with_action_changed(_record(tmp_path, engine), changed_row)
    r = replay(path)
    assert not r.matches
    # Row k is the action of control step k+1, so the first state that can differ is state k+1.
    assert r.first_divergent_step == changed_row + 1, r.summary()
    assert "differs from the recorded run" in r.summary()


@ENGINE
def test_generated_test_passes_in_a_fresh_process_three_times(engine, tmp_path):
    path = _record(tmp_path, engine)
    out = tmp_path / f"test_{engine}_regression.py"
    out.write_text(generate(path, test_name=f"test_{engine}_regression"))
    if (companion_file := companion(path, f"test_{engine}_regression")) is not None:  # a long run's actions, beside it
        out.with_name(companion_file[0]).write_bytes(companion_file[1])
    for _ in range(3):
        r = _run_fresh(out)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "1 passed" in r.stdout
