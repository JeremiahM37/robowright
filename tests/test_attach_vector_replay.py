"""Replay and codegen for a vector environment attached with rw.attach: each sub-environment's trace is a run of its own.
Every test runs on a SyncVectorEnv and on an AsyncVectorEnv (sub-environments in worker processes, rw.attachable).

The sub-environments are gymnasium's HalfCheetah-v5, built the way a user builds them, so the traces come from the
same code path as a real vector run. Each index's trace is replayed on a freshly made environment and compared with
the recorded state at every step; the generated test is run in a subprocess. Skipped where gymnasium is absent.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.codegen import generate
from robowright.replay import replay
from robowright.trace import Trace

gym = pytest.importorskip("gymnasium")

ACTION_SEED = 0
SRC = Path(rw.__file__).resolve().parent.parent


def sync_vec(n: int, autoreset_mode=None, **kwargs):
    extra = {} if autoreset_mode is None else {"autoreset_mode": autoreset_mode}
    return gym.vector.SyncVectorEnv([lambda: gym.make("HalfCheetah-v5", **kwargs) for _ in range(n)], **extra)


@pytest.fixture(params=["sync", "async"])
def mode(request):
    return request.param


def make_vector(mode: str, n: int, autoreset_mode=None, **kwargs):
    if mode == "sync":
        return sync_vec(n, autoreset_mode, **kwargs)
    extra = {} if autoreset_mode is None else {"autoreset_mode": autoreset_mode}
    return gym.make_vec("HalfCheetah-v5", n, vectorization_mode="async", vector_kwargs=extra, wrappers=[rw.attachable], **kwargs)


def run_vector(tmp_path, mode, *, n=2, steps=12, limit=None, autoreset_mode=None, seeded=True, invariant=False):
    """Attach to a vector env, run ``steps`` vector steps, and return the worlds and each index's trace path."""
    kwargs = {} if limit is None else {"max_episode_steps": limit}
    vec = make_vector(mode, n, autoreset_mode, **kwargs)
    worlds = rw.attach(vec, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    if invariant:
        for w in worlds:
            expect(w.body("torso")).always.to_be_above(0.0)
    vec.reset(seed=list(range(n)) if seeded else None)
    rng = np.random.default_rng(ACTION_SEED)
    for _ in range(steps):
        vec.step(rng.uniform(-1, 1, size=(n, 6)))
    paths = [w.close() for w in worlds]
    vec.close()
    return worlds, paths


def test_each_index_replays_on_its_own_state_for_state(tmp_path, mode):
    worlds, paths = run_vector(tmp_path, mode, n=3, steps=8)
    assert [w.index for w in worlds] == [0, 1, 2]
    for i, path in enumerate(paths):
        result = replay(path)
        assert result.identical, f"index {i}: {result.summary()}"
        assert result.steps == worlds[i].step_count == 8
        assert result.first_divergent_step is None


def test_an_index_that_autoresets_replays_its_resets_at_the_same_steps(tmp_path, mode):
    # A 5-step limit makes both indices end and autoreset inside the vector steps, in the same vector steps.
    worlds, paths = run_vector(tmp_path, mode, n=2, steps=12, limit=5)
    for i, path in enumerate(paths):
        resets = [e["step"] for e in Trace(path).events if e["type"] == "log" and e["name"] == "env.reset"]
        assert resets == [0, 5, 10], f"index {i} reset at {resets}"
        result = replay(path)
        assert result.identical, f"index {i}: {result.summary()}"
        assert result.steps == worlds[i].step_count == 10  # 12 vector steps, two of them autoresets


def test_same_step_autoreset_replays_too(tmp_path, mode):
    # SAME_STEP resets inside the vector step that ended the episode, so the reset follows that step's physics step.
    worlds, paths = run_vector(tmp_path, mode, n=2, steps=12, limit=5, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    for i, path in enumerate(paths):
        resets = [e["step"] for e in Trace(path).events if e["type"] == "log" and e["name"] == "env.reset"]
        assert resets == [0, 5, 10], f"index {i} reset at {resets}"
        result = replay(path)
        assert result.identical, f"index {i}: {result.summary()}"
        assert result.steps == worlds[i].step_count == 12


def test_a_vector_run_first_reset_without_a_seed_is_refused_by_replay_and_codegen(tmp_path, mode):
    # gymnasium seeds an unseeded generator from the operating system: the start cannot be repeated, so the trace says so.
    _, paths = run_vector(tmp_path, mode, n=2, steps=6, seeded=False)
    for path in paths:
        tr = Trace(path)
        assert tr.meta["env"]["actions_recorded"] is False
        assert "first reset without a seed" in tr.meta["env"]["note"]
        with pytest.raises(ValueError, match="first reset without a seed"):
            replay(path)
        with pytest.raises(ValueError, match="first reset without a seed"):
            generate(tr, test_name="refused")


def test_codegen_on_an_index_writes_a_test_that_passes_with_an_autoreset(tmp_path, mode):
    _, paths = run_vector(tmp_path, mode, n=2, steps=12, limit=5, invariant=True)
    test_file = tmp_path / "test_index_one.py"
    test_file.write_text(generate(Trace(paths[1]), test_name="test_index_one"))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(SRC), env.get("PYTHONPATH", "")] if p)
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed" in run.stdout, run.stdout
