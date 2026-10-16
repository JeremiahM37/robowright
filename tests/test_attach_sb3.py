"""robowright attached to Stable-Baselines3's vector environments: a DummyVecEnv (in this process) and a SubprocVecEnv
(sub-environments in worker processes, wrapped there by rw.attachable), on gymnasium's HalfCheetah-v5.

Each index of a SubprocVecEnv must read exactly what the same index of a DummyVecEnv reads. Stable-Baselines3 resets a
sub-environment inside the step that ended its episode, so the step and the reset are both on that index's trace, and
the trace replays. Skipped where stable-baselines3 is not installed.
"""

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.errors import CapabilityError, InvariantViolation
from robowright.replay import replay
from robowright.trace import Trace

gym = pytest.importorskip("gymnasium")
vec_env = pytest.importorskip("stable_baselines3.common.vec_env")
env_util = pytest.importorskip("stable_baselines3.common.env_util")

ACTION_SEED = 0


@pytest.fixture
def make():
    vecs = []

    def make(kind: str, n: int = 3, wrap: bool = True, **kwargs):
        def one():
            env = gym.make("HalfCheetah-v5", **kwargs)
            return rw.attachable(env) if wrap else env

        cls = vec_env.DummyVecEnv if kind == "dummy" else vec_env.SubprocVecEnv
        vec = cls([one for _ in range(n)])
        vecs.append(vec)
        return vec

    yield make
    for vec in vecs:
        vec.close()


@pytest.fixture
def settings(tmp_path):
    return rw.Settings(trace="on", trace_dir=str(tmp_path))


def run(vec, settings, steps=20):
    n = vec.num_envs
    worlds = rw.attach(vec, settings=settings)
    vec.seed(0)  # index i is seeded 0 + i at the next reset
    vec.reset()
    rng = np.random.default_rng(ACTION_SEED)
    out = []
    for _ in range(steps):
        vec.step(rng.uniform(-1, 1, size=(n, 6)))
        out.append([w.body("torso").position.copy() for w in worlds])
    return worlds, np.array(out)


def test_a_subproc_index_reads_what_the_same_dummy_index_reads(make, settings):
    dummy_worlds, dummy = run(make("dummy"), settings)
    worlds, sub = run(make("subproc"), settings)
    assert np.array_equal(dummy, sub), f"max difference {np.abs(dummy - sub).max()}"
    for d, w in zip(dummy_worlds, worlds, strict=True):
        assert w.step_count == d.step_count == 20 and len(w.trace.t) == 21
        assert w.backend.remote_engine == "mujoco" and d.backend.engine == "mujoco"


def test_a_reset_inside_the_step_is_on_the_trace_and_the_trace_replays(make, settings):
    # A 5-step limit: Stable-Baselines3 resets each index inside the step that ended its episode (steps 5 and 10).
    worlds, _ = run(make("subproc", n=2, max_episode_steps=5), settings, steps=12)
    for i, w in enumerate(worlds):
        resets = [e.step for e in w.trace.events if e.type == "log" and e.name == "env.reset"]
        assert resets == [0, 5, 10], f"index {i} reset at steps {resets}"
        assert w.step_count == 12
    paths = worlds.close()
    for i, path in enumerate(paths):
        result = replay(path)
        assert result.identical, f"index {i}: {result.summary()}"
        assert Trace(path).meta["env"]["actions_recorded"] is True


def test_an_invariant_in_one_subproc_index_is_reported_for_that_index(make, settings):
    vec = make("subproc", n=3)
    worlds = rw.attach(vec, settings=settings)
    vec.seed(0)
    vec.reset()
    start = worlds[1].body("torso").position.copy()
    expect(worlds[1].body("torso")).always.to_be_near(start, tol=0.05)
    rng = np.random.default_rng(ACTION_SEED)
    with pytest.raises(InvariantViolation, match="sub-environment 1: invariant violated"):
        for _ in range(50):
            vec.step(rng.uniform(-1, 1, size=(3, 6)))
    vec.step(np.zeros((3, 6)))  # the workers go on: the violation was raised in this process


def test_a_wrapped_subproc_vec_env_is_attached_through_its_wrappers(make, settings):
    vec = vec_env.VecNormalize(make("subproc", n=2))
    worlds = rw.attach(vec, settings=settings)
    vec.reset()
    vec.step(np.zeros((2, 6)))
    assert [w.step_count for w in worlds] == [1, 1]


def test_make_vec_env_with_rw_attachable_as_its_wrapper(settings):
    vec = env_util.make_vec_env("HalfCheetah-v5", n_envs=2, seed=0, vec_env_cls=vec_env.SubprocVecEnv, wrapper_class=rw.attachable)
    try:
        worlds = rw.attach(vec, settings=settings)
        vec.reset()
        for _ in range(3):
            vec.step(np.zeros((2, 6)))
        assert [w.step_count for w in worlds] == [3, 3]
        assert worlds[1].body("torso").position[2] > 0
    finally:
        vec.close()


def test_a_subproc_vec_env_without_robowrights_wrapper_is_refused_and_keeps_running(make):
    vec = make("subproc", n=2, wrap=False)
    with pytest.raises(CapabilityError) as info:
        rw.attach(vec)
    assert "SubprocVecEnv" in str(info.value) and "rw.attachable" in str(info.value)
    vec.reset()
    vec.step(np.zeros((2, 6)))
