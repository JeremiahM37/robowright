"""robowright attached to a gymnasium vector environment: a SyncVectorEnv gives one world per sub-environment.

The sub-environments are gymnasium's own MuJoCo tasks (HalfCheetah-v5; FetchReach-v4 when gymnasium-robotics is
installed), each read through its own model and data. An AsyncVectorEnv is refused: its sub-environments run in
other processes. Each test is skipped where its package is not installed.
"""

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.errors import CapabilityError, InvariantViolation

gym = pytest.importorskip("gymnasium")

ACTION_SEED = 0


def sync_vec(n: int, env_id: str = "HalfCheetah-v5", **kwargs):
    return gym.vector.SyncVectorEnv([lambda: gym.make(env_id, **kwargs) for _ in range(n)])


@pytest.fixture
def make_vec():
    vecs = []

    def make(n: int = 3, env_id: str = "HalfCheetah-v5", **kwargs):
        vec = sync_vec(n, env_id, **kwargs)
        vecs.append(vec)
        return vec

    yield make
    for vec in vecs:
        vec.close()


@pytest.fixture
def settings(tmp_path):
    return rw.Settings(trace_dir=str(tmp_path))


def torso_of(sub):
    """The sub-environment's own MuJoCo data for its torso, read without robowright."""
    core = sub.unwrapped
    return core.data.xpos[core.model.body("torso").id].copy()


def test_one_world_per_index_reads_that_sub_environments_own_data(make_vec, settings):
    vec = make_vec(4)
    worlds = rw.attach(vec, settings=settings)
    assert len(worlds) == 4 and [w.index for w in worlds] == [0, 1, 2, 3]
    assert worlds[3] is worlds.worlds[3] and [w.name.rsplit("__", 1)[-1][-3:] for w in worlds][:2] == ["[0]", "[1]"]
    vec.reset(seed=[0, 1, 2, 3])
    for i, w in enumerate(worlds):
        assert np.allclose(w.body("torso").position, torso_of(vec.envs[i])), f"index {i} read another environment's torso"
    positions = np.array([w.body("torso").position for w in worlds])
    assert not np.allclose(positions[0], positions[1]), "different seeds should start in different places"


def test_vector_step_steps_every_index_once_and_each_index_records_its_own_trace(make_vec, settings):
    vec = make_vec(3)
    worlds = rw.attach(vec, settings=settings)
    vec.reset(seed=[0, 1, 2])
    rng = np.random.default_rng(ACTION_SEED)
    for _ in range(5):
        vec.step(rng.uniform(-1, 1, size=(3, 6)))
    for i, w in enumerate(worlds):
        assert w.step_count == 5, f"index {i} took {w.step_count} steps for 5 vector steps"
        assert len(w.trace.t) == 6  # the start state, then one sample per step
        assert np.allclose(w.body("torso").position, torso_of(vec.envs[i]))


def test_an_invariant_violated_in_one_index_is_reported_for_that_index(make_vec, settings):
    vec = make_vec(4)
    worlds = rw.attach(vec, settings=settings)
    vec.reset(seed=[0, 1, 2, 3])
    for i, w in enumerate(worlds):
        if i == 2:
            # Random torques move the torso more than 5 cm from its start within a few steps (measured: 2 to 6 steps).
            start = w.body("torso").position.copy()
            expect(w.body("torso")).always.to_be_near(start, tol=0.05)
        else:
            expect(w.body("torso")).always.to_be_above(0.0)
    rng = np.random.default_rng(ACTION_SEED)
    with pytest.raises(InvariantViolation, match="sub-environment 2: invariant violated") as info:
        for _ in range(50):
            vec.step(rng.uniform(-1, 1, size=(4, 6)))
    assert "sub-environment 2" in str(info.value)
    assert worlds[2].step_count < 50  # the vector step stopped at the violation
    violations = [e for e in worlds[2].trace.events if e.type == "violation"]
    assert len(violations) == 1 and violations[0].status == "failed"
    for i in (0, 1, 3):
        assert not [e for e in worlds[i].trace.events if e.type == "violation"], f"index {i} recorded a violation it did not have"


def test_autoreset_is_recorded_on_its_index_and_is_not_a_physics_step(make_vec, settings):
    # A 5-step time limit makes every index end and autoreset inside the vector step, in the same vector steps.
    vec = make_vec(2, max_episode_steps=5)
    worlds = rw.attach(vec, settings=settings)
    for w in worlds:
        expect(w.body("torso")).always.to_be_above(0.0)  # holds across the resets: no invariant fires on the reset jump
    vec.reset(seed=[0, 1])
    for _ in range(12):
        vec.step(np.zeros((2, 6)))
    for i, w in enumerate(worlds):
        resets = [e.step for e in w.trace.events if e.type == "log" and e.name == "env.reset"]
        assert resets == [0, 5, 10], f"index {i} reset at steps {resets}"
        assert w.step_count == 10  # 12 vector steps, of which two were autoresets
        assert len(w.trace.t) == 11
    assert all(not [e for e in w.trace.events if e.type == "violation"] for w in worlds)


def test_a_wrapped_vector_env_is_attached_through_its_wrapper(make_vec, settings):
    vec = make_vec(2)
    wrapped = gym.wrappers.vector.RecordEpisodeStatistics(vec)
    worlds = rw.attach(wrapped, settings=settings)
    assert len(worlds) == 2
    wrapped.reset(seed=[0, 1])
    wrapped.step(np.zeros((2, 6)))
    assert [w.step_count for w in worlds] == [1, 1]


def test_argument_errors_are_said_in_one_line(make_vec, settings):
    vec = make_vec(2)
    with pytest.raises(ValueError, match="attached by itself"):
        rw.attach(vec, model=object(), settings=settings)
    with pytest.raises(ValueError, match="attached by itself"):
        rw.attach(vec, step=lambda: None, settings=settings)


def test_an_async_vector_env_is_refused_with_the_reason_and_the_alternative():
    vec = gym.vector.AsyncVectorEnv([lambda: gym.make("HalfCheetah-v5") for _ in range(2)])
    try:
        with pytest.raises(CapabilityError) as info:
            rw.attach(vec)
        msg = str(info.value)
        assert "AsyncVectorEnv" in msg and "worker process" in msg, msg
        assert "SyncVectorEnv" in msg and "rw.attach(gymnasium.make(env_id))" in msg, msg
    finally:
        vec.close()


def test_fetch_sub_environments_read_their_own_gripper_site(make_vec, settings):
    pytest.importorskip("gymnasium_robotics")
    vec = make_vec(3, "FetchReach-v4")
    worlds = rw.attach(vec, settings=settings)
    vec.reset(seed=[0, 1, 2])
    # Different moves per index, so the gripper positions differ and a read of the wrong index would show.
    vec.step(np.array([[0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]]))
    grips = []
    for i, w in enumerate(worlds):
        core = vec.envs[i].unwrapped
        site = core.model.site("robot0:grip").id  # a site: the gripper's own point, as Fetch's reward measures it
        assert np.allclose(w.body("robot0:grip").position, core.data.site_xpos[site])
        grips.append(w.body("robot0:grip").position)
    assert not np.allclose(grips[0], grips[1]) and not np.allclose(grips[0], grips[2])
