"""robowright attached to a gymnasium vector environment: a SyncVectorEnv gives one world per sub-environment.

The sub-environments are gymnasium's own MuJoCo tasks (HalfCheetah-v5; FetchReach-v4 when gymnasium-robotics is
installed), each read through its own model and data. An AsyncVectorEnv's sub-environments run in worker processes:
made with rw.attachable, each index is read in its worker and must read what a SyncVectorEnv's same index
reads. Each test is skipped where its package is not installed.
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


@pytest.fixture
def make_async():
    vecs = []

    def make(n: int = 3, env_id: str = "HalfCheetah-v5", wrap: bool = True, **kwargs):
        wrappers = [rw.attachable] if wrap else None
        vec = gym.make_vec(env_id, n, vectorization_mode="async", wrappers=wrappers, **kwargs)
        vecs.append(vec)
        return vec

    yield make
    for vec in vecs:
        vec.close()


def run_torsos(vec, settings, steps=20, n=3):
    """Attach, reset with seeds 0..n-1, take ``steps`` seeded random vector steps: each step's torso per index."""
    worlds = rw.attach(vec, settings=settings)
    vec.reset(seed=list(range(n)))
    rng = np.random.default_rng(ACTION_SEED)
    out = []
    for _ in range(steps):
        vec.step(rng.uniform(-1, 1, size=(n, 6)))
        out.append([w.body("torso").position.copy() for w in worlds])
    return worlds, np.array(out)


def test_an_async_index_reads_what_the_same_sync_index_reads(make_vec, make_async, settings):
    sync_worlds, sync = run_torsos(make_vec(3), settings)
    worlds, remote = run_torsos(make_async(3), settings)
    assert np.array_equal(sync, remote), f"max difference {np.abs(sync - remote).max()}"
    assert [w.index for w in worlds] == [0, 1, 2]
    for s, w in zip(sync_worlds, worlds, strict=True):
        assert w.step_count == s.step_count == 20 and len(w.trace.t) == 21
        assert w.time == pytest.approx(s.time) and w.backend.control_dt == pytest.approx(s.backend.control_dt)
        assert w.bodies() == s.bodies() and w.backend.contacts() == s.backend.contacts()
        assert w.backend.remote_engine == "mujoco"


def test_a_push_on_one_async_index_moves_that_index_only(make_async, settings):
    def run(push: bool):
        vec = make_async(3, env_id="Ant-v5")  # the Ant's torso is a free body
        worlds = rw.attach(vec, settings=settings)
        vec.reset(seed=[0, 1, 2])
        if push:
            worlds[1].push("torso", (300.0, 0.0, 0.0), at=0.0, duration=0.1)
        for _ in range(10):
            vec.step(np.zeros((3, 8)))
        out = [w.body("torso").position.copy() for w in worlds]
        worlds.close()
        return out

    still, pushed = run(False), run(True)
    assert np.allclose(still[0], pushed[0]) and np.allclose(still[2], pushed[2]), "the push reached another index"
    assert np.linalg.norm(pushed[1] - still[1]) > 0.01, "the push did not move the pushed index"


def test_an_invariant_violated_in_one_async_index_is_reported_for_that_index(make_async, settings):
    vec = make_async(3)
    worlds = rw.attach(vec, settings=settings)
    vec.reset(seed=[0, 1, 2])
    start = worlds[2].body("torso").position.copy()
    expect(worlds[2].body("torso")).always.to_be_near(start, tol=0.05)
    rng = np.random.default_rng(ACTION_SEED)
    with pytest.raises(InvariantViolation, match="sub-environment 2: invariant violated"):
        for _ in range(50):
            vec.step(rng.uniform(-1, 1, size=(3, 6)))
    assert worlds[2].step_count < 50
    vec.step(np.zeros((3, 6)))  # the workers are untouched by the violation, which was raised here


def test_an_async_autoreset_is_recorded_on_its_index(make_async, settings):
    vec = make_async(2, max_episode_steps=5)
    worlds = rw.attach(vec, settings=settings)
    vec.reset(seed=[0, 1])
    for _ in range(12):
        vec.step(np.zeros((2, 6)))
    for i, w in enumerate(worlds):
        resets = [e.step for e in w.trace.events if e.type == "log" and e.name == "env.reset"]
        assert resets == [0, 5, 10], f"index {i} reset at steps {resets}"
        assert w.step_count == 10 and len(w.trace.t) == 11


def test_closing_async_worlds_releases_the_vector_env(make_async, settings):
    vec = make_async(2)
    worlds = rw.attach(vec, settings=settings)
    with pytest.raises(CapabilityError, match="attached already"):
        rw.attach(vec, settings=settings)
    vec.reset(seed=[0, 1])
    vec.step(np.zeros((2, 6)))
    worlds.close()
    vec.step(np.zeros((2, 6)))  # no longer observed
    assert [w.step_count for w in worlds] == [1, 1]
    again = rw.attach(vec, settings=settings)
    vec.step(np.zeros((2, 6)))
    assert [w.step_count for w in again] == [1, 1]


def test_an_async_vector_env_without_robowrights_wrapper_is_refused_and_keeps_running(make_async):
    vec = make_async(2, wrap=False)
    with pytest.raises(CapabilityError) as info:
        rw.attach(vec)
    msg = str(info.value)
    assert "worker process" in msg and "wrappers=[rw.attachable]" in msg, msg
    vec.reset(seed=[0, 1])  # asking did not end a worker (an error in one would)
    vec.step(np.zeros((2, 6)))


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
