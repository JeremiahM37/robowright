"""rw.attach, out of the box, on all fifty Meta-World tasks, made as Meta-World's README makes them
(metaworld.make_mt_envs): each attaches, reads its bodies and passes an assertion. A run made that way cannot be
rebuilt (the maker keeps no record of its seed, and the seed places the task's goal), so replay refuses it and says
how to make the environment so that it can: through gymnasium.make, whose runs replay bit-for-bit. A cold trial
(2026-10-15) attached all fifty first time. Skipped where Meta-World is absent."""

import numpy as np
import pytest

import robowright as rw
from robowright import expect

metaworld = pytest.importorskip("metaworld")
gym = pytest.importorskip("gymnasium")
from metaworld import env_dict  # noqa: E402

TASKS = sorted(env_dict.ALL_V3_ENVIRONMENTS)


def _run(env, steps=30):
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        env.step(rng.uniform(env.action_space.low, env.action_space.high).astype(env.action_space.dtype))


@pytest.mark.parametrize("task", TASKS)
def test_attach_to_every_metaworld_task_as_its_readme_makes_it(task, tmp_path):
    env = metaworld.make_mt_envs(task, seed=0)
    w = rw.attach(env, settings=rw.Settings(trace="off"), name=task)
    _run(env)
    bodies = w.bodies()
    assert len(bodies) > 5  # the Sawyer arm's links and the task's objects, found by itself
    expect(w.body(bodies[-1])).to_be_above(-100.0, timeout=0)
    w.close()
    env.close()


def test_a_make_mt_envs_run_is_refused_with_the_way_to_make_it_replayable(tmp_path):
    from robowright.replay import replay

    env = metaworld.make_mt_envs("door-open-v3", seed=0)
    w = rw.attach(env, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    _run(env)
    path = w.close(trace_path=tmp_path / "t.zip")
    with pytest.raises(ValueError, match=r"keeps no record of its seed.*gymnasium\.make\('Meta-World/MT1'"):
        replay(path)


@pytest.mark.parametrize("task", ["door-open-v3", "pick-place-v3", "assembly-v3", "hammer-v3", "shelf-place-v3"])
def test_a_gymnasium_made_run_replays_bit_for_bit(task, tmp_path):
    from robowright.replay import replay

    env = gym.make("Meta-World/MT1", env_name=task, seed=0)
    w = rw.attach(env, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    _run(env)
    r = replay(w.close(trace_path=tmp_path / f"{task}.zip"))
    assert r.matches, r.summary()
