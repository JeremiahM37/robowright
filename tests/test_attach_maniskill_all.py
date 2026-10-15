"""rw.attach, out of the box, on every task ManiSkill 3 registers, made as its docs make them (gymnasium.make, state
observations, the CPU simulation): each attaches, steps under the user's own loop, reads its bodies and passes an
assertion. A task whose assets are not on this machine (ManiSkill asks to download them, or a dataset is missing) is
skipped with ManiSkill's own reason: that is ManiSkill's to fix, before robowright is involved. A cold trial
(2026-10-15) attached all 45 tasks whose assets were here, with no failure on robowright's side. Skipped where
ManiSkill is absent."""

import pytest

import robowright as rw
from robowright import expect

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill.envs")
from mani_skill.utils.registration import REGISTERED_ENVS  # noqa: E402

TASKS = sorted(REGISTERED_ENVS)


@pytest.mark.parametrize("task", TASKS)
def test_attach_to_every_maniskill_task(task):
    try:
        env = gym.make(task, obs_mode="state", sim_backend="cpu", render_mode=None)
    except (EOFError, OSError, FileNotFoundError, ImportError, RuntimeError) as e:  # assets or extras not installed
        pytest.skip(f"ManiSkill cannot make {task} here: {type(e).__name__}: {str(e)[:120]}")
    w = rw.attach(env, settings=rw.Settings(trace="off"), name=task)
    try:
        env.reset(seed=0)
        for _ in range(5):
            env.step(env.action_space.sample())
        bodies = w.bodies()
        assert bodies  # found by itself
        expect(w.body(bodies[-1])).to_be_above(-100.0, timeout=0)
    finally:
        w.close()
        env.close()
