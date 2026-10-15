"""rw.attach, out of the box, on gymnasium's own MuJoCo environments robowright was never set up for: each is
built as a user builds it (gymnasium.make), stepped by the user's own loop with random actions, read body by body,
asserted on, and its trace replayed bit-for-bit. A cold trial (2026-10-15) passed all nine first time; this keeps
them passing. Skipped where gymnasium's MuJoCo environments are absent."""

import numpy as np
import pytest

import robowright as rw
from robowright import expect

gym = pytest.importorskip("gymnasium")

ENVS = [
    "Ant-v5",
    "Hopper-v5",
    "Humanoid-v5",
    "HumanoidStandup-v5",
    "Pusher-v5",
    "Walker2d-v5",
    "Swimmer-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
]


@pytest.mark.parametrize("env_id", ENVS)
def test_attach_reads_asserts_and_replays_an_environment_it_was_never_set_up_for(env_id, tmp_path):
    from robowright.replay import replay

    try:
        env = gym.make(env_id)
    except Exception as e:  # noqa: BLE001 - gymnasium without its MuJoCo extra
        pytest.skip(f"{env_id}: {e}")
    w = rw.attach(env, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)), name=env_id)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        env.step(rng.uniform(env.action_space.low, env.action_space.high).astype(env.action_space.dtype))
    bodies = w.bodies()
    assert len(bodies) >= 3  # the world and the robot's links, found by itself
    moving = bodies[-1]
    assert np.all(np.isfinite(w.body(moving).position))
    expect(w.body(moving)).to_be_above(-100.0, timeout=0)
    path = w.close(trace_path=tmp_path / f"{env_id}.zip")
    r = replay(path)
    assert r.matches, r.summary()  # the user's run, re-simulated from the trace, bit for bit
    env.close()
