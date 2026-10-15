"""rw.attach, out of the box, on every domain of DeepMind's dm_control suite (one task each): loaded as a user loads
it (suite.load), stepped by the user's own loop with random actions, read body by body, asserted on, and its trace
replayed bit-for-bit, the environment rebuilt by itself from what the trace recorded (no factory). A cold trial
(2026-10-15) found dm_control runs could not be replayed without one, and their resets drew from the task's own
random state; both are fixed. lqr builds its model from random numbers when loaded, so a rebuilt one differs: the
replay says so. Skipped where dm_control is absent."""

import numpy as np
import pytest

import robowright as rw
from robowright import expect

suite = pytest.importorskip("dm_control.suite")

DOMAINS = sorted({d for d, _ in suite.ALL_TASKS})


def _first_task(domain):
    return next(t for d, t in suite.ALL_TASKS if d == domain)


@pytest.mark.parametrize("domain", DOMAINS)
def test_attach_and_replay_every_dm_control_domain(domain, tmp_path):
    from robowright.replay import replay

    task = _first_task(domain)
    env = suite.load(domain, task, task_kwargs={"random": 0})
    w = rw.attach(env, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)), name=f"{domain}-{task}")
    env.reset()
    spec = env.action_spec()
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(rng.uniform(spec.minimum, spec.maximum, spec.shape))
    bodies = w.bodies()
    assert len(bodies) >= 2
    expect(w.body(bodies[-1])).to_be_above(-100.0, timeout=0)
    r = replay(w.close(trace_path=tmp_path / f"{domain}.zip"))  # no make_env: the trace names the suite task
    if domain == "lqr":
        assert not r.matches and any("random numbers" in n for n in r.notes), r.summary()
    else:
        assert r.matches, r.summary()
