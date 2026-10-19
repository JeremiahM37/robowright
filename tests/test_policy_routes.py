"""run_policy(route=...): what a policy reads or sends changed on the way, as page.route does for a page."""

import numpy as np
import pytest

import robowright as rw
from robowright.codegen import generate


class Recorder:
    """A policy that holds the arm where it is and remembers what it saw."""

    def __init__(self):
        self.seen = []

    def __call__(self, obs):
        self.seen.append(np.array(obs["qpos"], float))
        return np.array(obs["qpos"], float)


def _world(trace_dir=None):
    w = rw.launch(settings=rw.Settings(trace="on" if trace_dir else "off", trace_dir=str(trace_dir or "robowright-traces")))
    w.robot.reset_to()
    return w


def test_a_cut_observation_leaves_the_policy_its_last_reading(tmp_path):
    w = _world(tmp_path)
    p = Recorder()
    w.robot.run_policy(p, timeout=1.0, route={"observation": rw.route.cut(at=0.3, duration=0.4)})
    t = np.arange(len(p.seen)) * w.backend.control_dt
    during = [s for s, ti in zip(p.seen, t) if 0.32 < ti < 0.68]
    assert during and all(np.array_equal(s, during[0]) for s in during)  # frozen while cut
    assert any(e.name.startswith("route observation: cut") for e in w.trace.events)
    w.close()


def test_a_dropped_action_leaves_the_last_command_in_force():
    w = _world()
    q0 = w.robot.qpos().copy()

    def wave(obs):
        out = np.array(obs["qpos"], float)
        out[0] = q0[0] + 0.3  # every command asks the first joint to turn
        return out

    w.robot.run_policy(wave, timeout=1.0, route={"action": rw.route.abort()})
    assert abs(w.robot.qpos()[0] - q0[0]) < 0.01  # nothing got through: it held
    w.close()


def test_a_function_edits_what_the_policy_sends():
    w = _world()
    q0 = w.robot.qpos().copy()

    def wave(obs):
        out = np.array(obs["qpos"], float)
        out[0] = q0[0] + 0.3
        return out

    def clamp(act):
        act = act.copy()
        act[0] = min(act[0], q0[0] + 0.1)
        return act

    w.robot.run_policy(wave, timeout=2.0, route={"action": clamp})
    assert q0[0] + 0.05 < w.robot.qpos()[0] < q0[0] + 0.12  # went, but only as far as the clamp let it
    w.close()


def test_a_delay_reaches_the_policy_late_and_codegen_says_the_route_is_not_kept(tmp_path):
    w = _world(tmp_path)
    p = Recorder()
    w.robot.run_policy(p, timeout=0.5, route={"observation": rw.route.delay(0.2)})
    dt = w.backend.control_dt
    assert len(p.seen) == pytest.approx((0.5 - 0.2) / dt, abs=2)  # nothing to read for the first 0.2 s
    path = w.close()
    assert "its routes are not written again" in generate(rw.Trace(path))


def test_route_keys_are_checked():
    w = _world()
    with pytest.raises(ValueError, match="'observation' and 'action'"):
        w.robot.run_policy(Recorder(), timeout=0.1, route={"obs": rw.route.abort()})
    w.close()
