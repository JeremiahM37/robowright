import pytest

import robowright as rw
from robowright import Point, condition, expect
from robowright.locators import Subject


def test_already_true_costs_no_time(quiet_world):
    w = quiet_world()
    t0 = w.time
    expect(w.scene["cube"]).to_be_near((0.22, -0.06, 0.0125), tol=0.005)
    assert w.time == t0


def test_failure_waits_for_timeout_and_reports_state(quiet_world):
    w = quiet_world()
    t0 = w.time
    with pytest.raises(rw.ExpectationError) as e:
        expect(w.scene["cube"]).to_be_inside(w.scene["bin"], timeout=0.5)
    assert w.time - t0 == pytest.approx(0.5, abs=w.dt)
    msg = str(e.value)
    assert "to_be_inside" in msg and "cube at" in msg and "bin spans" in msg


def test_waits_for_condition_to_become_true(quiet_world):
    w = quiet_world()
    w.robot.gripper.close()
    w.robot._target[-1] = 1.0  # start opening without waiting
    expect(w.robot.gripper).to_be_open(timeout=2.0)
    assert w.robot.gripper.opening >= 0.8


def test_hold_requires_condition_to_stay_true(quiet_world):
    w = quiet_world()
    t0 = w.time
    expect(w.scene["cube"]).to_be_at_rest(hold=0.5)
    assert w.time - t0 >= 0.5 - 1e-9


def test_not_negates(quiet_world):
    w = quiet_world()
    expect(w.scene["cube"]).not_.to_be_inside(w.scene["bin"], timeout=0)
    with pytest.raises(rw.ExpectationError):
        expect(w.scene["cube"]).not_.to_be_at_rest(timeout=0.1)


def test_soft_expectations_fail_at_close(quiet_world):
    w = quiet_world()
    expect.soft(w.scene["cube"]).to_be_inside(w.scene["bin"], timeout=0)
    expect.soft(w.scene["cube"]).to_be_above(0.5, timeout=0)
    with pytest.raises(rw.ExpectationError, match="2 soft expectation"):
        w.close()


def test_always_registers_an_invariant(quiet_world):
    w = quiet_world()
    expect(w.scene["cube"]).always.to_be_near((0.22, -0.06, 0.0125), tol=0.01)
    w.wait(0.2)  # holds
    w.faults.push("cube", (3.0, 0, 0), duration=0.2)
    with pytest.raises(rw.InvariantViolation, match="cube"):
        w.wait(1.0)


def test_condition_builds_a_predicate(quiet_world):
    w = quiet_world()
    inside = condition(w.scene["cube"], "to_be_inside", w.scene["bin"])
    near = condition(w.scene["cube"], "to_be_near", Point((0.22, -0.06, 0.0125)), tol=0.01)
    assert inside() is False and near() is True
    assert inside.__robowright__["matcher"] == "to_be_inside"


def test_to_satisfy_and_custom_message(quiet_world):
    w = quiet_world()
    expect(w.scene["cube"]).to_satisfy(lambda c: c.position[2] < 0.02, "on the table")
    with pytest.raises(rw.ExpectationError, match="^lifted: "):
        expect(w.scene["cube"], message="lifted").to_be_above(0.1, timeout=0)


def test_holding_and_collisions(quiet_world):
    w = quiet_world()
    cube = w.scene["cube"]
    expect(w.robot.gripper).not_.to_be_holding(cube, timeout=0)
    w.robot.pick(cube)
    expect(w.robot.gripper).to_be_holding(cube)
    expect(w.robot).to_have_no_collisions()
    expect(cube).to_be_touching("robot:left_finger")


def test_joint_matcher(quiet_world):
    w = quiet_world()
    q = w.robot.joints["shoulder_pan"]
    expect(w.robot).to_have_joint("shoulder_pan", q, tol=0.01)


class _Topic(Subject):
    """A ROS topic's handle as far as ``to_publish`` reads it: a name, a count and a rate."""

    def __init__(self, rate, count=10, name="/scan"):
        self.name, self._rate, self.count = name, rate, count

    @property
    def hz(self):
        return self._rate() if callable(self._rate) else self._rate


def test_a_topic_at_its_rate_passes_at_once(quiet_world):
    w = quiet_world()
    t0 = w.time
    expect(_Topic(4.0)).to_publish(hz=4.0, timeout=1)
    assert w.time == t0


def test_a_topic_off_its_rate_fails_with_the_rate_it_measured(quiet_world):
    w = quiet_world()
    t0 = w.time
    with pytest.raises(rw.ExpectationError) as e:
        expect(_Topic(1.0)).to_publish(hz=4.0, timeout=0.5)
    assert w.time - t0 == pytest.approx(0.5, abs=w.dt)
    assert "/scan measured 1.00 Hz (10 messages so far), want 4.0 Hz +/- 25%" in str(e.value)


def test_a_topic_within_tolerance_passes(quiet_world):
    quiet_world()
    expect(_Topic(3.2)).to_publish(hz=4.0, tol=0.25, timeout=0.1)
    with pytest.raises(rw.ExpectationError):
        expect(_Topic(2.9)).to_publish(hz=4.0, tol=0.25, timeout=0)


def test_a_topic_with_no_messages_fails_even_at_rate_zero(quiet_world):
    quiet_world()
    with pytest.raises(rw.ExpectationError, match="0 messages so far"):
        expect(_Topic(0.0, count=0)).to_publish(hz=0.0, timeout=0)


def test_to_publish_retries_until_the_rate_arrives(quiet_world):
    w = quiet_world()
    t0 = w.time
    expect(_Topic(lambda: 4.0 if w.time - t0 >= 0.5 else 0.0)).to_publish(hz=4.0, timeout=2)
    assert 0.5 - w.dt <= w.time - t0 <= 0.5 + 2 * w.dt


def test_to_publish_needs_a_topic(quiet_world):
    quiet_world()
    with pytest.raises(TypeError, match="ROS topic"):
        expect(Point((0.0, 0.0, 0.0))).to_publish(hz=1.0, timeout=0)
