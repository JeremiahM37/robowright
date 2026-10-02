import numpy as np
import pytest

import robowright as rw
from robowright import expect


def test_move_to_reaches_target(quiet_world):
    w = quiet_world()
    target = np.array([0.24, 0.05, 0.05])
    w.robot.arm.move_to(target)
    assert np.linalg.norm(w.robot.tcp.position - target) < 0.005


def test_linear_move_stays_on_line(quiet_world):
    w = quiet_world()
    w.robot.arm.move_to((0.22, 0.0, 0.07))
    start = w.robot.tcp.position.copy()
    end = np.array([0.22, 0.0, 0.025])
    worst = []
    w._step_hooks.append(lambda world: worst.append(_dist_to_segment(world.robot.tcp.position, start, end)))
    w.robot.arm.move_to(end, linear=True)
    assert max(worst) < 0.006


def _dist_to_segment(p, a, b):
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def test_unreachable_target_raises(quiet_world):
    w = quiet_world()
    with pytest.raises(rw.UnreachableError, match="mm"):
        w.robot.arm.move_to((0.5, 0.0, 0.3))


def test_weak_servo_times_out_with_diagnostic(quiet_world):
    w = quiet_world()
    w.faults.weak_joint("shoulder_lift", 0.001)
    with pytest.raises(rw.ActionTimeoutError, match="shoulder_lift|elbow_flex|wrist_flex"):
        w.robot.arm.move_to((0.28, 0.0, 0.03), timeout=1.0)


def test_gripper_opening(quiet_world):
    w = quiet_world()
    w.robot.gripper.close()
    assert w.robot.gripper.opening < 0.1
    w.robot.gripper.open()
    assert w.robot.gripper.opening > 0.9
    w.robot.gripper.open(0.5)
    assert 0.4 < w.robot.gripper.opening < 0.6


def test_pick_and_place_skill(quiet_world):
    w = quiet_world()
    w.robot.pick(w.scene["cube"])
    assert w.robot.gripper.holding() == "cube"
    w.robot.place(on=w.scene["bin"])
    expect(w.scene["cube"]).to_be_inside(w.scene["bin"])
    assert w.robot.gripper.holding() is None


def test_joint_readings_and_observation(quiet_world):
    w = quiet_world()
    assert list(w.robot.joints) == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    obs = w.robot.observe(cameras=("front",), privileged=True, task="t")
    assert obs["images"]["front"].shape == (240, 320, 3)
    assert set(obs["objects"]) == {"cube", "bin"} and obs["task"] == "t"


def test_run_policy_executes_action_chunks(quiet_world):
    w = quiet_world()
    calls = []
    hold = np.concatenate([w.robot.home_q, [1.2]])

    def policy(obs):
        calls.append(obs["t"])
        return np.tile(hold, (5, 1))

    r = w.robot.run_policy(policy, timeout=1.0)
    assert r.success and r.steps == 50 and len(calls) == 10


def test_run_policy_stops_when_condition_met(quiet_world):
    w = quiet_world()
    r = w.robot.run_policy(lambda obs: obs["qpos"], until=lambda: w.time > 0.3, timeout=5)
    assert r.success and 0.3 < r.sim_seconds < 0.4
