"""backend="lerobot": a real arm through LeRobot's own driver, here a stand-in with LeRobot's Robot interface and
the SO follower's motor names and units (degrees, gripper 0..100) whose servos move toward their targets in real
time. On the real arm, the stand-in is replaced by LeRobot's SOFollower(config) and nothing else changes."""

import math
import time

import numpy as np
import pytest

import robowright as rw
from robowright import expect

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class StandInArm:
    """LeRobot's Robot interface (connect, get_observation, send_action, disconnect), no serial port."""

    action_features = {f"{m}.pos": float for m in MOTORS}
    observation_features = action_features

    def __init__(self, start_deg=None, tau=0.04):
        self.pos = {m: 0.0 for m in MOTORS} | dict(start_deg or {})
        self.goal = dict(self.pos)
        self.tau, self.t = tau, time.monotonic()
        self.is_connected = False
        self.sent: list[dict] = []

    def connect(self, calibrate=True):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def _advance(self):
        now = time.monotonic()
        a = 1 - math.exp(-(now - self.t) / self.tau)
        self.t = now
        for m in MOTORS:
            self.pos[m] += (self.goal[m] - self.pos[m]) * a

    def get_observation(self):
        assert self.is_connected
        self._advance()
        return {f"{m}.pos": v for m, v in self.pos.items()}

    def send_action(self, action):
        assert self.is_connected
        self._advance()
        self.sent.append(dict(action))
        for k, v in action.items():
            self.goal[k.removesuffix(".pos")] = float(v)
        return action


def _launch(arm, **kw):
    return rw.launch(backend="lerobot", robot="so101", lerobot=arm, settings=rw.Settings(trace="off"), **kw)


def test_opening_the_world_moves_nothing_and_reads_the_arm_in_its_units():
    arm = StandInArm({"shoulder_pan": 30.0, "gripper": 100.0})
    w = _launch(arm)
    try:
        assert arm.is_connected and not arm.sent  # connected, and not one command on open
        assert w.robot.qpos()[0] == pytest.approx(math.radians(30.0))
        assert w.robot.qpos()[-1] == pytest.approx(1.0)  # gripper 100 of 0..100: fully open
        assert w.object_names == []  # a real arm's world has no objects robowright knows of
    finally:
        w.close()
    assert not arm.is_connected  # robowright connected it, so it disconnects it


def test_robowright_drives_the_arm_through_lerobot_inside_its_limits():
    arm = StandInArm()
    w = _launch(arm, limits={"joint_speed": 0.8})
    try:
        w.robot.reset_to()
        home = w.robot.qpos()[: w.robot.n_arm].copy()
        goal = home.copy()
        goal[0] += 0.4
        t0 = w.time
        w.robot.arm.move_joints(goal, speed=5.0)  # asked for far more than the limit
        assert abs(w.robot.qpos()[0] - goal[0]) < 0.03
        assert w.time - t0 >= 0.4 / 0.8 * 0.9
        speeds = [abs(b["shoulder_pan.pos"] - a["shoulder_pan.pos"]) for a, b in zip(arm.sent, arm.sent[1:])]
        assert max(speeds) <= math.degrees(0.8) / 30 + 1e-6  # no command step faster than the limit, in LeRobot's degrees
        w.robot.gripper.close()
        expect(w.robot.gripper).to_be_closed()
        assert arm.sent[-1]["gripper.pos"] == pytest.approx(0.0, abs=1.0)
    finally:
        w.close()


def test_a_policy_jump_is_refused_before_it_reaches_the_motors():
    arm = StandInArm()
    w = _launch(arm)  # the default limit: 1 rad/s
    try:
        here = w.robot.qpos()[: w.robot.n_arm].copy()
        jump = w.robot._target.copy()
        jump[1] += 1.0
        n = len(arm.sent)
        with pytest.raises(rw.SafetyLimitError, match="shoulder_lift would move at"):
            w.robot.run_policy(lambda obs: jump, timeout=1.0)
        assert all(abs(a["shoulder_lift.pos"] - math.degrees(here[1])) < 1.0 for a in arm.sent[n:])  # never sent
        w.step(10)
        assert np.abs(w.robot.qpos()[: w.robot.n_arm] - here).max() < 0.01  # and the arm holds
    finally:
        w.close()


def test_calibration_and_names_map_lerobot_onto_the_model():
    arm = StandInArm({"elbow_flex": 10.0})
    w = _launch(arm, calibration={"elbow_flex": (-1.0, 0.5)})
    try:
        assert w.robot.qpos()[2] == pytest.approx(-math.radians(10.0) + 0.5)
    finally:
        w.close()
    with pytest.raises(ValueError, match=r"has no wrist_twist\.pos"):
        _launch(StandInArm(), joints={"wrist_roll": "wrist_twist"})


def test_a_robot_handed_over_connected_stays_connected():
    arm = StandInArm()
    arm.connect()
    _launch(arm).close()
    assert arm.is_connected


def test_without_a_robot_says_how_to_give_one():
    with pytest.raises(ValueError, match="lerobot=arm"):
        rw.launch(backend="lerobot", robot="so101")


class _Bus:
    """LeRobot's Feetech bus, with its serial port taken away: what SOFollower calls on it, in its units."""

    def __init__(self, motors):
        self.motors = motors
        self.is_connected = False
        self.is_calibrated = True
        self.present = {m: 0.0 for m in motors}
        self.goals: list[dict] = []

    def connect(self, *a, **k):
        self.is_connected = True

    def disconnect(self, *a, **k):
        self.is_connected = False

    def configure_motors(self, *a, **k):
        pass

    def write(self, *a, **k):
        pass

    def torque_disabled(self, *a, **k):
        import contextlib

        return contextlib.nullcontext()

    def sync_read(self, register, *a, **k):
        assert register == "Present_Position"
        return dict(self.present)

    def sync_write(self, register, values, *a, **k):
        assert register == "Goal_Position"
        self.goals.append(dict(values))
        self.present.update(values)  # a servo that is where it was told: the bus is what is under test here


def test_lerobots_own_so_follower_driven_by_robowright(tmp_path):
    """LeRobot's SOFollower as LeRobot makes it, its serial bus replaced: robowright drives LeRobot's own code,
    including its max_relative_target clamp, and only the port is not real."""
    so = pytest.importorskip("lerobot.robots.so_follower")
    pytest.importorskip("scservo_sdk", reason="LeRobot's SO follower needs lerobot[feetech]")
    config = so.SOFollowerRobotConfig(port="/dev/null", id="robowright_test", calibration_dir=tmp_path, max_relative_target=20.0)
    arm = so.SOFollower(config)
    arm.bus = _Bus(arm.bus.motors)
    w = _launch(arm, limits={"joint_speed": 1.0})
    try:
        assert arm.is_connected and not arm.bus.goals  # LeRobot's connect ran; nothing commanded on open
        q = w.robot.qpos()[: w.robot.n_arm].copy()
        q[0] += 0.3
        w.robot.arm.move_joints(q, speed=0.5)
        assert arm.bus.present["shoulder_pan"] == pytest.approx(math.degrees(0.3), abs=0.5)  # in LeRobot's degrees
    finally:
        w.close()
    assert not arm.bus.is_connected
