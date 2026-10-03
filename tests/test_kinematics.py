import numpy as np
import pytest

from robowright import robots
from robowright.robot import DOWN, _kinematics, home_q, solve_ik

ALL = robots.names("arm")


@pytest.mark.parametrize("name", ALL)
def test_home_pose_points_the_fingers_down(name):
    kin = _kinematics(name)
    q = home_q(name)
    T = kin.fk(q)
    assert np.linalg.norm(T[:3, 3] - robots.get(name).home) < 1e-3
    assert T[:3, :3] @ kin.tool_axis @ DOWN > 0.999


@pytest.mark.parametrize("name", ALL)
def test_ik_round_trip_over_the_task_area(name):
    kin = _kinematics(name)
    home = home_q(name)
    rng = np.random.default_rng(1)
    # SO-101's top-down workspace is small; the rest cover the whole task area.
    lo, hi = ((0.16, -0.08, 0.02), (0.26, 0.1, 0.07)) if name == "so101" else ((0.14, -0.12, 0.02), (0.3, 0.18, 0.12))
    for _ in range(25):
        target = rng.uniform(lo, hi)
        q, err = solve_ik(kin, target, home, home, DOWN, yaw=rng.uniform(-np.pi, np.pi))
        assert err < 1e-3, f"{name} missed {target.round(3)} by {err * 1000:.1f} mm"
        T = kin.fk(q)
        assert T[:3, :3] @ kin.tool_axis @ DOWN > 0.99
        assert np.all(q >= kin.lower - 1e-9) and np.all(q <= kin.upper + 1e-9)


@pytest.mark.parametrize("name", ["so101", "panda", "ur5e"])
def test_kinematics_match_the_simulated_robot(name, quiet_world):
    """The IK model and the simulator agree on where the TCP is."""
    w = quiet_world(robot=name)
    kin = _kinematics(name)
    rng = np.random.default_rng(0)
    b = w.backend
    hand = b.model.body(robots.PREFIX + robots.get(name).hand).id
    for _ in range(20):
        q = rng.uniform(kin.lower, kin.upper)
        w.robot.reset_to(q)
        R = b.data.xmat[hand].reshape(3, 3)
        sim_tcp = b.data.xpos[hand] + R @ kin.tcp_offset
        assert np.linalg.norm(kin.tcp(q) - sim_tcp) < 1e-6


def test_unreachable_reports_error():
    kin = _kinematics("so101")
    _, err = kin.ik((0.6, 0, 0.3), home_q("so101"), DOWN)
    assert err > 0.1


def test_unknown_robot_lists_the_known_ones():
    with pytest.raises(ValueError, match="panda"):
        robots.get("no_such_robot")


@pytest.mark.parametrize("name", ALL)
def test_gripper_calibration_is_sane(name):
    d = robots.get(name).derived
    assert 0.03 < d.max_aperture < 0.15
    assert 0.005 < d.finger_reach < 0.05
    closed, opened = d.gripper_joints[d.gripper_joint]
    assert abs(opened - closed) > 1e-3
    assert abs(np.linalg.norm(d.tool_axis) - 1) < 1e-9 and abs(d.tool_axis @ d.grip_axis) < 1e-6
