"""Legged robots: the contract for quadrupeds and humanoids.

pytest tests/test_legged.py --rw-robot legged
"""

import numpy as np
import pytest

from robowright import expect


@pytest.fixture(autouse=True)
def _legged_only(rw_robot):
    from robowright import robots

    if robots.get(rw_robot).family != "legged":
        pytest.skip("legged robots only (run with --rw-robot legged)")


def test_stands_on_its_own(world, robot):
    """Joint servos alone hold the standing pose: no fall, no collapse, feet on the floor."""
    expect(robot.base).always.to_be_upright(tol_deg=10)
    world.wait(2.0)
    assert robot.base.height > 0.8 * robot.model.stand_height
    assert "floor" in robot.base.contacts()
    expect(robot.base).to_be_at_rest(lin_tol=0.02, ang_tol=0.05)


def test_crouches_and_stands_back_up(world, robot):
    h = robot.base.height
    robot.crouch(0.4)
    assert robot.base.height < h - 0.02
    robot.stand()
    expect(robot.base).to_be_upright(tol_deg=10)
    expect(robot.base).to_be_above(0.85 * h, timeout=2.0)


def test_recovers_from_a_shove(world, robot):
    """A sideways push scaled to the robot's weight: stumble allowed, falling over is not."""
    weight = robot.total_mass * 9.81
    world.wait(0.5)
    expect(robot.base).always.to_be_upright(tol_deg=30)
    world.faults.push("robot", force=(0.0, 0.3 * weight, 0.0), duration=0.1)
    world.wait(1.5)
    expect(robot.base).to_be_upright(tol_deg=10, timeout=2.0)


def test_falling_is_caught_by_an_invariant(world, robot):
    """A hard enough shove topples it, and the invariant fails at the moment it tips."""
    from robowright.errors import InvariantViolation

    weight = robot.total_mass * 9.81
    expect(robot.base).always.to_be_upright(tol_deg=30)
    world.faults.push("robot", force=(0.0, 3.0 * weight, 0.0), duration=0.3)
    with pytest.raises(InvariantViolation, match="tilted"):
        world.wait(3.0)


def test_policy_sees_imu_and_joint_state(world, robot):
    seen = {}

    def hold(obs):
        seen.update(obs)
        return robot.home_q

    rollout = robot.run_policy(hold, timeout=0.5)
    assert rollout.steps == 25
    assert {"qpos", "qvel", "base_quat", "base_lin_vel", "base_ang_vel"} <= set(seen)
    assert np.linalg.norm(seen["base_quat"]) == pytest.approx(1.0)


def test_state_restore_mid_stumble_is_exact(world, robot):
    """A state captured while the robot staggers from a shove replays the same future, even after the run moved on."""
    from robowright.backends.base import DETERMINISTIC, STATE

    b = world.backend
    if STATE not in b.capabilities or DETERMINISTIC not in b.capabilities:
        pytest.skip("no deterministic state save/restore")
    world.faults.push("robot", force=(0, 0.3 * robot.total_mass * 9.81, 0), duration=0.1)
    world.wait(0.15)

    def run():
        out = []
        for _ in range(30):
            b.step()
            out.append(np.concatenate([b.qpos(), *b.base_pose()]))
        return np.array(out)

    s = b.get_state()
    first = run()
    world.wait(0.5)
    b.set_state(s)
    assert np.array_equal(run(), first)
