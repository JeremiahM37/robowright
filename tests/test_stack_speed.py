"""``speed=`` on a ROS 2 stack: Gazebo run faster than real time, for a stack on simulation time. Tests wait in
simulated time, so the same test takes less wall time; ros2_control's controllers run inside Gazebo's step and keep
pace exactly. Against Universal Robots' own Gazebo simulation, where a ROS 2 environment has it.
"""

import importlib.util
import time

import numpy as np
import pytest

import robowright as rw

pytestmark = pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)")


@pytest.fixture(scope="module")
def fast():
    from test_stack import UR, _ur_env

    try:
        import ament_index_python.packages as ament

        ament.get_package_share_directory("ur_simulation_gz")
    except Exception:
        pytest.skip("needs ur_simulation_gz")
    with rw.ros2_stack(UR, env=_ur_env(), speed=4.0) as st:
        yield st


def _rate(gz, seconds: float = 3.0) -> float:
    w0, s0 = time.monotonic(), gz.time
    time.sleep(seconds)
    return (gz.time - s0) / (time.monotonic() - w0)


def test_the_world_runs_faster_than_real_time(fast):
    rate = _rate(fast._gz)
    print(f"\nmeasured: Gazebo ran at {rate:.2f}x real time (asked 4x)")
    assert rate > 2.0, rate  # well past real time; how close to 4x depends on the machine


def test_the_arm_follows_its_controller_and_the_move_takes_less_wall_time(fast):
    fast.reset()
    scene = rw.SceneSpec(robot=None, objects=[], cameras=[])
    with rw.launch(scene, backend="ros2", use_sim_time=True, gazebo=True, settings=rw.Settings(trace="off")) as w:
        q = w.robot.qpos()[: w.robot.n_arm].copy()
        target = q + np.array([0.6, -0.3, 0.3, 0.0, 0.0, 0.4])
        s0, w0 = w.time, time.monotonic()
        w.robot.arm.move_joints(target)
        sim, wall = w.time - s0, time.monotonic() - w0
        print(f"\nmeasured: the move took {sim:.2f} s of simulated time in {wall:.2f} s of wall time")
        assert np.allclose(w.robot.qpos()[: w.robot.n_arm], target, atol=0.02)  # the controller kept pace
        assert wall < 0.75 * sim, (wall, sim)


def test_a_speed_must_be_above_zero():
    from robowright.gazebo import Gazebo

    with pytest.raises(ValueError, match="above 0"):
        Gazebo.set_speed(object(), 0)  # checked before anything is sent
