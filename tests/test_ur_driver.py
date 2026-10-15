"""Universal Robots' own driver (ur_robot_driver), as it runs a real UR arm, with its mock hardware in
place of the arm: robowright finds the robot and its controller with no settings, drives it inside
[ros2] limits, and refuses a policy's target that would break them. On a real UR the only change is
use_mock_hardware:=false and the robot's address.

Needs a ROS 2 environment with ur_robot_driver: ``scripts/ros2_env.sh pytest tests/test_ur_driver.py``.
"""

import importlib.util
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import robowright as rw

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rclpy") is None or shutil.which("ros2") is None, reason="needs a ROS 2 environment"
)


@pytest.fixture(scope="module")
def ur_driver(tmp_path_factory):
    share = Path(os.environ.get("CONDA_PREFIX", "/nonexistent")) / "share" / "ur_robot_driver"
    if not share.is_dir():
        pytest.skip("ur_robot_driver is not installed in this ROS 2 environment")
    # The driver on this process's own ROS domain: rclpy, started by an earlier test in the same session,
    # keeps the domain it started with, so a driver moved to another one was never seen (found in the gate).
    os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    log = tmp_path_factory.mktemp("ur") / "driver.log"
    cmd = [
        "ros2", "launch", "ur_robot_driver", "ur_control.launch.py", "ur_type:=ur5e", "robot_ip:=192.0.2.1",
        "use_mock_hardware:=true", "launch_rviz:=false", "headless_mode:=true",
        "initial_joint_controller:=scaled_joint_trajectory_controller",
    ]  # fmt: skip
    p = subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    yield log
    os.killpg(p.pid, signal.SIGINT)
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def test_robowright_drives_urs_driver_inside_its_limits(ur_driver):
    w = rw.launch(backend="ros2", robot=None, settings=rw.Settings(trace="off"), timeout=90.0, limits={"joint_speed": 0.5})
    try:
        assert w.backend.ros_names[0] == "shoulder_pan_joint"
        assert w.backend.cfg["arm"]["controller"] == "scaled_joint_trajectory_controller"  # found, not configured
        n = w.robot.n_arm
        q0 = w.robot.qpos()[:n].copy()
        goal = q0.copy()
        goal[0] += 0.3
        t0 = w.time
        w.robot.arm.move_joints(goal, speed=3.0)  # asked for far more than the limit
        assert abs(w.robot.qpos()[0] - goal[0]) < 0.02
        assert w.time - t0 >= 0.3 / 0.5

        here = w.robot.qpos()[:n].copy()
        jump = w.robot._target.copy()
        jump[0] += 1.0
        with pytest.raises(rw.SafetyLimitError, match="shoulder_pan_joint"):
            w.robot.run_policy(lambda obs: jump, timeout=1.0)
        time.sleep(0.5)
        assert np.abs(w.robot.qpos()[:n] - here).max() < 0.01
    finally:
        w.close()
