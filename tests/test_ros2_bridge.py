"""robowright's simulation served as a ROS 2 robot, driven by code that only speaks ROS 2.

Needs a ROS 2 environment (``scripts/ros2_env.sh pytest tests/test_ros2_bridge.py``); skipped
elsewhere.
"""

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from robowright import expect

pytestmark = pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)")
HERE = Path(__file__).parent
if "PYTEST_XDIST_WORKER" in os.environ:  # a ROS 2 domain of this worker's own
    os.environ["ROS_DOMAIN_ID"] = str(100 + int(os.environ["PYTEST_XDIST_WORKER"].removeprefix("gw")) % 50)
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


def _client(action_type, name):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor

    node = rclpy.create_node(f"test_client_{int(time.time() * 1000) % 100000}")
    client = ActionClient(node, action_type, name)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    from robowright.ros2_bridge import _spin

    threading.Thread(target=_spin, args=(ex,), daemon=True).start()
    return node, client


def test_a_trajectory_goal_moves_the_arm(world, robot, ros2):
    from control_msgs.action import FollowJointTrajectory
    from rclpy.duration import Duration
    from trajectory_msgs.msg import JointTrajectoryPoint

    node, client = _client(FollowJointTrajectory, "arm_controller/follow_joint_trajectory")
    assert client.wait_for_server(timeout_sec=10)
    target = robot.qpos()[: robot.n_arm] + 0.15
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(robot.model.arm_joints)
    point = JointTrajectoryPoint(positions=[float(v) for v in target])
    point.time_from_start = Duration(seconds=1.0).to_msg()
    goal.trajectory.points = [point]
    sent = client.send_goal_async(goal)
    world.run_until(lambda: sent.done() and sent.result().get_result_async().done(), timeout=5)
    expect(robot).to_have_joint(robot.model.arm_joints[0], float(target[0]), tol=0.02)
    assert np.allclose(robot.qpos()[: robot.n_arm], target, atol=0.03)
    assert any("FollowJointTrajectory" in g for g in ros2.goals)  # in the trace's timeline too
    node.destroy_node()


def test_your_stack_puts_the_cube_in_the_bin(world, scene, ros2):
    """The test drives nothing: a separate ROS 2 node (examples/ros2_stack/pick_node.py, standing
    in for your software) does the work; the test watches and asserts."""
    node = HERE.parent / "examples" / "ros2_stack" / "pick_node.py"
    with ros2.run([sys.executable, str(node), "--robot", world.backend.robot_model.name]):
        expect(scene["cube"]).to_be_inside(scene["bin"], timeout=60)
        expect(scene["cube"]).to_be_at_rest()
