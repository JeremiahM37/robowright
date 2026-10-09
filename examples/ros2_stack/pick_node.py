"""A stand-in for *your* robot software: a ROS 2 node that puts the cube in the bin.

It knows nothing about the simulation. Like a stack on a real arm, it

* finds the cube and the bin as TF frames,
* plans joint positions with an IK library (robowright's kinematics here, standing in for
  MoveIt or KDL: any IK would do), and
* drives the arm through ``control_msgs/FollowJointTrajectory`` and the gripper through
  ``control_msgs/GripperCommand``, the interfaces a ``ros2_control`` arm has.

So it runs unchanged against robowright's simulation served over ROS 2
(``tests/test_ros2_bridge.py``), against Gazebo, or against the real arm. The test only watches
and asserts. Run it as ``python pick_node.py --robot so101``.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from robowright import robots
from robowright.robot import DOWN, solve_ik
from robowright.robots.model import Kinematics


class PickNode:
    def __init__(self, robot: str, arm: str, gripper: str):
        self.model = m = robots.get(robot)
        self.kin = Kinematics(m)
        self.home = np.zeros(m.n_arm)
        grip, (self.closed, self.opened) = next(iter(m.derived.gripper_joints.items()))
        self.node = node = rclpy.create_node("pick_node", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.q = None
        node.create_subscription(JointState, "joint_states", self._on_joints, 10)
        self.tf = Buffer()
        self._listener = TransformListener(self.tf, node)
        self.arm = ActionClient(node, FollowJointTrajectory, f"{arm}/follow_joint_trajectory")
        self.gripper = ActionClient(node, GripperCommand, f"{gripper}/gripper_cmd")
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(node)
        threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self):
        from rclpy.executors import ExternalShutdownException

        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass

    def _on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in self.model.arm_joints):
            self.q = np.array([pos[j] for j in self.model.arm_joints])

    def where(self, frame: str) -> tuple[np.ndarray, float]:
        """An object's position and yaw in the world frame, from TF."""
        t = self.tf.lookup_transform("world", frame, Time(), timeout=Duration(seconds=5.0)).transform
        r = t.rotation
        yaw = float(np.arctan2(2 * (r.w * r.z + r.x * r.y), 1 - 2 * (r.y * r.y + r.z * r.z)))
        return np.array([t.translation.x, t.translation.y, t.translation.z]), yaw

    def move(self, p, yaw: float, seconds: float = 1.5):
        q, err = solve_ik(self.kin, p, self.q, self.home, DOWN, yaw)
        if err > 5e-3:
            raise RuntimeError(f"cannot reach {np.round(p, 3)}")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.model.arm_joints)
        point = JointTrajectoryPoint(positions=[float(v) for v in q])
        point.time_from_start = Duration(seconds=seconds).to_msg()
        goal.trajectory.points = [point]
        self._call(self.arm, goal)
        time.sleep(0.3)  # settle (wall time; the simulation runs in real time)

    def reachable(self, p, yaw: float, heights) -> np.ndarray:
        """The first of ``p`` raised by each of ``heights`` that the arm can reach."""
        for h in heights:
            q = p + [0, 0, h]
            if solve_ik(self.kin, q, self.q, self.home, DOWN, yaw)[1] <= 5e-3:
                return q
        raise RuntimeError(f"cannot reach above {np.round(p, 3)}")

    def grip(self, opening: float):
        goal = GripperCommand.Goal()
        goal.command.position = float(self.closed + opening * (self.opened - self.closed))
        goal.command.max_effort = 0.0
        self._call(self.gripper, goal)
        time.sleep(0.3)

    def _call(self, client, goal):
        if not client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError(f"no action server {client._action_name}")
        handle = _wait(client.send_goal_async(goal))
        if not handle.accepted:
            raise RuntimeError("goal rejected")
        _wait(handle.get_result_async())

    def run(self):
        deadline = time.monotonic() + 20
        while self.q is None:
            if time.monotonic() > deadline:
                raise RuntimeError("no joint states")
            time.sleep(0.05)
        cube, yaw = self.where("cube")
        box, _ = self.where("bin")
        self.grip(1.0)
        self.move(cube + [0, 0, 0.06], yaw)
        self.move(cube + [0, 0, 0.002], yaw, seconds=1.0)
        self.grip(0.0)
        self.move(cube + [0, 0, 0.08], yaw, seconds=1.0)
        # Over the bin as high as the arm reaches there (a small arm's reach ends lower), then down.
        above = self.reachable(box, yaw, heights=(0.10, 0.09, 0.08, 0.075))
        self.move(above, yaw)
        self.move(box + [0, 0, 0.065], yaw, seconds=1.0)
        self.grip(1.0)
        self.move(above, yaw, seconds=1.0)
        print("done", flush=True)


def _wait(future, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while not future.done():
        if time.monotonic() > deadline:
            raise TimeoutError("an action did not answer")
        time.sleep(0.01)
    return future.result()


def main(argv=None) -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--robot", default="so101")
    a.add_argument("--arm", default="arm_controller")
    a.add_argument("--gripper", default="gripper_controller")
    args = a.parse_args(argv)
    rclpy.init()
    try:
        PickNode(args.robot, args.arm, args.gripper).run()
    finally:
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
