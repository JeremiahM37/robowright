"""A stand-in for *your* code on a MoveIt arm: ask ``move_group`` to put the tool somewhere.

It is a plain MoveIt client and knows nothing about robowright: it sends a ``MoveGroup`` goal
(a pose for ``tool0``, the tool pointing down) and MoveIt plans it, and executes it through the
robot's own controllers. ``tests/test_moveit_gazebo.py`` runs it against Universal Robots' Gazebo
simulation and MoveIt configuration, and robowright only watches.

    python reach.py X Y Z [--link tool0] [--group ur_manipulator] [--frame base_link]
"""

from __future__ import annotations

import argparse
import sys

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, MotionPlanRequest, MoveItErrorCodes, OrientationConstraint, PositionConstraint
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from shape_msgs.msg import SolidPrimitive

# What a plan that might be found next time fails with (OMPL ran out of time, a sampled goal had no IK).
_REPLAN = {MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.FAILURE, MoveItErrorCodes.TIMED_OUT, MoveItErrorCodes.NO_IK_SOLUTION}


def goal(x: float, y: float, z: float, link: str, group: str, frame: str, speed: float) -> MoveGroup.Goal:
    req = MotionPlanRequest(
        group_name=group,
        num_planning_attempts=5,
        allowed_planning_time=10.0,
        max_velocity_scaling_factor=speed,
        max_acceleration_scaling_factor=speed,
    )
    where = PositionConstraint(link_name=link, weight=1.0)
    where.header.frame_id = frame
    where.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.001])]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    where.constraint_region.primitive_poses = [pose]
    down = OrientationConstraint(link_name=link, weight=1.0)
    down.header.frame_id = frame
    down.orientation.x, down.orientation.w = 1.0, 0.0  # half a turn about x: the tool's z points down
    down.absolute_x_axis_tolerance = down.absolute_y_axis_tolerance = down.absolute_z_axis_tolerance = 0.01
    req.goal_constraints = [Constraints(position_constraints=[where], orientation_constraints=[down])]
    return MoveGroup.Goal(request=req)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", type=float, nargs=3)
    ap.add_argument("--link", default="tool0")
    ap.add_argument("--group", default="ur_manipulator")
    ap.add_argument("--frame", default="base_link")
    ap.add_argument("--speed", type=float, default=0.3, help="velocity and acceleration scaling")
    ap.add_argument("--attempts", type=int, default=3, help="plans to try before giving up")
    a = ap.parse_args(argv)
    rclpy.init()
    node = rclpy.create_node("reach", parameter_overrides=[Parameter("use_sim_time", value=True)])
    client = ActionClient(node, MoveGroup, "move_action")
    if not client.wait_for_server(timeout_sec=30):
        print("no move_group", file=sys.stderr)
        return 2
    # OMPL's planners sample at random, and a narrow pose goal is sometimes not solved in the
    # time given: plan again, as a MoveIt client should, before giving up.
    for attempt in range(1, a.attempts + 1):
        sent = client.send_goal_async(goal(*a.xyz, a.link, a.group, a.frame, a.speed))
        rclpy.spin_until_future_complete(node, sent, timeout_sec=30)
        handle = sent.result()
        if handle is None or not handle.accepted:
            print("move_group rejected the goal", file=sys.stderr)
            return 1
        done = handle.get_result_async()
        rclpy.spin_until_future_complete(node, done, timeout_sec=120)
        code = done.result().result.error_code.val if done.result() else None
        print(f"move_group: attempt {attempt}: error code {code}", flush=True)
        if code not in _REPLAN:
            break
    node.destroy_node()
    rclpy.shutdown()
    return 0 if code == MoveItErrorCodes.SUCCESS else 1


if __name__ == "__main__":
    sys.exit(main())
