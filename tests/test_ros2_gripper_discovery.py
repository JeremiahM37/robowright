"""Which controller robowright drives a gripper through, from a controller manager's list alone (no ROS 2
needed): a gripper that is one more joint of a joint trajectory controller, as xArm's is, is found; an
arm controller and a gripper action controller are not mistaken for it."""

from robowright.ros2_discover import Controller, Graph, gripper_trajectory


def _graph(*controllers):
    return Graph(controllers=list(controllers))


def _jtc(name, joints, state="active"):
    return Controller(name=name, type="joint_trajectory_controller/JointTrajectoryController", state=state, joints=list(joints))


def test_finds_a_gripper_claimed_by_a_joint_trajectory_controller():
    arm = _jtc("xarm6_traj_controller", [f"joint{i}" for i in range(1, 7)])
    grip = _jtc("xarm_gripper_traj_controller", ["drive_joint"])
    g = _graph(arm, grip)
    assert gripper_trajectory(g, "xarm6_traj_controller", "drive_joint") is grip


def test_never_takes_the_arm_controller_or_an_inactive_one():
    arm = _jtc("xarm6_traj_controller", ["joint1", "drive_joint"])
    idle = _jtc("xarm_gripper_traj_controller", ["drive_joint"], state="inactive")
    assert gripper_trajectory(_graph(arm), "xarm6_traj_controller", "drive_joint") is None
    assert gripper_trajectory(_graph(arm, idle), "xarm6_traj_controller", "drive_joint") is None


def test_no_gripper_joint_means_no_gripper():
    arm = _jtc("xarm6_traj_controller", [f"joint{i}" for i in range(1, 7)])
    assert gripper_trajectory(_graph(arm), "xarm6_traj_controller", None) is None
    assert gripper_trajectory(_graph(arm), "xarm6_traj_controller", "drive_joint") is None
