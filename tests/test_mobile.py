"""Mobile robots without ROS: a wheeled model is detected as one, and its API has a base and wheels, not an arm."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from robowright import robots
from robowright.legged import Base
from robowright.mobile import MobileRobot
from robowright.robots.detect import _wheeled, build
from robowright.robots.from_urdf import to_mjcf
from robowright.robots.urdf import cache_root

WHEELED = """<?xml version="1.0"?>
<robot name="rover">
  <link name="base_link">
    <inertial><origin xyz="0 0 0.05"/><mass value="5"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.3 0.2 0.1"/></geometry></visual>
    <collision><geometry><box size="0.3 0.2 0.1"/></geometry></collision>
  </link>
  <link name="left_wheel">
    <inertial><mass value="0.5"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.05" length="0.03"/></geometry></collision>
  </link>
  <link name="right_wheel">
    <inertial><mass value="0.5"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.05" length="0.03"/></geometry></collision>
  </link>
  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/><child link="left_wheel"/><origin xyz="0 0.12 0"/><axis xyz="0 1 0"/>
    <limit effort="1.0" velocity="10"/>
  </joint>
  <joint name="wheel_right_joint" type="continuous">
    <parent link="base_link"/><child link="right_wheel"/><origin xyz="0 -0.12 0"/><axis xyz="0 1 0"/>
    <limit effort="1.0" velocity="10"/>
  </joint>
</robot>
"""


@pytest.fixture
def rover(tmp_path):
    path = tmp_path / "rover.urdf"
    path.write_text(WHEELED)
    return path


def test_a_model_with_only_wheels_is_a_mobile_robot(rover):
    """No arm to find and no gripper to attach: the wheels are the controlled joints."""
    model = build(rover).model
    assert model.family == "mobile"
    assert model.arm_joints == ("wheel_left_joint", "wheel_right_joint")
    assert model.base_body == "base_link"
    assert model.gripper_actuator is None and model.hand == ""


def test_detection_says_why_it_is_mobile(rover):
    built = build(rover)
    assert any(n.startswith("family: mobile") and "wheel_left_joint" in n for n in built.notes)


def test_a_mobile_family_can_be_given_by_hand(rover):
    assert robots.load(str(rover), name="rover_by_hand", family="mobile").family == "mobile"


def test_a_limited_joint_is_an_arm_not_a_wheel(tmp_path):
    """Only unlimited hinges are wheels: a joint with limits is what an arm's joints have."""
    path = tmp_path / "limited.urdf"
    limited = WHEELED.replace('type="continuous"', 'type="revolute"').replace(
        '<limit effort="1.0" velocity="10"/>', '<limit lower="-1" upper="1" effort="1.0" velocity="10"/>'
    )
    path.write_text(limited)
    m = mujoco.MjModel.from_xml_path(str(to_mjcf(path, cache_root().parent / "from_urdf")[0]))
    assert not _wheeled(m)
    assert _wheeled(mujoco.MjModel.from_xml_path(str(to_mjcf(_write(tmp_path / "w.urdf"), cache_root().parent / "from_urdf")[0])))


def _write(path):
    path.write_text(WHEELED)
    return path


def _model(rover):
    return build(rover).model


class _Backend:
    def __init__(self, model, q):
        self.robot_model = model
        self._q = np.asarray(q, float)

    def qpos(self):
        return self._q.copy()


def _robot(rover, q=(0.25, -0.5)):
    model = _model(rover)
    return MobileRobot(SimpleNamespace(backend=_Backend(model, q)))


def test_a_mobile_robot_has_a_base_its_estimate_and_its_wheels(rover):
    r = _robot(rover)
    assert isinstance(r.base, Base)
    assert r.base.name == "base"
    assert r.base.estimate.name == "base.estimate"
    assert r.joints == {"wheel_left_joint": 0.25, "wheel_right_joint": -0.5}


@pytest.mark.parametrize("name", ["arm", "gripper", "tcp", "pick", "place", "stand", "move_joints", "reset_to"])
def test_arm_only_things_say_what_a_mobile_robot_has(rover, name):
    r = _robot(rover)
    with pytest.raises(AttributeError, match="mobile robot"):
        getattr(r, name)


@pytest.mark.parametrize("name", ["position", "up_axis", "yaw", "velocity"])
def test_pose_queries_point_at_the_base(rover, name):
    r = _robot(rover)
    with pytest.raises(AttributeError, match="robot.base"):
        getattr(r, name)


def test_the_built_in_arms_are_still_arms():
    assert robots.get("so101").family == "arm"
    assert all(robots.get(n).family != "mobile" for n in robots.names())


def test_mujoco_reads_the_wheeled_model(rover):
    """The two wheels, and the free joint a wheeled base rolls on (added to the URDF's fixed base)."""
    mjcf = to_mjcf(rover, cache_root().parent / "from_urdf")[0]
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    assert m.njnt == 3 and m.nu == 2
    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE


def test_a_chain_of_unlimited_joints_is_an_arm_not_wheels(tmp_path):
    """Joints one after another are a chain, whatever their limits: a description that leaves an
    arm's joints unlimited is still an arm (the EDO's is)."""
    chain = WHEELED.replace(
        '<parent link="base_link"/><child link="right_wheel"/><origin xyz="0 -0.12 0"/>',
        '<parent link="left_wheel"/><child link="right_wheel"/><origin xyz="0 -0.12 0"/>',
    )
    path = tmp_path / "chain.urdf"
    path.write_text(chain)
    m = mujoco.MjModel.from_xml_path(str(to_mjcf(path, cache_root().parent / "from_urdf")[0]))
    assert not _wheeled(m)


class _NavWorld:
    """A world whose stack never reports the goal done, at the pose it is given (no simulator needed)."""

    def __init__(self, pos):
        self.settings = SimpleNamespace(action_timeout=10.0)
        self.dt = 0.02
        self.backend = self
        self.trace = None  # no trace: the action is not recorded
        self.commands = True
        self.status = "running"
        self.pos = np.asarray(pos, float)

    def start_navigation(self, x, y, yaw):
        pass

    def navigation_done(self):
        return False

    def cancel_navigation(self):
        pass

    def base_pose(self):
        return self.pos, np.array([1.0, 0.0, 0.0, 0.0])

    def step(self, n=1):
        pass


def test_a_goal_the_stack_gives_up_on_says_where_the_base_is():
    """The timeout names the base's pose and how far it is from the goal, so a stall and a slow approach read differently."""
    from robowright.errors import ActionTimeoutError
    from robowright.mobile import MobileBase

    base = MobileBase(SimpleNamespace(world=_NavWorld((-1.22, -0.5, 0.0))))
    with pytest.raises(ActionTimeoutError, match=r"it is at \(-1\.220, -0\.500\), 0\.220 m from the goal"):
        base.go_to((-1.0, -0.5))
