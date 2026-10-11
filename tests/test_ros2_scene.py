"""Which robot a ROS 2 world runs, from its scene (no ROS 2 needed: the stack's description is stubbed):
a scene with no robot takes the one the stack publishes, with where it stands; a named robot is kept."""

import pytest

from robowright import world
from robowright.scene import SceneSpec


@pytest.fixture
def described(monkeypatch):
    calls = []

    def fake(robot, options, load=None, as_name=None):
        calls.append((robot, dict(options), load))
        return "described_arm"

    monkeypatch.setattr(world, "_ros_robot", fake)
    return calls


def test_a_scene_with_no_robot_takes_the_robot_the_stack_publishes(described):
    scene = world.ros2_scene(
        SceneSpec(robot=None, objects=[]), "ros2", None, {"use_sim_time": True}, {"base_pos": [1, 2, 3], "base_yaw": 1.5}
    )
    assert scene.robot == "described_arm"
    assert described == [(None, {"use_sim_time": True}, {"base_pos": (1, 2, 3), "base_yaw": 1.5})]


def test_a_named_robot_is_kept_and_not_read_from_the_stack(described):
    scene = world.ros2_scene(SceneSpec(robot="panda", objects=[]), "ros2", None, {}, None)
    assert scene.robot == "panda"
    assert described == []


def test_no_scene_on_ros2_reads_the_stacks_robot(described):
    scene = world.ros2_scene(None, "ros2", None, {}, None)
    assert scene.robot == "described_arm" and scene.objects == []
