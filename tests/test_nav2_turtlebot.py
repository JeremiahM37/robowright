"""robowright as the harness around Nav2's own TurtleBot3 simulation.

Nav2's ``nav2_bringup`` runs a TurtleBot3 Waffle in Gazebo (headless) with its localization (AMCL)
and navigation stacks. robowright does not command the robot (``command=False``): it reads the
robot's stack and Gazebo's ground truth, and asserts. See tests/nav2_rig.py for the stack, and for
the one change to Gazebo's world (its SceneBroadcaster, which the shipped headless world leaves out,
so that the robot's pose can be read).

Nav2's example programs (``nav2_simple_commander``) are written for the depot world, not this sandbox,
so they are not run here: they are tested as shipped, with their own launch files, in
tests/test_nav2_examples.py.
"""

import importlib.util
import os
from pathlib import Path

import pytest

import robowright as rw
from robowright import expect
from robowright.errors import CapabilityError
from robowright.session import Session

HERE = Path(__file__).parent


def _rig():
    spec = importlib.util.spec_from_file_location("nav2_rig", HERE / "nav2_rig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rig = _rig()
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)"),
    pytest.mark.skipif(not rig.installed(), reason="needs nav2_bringup, nav2_minimal_tb3_sim and nav2_simple_commander"),
]
_worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
# This module's own ROS graph and Gazebo, and nothing else on the machine
os.environ["ROS_DOMAIN_ID"] = str(120 + (int(_worker.removeprefix("gw")) * 7 + os.getpid()) % 30)
os.environ["GZ_PARTITION"] = f"rw-nav2-{_worker}-{os.getpid()}"
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

SPAWN = (-2.0, -0.5, 0.0)  # where nav2_bringup's launch file puts the robot in the sandbox world (z: on the floor)
SCAN_HZ = 5.0  # the lidar's update_rate in nav2_minimal_tb3_sim's waffle description


@pytest.fixture
def stack(tmp_path):
    # One simulation per test, so that each starts from the sandbox as the launch file puts it.
    with rig.nav2_stack(tmp_path) as st:
        yield st


@pytest.fixture
def world(stack):
    w = rw.launch(
        backend="ros2",
        robot=None,  # the robot the stack publishes (its robot_description)
        command=False,  # observe only: Nav2 drives the robot
        use_sim_time=True,
        gazebo=True,
        base={"frame": "base_footprint", "in": "map"},
        settings=rw.Settings(trace="off"),
    )
    yield w
    w.close(failed=False)


def test_the_robot_is_the_one_the_stack_publishes(world):
    r = world.robot
    assert type(r).__name__ == "MobileRobot" and r.model.family == "mobile"
    assert set(r.joints) == {"wheel_left_joint", "wheel_right_joint"}
    expect(r.base).to_be_near(SPAWN, tol=0.05, timeout=30)  # Gazebo's ground truth, where the launch put it
    expect(r.base).to_be_upright(tol_deg=5, timeout=10)


def test_the_lidar_publishes_at_its_rate(world):
    expect(world.ros.topic("/scan")).to_publish(hz=SCAN_HZ, tol=0.25, timeout=10)


def test_a_place_the_examples_did_not_go_fails_with_the_robot_where_it_is(world):
    with pytest.raises(rw.ExpectationError) as e:
        expect(world.robot.base).to_be_near((5.0, 5.0, 0.0), tol=0.25, timeout=1.0)
    assert "base at (-2." in str(e.value)  # where the robot is, not only where it was asked to go


def test_contacts_need_a_simulator_that_reports_them(world):
    """Gazebo's contacts are not read here: the robot's contacts say so, rather than report none."""
    with pytest.raises(CapabilityError, match="contacts"):
        world.robot.base.contacts()


def test_a_session_observes_the_mobile_robot(stack):
    s = Session()
    try:
        text = s.launch(
            robot=None,
            backend="ros2",
            ros2={"command": False, "use_sim_time": True, "gazebo": True, "base": {"frame": "base_footprint", "in": "map"}},
        )
        assert "robot (mobile):" in text and "base: position" in text and "wheels:" in text
    finally:
        s.close()
