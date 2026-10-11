"""robowright as the harness around Nav2's own TurtleBot3 simulation.

Nav2's ``nav2_bringup`` runs a TurtleBot3 Waffle in Gazebo (headless) with its localization (AMCL)
and navigation stacks, and ``nav2_simple_commander``'s example programs drive it. Those programs
are the code under test, run as published (``ros2 run nav2_simple_commander example_...``); robowright
does not command the robot (``command=False``): it reads the robot's stack and Gazebo's ground truth,
and asserts. See tests/nav2_rig.py for the stack, and for the one change to Gazebo's world (its
SceneBroadcaster, which the shipped headless world leaves out, so that the robot's pose can be read).

Measured with this stack (see the report with the module): the examples' goals and start pose are
written for another map. ``example_nav_to_pose`` sends the robot to (17.86, -0.77), outside the
sandbox's 20 m map, and every example sets Nav2's initial pose to (0, 0), where the simulation does
not put the robot (it spawns it at (-2.0, -0.5)), and where the costmap leaves no plan out. The tests
that run the examples assert what they should do, and fail where they do not.

Needs a ROS 2 environment with nav2_bringup, nav2_minimal_tb3_sim and nav2_simple_commander
(``ROBOWRIGHT_ROS2_ENV=/mnt/bulk/ros2-env/jazzy-nav2``); skipped elsewhere.
"""

import importlib.util
import os
import time
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
EXAMPLE_START = (0.0, 0.0, 0.0)  # the initial pose every nav2_simple_commander example sets
EXAMPLE_GOAL = (17.86, -0.77, 0.0)  # example_nav_to_pose's goal
SCAN_HZ = 5.0  # the lidar's update_rate in nav2_minimal_tb3_sim's waffle description


@pytest.fixture
def stack(tmp_path):
    # One stack per test: an example ends by shutting Nav2 down (BasicNavigator.lifecycleShutdown),
    # so a second example on the same stack finds no Nav2 and waits for it forever (measured).
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


def _run_example(stack, world, name: str, timeout: float = 240.0) -> str:
    """Run one of nav2_simple_commander's examples as shipped, with the world stepping meanwhile
    (so invariants hold through it); its output."""
    p = stack.example(name)
    deadline = time.monotonic() + timeout
    while p.poll() is None:
        if time.monotonic() > deadline:
            raise TimeoutError(f"{name} did not finish within {timeout} s:\n{stack.example_log(name)[-2000:]}")
        world.wait(0.2)
    return stack.example_log(name)


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


def test_nav_to_pose_example_reaches_its_goal(stack, world):
    """example_nav_to_pose, as shipped, must take the robot to its goal. It cannot: the goal lies
    outside the sandbox map. The example's own output is part of the failure."""
    r = world.robot
    log = _run_example(stack, world, "example_nav_to_pose")
    assert "Goal succeeded!" in log, log[-1500:]
    expect(r.base).to_be_near(EXAMPLE_GOAL, tol=0.25, timeout=5)


def test_nav2_believes_where_the_robot_is(stack, world):
    """Nav2's localization (the example's initial pose, then AMCL) against Gazebo's truth, after the
    example has run. The example's start is (0, 0); the robot is at (-2.0, -0.5)."""
    _run_example(stack, world, "example_nav_to_pose")
    expect(world.robot.base.estimate).to_be_near(world.robot.base, tol=0.25, timeout=5)


def test_the_robot_stays_upright_through_an_example(stack, world):
    expect(world.robot.base).always.to_be_upright(tol_deg=5)
    _run_example(stack, world, "example_nav_to_pose")


def test_follow_path_example_follows_its_path(stack, world):
    """example_follow_path plans from the example's start to (7.0, 0.5) and follows it. Its planner
    finds no path from (0, 0) in this map, and the example stops on the empty path."""
    log = _run_example(stack, world, "example_follow_path")
    assert "Estimated distance remaining" in log, log[-1500:]
    expect(world.robot.base).to_be_near((7.0, 0.5, 0.0), tol=0.5, timeout=5)
