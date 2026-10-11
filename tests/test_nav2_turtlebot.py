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

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.errors import ActionError, CapabilityError
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
# The sandbox map (nav2_bringup's tb3_sandbox.pgm, 0.05 m cells): (0.5, -0.5) is open floor in the room
# the robot starts in; (0.0, -2.6) is an occupied cell of the room's south wall; (30.0, 0.0) is off the map.
OPEN_GOAL = (0.5, -0.5)
WALL = (0.0, -2.6)
OFF_MAP = (30.0, 0.0)
GOAL_TOL = 0.25  # xy_goal_tolerance in nav2_bringup's nav2_params.yaml: where Nav2 itself counts a goal reached
BELIEF_ERROR = 0.1  # how far AMCL's belief is from Gazebo's truth at rest, with margin (measured: at most 0.035 m)


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


@pytest.fixture
def navigating(tmp_path):
    # Navigation needs AMCL's initial pose and the navigation lifecycle active: a stack of its own.
    with rig.nav2_stack(tmp_path, navigation=True) as st:
        yield st


@pytest.fixture
def driven(navigating):
    """The robot with robowright allowed to command its base (and Nav2 running it too, as shipped)."""
    w = rw.launch(
        backend="ros2",
        robot=None,
        command=True,
        use_sim_time=True,
        gazebo=True,
        base={"frame": "base_footprint", "in": "map"},
        settings=rw.Settings(trace="off"),
    )
    yield w
    w.close(failed=False)


def _heading(base) -> float:
    return float(base.yaw)


def test_the_base_drives_forward_at_the_commanded_speed(driven):
    """1 s at 0.2 m/s: the distance is v * t (0.2 m) less the time the base takes to get up to speed."""
    b = driven.robot.base
    start = np.asarray(b.position, float)
    b.drive(0.2, 0.0, 1.0)
    moved = float(np.linalg.norm(np.asarray(b.position, float)[:2] - start[:2]))
    print(f"\nmeasured: drove 1.0 s at 0.2 m/s, Gazebo's truth moved {moved:.3f} m", flush=True)
    assert abs(moved - 0.2) <= 0.05, f"moved {moved:.3f} m, expected about 0.2 m"


def test_the_base_turns_in_place(driven):
    """0.5 rad/s for 1 s turns the base about half a radian, and it stays where it is."""
    b = driven.robot.base
    start, yaw0 = np.asarray(b.position, float), _heading(b)
    b.drive(0.0, 0.5, 1.0)
    turned = (_heading(b) - yaw0 + np.pi) % (2 * np.pi) - np.pi
    shift = float(np.linalg.norm(np.asarray(b.position, float)[:2] - start[:2]))
    print(f"\nmeasured: turned {turned:.3f} rad, moved {shift:.3f} m", flush=True)
    assert abs(turned - 0.5) <= 0.1 and shift <= 0.05


def test_the_base_goes_to_a_point_on_the_map(driven):
    """Nav2 plans to (0.5, -0.5) and drives there: it arrives by Gazebo's truth, and by the belief
    AMCL gives the robot."""
    b = driven.robot.base
    t0 = driven.time
    b.go_to(OPEN_GOAL, yaw=0.0, timeout=90)
    elapsed = driven.time - t0
    truth = np.asarray(b.position, float)[:2]
    belief = np.asarray(b.estimate.position, float)[:2]
    e_truth = float(np.linalg.norm(truth - OPEN_GOAL))
    e_belief = float(np.linalg.norm(belief - OPEN_GOAL))
    print(f"\nmeasured: arrived in {elapsed:.1f} s (world clock); truth error {e_truth:.3f} m, belief error {e_belief:.3f} m", flush=True)
    # Nav2 stops as soon as its BELIEF is inside its goal tolerance, so that is what the tolerance
    # bounds (0.195 to 0.220 m in three runs); the truth may be off by the belief's own error
    # besides (0.220 to 0.254 m: at most 0.035 m more).
    expect(b.estimate).to_be_near((*OPEN_GOAL, 0.0), tol=GOAL_TOL, timeout=5)  # the base frame is 1 cm above the floor
    assert e_truth <= GOAL_TOL + BELIEF_ERROR


def test_nav2_says_it_arrived_at_a_goal_inside_a_wall_and_the_robot_is_not_there(driven):
    """A goal inside a wall of the sandbox. Nav2 does not refuse it: it plans to the nearest free
    space, stops about 0.4 m short (three runs, truth and belief agreeing) and answers SUCCEEDED.
    Nav2's answer is not the outcome: the test of where the robot is fails."""
    b = driven.robot.base
    b.go_to(WALL, yaw=0.0, timeout=90)  # no ActionError: Nav2 reports success
    short = float(np.linalg.norm(np.asarray(b.position, float)[:2] - WALL))
    print(f"\nmeasured: Nav2 succeeded {short:.3f} m from the goal", flush=True)
    with pytest.raises(rw.errors.ExpectationError, match="to_be_near"):
        expect(b).to_be_near((*WALL, 0.0), tol=GOAL_TOL, timeout=2)
    assert short > GOAL_TOL


def test_a_goal_off_the_map_is_refused_with_nav2s_reason(driven):
    with pytest.raises(ActionError) as e:
        driven.robot.base.go_to(OFF_MAP, yaw=0.0, timeout=90)
    print(f"\nmeasured: ActionError: {e.value}", flush=True)
    assert "navigate_to_pose" in str(e.value)
