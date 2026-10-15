"""robowright as the harness around Nav2's own example programs, run as their launch files ship them.

Each ``nav2_simple_commander`` launch file (``*_example_launch.py``) starts the depot world with a
TurtleBot 4, Nav2's bringup on the depot map, and the example program as a node of that launch. The
test starts the launch headless and observes: ``command=False``, so the example and Nav2 drive the
robot, and robowright reads the robot's Gazebo ground truth (``robot.base``) and its localization
(``robot.base.estimate``). See tests/nav2_examples_rig.py for the one change to the launch (the
depot world's SceneBroadcaster, read at the top of :func:`observed_launch`).

Frames, measured (the examples' own numbers are map-frame numbers): the robot spawns at Gazebo's
(-8, 0), and Nav2's belief (``estimate``, in ``map``) is that point plus (8, 0) once AMCL has
converged. So the example's initial pose (0, 0) is the robot's spawn point, and the map frame is the
Gazebo world frame shifted by ``MAP_FROM_WORLD``. At rest the converged belief sits about 0.1 m
from that (the first probe), which is a bias in the frame offset, not a localization error the
bounds are measured against.

Each example runs once per module, and its stack is stopped before the next one starts. With the two
fixtures overlapping (both stacks up on one ROS domain), the second stack's bringup aborted in 3 of 3
runs (measured); that the first stack's nodes caused it is inferred, not separately measured.
"""

import importlib.util
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect

HERE = Path(__file__).parent


def _rig():
    spec = importlib.util.spec_from_file_location("nav2_examples_rig", HERE / "nav2_examples_rig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rig = _rig()
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)"),
    pytest.mark.skipif(not rig.installed(), reason="needs nav2_bringup, nav2_minimal_tb4_sim and nav2_simple_commander"),
]
# This module's own ROS graph and Gazebo, and nothing else on the machine. A domain picked per process
# collided with another agent's Nav2 example on domain 151 (measured: the stack never came up); 152 is
# the domain this worktree is given for its runs.
os.environ["ROS_DOMAIN_ID"] = "152"
os.environ["GZ_PARTITION"] = f"rw-nav2ex-{os.getpid()}"
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

MAP_FROM_WORLD = np.array([8.0, 0.0, 0.0])  # map = Gazebo world + this (see the docstring)
XY_GOAL_TOLERANCE = 0.25  # nav2_params.yaml, general_goal_checker: xy_goal_tolerance
BELIEF_TOLERANCE = 0.5  # the belief's error once AMCL has converged (the first probe: 0.24 m at most while moving, 0.1 m at rest)
ARRIVAL_TOL = XY_GOAL_TOLERANCE + BELIEF_TOLERANCE  # the goal checker judges the belief; the truth may differ by the belief's error
SETTLE = (
    5.0  # seconds after the start before the belief is judged: AMCL starts from the initial pose and converges (measured: 0.56 m at 1.3 s)
)
SCAN_HZ = 10.0  # the lidar's update_rate in nav2_minimal_tb4_description's rplidar.urdf.xacro
UPRIGHT_DEG = 5
DRIVE_MOVED = 0.05  # m from the spawn: the robot has begun to drive
# where the example sends the robot, in the map frame
NAV_TO_POSE_GOAL = (0.0, 0.0, 0.0)  # its last goal: see test_the_example_sends_the_robot_back_to_its_origin
FOLLOW_PATH_GOAL = (7.0, 0.5, 0.0)  # example_follow_path's goal


class Run:
    """What one example run measured: the example's log, the robot's truth and belief along the run,
    and what the robot is. ``samples``: (t, truth in map, belief, tilt in degrees)."""

    def __init__(self, ex, goal):
        self.ex = ex
        self.goal = goal
        self.samples: list[tuple[float, np.ndarray, np.ndarray, float]] = []
        self.family = ""
        self.n_joints = 0
        self.wall = 0.0
        self.scan_error = None  # why the lidar's rate was not as the description sets, if it was not
        # the drive's start, as the robot shows it: how far it was from its spawn at the first sample (nonzero:
        # the drive began before observation), and the observation time it first left the spawn
        self.moved_before_observation = 0.0
        self.drive_seen_at = None

    def record(self, world, t: float) -> None:
        base = world.robot.base
        truth = np.asarray(base.position, dtype=float) + MAP_FROM_WORLD
        belief = np.asarray(base.estimate.position, dtype=float)
        tilt = float(np.degrees(np.arccos(np.clip(base.up_axis[2], -1.0, 1.0))))
        self.samples.append((t, truth, belief, tilt))

    @property
    def final(self) -> np.ndarray:
        return self.samples[-1][1]

    @property
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.final[:2] - np.array(self.goal[:2])))

    @property
    def settled_errors(self) -> np.ndarray:
        """The belief's distance from the truth (map frame), after AMCL has had SETTLE seconds."""
        return np.array([float(np.linalg.norm(b[:2] - x[:2])) for t, x, b, _ in self.samples if t >= SETTLE])

    @property
    def max_tilt(self) -> float:
        return max(tilt for *_, tilt in self.samples)


def _example_run(name: str, goal) -> Run:
    """One example, as its launch file ships it, from start to its result, with the stack stopped
    again before this returns. The robot is observed throughout (upright, and the belief). The
    launch starts the stack and the example together, but the example drives only once Nav2 is
    active, which can come before the observation has begun: how much the robot had already moved
    is measured (see :class:`Run`), not assumed away."""
    with rig.example(name) as ex:
        world = rw.launch(
            backend="ros2",
            robot=None,  # the robot the stack publishes (its robot_description)
            command=False,  # observe only: the example and Nav2 drive the robot
            use_sim_time=True,
            gazebo=True,
            base={"frame": "base_footprint", "in": "map"},
            # The first joint states can come well after the lidar under load (seen in the full verify, where
            # the 10 s default expired while the rig had already heard them): the example's own stack is the
            # thing under test here, so this run allows the stack a longer start. Robowright's default is unchanged.
            timeout=60.0,
            settings=rw.Settings(trace="off"),
        )
        try:
            run = Run(ex, goal)
            run.family = world.robot.model.family
            run.n_joints = len(world.robot.joints)
            try:
                expect(world.ros.topic("/scan")).to_publish(hz=SCAN_HZ, tol=0.25, timeout=10)
            except rw.ExpectationError as e:
                run.scan_error = str(e)
            expect(world.robot.base).always.to_be_upright(tol_deg=UPRIGHT_DEG)
            started = time.monotonic()
            _run_to_result(world, ex, run, timeout=400.0)
            run.wall = time.monotonic() - started
        finally:
            world.close(failed=False)
    _report(run)
    return run


def _run_to_result(world, ex, run: Run, timeout: float) -> None:
    """Let the example run to its result, stepping the world meanwhile, and sample the robot. The
    result is the example's own line ("Goal succeeded!" and the rest): its process does not end on
    its own, since it stays in ``lifecycleShutdown`` after the result (measured: still there 100 s on)."""
    t = 0.0
    while not re.search(r"Goal (succeeded|failed|was canceled)!|invalid return status", ex.log):
        if t > timeout:
            raise TimeoutError(f"{ex.name} gave no result within {timeout} s:\n{ex.log[-2000:]}")
        world.wait(0.2)
        t += 0.2
        run.record(world, t)
        away = float(np.linalg.norm(run.samples[-1][1][:2]))  # the robot spawns at the map origin (see the module docstring)
        if len(run.samples) == 1:
            run.moved_before_observation = away
        if run.drive_seen_at is None and away > DRIVE_MOVED:
            run.drive_seen_at = round(t, 1)


def _report(run: Run) -> None:
    """What the run measured, printed (shown in the test report with ``-rA``)."""
    ok = "Goal succeeded!" in run.ex.log
    if not len(run.samples):
        raise AssertionError(f"{run.ex.name} had its result before the first observation sample; its log ends:\n{run.ex.log[-2500:]}")
    span = run.samples[-1][0]
    if not len(run.settled_errors):
        raise AssertionError(
            f"{run.ex.name} gave its result after {span:.1f} s of observation ({len(run.samples)} samples), before the "
            f"{SETTLE:.0f} s the belief needs to settle; its log ends:\n{run.ex.log[-2500:]}"
        )
    print(
        f"\n{run.ex.name}: result {'succeeded' if ok else 'not succeeded'}; {len(run.samples)} samples over {span:.1f} s "
        f"of observation before the result (wall {run.wall:.1f} s; {run.moved_before_observation:.2f} m from spawn at the first "
        f"sample; left the spawn at observation t={run.drive_seen_at} s); "
        f"final truth (map) {np.round(run.final[:2], 3).tolist()}, distance to goal {run.distance_to_goal:.3f} m; "
        f"belief error after {SETTLE:.0f} s: max {run.settled_errors.max():.3f} m, mean {run.settled_errors.mean():.3f} m; "
        f"max tilt {run.max_tilt:.2f} deg",
        flush=True,
    )


@pytest.fixture(scope="module")
def nav_to_pose():
    return _example_run("example_nav_to_pose", NAV_TO_POSE_GOAL)


@pytest.fixture(scope="module")
def follow_path():
    return _example_run("example_follow_path", FOLLOW_PATH_GOAL)


def test_the_robot_is_the_turtlebot4_the_stack_publishes(nav_to_pose):
    assert nav_to_pose.family == "mobile"
    assert nav_to_pose.n_joints == 2  # the two wheels


def test_the_lidar_publishes_at_its_rate(nav_to_pose):
    assert nav_to_pose.scan_error is None, nav_to_pose.scan_error


def test_nav_to_pose_example_reports_success(nav_to_pose):
    assert re.search(r"\[example_nav_to_pose-\d+\] Goal succeeded!", nav_to_pose.ex.log), nav_to_pose.ex.log[-2000:]


def test_the_example_sends_the_robot_back_to_its_origin(nav_to_pose):
    """The example's first goal is (17.86, -0.77). Its own preemption (once navigation has run for
    18 s, it re-sends goal (0, 0)) changes that: measured, the robot turned at map x of about 9 m and
    went back. Its final goal is (0, 0), which is where the robot is set to start."""
    assert "Navigating to goal: 17.86 -0.77" in nav_to_pose.ex.log
    assert "Navigating to goal: 0.0 0.0" in nav_to_pose.ex.log


def test_nav_to_pose_example_arrives_at_its_final_goal(nav_to_pose):
    assert nav_to_pose.distance_to_goal <= ARRIVAL_TOL, (
        f"{nav_to_pose.final[:2]} is {nav_to_pose.distance_to_goal:.3f} m from {NAV_TO_POSE_GOAL[:2]}"
    )


def test_nav2_believes_where_the_robot_is(nav_to_pose):
    assert nav_to_pose.settled_errors.max() <= BELIEF_TOLERANCE, f"belief error max {nav_to_pose.settled_errors.max():.3f} m"


def test_the_robot_stays_upright_through_nav_to_pose(nav_to_pose):
    assert nav_to_pose.max_tilt <= UPRIGHT_DEG


def test_follow_path_example_reports_success(follow_path):
    assert re.search(r"\[example_follow_path-\d+\] Goal succeeded!", follow_path.ex.log), follow_path.ex.log[-2000:]


def test_follow_path_example_arrives_at_its_goal(follow_path):
    assert follow_path.distance_to_goal <= ARRIVAL_TOL, (
        f"{follow_path.final[:2]} is {follow_path.distance_to_goal:.3f} m from {FOLLOW_PATH_GOAL[:2]}"
    )


def test_nav2_believes_where_the_robot_is_while_following_a_path(follow_path):
    assert follow_path.settled_errors.max() <= BELIEF_TOLERANCE, f"belief error max {follow_path.settled_errors.max():.3f} m"


def test_the_robot_stays_upright_while_following_a_path(follow_path):
    assert follow_path.max_tilt <= UPRIGHT_DEG
