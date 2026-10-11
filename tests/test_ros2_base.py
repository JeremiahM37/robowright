"""A mobile base's command topic and navigation action, as the ROS graph shows them: discovery,
settings, the report, and codegen of the base's actions. No ROS 2 needed (the graph is a fake one);
the base's own behaviour is tested on Nav2's simulation in tests/test_nav2_turtlebot.py.
"""

from pathlib import Path

import pytest

from robowright import Trace, ros2_discover
from robowright.backends.ros2_backend import settings
from robowright.codegen import generate

TWIST = ["geometry_msgs/msg/Twist"]
STAMPED = ["geometry_msgs/msg/TwistStamped"]
NAV = ["nav2_msgs/action/NavigateToPose"]


def test_a_command_topic_is_found_by_its_subscriber_and_type():
    topics = {"/cmd_vel": TWIST, "/scan": ["sensor_msgs/msg/LaserScan"]}
    assert ros2_discover.find_cmd_vel(topics, {"/cmd_vel": 1}, "/cmd_vel") == ("/cmd_vel", "geometry_msgs/msg/Twist")
    # Jazzy's stacks differ on the type: the topic's own type is what robowright publishes
    assert ros2_discover.find_cmd_vel({"/cmd_vel": STAMPED}, {"/cmd_vel": 2}, "/cmd_vel") == ("/cmd_vel", "geometry_msgs/msg/TwistStamped")


def test_a_command_topic_with_no_subscriber_or_another_type_is_not_one():
    assert ros2_discover.find_cmd_vel({"/cmd_vel": TWIST}, {"/cmd_vel": 0}, "/cmd_vel") is None  # nothing would hear it
    assert ros2_discover.find_cmd_vel({"/cmd_vel": TWIST}, {}, "/cmd_vel") is None
    assert ros2_discover.find_cmd_vel({"/cmd_vel": ["std_msgs/msg/String"]}, {"/cmd_vel": 1}, "/cmd_vel") is None
    assert ros2_discover.find_cmd_vel({"/other": TWIST}, {"/other": 1}, "/cmd_vel") is None


def test_a_navigation_action_is_found_by_its_type():
    assert ros2_discover.find_navigate({"/navigate_to_pose": NAV}, "/navigate_to_pose") == "/navigate_to_pose"
    assert ros2_discover.find_navigate({"/navigate_to_pose": ["nav2_msgs/action/Spin"]}, "/navigate_to_pose") is None
    assert ros2_discover.find_navigate({}, "/navigate_to_pose") is None


def test_the_base_settings_take_its_command_topic_and_navigation_action(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no project file: only the keywords count
    s = settings({"base": {"cmd_vel": "/robot/cmd", "navigate": "/robot/navigate", "frame": "base_footprint", "in": "map"}})
    assert s["base"]["cmd_vel"] == "/robot/cmd"
    with pytest.raises(ValueError, match="ros2 base takes frame and in"):
        settings({"base": {"speed": 1.0}})


def test_suggest_and_the_report_name_what_was_found():
    g = ros2_discover.Graph(namespace="", sim_time=True)
    g.cmd_vel = ("/cmd_vel", "geometry_msgs/msg/TwistStamped", 1)
    g.navigate = "/navigate_to_pose"
    out = ros2_discover.suggest(g)
    assert out["base"] == {"cmd_vel": "/cmd_vel", "navigate": "/navigate_to_pose"}
    assert "command" not in out  # a base that can be driven is not observe-only, though it has no arm controller
    text = ros2_discover.report(g, out)
    assert "base command: /cmd_vel (geometry_msgs/msg/TwistStamped, 1 subscriber)" in text
    assert "navigation: /navigate_to_pose (nav2_msgs/action/NavigateToPose)" in text
    assert 'base = { cmd_vel = "/cmd_vel", navigate = "/navigate_to_pose" }' in text


def test_a_base_suggests_the_frames_it_localizes_in_when_tf_has_them():
    """Without its frame the base's belief is not read at all, and the agent has to guess it from TF."""
    g = ros2_discover.Graph(namespace="", sim_time=True)
    g.cmd_vel = ("/cmd_vel", "geometry_msgs/msg/TwistStamped", 1)
    g.navigate = "/navigate_to_pose"
    g.frames = ["base_footprint", "base_link", "map", "odom", "wheel_left_link"]
    assert ros2_discover.suggest(g)["base"] == {
        "cmd_vel": "/cmd_vel",
        "navigate": "/navigate_to_pose",
        "frame": "base_footprint",
        "in": "map",
    }
    g.frames = ["base_link", "odom"]  # no map: localization is odometry only
    assert ros2_discover.suggest(g)["base"] == {"cmd_vel": "/cmd_vel", "navigate": "/navigate_to_pose", "frame": "base_link", "in": "odom"}


def test_a_robot_without_them_suggests_no_base():
    assert "base" not in ros2_discover.suggest(ros2_discover.Graph(namespace=""))


class _Recorded(Trace):
    """A trace's events as a recording makes them, without the archive it would be saved in."""

    def __init__(self, events):
        self.path = Path("recorded.zip")
        self.meta = {
            "name": "test_drive_then_go",
            "backend": "ros2",
            "seed": 0,
            "status": "ok",
            "scene": {},
            "events": events,
        }


def test_codegen_writes_the_base_actions_as_calls_on_the_base():
    def action(name, args):
        return {"type": "action", "name": name, "args": args, "status": "ok", "t": 0.0, "detail": ""}

    events = [
        action("base.drive", {"linear": 0.2, "angular": 0.0, "duration": 1.0}),
        action("base.go_to", {"xy": [1.0, 0.5], "yaw": None, "timeout": 60.0}),
    ]
    code = generate(_Recorded(events))
    assert "robot.base.drive(linear=0.2, angular=0.0, duration=1.0)" in code
    assert "robot.base.go_to(xy=[1.0, 0.5], timeout=60.0)\n" in code  # yaw=None is the default: codegen leaves it out


def test_a_step_after_idle_time_waits_for_the_clock_instead_of_bursting(monkeypatch):
    """An agent thinks between calls while the simulator runs on: the schedule must not stay behind the clock,
    or the next steps return without the simulator running at all."""
    from robowright.backends import ros2_backend

    class Clocked(ros2_backend.Ros2Backend):  # no node: a step on sim time only reads the clock
        commands = property(lambda self: False)
        control_dt = property(lambda self: 0.02)

    b = object.__new__(Clocked)
    clock = [10.0]
    b._now = lambda: clock[0]
    monkeypatch.setattr(ros2_backend._time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    b._base_cmd, b.cfg = None, {"use_sim_time": True, "clock_timeout": 30.0}
    b._tick = 2.0  # the schedule was last advanced 8 s of simulated time ago
    for _ in range(100):
        b.step()
    assert clock[0] == pytest.approx(12.0, abs=0.05)  # 100 control steps are 2 s of simulated time


def test_a_simulation_clock_that_stops_is_an_error_not_a_hang(monkeypatch):
    """A simulator that crashes or hangs publishes no more /clock: a step waiting for the next tick would wait
    for ever. After ``clock_timeout`` seconds of wall time with the clock standing still, it says so."""
    from robowright.backends import ros2_backend
    from robowright.errors import RobowrightError

    class Clocked(ros2_backend.Ros2Backend):
        commands = property(lambda self: False)
        control_dt = property(lambda self: 0.02)

    b = object.__new__(Clocked)
    wall = [0.0]
    b._now = lambda: 5.0  # the clock never moves
    monkeypatch.setattr(ros2_backend._time, "sleep", lambda s: wall.__setitem__(0, wall[0] + s))
    monkeypatch.setattr(ros2_backend._time, "monotonic", lambda: wall[0])
    b._base_cmd, b.cfg, b._tick, b._t0 = None, {"use_sim_time": True, "clock_timeout": 2.0}, 5.0, 1.0
    with pytest.raises(RobowrightError, match=r"clock \(/clock\) has not moved for 2 s.*t=4.000 s"):
        b.step()
    assert wall[0] == pytest.approx(2.0, abs=0.01)
