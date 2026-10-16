"""Routes (``ros2.run(..., route={topic: rw.route...})``): what the code under test receives or sends
on a topic, changed on the way, as Playwright's ``page.route`` changes what a page receives.

The routes themselves and the command rewriting are tested everywhere; the relaying, against a
real ROS 2 node (tests/route_node.py), where rclpy is installed.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.routes import as_route, private_name, remap_command

HERE = Path(__file__).parent
NODE = HERE / "route_node.py"
HAS_ROS = importlib.util.find_spec("rclpy") is not None
if "PYTEST_XDIST_WORKER" in os.environ:  # a ROS 2 domain of this worker's own
    os.environ["ROS_DOMAIN_ID"] = str(150 + int(os.environ["PYTEST_XDIST_WORKER"].removeprefix("gw")) % 50)
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


def _through(route, n=1000, dt=0.01):
    rng = np.random.default_rng(0)
    return [route._count(route.handle(i, i * dt, rng), i) for i in range(n)]


# -- the routes -------------------------------------------------------------------------


def test_abort_lets_nothing_through():
    r = rw.route.abort()
    assert all(out == [] for out in _through(r))
    assert r.stats["dropped"] == 1000 and r.stats["passed"] == 0


def test_drop_loses_about_its_share_and_is_seeded():
    r = rw.route.drop(0.3)
    lost = sum(out == [] for out in _through(r))
    assert 250 < lost < 350
    again = rw.route.drop(0.3)
    assert sum(out == [] for out in _through(again)) == lost  # same seed, same losses


def test_delay_marks_every_message_late():
    r = rw.route.delay(0.25)
    assert all(out == [(0.25, i)] for i, out in enumerate(_through(r, 10)))
    assert r.stats["delayed"] == 10


def test_cut_silences_a_window_only():
    r = rw.route.cut(at=2.0, duration=1.0)
    out = _through(r, 500)  # 0 .. 4.99 s
    quiet = [i for i, o in enumerate(out) if o == []]
    assert quiet == list(range(200, 300))


def test_cut_without_a_duration_lasts_to_the_end():
    assert sum(o == [] for o in _through(rw.route.cut(at=1.0), 300)) == 200


def test_throttle_caps_the_rate():
    out = _through(rw.route.throttle(10), 1000, dt=0.01)  # 100 Hz in, 10 s
    assert sum(o != [] for o in out) == 100


def test_a_function_edits_or_drops():
    r = as_route(lambda m: None if m % 2 else m * 10)
    out = _through(r, 4)
    assert out == [[(0.0, 0)], [], [(0.0, 20)], []]
    assert r.stats["changed"] == 1  # 2 -> 20 (0 * 10 is 0, the same object)


def test_routes_chain_and_delays_add():
    r = rw.route.delay(0.1) | rw.route.delay(0.2) | (lambda m: m)
    assert _through(r, 1) == [[(pytest.approx(0.3), 0)]]
    assert r.describe() == "delay(0.1) | delay(0.2) | <lambda>"


@pytest.mark.parametrize("bad", [lambda: rw.route.drop(1.5), lambda: rw.route.delay(-1), lambda: rw.route.throttle(0)])
def test_nonsense_is_refused(bad):
    with pytest.raises(ValueError):
        bad()


def test_a_non_route_is_refused():
    with pytest.raises(TypeError, match="a route is"):
        as_route(3)


# -- remapping the command --------------------------------------------------------------


def test_a_node_command_gets_remapping_arguments():
    args, wrapper = remap_command(["ros2", "run", "pkg", "node", "--ros-args", "-p", "x:=1"], ["/scan", "cmd_vel"])
    assert wrapper is None
    assert args[-5:] == ["--ros-args", "-r", f"/scan:={private_name('/scan')}", "-r", f"/cmd_vel:={private_name('cmd_vel')}"]
    assert args[:7] == ["ros2", "run", "pkg", "node", "--ros-args", "-p", "x:=1"]


def test_a_script_gets_them_too():
    args, _ = remap_command([sys.executable, "my_node.py", "--fast"], ["/scan"])
    assert args == [sys.executable, "my_node.py", "--fast", "--ros-args", "-r", f"/scan:={private_name('/scan')}"]


@pytest.mark.parametrize(
    "cmd",
    [["ros2", "launch", "my_pkg", "bringup.launch.py", "use_sim_time:=true"], ["ros2", "launch", "/abs/bringup.launch.py", "a:=b"]],
)
def test_a_launch_file_is_wrapped_with_the_remaps(cmd):
    args, wrapper = remap_command(cmd, ["/scan"])
    assert args == ["ros2", "launch", "{WRAPPER}"]
    compile(wrapper, "wrapper.launch.py", "exec")  # valid Python
    assert f"SetRemap(src='/scan', dst='{private_name('/scan')}')" in wrapper
    assert cmd[-1].split(":=")[0] in wrapper  # the launch arguments are passed through


def test_launch_options_are_refused_rather_than_dropped():
    with pytest.raises(ValueError, match="options"):
        remap_command(["ros2", "launch", "pkg", "f.launch.py", "--debug"], ["/scan"])


def test_bridge_stamps_never_read_a_second_early():
    """Found by the delay test: a time a hair under a whole second was stamped as the second before it."""
    from robowright.ros2_bridge import _stamp

    assert _stamp(2.9999999996) == {"sec": 3, "nanosec": 0}
    assert _stamp(1.25) == {"sec": 1, "nanosec": 250_000_000}
    for t in np.arange(0, 20, 0.02):  # every control step of a 50 Hz world for 20 s
        s = _stamp(float(t))
        assert abs(s["sec"] + s["nanosec"] * 1e-9 - t) < 1e-6


# -- relaying, against a real node ------------------------------------------------------

ros = pytest.mark.skipif(not HAS_ROS, reason="needs a ROS 2 environment (rclpy)")


def _listen(world, ros2, tmp_path, route=None, seconds=3.0, mode="listen"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "seen.json"
    with ros2.run([sys.executable, str(NODE), mode, str(out)], route=route):
        world.wait(seconds)
        expect(world).to_satisfy(lambda _w: out.exists(), "the node wrote what it saw", timeout=5.0)
        seen = json.loads(out.read_text())
    return seen


@ros
def test_without_a_route_the_node_hears_the_robot(world, ros2, tmp_path):
    seen = _listen(world, ros2, tmp_path)
    assert seen["count"] > 50 and seen["lag"] < 0.1


@ros
def test_abort_leaves_the_node_hearing_nothing(world, ros2, tmp_path):
    seen = _listen(world, ros2, tmp_path, route={"/joint_states": rw.route.abort()})
    assert seen["count"] == 0
    (r,) = ros2.routes
    assert r["direction"] == "to the code under test" and r["type"] == "sensor_msgs/msg/JointState"
    assert r["dropped"] > 50 and r["passed"] == 0


@ros
def test_delay_makes_the_readings_late_in_sim_time(world, ros2, tmp_path):
    seen = _listen(world, ros2, tmp_path, route={"/joint_states": rw.route.delay(0.5)})
    assert seen["count"] > 20
    # Each reading half a second behind the clock. The typical reading, not the earliest or latest: the node reads
    # its clock from /clock too, and one message reaching it late makes a reading look less or more delayed than it
    # was (seen: 0.4 s once, on a busy host). A route that delayed by a different amount moves the median.
    assert float(np.median(seen["lags"])) == pytest.approx(0.5, abs=0.02)


@ros
def test_a_function_changes_what_the_node_reads(world, ros2, tmp_path):
    def wrong_first_joint(msg):
        msg.position = [9.0, *msg.position[1:]]
        return msg

    seen = _listen(world, ros2, tmp_path, route={"/joint_states": wrong_first_joint})
    assert seen["first"][0] == 9.0
    assert ros2.routes[0]["passed"] > 0


@ros
def test_dropping_the_stacks_commands_leaves_the_arm_where_it_was(world, robot, ros2):
    start = float(robot.qpos()[0])
    with ros2.run([sys.executable, str(NODE), "command"], route={"/forward_position_controller/commands": rw.route.abort()}):
        world.wait(2.0)
        expect(robot).always.to_satisfy(lambda r: abs(float(r.qpos()[0]) - start) < 0.02, "arm holds", timeout=1.0)
    assert ros2.routes[0]["direction"] == "from the code under test"
    assert ros2.routes[0]["dropped"] > 10


@ros
def test_the_same_commands_without_a_route_move_the_arm(world, robot, ros2):
    start = float(robot.qpos()[0])
    with ros2.run([sys.executable, str(NODE), "command"]):
        expect(robot).to_satisfy(lambda r: float(r.qpos()[0]) - start > 0.2, "arm moved", timeout=5.0)


@ros
def test_a_node_that_ignores_the_remapping_is_reported(world, ros2, tmp_path):
    with pytest.raises(RuntimeError, match="nothing in the command subscribed to the routed topic"):
        _listen(world, ros2, tmp_path, route={"/joint_states": rw.route.abort()}, mode="deaf")


@ros
def test_a_route_on_a_topic_nobody_publishes_is_reported(world, ros2, tmp_path):
    with pytest.raises(RuntimeError, match="never engaged"):
        _listen(world, ros2, tmp_path, route={"/joint_statez": rw.route.abort()}, seconds=1.0)


@ros
def test_a_launch_file_is_routed_through_its_wrapper(world, ros2, tmp_path):
    """``ros2 launch``: every node the launch file starts is remapped by the wrapping launch file."""
    out = tmp_path / "seen.json"
    launch = tmp_path / "under_test.launch.py"
    launch.write_text(
        "from launch import LaunchDescription\n"
        "from launch.actions import DeclareLaunchArgument\n"
        "from launch.substitutions import LaunchConfiguration\n"
        "from launch_ros.actions import Node\n\n\n"
        "def generate_launch_description():\n"
        "    return LaunchDescription([\n"
        "        DeclareLaunchArgument('out'),\n"
        f"        Node(executable={sys.executable!r}, arguments=[{str(NODE)!r}, 'listen', LaunchConfiguration('out')]),\n"
        "    ])\n"
    )
    with ros2.run(["ros2", "launch", str(launch), f"out:={out}"], route={"/joint_states": rw.route.abort()}):
        world.wait(4.0)
        expect(world).to_satisfy(lambda _w: out.exists(), "the node wrote what it saw", timeout=10.0)
        seen = json.loads(out.read_text())
    assert seen["count"] == 0
    assert ros2.routes[0]["dropped"] > 50


# -- recording a topic, and playing it back (Playwright's recordHar / routeFromHAR) --------


def test_record_refuses_to_write_over_an_existing_directory(tmp_path):
    with pytest.raises(FileExistsError, match="exists"):
        rw.route.record(tmp_path)


def _bag_messages(path):
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path)), rosbag2_py.ConverterOptions("cdr", "cdr"))
    out = []
    while reader.has_next():
        name, _, t = reader.read_next()
        out.append((name, t))
    return out


@ros
def test_a_recorded_topic_plays_back_instead_of_the_live_one(world, ros2, tmp_path):
    def marked(msg):  # a first joint the live robot never has: what the second run hears shows where it came from
        msg.position = [9.0, *msg.position[1:]]
        return msg

    bag = tmp_path / "joint_states_bag"
    first = _listen(world, ros2, tmp_path / "rec", route={"/joint_states": rw.route.fulfill(marked) | rw.route.record(bag)})
    assert first["first"][0] == 9.0
    recorded = _bag_messages(bag)
    assert len(recorded) > 50 and {name for name, _ in recorded} == {"/joint_states"}
    stamps = [t for _, t in recorded]
    assert stamps == sorted(stamps) and (stamps[-1] - stamps[0]) * 1e-9 > 2.0  # simulated time, about the run's length

    second = _listen(world, ros2, tmp_path / "play", route={"/joint_states": rw.route.from_bag(bag)})
    assert second["first"][0] == 9.0, "the node heard the live robot, not the bag"
    (r,) = ros2.routes
    assert r["played"] > 50 and r["dropped"] > 50 and r["passed"] == 0  # the live messages went no further
    assert float(np.median(second["lags"])) == pytest.approx(0.0, abs=0.05)  # restamped as they are delivered


@ros
def test_a_bag_plays_at_its_recorded_pace_in_simulated_time(world, ros2, tmp_path):
    bag = tmp_path / "paced"
    _listen(world, ros2, tmp_path / "rec", route={"/joint_states": rw.route.record(bag)}, seconds=2.0)
    n = len(_bag_messages(bag))
    seen = _listen(world, ros2, tmp_path / "play", route={"/joint_states": rw.route.from_bag(bag, at=1.0)}, seconds=2.0)
    # Played from 1 s after the command started, for the 2 s waited (and the moments the node took to write): about
    # half the recording, not all of it at once.
    assert 0.25 * n < ros2.routes[0]["played"] < 0.75 * n, (ros2.routes[0], n)
    assert seen["count"] > 0


@ros
def test_a_bag_from_ros2_bag_record_plays_too(world, ros2, tmp_path):
    bag = tmp_path / "cli_bag"
    with ros2.run(["ros2", "bag", "record", "-o", str(bag), "/joint_states"]):
        world.wait(3.0)  # ros2 bag record stamps by the wall clock it received them on: offsets are what play uses
    assert len(_bag_messages(bag)) > 20
    seen = _listen(world, ros2, tmp_path / "play", route={"/joint_states": rw.route.from_bag(bag)}, seconds=2.0)
    assert seen["count"] > 10 and ros2.routes[0]["played"] > 10
