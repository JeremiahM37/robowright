"""Published wheeled robots, loaded as their makers ship them: TurtleBot 3 burger and waffle (ROBOTIS's
URDFs in the ROS 2 Jazzy environment) and TurtleBot 4 (Clearpath's Create 3 base, its xacro expanded to a
URDF with that environment's xacro). Nothing in their files is edited.

Measured when this file was written, from settled rest, on the MuJoCo backend (deterministic: the same
numbers each run). Drive: straight 0.1 and 0.2 m/s for 2 s; pivot 0.5 and 1.0 rad/s for 2 s; go_to three
points. The tests assert wide bands around those measurements, not the accuracy a robot would need.

Skipped when a file is absent. The TurtleBot 3 and 4 descriptions are expected in the ROS 2 Jazzy
environment at ROBOWRIGHT_ROS2_SHARE (default /mnt/bulk/ros2-env/jazzy-nav2/share).
"""

import os
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright.errors import ActionTimeoutError
from robowright.robots.detect import build

SHARE = Path(os.environ.get("ROBOWRIGHT_ROS2_SHARE", "/mnt/bulk/ros2-env/jazzy-nav2/share"))
TB4_EXPANDED = Path(os.environ.get("ROBOWRIGHT_TB4_URDF", "/mnt/bulk/rw-ext/wheeled/turtlebot4.urdf"))

# name: (file, published wheel radius m, published wheel separation m)
MODELS = {
    "burger": (SHARE / "turtlebot3_gazebo/urdf/turtlebot3_burger.urdf", 0.033, 0.160),
    "waffle": (SHARE / "turtlebot3_gazebo/urdf/turtlebot3_waffle.urdf", 0.033, 0.287),
    "turtlebot4": (TB4_EXPANDED, 0.03575, 0.233),  # create3.urdf.xacro's wheel_radius and wheel_separation
}


@pytest.fixture(autouse=True)
def ament(monkeypatch):
    """The packages' meshes resolve through AMENT_PREFIX_PATH, as a ROS install's do."""
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(SHARE.parent))


def _path(name):
    path = MODELS[name][0]
    if not path.exists():
        hint = (
            "expand the published xacro with the ROS 2 environment's own xacro: "
            f"{SHARE.parent}/bin/xacro {SHARE}/nav2_minimal_tb4_description/urdf/standard/turtlebot4.urdf.xacro > {path}"
            if name == "turtlebot4"
            else "install turtlebot3_gazebo in the ROS 2 Jazzy environment"
        )
        pytest.skip(f"{path} is not installed: {hint}")
    return str(path)


def _settled(path):
    w = rw.launch(robot=path, backend="mujoco")
    w.wait(1.0)  # the base settles on its wheels and caster before any command
    return w


@pytest.mark.parametrize("name", list(MODELS))
def test_loads_as_a_mobile_robot_with_a_free_base(name):
    model = build(_path(name)).model
    assert model.family == "mobile"
    assert model.floating and len(model.arm_joints) == 2


@pytest.mark.parametrize("name", list(MODELS))
def test_wheel_radius_and_track_match_the_published_figures(name):
    """Measured from the model's own geoms and joints; the published figure is the maker's."""
    path, radius, track = MODELS[name]
    w = rw.launch(robot=_path(name), backend="mujoco")
    wheels = w.backend.wheels
    assert wheels["radius"] == pytest.approx(radius, abs=5e-4)
    assert wheels["track"] == pytest.approx(track, abs=1.5e-3)  # the waffle's own joints say 0.288, published 0.287


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("speed", [0.1, 0.2])
def test_straight_drive_goes_forward_without_drift(name, speed):
    w = _settled(_path(name))
    b = w.robot.base
    p0 = b.position.copy()
    b.drive(speed, 0.0, 2.0)
    travelled = float(np.linalg.norm((b.position - p0)[:2]))
    assert 0.70 * speed * 2.0 <= travelled <= 1.05 * speed * 2.0
    assert float(np.linalg.norm((b.position - p0)[1:2])) < 0.02


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("rate", [0.5, 1.0])
def test_pivot_turns_on_the_spot_the_right_way(name, rate):
    w = _settled(_path(name))
    b = w.robot.base
    p0, y0 = b.position.copy(), b.yaw
    b.drive(0.0, rate, 2.0)
    turned = b.yaw - y0
    assert 0.70 * rate * 2.0 <= turned <= 1.05 * rate * 2.0
    assert float(np.linalg.norm((b.position - p0)[:2])) < 0.02


@pytest.mark.parametrize("name", list(MODELS))
def test_go_to_three_points_arrives(name):
    for goal in [(1.0, 0.0), (0.0, 1.0), (-0.5, -0.5)]:
        w = _settled(_path(name))
        w.robot.base.go_to(goal, yaw=0.0, timeout=60)  # ActionTimeoutError here would be a failure
        w.wait(1.0)
        assert float(np.linalg.norm(w.robot.base.position[:2] - np.array(goal))) < 0.05


@pytest.mark.parametrize("name", list(MODELS))
def test_a_goal_out_of_reach_in_the_time_given_times_out(name):
    """go_to does not hurry: a goal further than the time allows is ActionTimeoutError, not a hang."""
    w = _settled(_path(name))
    with pytest.raises(ActionTimeoutError):
        w.robot.base.go_to((3.0, 0.0), timeout=3.0)
