"""robowright against a third-party simulator: an SO-101 in Gazebo (gz sim), behind
``gz_ros2_control`` and ``ros2_control``'s stock controllers (tests/gazebo_rig.py).

robowright runs none of the physics here: it reads ``/joint_states`` and TF and sends commands
to the controllers, exactly as it would to a real arm. Needs a ROS 2 environment with Gazebo
(``ros-jazzy-ros-gz-sim``, ``ros-jazzy-gz-ros2-control``); skipped elsewhere.
"""

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.scene import default_scene

HERE = Path(__file__).parent
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)"),
    pytest.mark.skipif(shutil.which("gz") is None, reason="needs Gazebo (gz sim)"),
]
if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ["ROS_DOMAIN_ID"] = str(150 + int(os.environ["PYTEST_XDIST_WORKER"].removeprefix("gw")) % 50)
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


def _rig():
    spec = importlib.util.spec_from_file_location("gazebo_rig", HERE / "gazebo_rig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gazebo(tmp_path_factory):
    with _rig().gazebo("so101", workdir=tmp_path_factory.mktemp("gz")) as d:
        yield d


def test_robowright_drives_an_so101_in_gazebo(gazebo):
    w = rw.launch(
        scene=default_scene("so101"),
        backend="ros2",
        settings=rw.Settings(trace="on", trace_dir=str(gazebo)),
        frame="robowright_base",
        objects={"cube": "cube", "bin": "bin"},
        use_sim_time=True,
    )
    r = w.robot
    assert w.model_changes == []  # robowright changes nothing in a robot it reaches over ROS 2
    assert np.allclose(w.scene["cube"].position[:2], (0.22, -0.06), atol=0.005)  # seen through TF
    r.reset_to()
    goal = r.tcp.position + [0.03, 0.02, -0.03]
    r.arm.move_to(goal)
    expect(r.tcp).to_be_near(goal, tol=0.003)
    r.gripper.close()
    expect(r.gripper).to_be_closed()
    r.gripper.open()
    expect(r.gripper).to_be_open()
    r.pick(w.scene["cube"])
    r.place(on=w.scene["bin"])
    expect(w.scene["cube"]).to_be_inside(w.scene["bin"], timeout=5)
    w.close(failed=False)
