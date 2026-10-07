"""The ROS 2 backend, against real ROS 2: ``ros2_control``'s own controllers on mock hardware,
and a robowright scene simulated behind the same topics (tests/ros2_rig.py).

These need a ROS 2 environment (rclpy, ros2_control, ros2_controllers, tf2_ros): run them from
one, e.g. ``scripts/ros2_env.sh pytest tests/test_ros2.py``. Elsewhere they are skipped.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import condition, expect
from robowright.backends.ros2_backend import settings
from robowright.scene import default_scene

HERE = Path(__file__).parent
needs_ros = pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)")

# A ROS 2 domain of this worker's own, so parallel workers' robots never hear each other.
if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ["ROS_DOMAIN_ID"] = str(40 + int(os.environ["PYTEST_XDIST_WORKER"].removeprefix("gw")) % 50)
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


def _rig():
    spec = importlib.util.spec_from_file_location("ros2_rig", HERE / "ros2_rig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = {"frame": "robowright_base"}  # the root link of robowright's URDF export, which the rigs run
SEEN = {**BASE, "objects": {"cube": "cube", "bin": "bin"}, "cameras": {"front": "/camera/image_raw"}}


def _launch(robot="so101", **opts):
    return rw.launch(scene=default_scene(robot), backend="ros2", settings=rw.Settings(trace="off"), **opts)


def _moves_and_grips(w):
    w.robot.reset_to()
    goal = w.robot.tcp.position + [0.03, 0.02, -0.03]
    w.robot.arm.move_to(goal)
    expect(w.robot.tcp).to_be_near(goal, tol=0.002)
    w.robot.gripper.close()
    expect(w.robot.gripper).to_be_closed()
    w.robot.gripper.open()
    expect(w.robot.gripper).to_be_open()


def test_settings_come_from_the_project_and_are_checked(tmp_path, monkeypatch):
    (tmp_path / "robowright.toml").write_text('[ros2]\nnamespace = "/arm1"\narm = { controller = "my_controller" }\n')
    monkeypatch.chdir(tmp_path)
    s = settings({"timeout": 3.0})
    assert (s["namespace"], s["arm"], s["timeout"]) == ("/arm1", {"controller": "my_controller", "interface": "position"}, 3.0)
    with pytest.raises(ValueError, match="unknown ros2 setting 'topic'"):
        settings({"topic": "x"})
    with pytest.raises(ValueError, match="ros2 gripper interface is one of with_arm, position, trajectory, action, none, not 'effort'"):
        settings({"gripper": {"interface": "effort"}})


@needs_ros
def test_drives_a_forward_position_controller():
    """One JointGroupPositionController for the arm and gripper, the joints named as the
    driver names them (``[ros2] joints`` maps robowright's names to the driver's)."""
    rename = {"shoulder_pan": "joint1", "elbow_flex": "joint3", "gripper": "jaw"}
    with _rig().mock_hardware("so101", "position", rename=rename), _launch(joints=rename, **BASE) as w:
        assert w.backend.ros_names[0] == "joint1" and w.backend.capabilities == frozenset()
        _moves_and_grips(w)


@needs_ros
def test_drives_a_trajectory_controller_and_a_gripper_action():
    opts = {
        "arm": {"controller": "arm_controller", "interface": "trajectory"},
        "gripper": {"controller": "gripper_controller", "interface": "action"},
    }
    with _rig().mock_hardware("so101", "trajectory"), _launch(**opts, **BASE) as w:
        _moves_and_grips(w)


@needs_ros
def test_says_what_is_missing_when_no_robot_answers():
    with pytest.raises(TimeoutError, match=r"no joint states for .* on /nobody/joint_states within 1.0 s .* is the robot's driver running"):
        _launch(namespace="/nobody", timeout=1.0)


@needs_ros
def test_the_same_pick_test_runs_on_a_ros2_robot():
    """examples/test_pick_and_place.py's test, unchanged, on a robot behind ROS 2: objects seen
    through TF, a grasp judged without contact sensing, a camera through an image topic."""
    with _rig().physics("so101"), _launch(**SEEN) as w:
        w.robot.reset_to()
        cube, bin = w.scene["cube"], w.scene["bin"]
        w.robot.pick(cube)
        expect(w.robot.gripper).to_be_holding(cube)
        expect(cube).to_be_above(0.04)
        w.robot.place(on=bin)
        expect(cube).to_be_inside(bin)
        expect(cube).to_be_at_rest()
        with pytest.raises(rw.ExpectationError, match="no contact sensing on ros2: the jaws told to open"):
            expect(w.robot.gripper).to_be_holding(cube, timeout=0.1)
        img = w.robot.observe(cameras=["front"], image_size=(80, 60))["images"]["front"]
        assert img.shape == (60, 80, 3) and img.std() > 5  # a picture, not a blank


@needs_ros
def test_a_trace_records_where_the_robot_saw_the_objects(tmp_path):
    """No ground truth on a robot: its trace keeps what TF reported, so a failure shows the
    objects where the robot thought they were (not at the origin)."""
    with _rig().physics("so101"):
        w = rw.launch(scene=default_scene("so101"), backend="ros2", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)), **SEEN)
        w.robot.reset_to()
        w.robot.pick(w.scene["cube"])
        path = w.close(trace_path=tmp_path / "t.zip")
    pos = rw.Trace(path).arrays["obj_pos"]
    cube = w.object_names.index("cube")
    assert np.allclose(pos[0, cube], [0.22, -0.06, 0.0125], atol=2e-3)  # where it started
    assert pos[-1, cube, 2] > 0.04  # and lifted


@needs_ros
def test_object_poses_are_put_where_the_robot_is_mounted(tmp_path, monkeypatch):
    """TF reports objects from the robot's base; robowright mounts the robot (here turned a
    quarter and moved), and finds the objects where the scene put them."""
    from robowright import assets

    (tmp_path / "robowright.toml").write_text(
        f'[robots.turned]\nfile = "{assets.so101_mjcf()}"\nbase_pos = [0.02, -0.03, 0.0]\nbase_yaw = 1.5708\n'
    )
    monkeypatch.chdir(tmp_path)
    with _rig().physics("turned", cwd=tmp_path), _launch("turned", **SEEN) as w:
        assert np.allclose(w.scene["cube"].position, [0.22, -0.06, 0.0125], atol=1e-3)
        assert abs(w.scene["cube"].yaw) < 1e-3


@needs_ros
def test_a_learned_policy_drives_a_ros2_robot():
    pytest.importorskip("onnxruntime")
    sys.path.insert(0, str(HERE.parent / "examples"))
    from test_learned_policy import so101_pick

    with _rig().physics("so101"), _launch(**SEEN) as w:
        w.robot.reset_to()
        done = condition(w.scene["cube"], "to_be_inside", w.scene["bin"])
        rollout = w.robot.run_policy(so101_pick(), until=done, hold=1.0, timeout=20)
        assert rollout.success, rollout


@needs_ros
@pytest.mark.parametrize("rig", ["mock_hardware", "physics"])
def test_the_contract_runs_on_a_ros2_robot(rig, tmp_path):
    """``robowright check --backend ros2``: what a robot behind ROS 2 can do passes, and the rest
    (no ground truth, contacts, state) is skipped and says why."""
    seen = rig == "physics"
    cfg = '[ros2]\nframe = "robowright_base"\n' + (
        'objects = { cube = "cube", bin = "bin" }\ncameras = { front = "/camera/image_raw" }\n' if seen else ""
    )
    (tmp_path / "robowright.toml").write_text(cfg)
    with getattr(_rig(), rig)("so101"):
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "robowright.cli",
                "check",
                "--backend",
                "ros2",
                "-q",
                "-p",
                "no:cacheprovider",
                "-k",
                "test_arm",
                "--rw-trace",
                "off",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=900,
        )
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-2000:]
    assert ("9 passed, 7 skipped" if seen else "5 passed, 11 skipped") in out.stdout, out.stdout[-2000:]
