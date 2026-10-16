"""rw.ros2_stack: a ROS 2 stack started once and put back between tests (Playwright's webServer and
browser contexts). The process handling is tested everywhere; the stack itself against Universal
Robots' own Gazebo simulation (``ur_simulation_gz``), where a ROS 2 environment has it.
"""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright.stack import StackError, _spawners, _stop_group

HERE = Path(__file__).parent


# -- processes --------------------------------------------------------------------------


def test_a_stack_that_ignores_sigint_is_gone_within_its_grace(tmp_path):
    # ros2 launch waits 5 s before SIGTERM and 5 s more before SIGKILL; a simulation being thrown away need not
    proc = subprocess.Popen(["bash", "-c", "trap '' INT; sleep 60 & trap '' INT; wait"], start_new_session=True, stdout=subprocess.DEVNULL)
    time.sleep(0.2)
    t0 = time.monotonic()
    _stop_group(proc, grace=0.5)
    assert time.monotonic() - t0 < 2.0
    # the whole group, the background sleep included: gone once reaped (a killed process is a zombie until then, and
    # killpg still finds a zombie, so this polls briefly)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    with pytest.raises(ProcessLookupError):
        os.killpg(proc.pid, 0)


def test_spawners_are_counted_in_the_stacks_own_group_only(tmp_path):
    fake = tmp_path / "spawner"
    fake.write_text("#!/bin/bash\nsleep 30\n")
    fake.chmod(0o755)
    mine = subprocess.Popen(["bash", "-c", f"{fake} & {fake} & wait"], start_new_session=True)
    other = subprocess.Popen([str(fake)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while _spawners(mine.pid) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _spawners(mine.pid) == 2
        assert _spawners(other.pid) == 1
    finally:
        _stop_group(mine, grace=0.1)
        _stop_group(other, grace=0.1)
    assert _spawners(mine.pid) == 0


@pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="a stack is watched through ROS 2 (rclpy)")
def test_env_none_unsets_a_variable_for_the_stack(tmp_path, monkeypatch):
    monkeypatch.setenv("RW_STACK_PROBE", "set")
    st = rw.ros2_stack(
        ["bash", "-c", "echo probe=${RW_STACK_PROBE:-unset}; sleep 0.2"], env={"RW_STACK_PROBE": None}, timeout=5, log_dir=tmp_path
    )
    with pytest.raises(StackError, match="exited with 0 before it was up"):
        st.start()  # it exits before any robot appears: refused, with its output
    assert "probe=unset" in (tmp_path / "stack.log").read_text()


# -- a real stack -----------------------------------------------------------------------


def _ur_env() -> dict:
    sys.path.insert(0, str(HERE))
    import ur_moveit_rig as ur

    env = {"DISPLAY": None}  # UR's launch gives gz sim no headless flag: no display, no GLX failure
    if ur.CONTROLLER_MANAGER.exists():
        env["LD_PRELOAD"] = str(ur.CONTROLLER_MANAGER)
    return env


UR = "ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e launch_rviz:=false gazebo_gui:=false"


@pytest.fixture(scope="module")
def ur():
    if importlib.util.find_spec("rclpy") is None:
        pytest.skip("needs a ROS 2 environment (rclpy)")
    try:
        import ament_index_python.packages as ament

        ament.get_package_share_directory("ur_simulation_gz")
    except Exception:
        pytest.skip("needs ur_simulation_gz")
    with rw.ros2_stack(UR, env=_ur_env()) as st:
        yield st


def _world(objects=()):
    scene = rw.SceneSpec(robot=None, objects=list(objects), cameras=[])
    return rw.launch(scene, backend="ros2", use_sim_time=True, gazebo=True, settings=rw.Settings(trace="off"))


def test_the_stack_is_up_when_its_controllers_are(ur):
    snap = ur.snapshot
    assert "scaled_joint_trajectory_controller" in {c.name for c in snap.controllers}
    assert len(snap.joints) == 6 and {"ground_plane", "ur"} <= set(snap.models)
    assert not {m for m in snap.models if m.endswith("_link")}  # models, not the arm's links


def test_reset_puts_the_arm_back_and_removes_the_tests_objects(ur):
    ur.reset()
    with _world([rw.ObjectSpec("cube", pos=(0.5, 0.3, None))]) as w:
        q = w.robot.qpos()[: w.robot.n_arm].copy()
        w.robot.arm.move_joints(q + np.array([0.4, -0.2, 0.2, 0.0, 0.0, 0.3]))
        assert "cube" in w.object_names
    took = ur.reset()
    assert took < 5.0, took  # a reset is the motion back, not a restart
    with _world() as w:
        assert np.allclose(w.robot.qpos()[: w.robot.n_arm], q, atol=0.03), (w.robot.qpos(), q)
    assert {m["name"] for m in ur._gz.scene()} == set(ur.snapshot.models)
    assert ur.restarts == 0


def test_a_checkpoint_is_what_reset_puts_back(ur):
    """A box placed once and checkpointed is put back where it was after a test moves it; checked in Gazebo's own
    scene, not robowright's copy of the poses. The stack is left as it came up."""
    from robowright.gazebo import Gazebo
    from robowright.scene import ObjectSpec

    ur.reset()
    gz = ur._gz
    seen = Gazebo(ur.snapshot.world)  # a connection of its own: only Gazebo's pose stream, never robowright's copy

    def where(name):
        return seen.pose(name)[0]

    gz.spawn_object(ObjectSpec("crate", size=(0.05, 0.05, 0.05), pos=(0.6, -0.4, None), mass=0.5))
    time.sleep(0.5)  # let it settle on the ground
    ur.checkpoint()
    try:
        home = where("crate")
        gz.set_pose("crate", home + np.array([0.0, 0.3, 0.0]))
        deadline = time.monotonic() + 5
        while np.linalg.norm(where("crate") - home) < 0.2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert np.linalg.norm(where("crate") - home) > 0.2  # moved, in Gazebo
        ur.reset()
        deadline = time.monotonic() + 5
        while not np.allclose(where("crate"), home, atol=2e-3) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert np.allclose(where("crate"), home, atol=2e-3), (where("crate"), home)
    finally:
        seen.close()
        gz.remove("crate")
        ur.checkpoint()
    assert "crate" not in ur.snapshot.models


def test_a_stack_a_reset_cannot_put_back_is_restarted(ur):
    from controller_manager_msgs.srv import SwitchController

    ur.reset()
    link = ur._link
    client = link.node.create_client(SwitchController, "/controller_manager/switch_controller")
    assert client.wait_for_service(timeout_sec=10)
    req = SwitchController.Request(deactivate_controllers=["scaled_joint_trajectory_controller"], strictness=2)
    f = client.call_async(req)
    end = time.monotonic() + 10
    while not f.done() and time.monotonic() < end:
        time.sleep(0.05)
    assert f.done() and f.result().ok
    assert {c.name: c.state for c in link.controller_list()}["scaled_joint_trajectory_controller"] == "inactive"
    before = ur.restarts
    ur.reset()
    assert ur.restarts == before + 1
    assert "scaled_joint_trajectory_controller" in {c.name for c in ur.snapshot.controllers}
    with _world() as w:
        w.robot.arm.move_joints(w.robot.qpos()[: w.robot.n_arm] + 0.1)  # the restarted stack drives the arm


def test_a_stack_does_not_outlive_a_test_run_that_was_killed(tmp_path):
    """A run killed outright (SIGKILL: no cleanup runs) took its stack with it: the kernel signals the launch process.
    Before, Gazebo stacks from a run a timeout killed were still running hours later."""
    script = tmp_path / "parent.py"
    script.write_text(
        "import os, signal, subprocess, sys\n"
        "from robowright.stack import _die_with_parent\n"
        "p = subprocess.Popen(['sleep', '60'], start_new_session=True, preexec_fn=_die_with_parent())\n"
        "print(p.pid, flush=True)\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=30)
    child = int(run.stdout.split()[0])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(child, 9)
    pytest.fail("the launch process outlived the run that started it")
