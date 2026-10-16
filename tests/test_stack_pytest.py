"""``[ros2] stack`` in a project's robowright.toml: ``pytest --rw-backend ros2`` starts the stack once per worker and
each test gets it as it came up, the way Playwright's webServer and browser contexts work. A project of its own (a
robowright.toml and three plain tests) is run in a fresh pytest against Universal Robots' own Gazebo simulation:
each test finds the arm where the stack started it and none of the objects the test before it added, and the stack
is started once. Needs a ROS 2 environment with ur_simulation_gz; skipped elsewhere.
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).parent

TESTS = """
import json
import os
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright.gazebo import Gazebo

LOG = Path(os.environ["RW_STACK_LOG"])
CUBE = rw.SceneSpec(robot=None, objects=[rw.ObjectSpec("cube", pos=(0.5, 0.3, None))], cameras=[])


def _check_clean_then_dirty(world, name):
    q = world.robot.qpos()[: world.robot.n_arm]
    seen = json.loads(LOG.read_text()) if LOG.exists() else {}
    start = seen.get("start")
    if start is None:
        seen["start"] = [float(v) for v in q]
    else:
        assert np.allclose(q, start, atol=0.03), (q, start)  # where the stack started it, not where the last test left it
    gz = Gazebo()
    try:
        stray = {m["name"] for m in gz.scene()} - {"ground_plane", "ur", "cube"}
    finally:
        gz.close()
    assert not stray, stray
    world.robot.arm.move_joints(q + 0.3)
    seen.setdefault("ran", []).append(name)
    LOG.write_text(json.dumps(seen))


@pytest.mark.scene(CUBE)
def test_one(world):
    _check_clean_then_dirty(world, "one")


@pytest.mark.scene(CUBE)
def test_two(world):
    _check_clean_then_dirty(world, "two")


@pytest.mark.scene(CUBE)
def test_three(world):
    _check_clean_then_dirty(world, "three")
"""


def _have_ur() -> bool:
    if importlib.util.find_spec("rclpy") is None:
        return False
    try:
        import ament_index_python.packages as ament

        ament.get_package_share_directory("ur_simulation_gz")
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _have_ur(), reason="needs a ROS 2 environment with ur_simulation_gz")
def test_a_project_runs_its_tests_against_one_stack_reset_between_them(tmp_path):
    sys.path.insert(0, str(HERE))
    import ur_moveit_rig as ur

    env_lines = ["DISPLAY = false"]  # UR's launch gives gz sim no headless flag: no display, no GLX failure
    if ur.CONTROLLER_MANAGER.exists():
        env_lines.append(f'LD_PRELOAD = "{ur.CONTROLLER_MANAGER}"')
    (tmp_path / "robowright.toml").write_text(
        "[ros2]\n"
        "use_sim_time = true\n"
        "gazebo = true\n"
        "[ros2.stack]\n"
        'command = "ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e launch_rviz:=false gazebo_gui:=false"\n'
        f"env = {{ {', '.join(env_lines)} }}\n"
    )
    (tmp_path / "test_arm.py").write_text(TESTS)
    log = tmp_path / "seen.json"
    # A fixed domain no other rig uses: one counted up from this run's went past 232, the last Fast DDS takes, when
    # another module had set 230-232, and the child's ROS failed to start (with no output at all).
    env = {**os.environ, "RW_STACK_LOG": str(log), "ROS_DOMAIN_ID": "22"}
    env["PYTHONUNBUFFERED"] = "1"  # a child ended by a native library's exit() keeps what it printed
    env.pop("GZ_PARTITION", None)
    t0 = time.monotonic()
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:launch_testing",
            "-p",
            "no:launch_ros",
            "--rw-backend",
            "ros2",
            "--rw-trace",
            "on",
            "--durations=0",
            "test_arm.py",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    took = time.monotonic() - t0
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]
    assert json.loads(log.read_text())["ran"] == ["one", "two", "three"]
    # One stack for the three: a start each would be three times its startup (~12 s here) plus a stop each.
    traces = sorted((tmp_path / "robowright-traces").glob("*.zip"))
    assert len(traces) == 3
    from robowright.trace import Trace

    resets = [next(e["args"] for e in Trace(t).events if e["name"] == "ros2_stack_reset") for t in traces]
    assert not any(r["restarted"] for r in resets), resets
    assert sum(r["seconds"] for r in resets) < 15, resets
    print(f"three tests against one stack: {took:.1f} s; resets {[(r['seconds'], r['parts']) for r in resets]}")
    print(run.stdout[-1500:])
