"""Kinova's own simulation of a Gen3 (7 DoF) with a Robotiq 2F-85, headless: nothing of robowright's.

* ``kortex_bringup``'s ``kortex_sim_control.launch.py``, from ros2_kortex at a pinned commit
  (``scripts/ros2_env.sh kortex``): it spawns ``kortex_description``'s arm and gripper into Gazebo
  on ``gz_ros2_control``, and starts Kinova's ``joint_trajectory_controller``,
  ``robotiq_gripper_controller``, ``joint_state_broadcaster`` and (inactive) ``twist_controller``;
* the clock: Gazebo's, bridged to ``/clock`` by the launch file itself (``use_sim_time``).

The launch is run from a copy in the rig's workdir, with one change from "as shipped": its Gazebo
arguments are ``-r -v 3 empty.sdf ...``, which starts the GUI, and the GUI cannot start without a
display. The copy adds ``-s`` (Gazebo's server only) and ``--headless-rendering`` (EGL) and nothing
else. With ``-s`` alone, the server segfaulted in OGRE (no GL display) on the first trajectory, when
the wrist camera's sensor was first rendered. ``kortex_bringup`` is launched by path, so the ament
index needs no entry for it.

Two more things differ from the packages as installed, both in the ament index this rig makes:

* ``robotiq_description``: the installed one (RoboStack's) has no ``mock_sensor_commands``
  argument, which ``kortex_description``'s Robotiq 2F-85 macro passes, so the description does not
  expand. The rig puts the ros2_robotiq_gripper checkout (``scripts/ros2_env.sh kortex``) first on
  the ament index and on ``ROS_PACKAGE_PATH`` (which robowright's own xacro expansion reads), so the
  stack and robowright load the same Robotiq description.
* controller time: the patched controller manager of ``scripts/ros2_env.sh ros2_control`` is
  preloaded, as in ``ur_moveit_rig.py`` (see its docstring).

Run as ``python tests/kinova_gz_rig.py`` to start it on its own (Ctrl-C stops it).
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SHARE = Path(sys.prefix) / "share"
SRC = Path(sys.prefix) / "src"  # the checkouts of scripts/ros2_env.sh kortex
KORTEX_LAUNCH = SRC / "ros2_kortex" / "kortex_bringup" / "launch" / "kortex_sim_control.launch.py"
ROBOTIQ = SRC / "ros2_robotiq_gripper" / "robotiq_description"
CONTROLLER_MANAGER = Path(sys.prefix) / "overlay" / "lib" / "libcontroller_manager.so"
CONTROLLERS = SHARE / "kortex_description" / "arms" / "gen3" / "7dof" / "config" / "ros2_controllers.yaml"
JOINT_LIMITS = SHARE / "kortex_description" / "arms" / "gen3" / "7dof" / "config" / "joint_limits.yaml"
# The Gen3's base stands 30 cm above Gazebo's floor: kortex's launch spawns it at z = 0.3.
BASE_POS = (0.0, 0.0, 0.3)
READY = {
    "control.log": (
        "Configured and activated joint_trajectory_controller",
        "Configured and activated robotiq_gripper_controller",
        "Configured and activated joint_state_broadcaster",
    ),
}


def installed() -> bool:
    packages = ("kortex_description", "robotiq_description", "gripper_controllers", "picknik_twist_controller", "ros_gz_sim")
    return (
        shutil.which("ros2") is not None
        and shutil.which("gz") is not None
        and KORTEX_LAUNCH.is_file()
        and (ROBOTIQ / "package.xml").is_file()
        and all((SHARE / p).is_dir() for p in packages)
    )


def description() -> str:
    """kortex_description's arm and gripper as its launch expands them (``kinova.urdf.xacro`` with
    the launch file's arguments: the Gen3, 7 DoF, in Gazebo, with the Robotiq 2F-85)."""
    args = {
        "robot_ip": "xxx.yyy.zzz.www",
        "name": "gen3",
        "arm": "gen3",
        "dof": "7",
        "vision": "false",
        "prefix": "",
        "sim_gazebo": "true",
        "simulation_controllers": str(CONTROLLERS),
        "gripper": "robotiq_2f_85",
    }
    query = "&".join(f"{k}={v}" for k, v in args.items())
    return f"{SHARE / 'kortex_description' / 'robots' / 'kinova.urdf.xacro'}?{query}"


def ament_overlay(workdir: Path) -> Path:
    """An ament prefix in ``workdir`` holding ``robotiq_description`` from the checkout."""
    prefix = workdir / "ament"
    (prefix / "share" / "ament_index" / "resource_index" / "packages").mkdir(parents=True, exist_ok=True)
    (prefix / "share" / "ament_index" / "resource_index" / "packages" / "robotiq_description").touch()
    link = prefix / "share" / "robotiq_description"
    if not link.exists():
        link.symlink_to(ROBOTIQ)
    return prefix


@contextlib.contextmanager
def ros_paths(workdir: Path):
    """This process's ament and package paths with the overlay first, as the stack and robowright
    need them to load the description; restored on exit."""
    overlay = ament_overlay(workdir)
    saved = {k: os.environ.get(k) for k in ("AMENT_PREFIX_PATH", "ROS_PACKAGE_PATH")}
    os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(filter(None, [str(overlay), saved["AMENT_PREFIX_PATH"]]))
    os.environ["ROS_PACKAGE_PATH"] = str(overlay / "share")
    try:
        yield overlay
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def headless_launch(workdir: Path) -> Path:
    """kortex_bringup's simulation launch as shipped, with Gazebo's server only, rendering headless."""
    text = KORTEX_LAUNCH.read_text()
    shipped = '" -r -v 3 empty.sdf'
    if text.count(shipped) != 1:
        raise RuntimeError(f"unexpected Gazebo arguments in {KORTEX_LAUNCH}")
    launch = workdir / "kortex_sim_control.launch.py"
    launch.write_text(text.replace(shipped, '" -s --headless-rendering -r -v 3 empty.sdf'))
    return launch


def _start(cmd, log: Path, env=None) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True, env=env)


def _environment() -> dict:
    """What kortex's launch runs with: the environment as it is, ``DISPLAY`` included, as in
    ``gazebo_rig.py`` (``ur_moveit_rig.py`` removes it; with the camera crash of
    ``tests/test_kinova_gazebo.py`` measured both ways, it made no difference here), and the
    patched controller manager preloaded where it has been built."""
    env = dict(os.environ)  # ros_paths() has the overlay on AMENT_PREFIX_PATH
    if CONTROLLER_MANAGER.exists():
        env["LD_PRELOAD"] = os.pathsep.join(filter(None, [str(CONTROLLER_MANAGER), env.get("LD_PRELOAD")]))
    return env


def _stop(procs) -> None:
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGINT)
    deadline = time.monotonic() + 20
    for p in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(max(0.0, deadline - time.monotonic()))
    # Gazebo's server is a child of the launch's shell, in its group: wait for the whole group to go
    # (a launch can exit before what it started has), then end whatever is left of it.
    while time.monotonic() < deadline and any(_alive(p.pid) for p in procs):
        time.sleep(0.2)
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def _alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True


@contextlib.contextmanager
def kinova_sim(workdir: Path | None = None, timeout: float = 240.0):
    """Kinova's Gen3 and Robotiq 2F-85 in headless Gazebo, with Kinova's controllers up."""
    d = Path(workdir or tempfile.mkdtemp(prefix="rw-kinova-"))
    with ros_paths(d):
        launch = headless_launch(d)
        proc = _start(
            ["ros2", "launch", str(launch), "launch_rviz:=false", "gripper:=robotiq_2f_85"],
            d / "control.log",
            _environment(),
        )
        try:
            deadline = time.monotonic() + timeout
            while not (all(text in _read(d / log) for log, texts in READY.items() for text in texts) and _spawned(d / "control.log")):
                if proc.poll() is not None or time.monotonic() > deadline:
                    why = f"exited {proc.returncode}" if proc.poll() is not None else "timed out"
                    raise RuntimeError(f"Kinova simulation did not come up ({why}):\n{_read(d / 'control.log')[-4000:]}")
                time.sleep(0.5)
            yield d
        finally:
            _stop([proc])


def _spawned(log: Path) -> bool:
    """Whether every controller spawner the launch started has finished (the arm's controller is
    active before the others are loaded, and a move sent while the controller manager is loading
    one is aborted, as in ``ur_moveit_rig.py``)."""
    text = _read(log)
    started = set(re.findall(r"\[(spawner-\d+)\]: process started", text))
    return bool(started) and all(f"[{s}]: process has finished cleanly" in text for s in started)


def _read(path: Path) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")) if path.exists() else ""


if __name__ == "__main__":
    with kinova_sim() as d:
        print("serving", d, flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(1)
