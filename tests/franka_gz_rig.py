"""Franka Robotics' own Gazebo simulation of the FR3, headless: nothing of robowright's.

* ``franka_gazebo_bringup``'s ``gazebo_franka_arm_example_controller.launch.py``, from franka_ros2 at a
  pinned commit (``scripts/ros2_env.sh franka``, built with colcon into ``ROBOWRIGHT_FRANKA_WS``): it
  spawns ``franka_description``'s FR3 into Gazebo on ``gz_ros2_control``, starts ``joint_state_broadcaster``
  and one of Franka's example controllers, and the clock is Gazebo's, bridged to ``/clock`` by the launch.
* The one change from "as shipped" is the Gazebo arguments: ``-r empty.sdf`` starts the GUI, which needs a
  display, so the rig runs ``-r -s empty.sdf`` (Gazebo's server only, as ``kinova_gz_rig.py`` does). RViz is
  off (``rviz:=false``). Nothing else of the launch is changed; the files are not copied or edited.

The example controller is chosen by the launch's ``controller:=`` argument, as the launch is written; each
one is a file of ``franka_example_controllers`` with its own motion (see the tests for each one's source).

Run as ``python tests/franka_gz_rig.py`` to start it on its own (Ctrl-C stops it).
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

# The workspace ``scripts/ros2_env.sh franka`` builds; the default is next to the environment it runs in.
WS = Path(os.environ.get("ROBOWRIGHT_FRANKA_WS", f"{sys.prefix}-ws"))
INSTALL = WS / "install"
LAUNCH_FILE = "gazebo_franka_arm_example_controller.launch.py"
# The three example controllers of franka_gazebo_bringup's config that the rig starts headless.
CONTROLLERS = (
    "joint_position_example_controller",
    "joint_velocity_example_controller",
    "joint_impedance_example_controller",
)
# The FR3's base stands on Gazebo's ground at the origin (the launch spawns the description at the world's origin).
BASE_POS = (0.0, 0.0, 0.0)


def installed() -> bool:
    return (
        shutil.which("ros2") is not None
        and shutil.which("gz") is not None
        and (INSTALL / "franka_gazebo_bringup" / "share" / "franka_gazebo_bringup" / "launch" / LAUNCH_FILE).is_file()
        and (INSTALL / "franka_description").is_dir()
    )


@contextlib.contextmanager
def ros_paths():
    """This process's ament prefixes with the workspace first (robowright and ``ros2`` find Franka's
    packages through them); restored on exit."""
    saved = os.environ.get("AMENT_PREFIX_PATH")
    prefixes = [str(p) for p in sorted(INSTALL.iterdir()) if p.is_dir()]
    os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(filter(None, prefixes + [saved or ""]))
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("AMENT_PREFIX_PATH", None)
        else:
            os.environ["AMENT_PREFIX_PATH"] = saved


def launch_command(controller: str) -> list[str]:
    """``ros2 launch`` of the bringup as shipped (its own arguments, plus the two above), in the
    workspace's setup environment."""
    if controller not in CONTROLLERS:
        raise ValueError(f"no headless example controller {controller!r}")
    args = f'rviz:=false gz_args:="-r -s empty.sdf" controller:={controller}'
    script = f'source "{INSTALL}/setup.bash" && exec ros2 launch franka_gazebo_bringup {LAUNCH_FILE} {args}'
    return ["bash", "-c", script]


def _start(cmd, log: Path, env=None) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True, env=env)


def _stop(procs) -> None:
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGINT)
    deadline = time.monotonic() + 20
    for p in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(max(0.0, deadline - time.monotonic()))
    # Gazebo's server is a child of the launch's shell, in its group: wait for the whole group to go.
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


def _read(path: Path) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")) if path.exists() else ""


def _up(log: Path, controller: str) -> bool:
    """Whether the launch's spawners have all finished and the example controller is active."""
    text = _read(log)
    started = set(re.findall(r"\[(spawner-\d+)\]: process started", text))
    done = bool(started) and all(f"[{s}]: process has finished cleanly" in text for s in started)
    return done and f"Configured and activated {controller}" in text


@contextlib.contextmanager
def franka_sim(controller: str, workdir: Path | None = None, timeout: float = 240.0):
    """Franka's FR3 in headless Gazebo with one of its example controllers active; yields the work dir."""
    d = Path(workdir or tempfile.mkdtemp(prefix="rw-franka-"))
    d.mkdir(parents=True, exist_ok=True)
    # robowright finds the world in this process by GZ_PARTITION, so the partition is set here as well
    os.environ["GZ_PARTITION"] = f"rw-franka-{os.getpid()}"
    os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    env = dict(os.environ)
    with ros_paths():
        proc = _start(launch_command(controller), d / "launch.log", env)
        try:
            deadline = time.monotonic() + timeout
            while not _up(d / "launch.log", controller):
                if proc.poll() is not None or time.monotonic() > deadline:
                    why = f"exited {proc.returncode}" if proc.poll() is not None else "timed out"
                    raise RuntimeError(f"Franka simulation did not come up ({why}):\n{_read(d / 'launch.log')[-4000:]}")
                time.sleep(0.5)
            yield d
        finally:
            _stop([proc])


if __name__ == "__main__":
    with franka_sim(CONTROLLERS[0]) as d:
        print("serving", d, flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(1)
