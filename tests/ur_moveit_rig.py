"""Universal Robots' own simulation of a UR arm, with MoveIt: nothing of robowright's in it.

* Gazebo and ``ros2_control``: ``ur_simulation_gz``'s ``ur_sim_control.launch.py``, which spawns
  UR's description on ``gz_ros2_control`` with UR's controllers
  (``scaled_joint_trajectory_controller`` among them);
* MoveIt: ``ur_moveit_config``'s ``ur_moveit.launch.py`` (``move_group``, OMPL), on sim time.

Both are the packages' launch files as shipped, headless (no Gazebo GUI, no RViz). Run as
``python tests/ur_moveit_rig.py [ur5e]`` to start it on its own (Ctrl-C stops it).
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
READY = {
    "control.log": "Configured and activated scaled_joint_trajectory_controller",
    "moveit.log": "You can start planning now",
}


def installed() -> bool:
    packages = ("ur_simulation_gz", "ur_moveit_config", "ur_description")
    return shutil.which("ros2") is not None and all((SHARE / p).is_dir() for p in packages)


def description(ur_type: str = "ur5e") -> str:
    """UR's own description of the arm, as robowright loads a xacro with its arguments."""
    return f"{SHARE / 'ur_description' / 'urdf' / 'ur.urdf.xacro'}?ur_type={ur_type}&name=ur"


def _start(cmd, log: Path) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True)


def _stop(procs) -> None:
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGINT)
    for p in procs:
        try:
            p.wait(20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()


@contextlib.contextmanager
def ur_moveit(ur_type: str = "ur5e", workdir: Path | None = None, timeout: float = 180.0):
    d = Path(workdir or tempfile.mkdtemp(prefix="rw-ur-"))
    procs = [
        _start(
            ["ros2", "launch", "ur_simulation_gz", "ur_sim_control.launch.py", f"ur_type:={ur_type}"]
            + ["launch_rviz:=false", "gazebo_gui:=false"],
            d / "control.log",
        )
    ]
    try:
        procs.append(
            _start(
                ["ros2", "launch", "ur_moveit_config", "ur_moveit.launch.py", f"ur_type:={ur_type}"]
                + ["use_sim_time:=true", "launch_rviz:=false"],
                d / "moveit.log",
            )
        )
        deadline = time.monotonic() + timeout
        while not all(text in _read(d / log) for log, text in READY.items()):
            dead = [p.args[3] for p in procs if p.poll() is not None]
            if dead or time.monotonic() > deadline:
                tails = "\n".join(f"--- {log}\n{_read(d / log)[-2500:]}" for log in READY)
                raise RuntimeError(f"UR simulation did not come up ({'exited: ' + ', '.join(dead) if dead else 'timed out'}):\n{tails}")
            time.sleep(0.5)
        yield d
    finally:
        _stop(procs[::-1])


def _read(path: Path) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")) if path.exists() else ""


if __name__ == "__main__":
    with ur_moveit(sys.argv[1] if len(sys.argv) > 1 else "ur5e") as d:
        print("serving", d, flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(1)
