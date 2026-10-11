"""MoveIt's own Panda demo, headless: nothing of robowright's in it.

``moveit_resources_panda_moveit_config``'s ``demo.launch.py`` as shipped: ``move_group`` (OMPL,
the stack's default pipeline), ``ros2_control`` on mock hardware, the arm and hand controllers,
and the robot's own TF. The demo's one RViz node is left out, and nothing else: the launch file
has no argument for it, so :func:`launch_file` includes the demo's own ``generate_launch_description``
and drops that node. Run as ``python tests/panda_moveit_rig.py`` to start it on its own (Ctrl-C stops it).

Mock hardware is not a simulator: nothing publishes ``/clock``, and the arm moves on the wall clock,
so the stack, its clients and robowright all run without sim time (``use_sim_time`` false).
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
PYMOVEIT2 = Path(os.environ.get("ROBOWRIGHT_PYMOVEIT2", Path(sys.prefix) / "src" / "pymoveit2"))
PACKAGE = "moveit_resources_panda_moveit_config"
READY = {"panda.log": "You can start planning now"}


def installed() -> bool:
    packages = (PACKAGE, "moveit_resources_panda_description")
    return shutil.which("ros2") is not None and all((SHARE / p).is_dir() for p in packages)


def start_pose() -> list[float]:
    """The joint positions the mock hardware starts the arm in (the package's ``initial_positions.yaml``)."""
    import yaml

    q = yaml.safe_load((SHARE / PACKAGE / "config" / "initial_positions.yaml").read_text())
    return [float(q[f"panda_joint{i}"]) for i in range(1, 8)]


def description() -> str:
    """The description the stack publishes as ``robot_description``: its own xacro, with the mock hardware."""
    return f"{SHARE / PACKAGE / 'config' / 'panda.urdf.xacro'}?ros2_control_hardware_type=mock_components"


def launch_file(workdir: Path) -> Path:
    """A launch file that runs ``demo.launch.py`` as shipped, without its RViz node."""
    demo = SHARE / PACKAGE / "launch" / "demo.launch.py"
    launch = workdir / "panda_moveit_headless.launch.py"
    launch.write_text(
        "import importlib.util\n"
        "from launch import LaunchDescription\n"
        "from launch_ros.actions import Node\n\n\n"
        "def generate_launch_description():\n"
        f"    spec = importlib.util.spec_from_file_location('panda_demo', {str(demo)!r})\n"
        "    demo = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(demo)\n"
        "    kept = [e for e in demo.generate_launch_description().entities\n"
        "            if not (isinstance(e, Node) and e.node_executable == 'rviz2')]\n"
        "    return LaunchDescription(kept)\n"
    )
    return launch


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
    # A launch can exit before everything it started has: wait for its whole process group, then end
    # whatever is left of it.
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
def panda_moveit(workdir: Path | None = None, timeout: float = 180.0):
    """The demo stack, up: yields its work directory once ``move_group`` plans and every controller is active."""
    d = Path(workdir or tempfile.mkdtemp(prefix="rw-panda-"))
    proc = _start(["ros2", "launch", str(launch_file(d))], d / "panda.log")
    try:
        deadline = time.monotonic() + timeout
        while not (all(text in _read(d / log) for log, text in READY.items()) and _controllers_up(d / "panda.log")):
            if proc.poll() is not None or time.monotonic() > deadline:
                state = "exited" if proc.poll() is not None else "timed out"
                raise RuntimeError(f"Panda demo did not come up ({state}):\n{_read(d / 'panda.log')[-4000:]}")
            time.sleep(0.5)
        yield d
    finally:
        _stop([proc])


def _controllers_up(log: Path) -> bool:
    """Whether the arm and hand controllers are active and every spawner the demo started has finished."""
    text = _read(log)
    started = set(re.findall(r"\[(spawner-\d+)\]: process started", text))
    return (
        "Configured and activated panda_arm_controller" in text
        and "Configured and activated panda_hand_controller" in text
        and len(started) == 3
        and all(f"[{s}]: process has finished cleanly" in text for s in started)
    )


def _read(path: Path) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")) if path.exists() else ""


if __name__ == "__main__":
    with panda_moveit() as d:
        print("serving", d, flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(1)
