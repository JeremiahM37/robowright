"""UFACTORY's own Gazebo simulation of an xArm 6, with its gripper if asked, headless: nothing of robowright's.

* ``xarm_gazebo``'s ``_robot_beside_table_gazebo.launch.py``, from xarm_ros2 at a pinned commit
  (``scripts/ros2_env.sh xarm``), run with ``robot_type:=xarm dof:=6 gz_type:=gz load_controller:=true``.
  It spawns ``xarm_description``'s arm (and gripper) into Gazebo on ``gz_ros2_control`` and starts
  ``joint_state_broadcaster``, ``xarm6_traj_controller`` and, with ``add_gripper:=true``,
  ``xarm_gripper_traj_controller``. The hardware is gz_ros2_control's, not ``xarm_controller``'s.
* the clock: Gazebo's, bridged to ``/clock`` by the launch file itself (``use_sim_time``).

Two things differ from "as shipped":

* the Gazebo arguments are ``-r -v 3 table_gz.world ...``, which starts the GUI. The launch is run
  from a copy in the rig's workdir with ``-s --headless-rendering`` added (server only, EGL), and
  nothing else.
* the ament index. Only three packages of xarm_ros2 are needed at runtime: ``xarm_description``
  (the xacro and meshes), ``xarm_controller`` (its controller configs, which the launch reads) and
  ``xarm_gazebo`` (the launch files and worlds). The rig makes them visible through an ament prefix
  of its own, as ``kinova_gz_rig.py`` does, and puts ``uf_ros_lib`` (pure Python, which the launch
  imports) on ``PYTHONPATH``. No xarm C++ package is installed; the one library that is built is
  ``xarm_gazebo``'s mimic-joint plugin (``GZ_SIM_SYSTEM_PLUGIN_PATH``), which the gripper needs.

Run as ``python tests/xarm_gz_rig.py`` to start it on its own (Ctrl-C stops it).
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

from robowright.stack import popen_reaped

SHARE = Path(sys.prefix) / "share"
SRC = Path(sys.prefix) / "src" / "xarm_ros2"  # the checkout of scripts/ros2_env.sh xarm
XARM_WS = Path(os.environ.get("ROBOWRIGHT_XARM_WS", "/mnt/bulk/rw-ext/xarm-ws"))
XARM_LAUNCH = SRC / "xarm_gazebo" / "launch" / "_robot_beside_table_gazebo.launch.py"
MIMIC_PLUGIN = XARM_WS / "lib" / "libgazebo_mimic_joint_plugin.so"
CONTROLLER_MANAGER = Path(sys.prefix) / "overlay" / "lib" / "libcontroller_manager.so"
# The arm stands on the table: xarm_gazebo's launch spawns it at (-0.2, -0.5, 1.021), yawed 90 degrees.
BASE_POS = (-0.2, -0.5, 1.021)
READY = (
    "Configured and activated joint_state_broadcaster",
    "Configured and activated xarm6_traj_controller",
)
GRIPPER_READY = "Configured and activated xarm_gripper_traj_controller"


def installed() -> bool:
    """The checkout and the three packages it needs (the rig's ament index is made from them)."""
    return (
        shutil.which("ros2") is not None
        and shutil.which("gz") is not None
        and XARM_LAUNCH.is_file()
        and (SRC / "uf_ros_lib" / "uf_ros_lib").is_dir()
        and all((SRC / p / "package.xml").is_file() for p in ("xarm_description", "xarm_controller", "xarm_gazebo"))
    )


def gripper_installed() -> bool:
    """Whether the mimic-joint plugin the gripper needs is built (``scripts/ros2_env.sh xarm``)."""
    return MIMIC_PLUGIN.is_file()


def description(gripper: bool = False) -> str:
    """xarm_description's xacro as the launch expands it (``xarm_device.urdf.xacro``, the xArm 6 in Gazebo)."""
    query = "dof=6&robot_type=xarm&ros2_control_plugin=gz_ros2_control/GazeboSimSystem"
    query += "&add_gripper=true" if gripper else ""
    return f"{SRC / 'xarm_description' / 'urdf' / 'xarm_device.urdf.xacro'}?{query}"


def ament_overlay(workdir: Path) -> Path:
    """An ament prefix in ``workdir`` holding the three xarm_ros2 packages, as the launch finds them."""
    prefix = workdir / "ament"
    index = prefix / "share" / "ament_index" / "resource_index" / "packages"
    index.mkdir(parents=True, exist_ok=True)
    for pkg in ("xarm_description", "xarm_controller", "xarm_gazebo"):
        (index / pkg).touch()
        link = prefix / "share" / pkg
        if not link.exists():
            link.symlink_to(SRC / pkg)
    return prefix


@contextlib.contextmanager
def ros_paths(workdir: Path):
    """This process's ament and package paths with the overlay first, restored on exit."""
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
    """xarm_gazebo's launch as shipped, with Gazebo's server only, rendering headless."""
    text = XARM_LAUNCH.read_text()
    shipped = "' -r -v 3 {} --physics"
    if text.count(shipped) != 1:
        raise RuntimeError(f"unexpected Gazebo arguments in {XARM_LAUNCH}")
    launch = workdir / "xarm_gz_launch.py"
    launch.write_text(text.replace(shipped, "' -s --headless-rendering -r -v 3 {} --physics"))
    return launch


def _environment(gripper: bool) -> dict:
    """What the launch runs with: the environment as it is, with uf_ros_lib importable and, for the
    gripper, the mimic-joint plugin on Gazebo's system plugin path."""
    env = dict(os.environ)  # ros_paths() has the overlay on AMENT_PREFIX_PATH
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC / "uf_ros_lib"), env.get("PYTHONPATH")]))
    # The patched controller manager of scripts/ros2_env.sh ros2_control, as the UR and Kinova rigs
    # preload it: the installed one times controllers by /clock messages, which go late in simulation.
    if CONTROLLER_MANAGER.exists():
        env["LD_PRELOAD"] = os.pathsep.join(filter(None, [str(CONTROLLER_MANAGER), env.get("LD_PRELOAD")]))
    if gripper:
        env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.pathsep.join(filter(None, [str(MIMIC_PLUGIN.parent), env.get("GZ_SIM_SYSTEM_PLUGIN_PATH")]))
    return env


def _start(cmd, log: Path, env=None) -> subprocess.Popen:
    return popen_reaped(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env)  # gone with this process, however it ends


def _stop(procs) -> None:
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGINT)
    deadline = time.monotonic() + 20
    for p in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(max(0.0, deadline - time.monotonic()))
    # Gazebo's server is in the launch's process group and can outlive the launch: end what is left.
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
def xarm_sim(workdir: Path | None = None, gripper: bool = False, timeout: float = 240.0):
    """xArm 6 (with its gripper if ``gripper``) in headless Gazebo, with UFACTORY's controllers up."""
    d = Path(workdir or tempfile.mkdtemp(prefix="rw-xarm-"))
    with ros_paths(d):
        launch = headless_launch(d)
        cmd = [
            "ros2",
            "launch",
            str(launch),
            "robot_type:=xarm",
            "dof:=6",
            "gz_type:=gz",
            "load_controller:=true",
            f"add_gripper:={'true' if gripper else 'false'}",
        ]
        proc = _start(cmd, d / "control.log", _environment(gripper))
        ready = READY + ((GRIPPER_READY,) if gripper else ())
        try:
            deadline = time.monotonic() + timeout
            while not (all(t in _read(d / "control.log") for t in ready) and _spawned(d / "control.log")):
                if proc.poll() is not None or time.monotonic() > deadline:
                    why = f"exited {proc.returncode}" if proc.poll() is not None else "timed out"
                    raise RuntimeError(f"xArm simulation did not come up ({why}):\n{_read(d / 'control.log')[-4000:]}")
                time.sleep(0.5)
            yield d
        finally:
            _stop([proc])


def _spawned(log: Path) -> bool:
    """Whether every controller spawner the launch started has finished cleanly."""
    text = _read(log)
    started = set(re.findall(r"\[(spawner-\d+)\]: process started", text))
    return bool(started) and all(f"[{s}]: process has finished cleanly" in text for s in started)


def _read(path: Path) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")) if path.exists() else ""


if __name__ == "__main__":
    with xarm_sim(gripper="--gripper" in sys.argv) as d:
        print("serving", d, flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(1)
