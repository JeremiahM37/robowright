"""Universal Robots' own simulation of a UR arm, with MoveIt: nothing of robowright's in it.

* Gazebo and ``ros2_control``: ``ur_simulation_gz``'s ``ur_sim_control.launch.py``, which spawns
  UR's description on ``gz_ros2_control`` with UR's controllers
  (``scaled_joint_trajectory_controller`` among them);
* MoveIt: ``ur_moveit_config``'s ``ur_moveit.launch.py`` (``move_group``, OMPL), on sim time.

Both are the packages' launch files as shipped, headless (no Gazebo GUI, no RViz), with two
settings a cell needs and UR's defaults leave out (``floor_mounted`` and ``finer_checks``,
below), each through UR's or MoveIt's own parameters. Run as
``python tests/ur_moveit_rig.py [ur5e]`` to start it on its own (Ctrl-C stops it).

**Floor mounting.** Every UR joint but the elbow is allowed two full turns (-360 to 360 degrees),
the arm's real limits. UR's simulated cell puts a floor 1 cm under the base (the description's
``ground_plane`` link, which MoveIt checks the arm against). A shoulder angle a turn away from
the one the arm is at is then a pose the arm can only reach by swinging the upper arm through
the floor, and MoveIt's KDL solver returns such angles: planning for the tests' poses failed 14
to 18 times in every 40, each after its full 10 s, and 40 of 40 plans to one such goal failed.
A floor-mounted cell keeps the upper arm above the floor, which UR's description takes as a
joint-limits file (``joint_limit_params``); ``floor_mounted`` writes UR's own file with the
shoulder lift held to -180..0 degrees. With it no plan timed out.

**Finer collision checks.** OMPL checks a motion for collisions at points 0.5% of the joint
space's extent apart, and two turns per joint make that extent large: 15 of 600 plans then
failed MoveIt's own check of the finished path (the forearm through the wrist).
``finer_checks`` sets 0.05% for ``ur_manipulator``, and 600 of 600 plans were found.

**Gazebo's floor.** UR's ``ground_plane`` is 1 cm below the base, but Gazebo's world has its own
ground at the base (``empty.sdf``): MoveIt accepted motions through a centimetre that Gazebo's
floor stops. The first run of all saw one execution aborted (``PATH_TOLERANCE_VIOLATED``, a wrist
0.23 rad behind, every joint jolted at once, as by a contact) that none of about 80 since has repeated.
``ground`` adds Gazebo's floor to MoveIt's planning scene with pymoveit2's own collision
example, its top 1 mm under the base.
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
# pymoveit2, whose examples are the MoveIt clients under test (scripts/ros2_env.sh pymoveit2)
PYMOVEIT2 = Path(os.environ.get("ROBOWRIGHT_PYMOVEIT2", Path(sys.prefix) / "src" / "pymoveit2"))
READY = {
    "control.log": "Configured and activated scaled_joint_trajectory_controller",
    "moveit.log": "You can start planning now",
}


def installed() -> bool:
    packages = ("ur_simulation_gz", "ur_moveit_config", "ur_description")
    return shutil.which("ros2") is not None and all((SHARE / p).is_dir() for p in packages)


def start_pose() -> list[float]:
    """The joint positions UR's simulation starts the arm in (``ur_description``'s
    ``initial_positions.yaml``), in the arm's joint order."""
    import yaml

    q = yaml.safe_load((SHARE / "ur_description" / "config" / "initial_positions.yaml").read_text())
    joints = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    return [float(q[j]) for j in joints]


def description(ur_type: str = "ur5e") -> str:
    """UR's own description of the arm, as robowright loads a xacro with its arguments."""
    return f"{SHARE / 'ur_description' / 'urdf' / 'ur.urdf.xacro'}?ur_type={ur_type}&name=ur"


def floor_mounted(ur_type: str, workdir: Path) -> Path:
    """UR's simulation description, with UR's joint limits for ``ur_type`` except the shoulder
    lift, held between -180 and 0 degrees: the upper arm stays above the floor."""
    limits = (SHARE / "ur_description" / "config" / ur_type / "joint_limits.yaml").read_text()
    lift = re.compile(r"(shoulder_lift_joint:.*?max_position: !degrees)\s*360\.0(.*?min_position: !degrees)\s*-360\.0", re.S)
    limits, n = lift.subn(r"\1 0.0\2 -180.0", limits)
    if n != 1:
        raise RuntimeError(f"unexpected layout of UR's joint limits for {ur_type}")
    (workdir / "joint_limits.yaml").write_text(limits)
    cell = workdir / "ur_floor_mounted.urdf.xacro"
    cell.write_text(
        '<?xml version="1.0"?>\n<robot xmlns:xacro="http://wiki.ros.org/xacro" name="$(arg name)">\n'
        f'  <xacro:arg name="joint_limit_params" default="{workdir / "joint_limits.yaml"}"/>\n'
        f'  <xacro:include filename="{SHARE / "ur_simulation_gz" / "urdf" / "ur_gz.urdf.xacro"}"/>\n</robot>\n'
    )
    return cell


def finer_checks(workdir: Path, fraction: float, **arguments: str) -> Path:
    """UR's ``ur_moveit.launch.py``, included unchanged, with OMPL checking ``ur_manipulator``'s
    motions for collisions every ``fraction`` of its joint space's extent (OMPL's default, which
    UR's configuration keeps, is 0.005)."""
    launch = workdir / "ur_moveit_checked.launch.py"
    ur_launch = str(SHARE / "ur_moveit_config" / "launch" / "ur_moveit.launch.py")
    launch.write_text(
        "from launch import LaunchDescription\n"
        "from launch.actions import IncludeLaunchDescription\n"
        "from launch.launch_description_sources import PythonLaunchDescriptionSource\n"
        "from launch_ros.actions import SetParameter\n\n\n"
        "def generate_launch_description():\n"
        "    return LaunchDescription([\n"
        f"        SetParameter(name='ompl.ur_manipulator.longest_valid_segment_fraction', value={fraction!r}),\n"
        f"        IncludeLaunchDescription(PythonLaunchDescriptionSource({ur_launch!r}),\n"
        f"                                 launch_arguments={arguments!r}.items()),\n"
        "    ])\n"
    )
    return launch


def ground(workdir: Path) -> None:
    """Gazebo's ground (``empty.sdf``: a plane at z = 0) in MoveIt's planning scene, with
    pymoveit2's ``ex_collision_primitive.py``, as published."""
    cmd = [sys.executable, str(PYMOVEIT2 / "examples" / "ex_collision_primitive.py"), "--ros-args"]
    cmd += ["-p", "shape:=box", "-p", "position:=[0.0, 0.0, -0.011]", "-p", "quat_xyzw:=[0.0, 0.0, 0.0, 1.0]"]
    cmd += ["-p", "dimensions:=[4.0, 4.0, 0.02]", "-p", "use_sim_time:=true"]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(PYMOVEIT2), os.environ.get("PYTHONPATH")])))
    with (workdir / "ground.log").open("w") as log:
        done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=120)
    if done.returncode != 0 or "Planning scene updated successfully" not in _read(workdir / "ground.log"):
        raise RuntimeError(f"could not add Gazebo's floor to MoveIt's scene:\n{_read(workdir / 'ground.log')[-2000:]}")


def _start(cmd, log: Path, env=None) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True, env=env)


def _headless() -> dict:
    """The environment without ``DISPLAY``: UR's launch starts ``gz sim`` with no way to pass
    ``--headless-rendering``, and with DISPLAY set to an X server without GLX, Gazebo's renderer
    fails to make a window and the server crashes once a camera renders."""
    return {k: v for k, v in os.environ.items() if k != "DISPLAY"}


def _stop(procs) -> None:
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGINT)
    deadline = time.monotonic() + 20
    for p in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(max(0.0, deadline - time.monotonic()))
    # A launch can exit before everything it started has: Gazebo's server, signalled while it was
    # still starting, outlived its launch and kept running. Wait for each launch's whole process
    # group, then end whatever is left of it.
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
def ur_moveit(
    ur_type: str = "ur5e", workdir: Path | None = None, timeout: float = 180.0, floor: bool = True, segment: float | None = 0.0005
):
    """UR's Gazebo simulation and MoveIt, with ``floor_mounted`` limits and Gazebo's ``ground`` in
    MoveIt's scene unless ``floor=False``, and ``finer_checks(segment)`` unless ``segment=None``
    (both off: UR's packages exactly as shipped)."""
    d = Path(workdir or tempfile.mkdtemp(prefix="rw-ur-"))
    cell = [f"description_file:={floor_mounted(ur_type, d)}"] if floor else []
    arguments = {"ur_type": ur_type, "use_sim_time": "true", "launch_rviz": "false"}
    if segment:
        moveit = [str(finer_checks(d, segment, **arguments))]
    else:
        moveit = ["ur_moveit_config", "ur_moveit.launch.py", *(f"{k}:={v}" for k, v in arguments.items())]
    procs = [
        _start(
            ["ros2", "launch", "ur_simulation_gz", "ur_sim_control.launch.py", f"ur_type:={ur_type}"]
            + ["launch_rviz:=false", "gazebo_gui:=false", *cell],
            d / "control.log",
            _headless(),
        )
    ]
    try:
        procs.append(
            _start(
                ["ros2", "launch", *moveit],
                d / "moveit.log",
            )
        )
        deadline = time.monotonic() + timeout
        while not (all(text in _read(d / log) for log, text in READY.items()) and _spawned(d / "control.log")):
            dead = [log for log, p in zip(READY, procs) if p.poll() is not None]
            if dead or time.monotonic() > deadline:
                tails = "\n".join(f"--- {log}\n{_read(d / log)[-2500:]}" for log in READY)
                raise RuntimeError(f"UR simulation did not come up ({'exited: ' + ', '.join(dead) if dead else 'timed out'}):\n{tails}")
            time.sleep(0.5)
        if floor:
            ground(d)
        yield d
    finally:
        _stop(procs[::-1])


def _spawned(log: Path) -> bool:
    """Whether every controller spawner UR's launch started has finished. The arm's controller
    is active before the others are loaded, and while the controller manager loads one, it
    skips reading and writing the hardware: a move started then stops dead and is aborted."""
    text = _read(log)
    started = set(re.findall(r"\[(spawner-\d+)\]: process started", text))
    return bool(started) and all(f"[{s}]: process has finished cleanly" in text for s in started)


def _read(path: Path) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")) if path.exists() else ""


if __name__ == "__main__":
    with ur_moveit(sys.argv[1] if len(sys.argv) > 1 else "ur5e") as d:
        print("serving", d, flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(1)
