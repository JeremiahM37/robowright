"""Nav2's own example launch files, as shipped, headless, with robowright as the observer.

Each of ``nav2_simple_commander``'s ``*_example_launch.py`` files starts the whole demo: Gazebo with
the ``depot`` world (``nav2_minimal_tb4_sim``), a TurtleBot 4 spawned at (-8, 0), its
robot_state_publisher, Nav2's bringup (localization and navigation, on the depot map), and the
example program itself as a node. :meth:`Example.run` starts one of those launch files with the
arguments a headless machine needs (``headless:=True use_rviz:=False``) and nothing else: no
world, robot or parameter is changed, and nothing of robowright's runs in it.

The one change from as shipped: the depot world is the same as shipped but with Gazebo's
SceneBroadcaster always on. The shipped world leaves it out when headless (its comment: it is for
the GUI), and without it Gazebo publishes no model poses, so there is no ground truth to read. The
launch file is copied into the test's directory with its world path pointing at that copy (as
:func:`nav2_rig.observed_world` does for the TurtleBot 3 sandbox, and the launch file has no world
argument to do it with). Robowright's own attempt to add the system to the running world did not
produce poses (measured: none in 15 s). Nothing else changes.

The example node runs in the launch, so its output is the launch log (``launch.log``, with the
node's name in front of each line).

Run as ``python tests/nav2_examples_rig.py example_nav_to_pose`` to start one on its own (Ctrl-C
stops it).
"""

from __future__ import annotations

import contextlib
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SHARE = Path(sys.prefix) / "share"
PACKAGES = ("nav2_bringup", "nav2_minimal_tb4_sim", "nav2_simple_commander", "nav2_minimal_tb4_description")
# example node -> the launch file nav2_simple_commander ships for it
LAUNCHES = {
    "example_nav_to_pose": "nav_to_pose_example_launch.py",
    "example_follow_path": "follow_path_example_launch.py",
    "example_waypoint_follower": "waypoint_follower_example_launch.py",
}


def _nav2_rig():
    spec = importlib.util.spec_from_file_location("nav2_rig", Path(__file__).parent / "nav2_rig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stack = _nav2_rig()
_environment, _read, _start, _stop, _first_message = _stack._environment, _stack._read, _stack._start, _stack._stop, _stack._first_message
LOCALIZED = re.compile(r"\[lifecycle_manager_localization\]: .*Managed nodes are active")
NAVIGATION = re.compile(r"\[lifecycle_manager_navigation\]: .*Managed nodes are active")


_last_scan_check = ""


def _scan_publishes(timeout: float) -> bool:
    """Whether the lidar publishes within ``timeout``, with what ``ros2`` last said kept for the
    error message. Asked until it does: ``ros2 topic echo`` gives up at once on a topic nobody
    advertises yet, and Nav2 can be active a moment before Gazebo's bridge has advertised /scan
    (a single try failed the whole module in two runs of four)."""
    global _last_scan_check
    deadline = time.monotonic() + timeout
    while True:
        left = deadline - time.monotonic()
        try:
            done = subprocess.run(
                ["ros2", "topic", "echo", "--no-daemon", "/scan", "--once"],
                capture_output=True,
                text=True,
                timeout=max(left, 1.0),
                env=_environment(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            _last_scan_check = "ros2 topic echo /scan timed out"
            return False
        _last_scan_check = f"ros2 topic echo /scan exit {done.returncode}: {done.stderr.strip()[-500:]}"
        if done.returncode == 0:
            return True
        if time.monotonic() + 1.0 > deadline:
            return False
        time.sleep(1.0)


def installed() -> bool:
    return shutil.which("ros2") is not None and all((SHARE / p).is_dir() for p in PACKAGES)


def launch_file(example: str) -> Path:
    return SHARE / "nav2_simple_commander" / LAUNCHES[example]


class Example:
    """One running example launch: its log, and whether its example program has finished."""

    def __init__(self, name: str, workdir: Path, launch: subprocess.Popen):
        self.name = name
        self.workdir = workdir
        self.launch = launch

    @property
    def log(self) -> str:
        return _read(self.workdir / "launch.log")

    def wait_for(self, pattern: re.Pattern, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not pattern.search(self.log):
            if self.launch.poll() is not None or time.monotonic() > deadline:
                return False
            time.sleep(0.5)
        return True


def observed_launch(name: str, workdir: Path) -> Path:
    """The example's launch file as shipped, except for the world it loads (see the docstring)."""
    src = launch_file(name)
    text = src.read_text()
    world = "os.path.join(sim_dir, 'worlds', 'depot.sdf')"
    if text.count(world) != 1:
        raise RuntimeError(f"unexpected layout of {src}: the depot world is not where it was")
    world_src = SHARE / "nav2_minimal_tb4_sim" / "worlds" / "depot.sdf"
    unless, end = '<xacro:unless value="$(arg headless)">', "</xacro:unless>"
    wtext = world_src.read_text()
    if wtext.count(unless) != 1 or wtext.count(end) != 1:
        raise RuntimeError(f"unexpected layout of {world_src}: the SceneBroadcaster block is not where it was")
    observed = workdir / "depot_observed.sdf"
    observed.write_text(wtext.replace(unless, '<xacro:if value="1">').replace(end, "</xacro:if>"))
    dst = workdir / src.name
    dst.write_text(text.replace(world, repr(str(observed))))
    return dst


@contextlib.contextmanager
def example(name: str, workdir: Path | None = None, timeout: float = 300.0):
    """The launch file of ``name``, started headless, until Nav2's navigation stack is active and the
    robot's lidar publishes. The example program has started by then, and waits for Nav2 itself.
    Its logs are kept in ``workdir`` (default: a directory of their own in /tmp, not pytest's, which
    a shared test user's other runs would clear)."""
    d = Path(workdir or tempfile.mkdtemp(prefix=f"rw-{name}-"))
    d.mkdir(parents=True, exist_ok=True)
    launch = _start(
        ["ros2", "launch", str(observed_launch(name, d)), "use_rviz:=False", "headless:=True"],
        d / "launch.log",
        _environment(),
    )
    ex = Example(name, d, launch)
    try:
        for what, reached in (
            ("localization active", lambda: ex.wait_for(LOCALIZED, timeout)),
            ("navigation active", lambda: ex.wait_for(NAVIGATION, timeout)),
            ("the lidar publishing", lambda: _scan_publishes(timeout)),
        ):
            if not reached():
                state = "exited" if launch.poll() is not None else "still running"
                raise RuntimeError(f"{name}'s launch: {what} not reached ({state}; {_last_scan_check}):\n{ex.log[-4000:]}")
        yield ex
    finally:
        _stop([launch])


if __name__ == "__main__":
    with example(sys.argv[1]) as ex:
        print("serving", ex.workdir, flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(1)
