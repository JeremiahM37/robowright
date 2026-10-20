"""Nav2's own TurtleBot3 simulation, headless, with robowright as the observer: nothing of
robowright's in it.

* Gazebo and the robot: ``nav2_bringup``'s ``tb3_simulation_launch.py`` as shipped
  (``gz sim`` with the sandbox world, a TurtleBot3 Waffle spawned by ``nav2_minimal_tb3_sim``,
  its ``ros_gz_bridge``, ``robot_state_publisher``);
* Nav2: the navigation and localization stacks of ``bringup_launch.py`` (AMCL, the planners,
  controllers, behaviours, ``bt_navigator``, ``lifecycle_manager``) in one container.

Both run with the launch arguments a headless machine needs (``headless:=True use_rviz:=False``)
and nothing else. Nav2's own example programs (``nav2_simple_commander``) are the clients under
test; the test starts one with :meth:`Nav2Stack.example`, as shipped.

Nav2 does not come up on its own. Its localization is AMCL, which publishes ``map -> odom`` only
once it has an initial pose, and the costmaps of the navigation stack wait for that transform:
about a minute after the stack is configured the navigation bringup gives up and aborts
(measured: the stack alone, with no initial pose, aborted at 63 s). The example sets the pose
(``setInitialPose``) and waits for Nav2 to be active, so the rig's readiness is the localization
stack being active, which needs no pose, and the test starts the example straight after.

Navigation (``navigation=True``) needs one more thing, which the shipped stack does not do for
itself: AMCL starts at the origin unless it is told where the robot is. The shipped way to tell it is
RViz's "2D pose estimate", which publishes on ``/initialpose`` (nav2_bringup's rviz_launch relays that
topic), so the rig publishes the same message with the spawn pose. Navigation's costmaps wait for the
``map -> odom`` transform that AMCL publishes once it has that pose, and the navigation lifecycle is
active only after that (measured: without the pose, the navigation bringup aborts about a minute on).

Run as ``python tests/nav2_rig.py`` to start it on its own (Ctrl-C stops it).
"""

from __future__ import annotations

import contextlib
import math
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
PACKAGES = ("nav2_bringup", "nav2_minimal_tb3_sim", "nav2_simple_commander", "turtlebot3_gazebo")
# The sandbox world's map is 20 m square, centred on the origin (nav2_bringup's tb3_sandbox.yaml).
LOCALIZED = re.compile(r"\[lifecycle_manager_localization\]: .*Managed nodes are active")
ACTIVE = re.compile(r"\[lifecycle_manager_navigation\]: .*Managed nodes are active")


def set_initial_pose(x: float, y: float, yaw: float = 0.0) -> None:
    """Publish the pose AMCL starts from, as RViz's "2D pose estimate" does: ``/initialpose`` in the map
    frame, with RViz's default covariance (0.25 m^2 in position, about 0.07 rad^2 in heading)."""
    cov = [0.25, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0685]
    z, w = math.sin(yaw / 2), math.cos(yaw / 2)
    msg = (
        f"{{header: {{frame_id: map}}, pose: {{pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, "
        f"orientation: {{z: {z}, w: {w}}}}}, covariance: {str(cov).replace(' ', '')}}}}}"
    )
    # Published by a short rclpy node rather than `ros2 topic pub --once -w 1`: that CLI publishes, then
    # does not exit (its shutdown hangs), and the 60 s timeout turned a delivered message into a
    # setup error. The node waits until AMCL subscribes, publishes once, and exits by itself.
    script = (
        "import sys, time, rclpy, yaml\n"
        "from geometry_msgs.msg import PoseWithCovarianceStamped\n"
        "from rosidl_runtime_py import set_message_fields\n"
        "from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy\n"
        "rclpy.init()\n"
        "node = rclpy.create_node('rw_initialpose')\n"
        "qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)\n"
        "pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', qos)\n"
        "deadline = time.monotonic() + 30\n"
        "while pub.get_subscription_count() == 0 and time.monotonic() < deadline:\n"
        "    time.sleep(0.1)\n"
        "if pub.get_subscription_count() == 0:\n"
        "    sys.exit('no subscriber on /initialpose within 30 s')\n"
        "msg = PoseWithCovarianceStamped()\n"
        "set_message_fields(msg, yaml.safe_load(sys.argv[1]))\n"
        "pub.publish(msg)\n"
        "time.sleep(0.5)\n"
        "node.destroy_node()\n"
        "rclpy.shutdown()\n"
    )
    subprocess.run([sys.executable, "-c", script, msg], check=True, capture_output=True, timeout=60, env=_environment())


def installed() -> bool:
    return shutil.which("ros2") is not None and all((SHARE / p).is_dir() for p in PACKAGES)


def _start(cmd, log: Path, env=None) -> subprocess.Popen:
    return popen_reaped(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env)  # gone with this process, however it ends


def _environment() -> dict:
    """Without ``DISPLAY``: the Gazebo server then renders headless (see ur_moveit_rig)."""
    return {k: v for k, v in os.environ.items() if k != "DISPLAY"}


def _first_message(topic: str, timeout: float) -> bool:
    """Whether the robot's sensor has published (the robot is spawned and its simulation runs).
    An example started before this lost its first service call to Nav2's AMCL once (measured: the
    example waited forever for AMCL's get_state), so the rig does not hand the stack over earlier."""
    deadline = time.monotonic() + timeout
    while True:  # asked until it does: echo gives up at once on a topic nobody advertises yet
        try:
            cmd = ["ros2", "topic", "echo", "--no-daemon", topic, "--once"]
            left = max(deadline - time.monotonic(), 1.0)
            if subprocess.run(cmd, capture_output=True, timeout=left, env=_environment(), check=False).returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            return False
        if time.monotonic() + 1.0 > deadline:
            return False
        time.sleep(1.0)


def _read(path: Path) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")) if path.exists() else ""


def _alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True


def _stop(procs) -> None:
    """SIGINT each launch's whole process group (Gazebo's server outlives its launch otherwise),
    then SIGKILL whatever is left of it. Only the groups started here are signalled."""
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGINT)
    deadline = time.monotonic() + 30
    for p in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(max(0.0, deadline - time.monotonic()))
    while time.monotonic() < deadline and any(_alive(p.pid) for p in procs):
        time.sleep(0.2)
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        p.wait()


class Nav2Stack:
    """A running stack: its logs, and the example programs that drive it."""

    def __init__(self, workdir: Path, launch: subprocess.Popen):
        self.workdir = workdir
        self.launch = launch
        self.examples: list[subprocess.Popen] = []

    @property
    def log(self) -> str:
        return _read(self.workdir / "stack.log")

    def wait_for(self, pattern: re.Pattern, timeout: float) -> bool:
        """Whether the stack's log matches ``pattern`` within ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while not pattern.search(self.log):
            if self.launch.poll() is not None or time.monotonic() > deadline:
                return False
            time.sleep(0.5)
        return True

    def example(self, name: str) -> subprocess.Popen:
        """Start one of nav2_simple_commander's example programs, as shipped (``ros2 run``), with
        its output in ``<name>.log``."""
        p = _start(["ros2", "run", "nav2_simple_commander", name], self.workdir / f"{name}.log", _environment())
        self.examples.append(p)
        return p

    def example_log(self, name: str) -> str:
        return _read(self.workdir / f"{name}.log")


def observed_world(workdir: Path) -> Path:
    """The sandbox world as shipped, with Gazebo's SceneBroadcaster always on. The shipped world
    leaves it out when headless (its comment: it is for the GUI), and without it Gazebo publishes
    no model poses at all, so there is no ground truth to read. Nothing else changes."""
    src = SHARE / "nav2_minimal_tb3_sim" / "worlds" / "tb3_sandbox.sdf.xacro"
    text = src.read_text()
    unless, end = '<xacro:unless value="$(arg headless)">', "</xacro:unless>"
    if text.count(unless) != 1 or text.count(end) != 1:
        raise RuntimeError(f"unexpected layout of {src}: the SceneBroadcaster block is not where it was")
    dst = workdir / "tb3_sandbox_observed.sdf.xacro"
    dst.write_text(text.replace(unless, '<xacro:if value="1">').replace(end, "</xacro:if>"))
    return dst


@contextlib.contextmanager
def nav2_stack(
    workdir: Path | None = None,
    timeout: float = 240.0,
    spawn: tuple[float, float] | None = None,
    poses: bool = True,
    navigation: bool = False,
):
    """Nav2's TurtleBot3 simulation, started headless, until its localization is active. ``spawn``:
    where the robot is put in the world (``x_pose``, ``y_pose``); default, as the launch file has it.
    ``poses``: the world with its SceneBroadcaster (:func:`observed_world`); False, as shipped.
    ``navigation``: also tell AMCL where the robot is (:func:`set_initial_pose`, at the spawn pose)
    and wait until the navigation stack is active, so that it accepts goals."""
    d = Path(workdir or tempfile.mkdtemp(prefix="rw-nav2-"))
    d.mkdir(parents=True, exist_ok=True)
    args = [f"x_pose:={spawn[0]}", f"y_pose:={spawn[1]}"] if spawn is not None else []
    if poses:
        args.append(f"world:={observed_world(d)}")
    launch = _start(
        ["ros2", "launch", "nav2_bringup", "tb3_simulation_launch.py", "headless:=True", "use_rviz:=False", *args],
        d / "stack.log",
        _environment(),
    )
    stack = Nav2Stack(d, launch)
    try:
        # The joint states too: robowright's ROS 2 backend waits 10 s for them on the first observation, and
        # the simulation's driver can publish them after the lidar (seen once under load in the full verify).
        if not stack.wait_for(LOCALIZED, timeout) or not _first_message("/scan", timeout) or not _first_message("/joint_states", timeout):
            state = "exited" if launch.poll() is not None else "timed out"
            raise RuntimeError(f"Nav2's simulation did not come up ({state}):\n{stack.log[-4000:]}")
        if navigation:
            x, y = spawn if spawn is not None else (-2.0, -0.5)  # the launch file's default spawn, in the sandbox
            set_initial_pose(x, y, 0.0)
            if not stack.wait_for(ACTIVE, timeout):
                state = "exited" if launch.poll() is not None else "timed out"
                raise RuntimeError(f"Nav2's navigation stack did not become active ({state}):\n{stack.log[-4000:]}")
        yield stack
    finally:
        _stop([launch, *stack.examples])


if __name__ == "__main__":
    with nav2_stack() as stack:
        print("serving", stack.workdir, flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(1)
