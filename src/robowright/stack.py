"""A ROS 2 stack started once and reset between tests, as Playwright starts a web server once and
gives each test a fresh browser context.

Starting a Gazebo or ROS 2 stack takes seconds to tens of seconds (measured: 7-12 s for xArm or
Nav2 in Gazebo, plus 10 s to stop, against 1-15 s for the test itself). Started once and put
back between tests, the cost is paid once::

    @pytest.fixture(scope="session")
    def ur():
        with rw.ros2_stack("ros2 launch ur_simulation_gz ur_sim_control.launch.py") as stack:
            yield stack

    @pytest.fixture
    def world(ur):
        ur.reset()  # as it was when it came up: arm, gripper, Gazebo's models; the test's objects removed
        with rw.launch(backend="ros2", use_sim_time=True, gazebo=True) as w:
            yield w

Ready is read from the ROS graph, not from the launch's log: joint states arriving; a controller
manager's controllers loaded, with no spawner still running and at least one active; and, where
the stack publishes ``/clock``, the clock moving. What the stack was then (each controller's
joints, Gazebo's models and where they stood) is what ``reset()`` puts back: models the test
added are removed, moved ones put back, and each arm and gripper controller driven to where it
started. A stack ``reset()`` cannot put back (a controller that is no longer active, a process
that exited) is restarted, and ``restarts`` counts it.

The stack runs in this process's ROS domain, so the worlds a test launches see it, and in a
Gazebo partition of its own, so parallel workers' simulations never meet. It is stopped with
SIGINT and, a second later, whatever is left of it is killed (``ros2 launch`` alone waits 5 s
before SIGTERM and 5 s more before SIGKILL, for a simulation being thrown away).
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import RobowrightError

SETTLED = 1.0  # seconds the controllers must stay as they are before the stack counts as up
_JOINT_TOL = 0.01  # rad (or m): a joint this close to where it started is back
_POSE_TOL = 1e-3  # m: a model this close to where it started is not moved back
_RESET_SPEED = 1.0  # rad/s (or m/s): how fast reset() drives a joint back


class StackError(RobowrightError):
    """The stack did not come up, or exited."""


@dataclass
class Snapshot:
    """The stack as it came up: what ``reset()`` puts back."""

    joints: dict = field(default_factory=dict)  # joint -> position
    controllers: list = field(default_factory=list)  # ros2_discover.Controller, active ones
    models: dict = field(default_factory=dict)  # Gazebo model -> (pos, quat)
    world: str | None = None  # the Gazebo world, when there is one
    localized: object = None  # AMCL's pose (geometry_msgs/PoseWithCovarianceStamped), where AMCL runs


class Ros2Stack:
    """A ROS 2 stack (your launch file) run for as long as it is open. See the module."""

    def __init__(
        self,
        command: str | list,
        *,
        env: dict | None = None,
        cwd: str | None = None,
        timeout: float = 240.0,
        namespace: str = "",
        log_dir: str | Path | None = None,
        ready=None,
        setup=None,
        speed: float | None = None,
    ):
        self.args = shlex.split(command) if isinstance(command, str) else list(command)
        self.env = dict(env or {})
        self.cwd = cwd
        self.timeout = timeout
        self.namespace = namespace
        # More the stack needs before a test may start, beyond what is read from the graph: a function of this stack
        # returning True, or a reason it is not ready yet (Nav2's lifecycle nodes active, a map loaded, ...).
        self.ready = ready
        # Run once when the stack is up, before what reset() puts back is taken (Playwright's globalSetup): tell a
        # localizer where the robot is, load a map, spawn a table. A function of this stack.
        self.setup = setup
        # Gazebo's real-time factor once the stack is up (inf: as fast as it steps). Tests wait in simulated time, so a
        # stack whose nodes run on it finishes the same test in less wall time. None leaves the world's own.
        self.speed = speed
        self.log_dir = Path(log_dir or tempfile.mkdtemp(prefix="robowright-stack-"))
        self.log = self.log_dir / "stack.log"
        self.proc: subprocess.Popen | None = None
        self.snapshot: Snapshot | None = None
        self.restarts = 0
        self.resets = 0
        self.startup_s = 0.0  # how long the last start took
        self._gz = None
        self.waited: list = []  # how the last start went: (seconds, what it was waiting for)
        self._domain_set = False
        self.last_reset: dict = {}  # seconds each part of the last reset took
        self._link = None  # one ROS node for the stack's life: reset() must not pay discovery each time

    # -- starting and stopping -----------------------------------------------------------
    def start(self) -> Ros2Stack:
        import importlib.util

        if importlib.util.find_spec("rclpy") is None:
            raise StackError(
                "rw.ros2_stack watches the stack through ROS 2: run it where rclpy is (a ROS 2 install, or scripts/ros2_env.sh)"
            )
        if "GZ_PARTITION" not in self.env and "GZ_PARTITION" not in os.environ:
            # Gazebo's transport is per partition: a partition of this stack's own keeps parallel workers' worlds apart,
            # and this process uses it too, so robowright's Gazebo connection finds this stack's world.
            worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
            os.environ["GZ_PARTITION"] = f"robowright-{os.getpid()}-{worker}"
        os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
        worker = os.environ.get("PYTEST_XDIST_WORKER")
        if worker and "ROS_DOMAIN_ID" not in self.env and not self._domain_set:
            # Each pytest-xdist worker runs a stack of its own: on one ROS domain they would answer each other's tests.
            # This process moves with it, so the worlds a test launches see its stack (rclpy reads it per context).
            base = int(os.environ.get("ROS_DOMAIN_ID", "0") or 0)
            os.environ["ROS_DOMAIN_ID"] = str((base + 1 + int(worker.removeprefix("gw") or 0)) % 101)
            self._domain_set = True
        domain = str(self.env.get("ROS_DOMAIN_ID") or os.environ.get("ROS_DOMAIN_ID") or "0")
        if not domain.isdigit() or int(domain) > 232:
            # Past 232 Fast DDS has no ports for the domain: every node fails to start, and a launch can say nothing.
            raise StackError(f"ROS_DOMAIN_ID={domain} is not a domain ROS 2 can use: 0-232 (0-101 to stay clear of ephemeral ports)")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            self.args,
            stdout=self.log.open("a"),
            stderr=subprocess.STDOUT,
            env={k: str(v) for k, v in {**os.environ, **self.env}.items() if v is not None and v is not False},  # None or false unsets one
            cwd=self.cwd,
            start_new_session=True,
        )
        try:
            self._link = _Link(self.namespace)
            self._link.listen()
            self._wait_ready(t0 + self.timeout)
            if self.setup is not None:
                self.setup(self)
            self.snapshot = self._take_snapshot()
            if self.speed is not None:
                if self.snapshot.world is None:
                    raise StackError("speed= needs a Gazebo world, and this stack runs none")
                self._gazebo(self.snapshot.world).set_speed(self.speed)
            self._link.quiet()
        except BaseException:
            self.stop()
            raise
        self.startup_s = time.monotonic() - t0
        return self

    def stop(self) -> None:
        if self._link is not None:
            self._link.close()
            self._link = None
        if self._gz is not None:
            with contextlib.suppress(Exception):
                self._gz.close()
            self._gz = None
        if self.proc is not None:
            _stop_group(self.proc)
            self.proc = None

    def restart(self) -> None:
        self.stop()
        self.restarts += 1
        self.start()

    def __enter__(self) -> Ros2Stack:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, chars: int = 3000) -> str:
        return _read(self.log)[-chars:]

    # -- ready ----------------------------------------------------------------------------
    def _wait_ready(self, deadline: float) -> None:
        link = self._link
        last, since = None, None
        why = "nothing seen yet"
        t0 = time.monotonic()
        self.waited = []  # (seconds since start, what it was waiting for), each time that changed
        while True:
            if not self.waited or self.waited[-1][1] != why:
                self.waited.append((round(time.monotonic() - t0, 1), why))
            if not self.alive:
                raise StackError(f"the stack exited with {self.proc.returncode} before it was up:\n{self.log_tail()}")
            if time.monotonic() > deadline:
                raise StackError(f"the stack was not up within {self.timeout:g} s ({why}):\n{self.log_tail()}")
            time.sleep(0.1)
            try:
                controllers = link.controller_list(timeout=1.0) if link.manager() else []
            except StackError:  # in the graph a moment before it answers, while the stack is still starting
                why = "the controller manager is not answering yet"
                last, since = None, None
                continue
            state = tuple(sorted((c.name, c.state) for c in controllers))
            spawning = _spawners(self.proc.pid)
            extra = self.ready(self) if self.ready is not None else True
            if not link.joints():
                why = "no joint states yet"
            elif link.manager() and not any(c.active for c in controllers):
                why = "no controller active yet"
            elif spawning:
                why = f"{spawning} controller spawner(s) still running"
            elif link.clock_published() and not link.clock_moving():
                why = "/clock is published but not moving"
            elif extra is not True:
                why = str(extra) if extra else "the stack's own ready check is not yet true"
            elif state != last:
                last, since = state, time.monotonic()
                why = "controllers settling"
                continue
            elif time.monotonic() - since >= SETTLED:
                return
            else:
                continue
            last, since = None, None

    # -- what reset() puts back -----------------------------------------------------------
    def _take_snapshot(self) -> Snapshot:
        from . import ros2_discover as discover

        link = self._link
        # Controllers only where there is a manager (a stack without ros2_control, Nav2's TurtleBot3, has none)
        controllers = [c for c in link.controller_list() if c.active] if link.manager() else []
        snap = Snapshot(joints=link.joints(timeout=3.0), controllers=controllers)
        snap.localized = link.amcl_pose() if link.has_topic("/amcl_pose") else None
        worlds = discover._gazebo_worlds()
        if worlds:
            snap.world = worlds[0]
            gz = self._gazebo(snap.world)
            snap.models = {m: gz.pose(m) for m in _models(gz)}
        return snap

    def checkpoint(self) -> Snapshot:
        """Make the stack as it is now what ``reset()`` puts back: a table spawned, a map loaded or the
        arm parked once, for every test after (Playwright's ``storageState``, for a robot). A restart
        brings back the stack as it came up, not the checkpoint, and says so in ``restarts``."""
        self._link.listen()
        try:
            self.snapshot = self._take_snapshot()
        finally:
            self._link.quiet()
        return self.snapshot

    def _gazebo(self, world: str):
        if self._gz is None:
            from .gazebo import Gazebo

            self._gz = Gazebo(world)
        return self._gz

    def reset(self) -> float:
        """Put the stack back as it came up (see the module), restarting it when that cannot be done.
        Returns the seconds it took."""
        t0 = time.monotonic()
        if not self.alive:
            self.restart()
            return time.monotonic() - t0
        try:
            self._reset()
        except StackError:
            self.restart()
        self.resets += 1
        return time.monotonic() - t0

    def _reset(self) -> None:
        snap, link = self.snapshot, self._link
        took = self.last_reset = {}  # seconds each part of the last reset took: where the time went
        t = time.monotonic()
        if link.manager():
            states = {c.name: c.state for c in link.controller_list()}
            lost = [c.name for c in snap.controllers if states.get(c.name) != "active"]
            if lost:
                raise StackError(f"controllers no longer active: {lost}")
        took["controllers"], t = round(time.monotonic() - t, 3), time.monotonic()
        link.cancel_all_goals()  # whatever the last test started (a navigation goal, a trajectory) stops first
        link.stop_base()
        took["stopped"], t = round(time.monotonic() - t, 3), time.monotonic()
        if snap.world is not None:
            gz = self._gazebo(snap.world)
            models = set(_models(gz))
            for m in models - set(snap.models):
                gz.remove(m)
            took["removed"], t = round(time.monotonic() - t, 3), time.monotonic()
            for m, (pos, quat) in snap.models.items():
                if m not in models:
                    raise StackError(f"Gazebo model {m!r} is gone")
                p, q = gz.pose(m)
                if np.linalg.norm(np.asarray(p) - pos) > _POSE_TOL or abs(abs(float(np.dot(q, quat))) - 1.0) > 1e-6:
                    gz.set_pose(m, pos, quat)
            took["moved"], t = round(time.monotonic() - t, 3), time.monotonic()
        link.listen()
        try:
            link.drive_home(self.namespace, snap)
        finally:
            link.quiet()
        took["joints"], t = round(time.monotonic() - t, 3), time.monotonic()
        if snap.localized is not None:
            link.relocalize(snap.localized)
            took["localized"] = round(time.monotonic() - t, 3)


def ros2_stack(command: str | list, **kwargs) -> Ros2Stack:
    """A :class:`Ros2Stack` for ``command`` (use it in a ``with``: started on entry, stopped on exit)."""
    return Ros2Stack(command, **kwargs)


# -- helpers ----------------------------------------------------------------------------


def _stop_group(proc: subprocess.Popen, grace: float = 1.0) -> None:
    """SIGINT the stack's process group; a second later, SIGKILL whatever is left of it."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGINT)
    end = time.monotonic() + grace
    while time.monotonic() < end and _group_alive(proc.pid):
        time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(5)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _spawners(pgid: int) -> int:
    """How many controller spawners (ros2_control's ``spawner``) are still running in the stack's process group.
    The program itself is matched (its executable, or the script a Python runs), never a word in another
    command line: a shell that started spawners names them too."""
    n = 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if os.getpgid(int(d.name)) != pgid:
                continue
            argv = [a.decode(errors="replace") for a in (d / "cmdline").read_bytes().split(b"\0") if a]
        except (OSError, ProcessLookupError):
            continue
        prog = argv[:1]
        if len(argv) > 1 and Path(argv[0]).name.startswith(("python", "bash", "sh")):  # a script, run by its interpreter
            prog = argv[1:2]
        if prog and Path(prog[0]).name in ("spawner", "spawner.py"):
            n += 1
    return n


def _models(gz) -> list[str]:
    """Gazebo's top-level models (its pose stream also names links, which reset must never move on their own)."""
    return [m["name"] for m in gz.scene()]


def _read(path: Path) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")) if path.exists() else ""


class _Link:
    """One ROS node, from the stack's start to its stop: the newest joint states and clock, the controller manager,
    and the controllers' actions and topics. Readiness and reset read it; neither pays for discovery again."""

    def __init__(self, namespace: str):
        import threading

        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        self.namespace = namespace
        self._ctx = rclpy.Context()
        rclpy.init(context=self._ctx)
        self.node = rclpy.create_node(f"robowright_stack_{os.getpid()}", context=self._ctx)
        self._lock = threading.Lock()
        self._joints: dict = {}
        self._clock = (None, 0.0)  # (newest /clock stamp, monotonic time it last moved on)
        self._subs: list = []
        self._manager: str | None = None
        self._list = None
        self._actions: dict = {}
        self._pubs: dict = {}
        self._ex = SingleThreadedExecutor(context=self._ctx)
        self._ex.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def listen(self) -> None:
        """Subscribe to the joint states and the clock: only while starting and resetting. Kept for the stack's life,
        a Python node taking /clock at 1 kHz and joint states at hundreds of Hz held the GIL so much that the test's own
        threads (a robot description being converted, robowright's own node) ran several times slower."""
        from rclpy.qos import qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState

        from .ros2_discover import _topic

        if self._subs:
            return
        with self._lock:
            self._joints, self._clock = {}, (None, 0.0)  # what arrives from now on, not what was true when last listened
        self._subs = [
            self.node.create_subscription(JointState, _topic(self.namespace, "joint_states"), self._on_joints, qos_profile_sensor_data),
            self.node.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data),
        ]

    def quiet(self) -> None:
        for sub in self._subs:
            self.node.destroy_subscription(sub)
        self._subs = []

    def _spin(self) -> None:
        from rclpy.executors import ExternalShutdownException

        with contextlib.suppress(ExternalShutdownException, Exception):
            self._ex.spin()

    def _on_joints(self, m) -> None:
        if m.position:
            with self._lock:
                self._joints.update(zip(m.name, map(float, m.position)))

    def _on_clock(self, m) -> None:
        t = m.clock.sec + m.clock.nanosec * 1e-9
        with self._lock:
            if self._clock[0] is None or t > self._clock[0]:
                self._clock = (t, time.monotonic())

    def joints(self, timeout: float = 0.0) -> dict:
        end = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._joints or time.monotonic() >= end:
                    return dict(self._joints)
            time.sleep(0.02)

    def clock_published(self) -> bool:
        return self.node.count_publishers("/clock") > 0

    def clock_moving(self, within: float = 0.5) -> bool:
        with self._lock:
            return self._clock[0] is not None and time.monotonic() - self._clock[1] < within

    def manager(self) -> str | None:
        """The controller manager's service prefix, once it is in the graph (None where there is none yet)."""
        if self._manager is None:
            from .ros2_discover import _manager

            self._manager = _manager(self.node, self.namespace)
        return self._manager

    def controller_list(self, timeout: float = 3.0) -> list:
        """Every controller the manager has (``ros2_discover.Controller``: name, type, state, joints)."""
        from controller_manager_msgs.srv import ListControllers

        from .ros2_discover import Controller

        if self._list is None:
            self._list = self.node.create_client(ListControllers, f"{self.manager()}/list_controllers")
        if not self._list.wait_for_service(timeout_sec=timeout):
            raise StackError("the controller manager no longer answers")
        f = self._list.call_async(ListControllers.Request())
        if not _wait(f.done, timeout):
            raise StackError("the controller manager did not answer list_controllers")
        out = []
        for c in f.result().controller:
            claimed = list(c.claimed_interfaces) or list(c.required_command_interfaces)
            out.append(Controller(c.name, c.type, c.state, list(dict.fromkeys(i.rsplit("/", 1)[0] for i in claimed))))
        return out

    def has_topic(self, name: str) -> bool:
        return any(n == name for n, _ in self.node.get_topic_names_and_types())

    def amcl_pose(self, timeout: float = 5.0):
        """AMCL's newest pose (it latches it: transient local), or None."""
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile

        got = []
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sub = self.node.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", got.append, qos)
        try:
            _wait(lambda: bool(got), timeout)
        finally:
            self.node.destroy_subscription(sub)
        return got[-1] if got else None

    def cancel_all_goals(self, timeout: float = 2.0) -> None:
        """Cancel every goal of every action server in the graph (a zero goal id and stamp means all of them)."""
        from action_msgs.srv import CancelGoal
        from rclpy.action import get_action_names_and_types

        pending = []
        for name, _ in get_action_names_and_types(self.node):
            client = self._pubs.get(("cancel", name))
            if client is None:
                client = self._pubs[("cancel", name)] = self.node.create_client(CancelGoal, f"{name}/_action/cancel_goal")
            if client.service_is_ready():
                pending.append(client.call_async(CancelGoal.Request()))
        _wait(lambda: all(f.done() for f in pending), timeout)

    def stop_base(self) -> None:
        """A zero velocity on cmd_vel, where something drives on it: a base the last test left moving stops."""
        from .ros2_discover import CMD_VEL_TYPES, _topic

        topic = _topic(self.namespace, "cmd_vel")
        types = dict(self.node.get_topic_names_and_types()).get(topic, [])
        if not self.node.count_subscribers(topic) or not types:
            return
        if types[0] == CMD_VEL_TYPES[1]:
            from geometry_msgs.msg import TwistStamped as T
        else:
            from geometry_msgs.msg import Twist as T
        key = ("cmd_vel", T.__name__)
        if key not in self._pubs:
            self._pubs[key] = self.node.create_publisher(T, topic, 10)
        self._pubs[key].publish(T())

    def relocalize(self, pose, timeout: float = 10.0) -> None:
        """Tell AMCL the robot is at ``pose`` again (RViz's "2D pose estimate"), clear Nav2's costmaps of what the
        last test's sensors marked, and wait until AMCL's estimate is back there."""
        from geometry_msgs.msg import PoseWithCovarianceStamped

        key = ("initialpose",)
        if key not in self._pubs:
            self._pubs[key] = self.node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = pose.header.frame_id or "map"
        msg.pose.pose = pose.pose.pose
        # Tight: the robot is exactly where it was put back. The pose kept at the start carries the spread of the first
        # estimate (RViz's 0.25 m^2), which AMCL narrows only as the robot moves: re-sent, every test started unsure.
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.0025  # (5 cm)^2
        cov[35] = 0.0025  # (0.05 rad)^2
        msg.pose.covariance = cov
        target = np.array([pose.pose.pose.position.x, pose.pose.pose.position.y])
        before = self.newest_map_to_odom()
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self._pubs[key].publish(msg)  # again until AMCL has it: it can miss a message sent as it re-plans
            now = self.amcl_pose(timeout=1.0)
            if now is not None and np.linalg.norm(np.array([now.pose.pose.position.x, now.pose.pose.position.y]) - target) < 0.05:
                break
        else:
            raise StackError("AMCL did not take the pose the stack started from")
        self._wait_map_to_odom(before, timeout=timeout)
        self._clear_costmaps()

    def _wait_map_to_odom(self, before: float | None, timeout: float) -> None:
        """Until a lookup of map -> odom at the current time gives AMCL's new transform. AMCL stamps it ahead of the
        clock (its transform_tolerance, 1 s in Nav2's defaults), so for that long a lookup still found the old one, or
        interpolated between old and new across the jump the reset made; a goal planned then went nowhere near the robot
        (Nav2's controller: "Resulting plan has 0 poses in it", FollowPath INVALID_PATH). ``before``: the newest stamp
        AMCL had published before the re-seed."""
        from rclpy.qos import qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from tf2_msgs.msg import TFMessage

        new: list = []
        clock: list = []

        def on_tf(m):
            for t in m.transforms:
                if t.header.frame_id.lstrip("/") == "map" and t.child_frame_id.lstrip("/") == "odom":
                    stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
                    if before is None or stamp > before:
                        new.append(stamp)

        subs = [
            self.node.create_subscription(TFMessage, "/tf", on_tf, 100),
            self.node.create_subscription(
                Clock, "/clock", lambda m: clock.append(m.clock.sec + m.clock.nanosec * 1e-9), qos_profile_sensor_data
            ),
        ]
        try:
            if not _wait(lambda: bool(new) and bool(clock) and clock[-1] >= new[0], timeout):
                raise StackError("AMCL's new map -> odom transform did not come into effect after it was re-seeded")
        finally:
            for sub in subs:
                self.node.destroy_subscription(sub)

    def newest_map_to_odom(self, timeout: float = 1.0) -> float | None:
        """The newest stamp of a map -> odom transform heard within ``timeout`` (None: none)."""
        from tf2_msgs.msg import TFMessage

        stamps: list = []

        def on_tf(m):
            stamps.extend(
                t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
                for t in m.transforms
                if t.header.frame_id.lstrip("/") == "map" and t.child_frame_id.lstrip("/") == "odom"
            )

        sub = self.node.create_subscription(TFMessage, "/tf", on_tf, 100)
        try:
            _wait(lambda: bool(stamps), timeout)
        finally:
            self.node.destroy_subscription(sub)
        return max(stamps) if stamps else None

    def _clear_costmaps(self, timeout: float = 3.0) -> None:
        try:
            from nav2_msgs.srv import ClearEntireCostmap
        except ImportError:
            return
        pending = []
        for name, types in self.node.get_service_names_and_types():
            # The local costmaps only: a rolling window of what the sensors saw, which the last test's run filled. Clearing
            # the global one wiped its static map layer too, until its next update: a goal planned in that moment went
            # through a wall, and the controller refused the path (FollowPath INVALID_PATH).
            if "nav2_msgs/srv/ClearEntireCostmap" in types and "local" in name:
                client = self._pubs.get(("clear", name)) or self.node.create_client(ClearEntireCostmap, name)
                self._pubs[("clear", name)] = client
                if client.service_is_ready():
                    pending.append(client.call_async(ClearEntireCostmap.Request()))
        _wait(lambda: all(f.done() for f in pending), timeout)

    def _action(self, kind, name: str):
        if name not in self._actions:
            from rclpy.action import ActionClient

            self._actions[name] = ActionClient(self.node, kind, name)
        return self._actions[name]

    def drive_home(self, namespace: str, snap: Snapshot, timeout: float = 30.0) -> None:
        """Each arm and gripper controller's joints back to where they were when the stack came up."""
        # Only joints a controller drives can be driven back: a wheel that turned is where its base drove, not a
        # joint to undo (the base itself is put back with Gazebo's models).
        driven = {j for c in snap.controllers if c.interface is not None for j in c.joints}
        now = self.joints(timeout=3.0)
        far = {j: q for j, q in snap.joints.items() if j in driven and j in now and abs(now[j] - q) > _JOINT_TOL}
        if not far:
            return
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import JointTrajectoryPoint

        results = []
        for c in snap.controllers:
            joints = [j for j in c.joints if j in snap.joints]
            if not joints or not any(j in far for j in joints) or c.interface is None:
                continue
            part, kind = c.interface
            base = f"{namespace.rstrip('/')}/{c.name}"
            target = [snap.joints[j] for j in joints]
            if kind == "trajectory":
                span = max(abs(now[j] - snap.joints[j]) for j in joints)
                sec, nanosec = divmod(round(max(0.3, span / _RESET_SPEED) * 1e9), 1_000_000_000)
                goal = FollowJointTrajectory.Goal()
                goal.trajectory.joint_names = joints
                goal.trajectory.points = [JointTrajectoryPoint(positions=target, time_from_start=Duration(sec=sec, nanosec=nanosec))]
                results.append(self._goal(FollowJointTrajectory, f"{base}/follow_joint_trajectory", goal, timeout))
            elif kind == "position":
                if base not in self._pubs:
                    self._pubs[base] = self.node.create_publisher(Float64MultiArray, f"{base}/commands", 10)
                self._pubs[base].publish(Float64MultiArray(data=target))
            elif kind == "action" and part == "gripper":
                goal = GripperCommand.Goal()
                goal.command.position = target[0]
                results.append(self._goal(GripperCommand, f"{base}/gripper_cmd", goal, timeout))
        end = time.monotonic() + timeout
        for r in results:
            r(end)
        while time.monotonic() < end:
            now = self.joints()
            if all(abs(now.get(j, q) - q) <= 5 * _JOINT_TOL for j, q in far.items()):
                return
            time.sleep(0.02)
        off = {j: round(now.get(j, float("nan")) - q, 3) for j, q in far.items()}
        raise StackError(f"joints did not return to where the stack started: {off}")

    def _goal(self, kind, name: str, goal, timeout: float):
        """Send ``goal``; returns a function that waits (until a deadline) for its result."""
        client = self._action(kind, name)
        if not client.wait_for_server(timeout_sec=timeout):
            raise StackError(f"{name} no longer answers")
        sent = client.send_goal_async(goal)

        def wait(end: float) -> None:
            if not _wait(sent.done, end - time.monotonic()):
                raise StackError(f"{name} did not take the reset goal")
            h = sent.result()
            if not h.accepted:
                raise StackError(f"{name} refused the reset goal")
            res = h.get_result_async()
            if not _wait(res.done, end - time.monotonic()):
                raise StackError(f"{name} did not finish the reset goal")

        return wait

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._ex.shutdown(timeout_sec=1.0)
        with contextlib.suppress(Exception):
            self.node.destroy_node()
        with contextlib.suppress(Exception):
            self._ctx.try_shutdown()
        self._thread.join(timeout=2.0)


def _wait(done, timeout: float) -> bool:
    end = time.monotonic() + max(0.0, timeout)
    while not done():
        if time.monotonic() >= end:
            return False
        time.sleep(0.005)
    return True
