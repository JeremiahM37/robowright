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
    ):
        self.args = shlex.split(command) if isinstance(command, str) else list(command)
        self.env = dict(env or {})
        self.cwd = cwd
        self.timeout = timeout
        self.namespace = namespace
        # More the stack needs before a test may start, beyond what is read from the graph: a function of this stack
        # returning True, or a reason it is not ready yet (Nav2's lifecycle nodes active, a map loaded, ...).
        self.ready = ready
        self.log_dir = Path(log_dir or tempfile.mkdtemp(prefix="robowright-stack-"))
        self.log = self.log_dir / "stack.log"
        self.proc: subprocess.Popen | None = None
        self.snapshot: Snapshot | None = None
        self.restarts = 0
        self.resets = 0
        self.startup_s = 0.0  # how long the last start took
        self._gz = None
        self.waited: list = []  # how the last start went: (seconds, what it was waiting for)
        self._link = None  # one ROS node for the stack's life: reset() must not pay discovery each time

    # -- starting and stopping -----------------------------------------------------------
    def start(self) -> Ros2Stack:
        if "GZ_PARTITION" not in self.env and "GZ_PARTITION" not in os.environ:
            # Gazebo's transport is per partition: a partition of this stack's own keeps parallel workers' worlds apart,
            # and this process uses it too, so robowright's Gazebo connection finds this stack's world.
            worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
            os.environ["GZ_PARTITION"] = f"robowright-{os.getpid()}-{worker}"
        os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            self.args,
            stdout=self.log.open("a"),
            stderr=subprocess.STDOUT,
            env={k: v for k, v in {**os.environ, **self.env}.items() if v is not None},  # None unsets a variable
            cwd=self.cwd,
            start_new_session=True,
        )
        try:
            self._link = _Link(self.namespace)
            self._wait_ready(t0 + self.timeout)
            self.snapshot = self._take_snapshot()
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
            controllers = link.controller_list() if link.manager() else []
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
        snap = Snapshot(joints=link.joints(timeout=3.0), controllers=[c for c in link.controller_list() if c.active])
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
        self.snapshot = self._take_snapshot()
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
        if link.manager():
            states = {c.name: c.state for c in link.controller_list()}
            lost = [c.name for c in snap.controllers if states.get(c.name) != "active"]
            if lost:
                raise StackError(f"controllers no longer active: {lost}")
        if snap.world is not None:
            gz = self._gazebo(snap.world)
            models = set(_models(gz))
            for m in models - set(snap.models):
                gz.remove(m)
            for m, (pos, quat) in snap.models.items():
                if m not in models:
                    raise StackError(f"Gazebo model {m!r} is gone")
                p, q = gz.pose(m)
                if np.linalg.norm(np.asarray(p) - pos) > _POSE_TOL or abs(abs(float(np.dot(q, quat))) - 1.0) > 1e-6:
                    gz.set_pose(m, pos, quat)
        link.drive_home(self.namespace, snap)


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
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState

        from .ros2_discover import _topic

        self.namespace = namespace
        self._ctx = rclpy.Context()
        rclpy.init(context=self._ctx)
        self.node = rclpy.create_node(f"robowright_stack_{os.getpid()}", context=self._ctx)
        self._lock = threading.Lock()
        self._joints: dict = {}
        self._clock = (None, 0.0)  # (newest /clock stamp, monotonic time it last moved on)
        self.node.create_subscription(JointState, _topic(namespace, "joint_states"), self._on_joints, qos_profile_sensor_data)
        self.node.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data)
        self._manager: str | None = None
        self._list = None
        self._actions: dict = {}
        self._pubs: dict = {}
        self._ex = MultiThreadedExecutor(num_threads=2, context=self._ctx)
        self._ex.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

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

    def _action(self, kind, name: str):
        if name not in self._actions:
            from rclpy.action import ActionClient

            self._actions[name] = ActionClient(self.node, kind, name)
        return self._actions[name]

    def drive_home(self, namespace: str, snap: Snapshot, timeout: float = 30.0) -> None:
        """Each arm and gripper controller's joints back to where they were when the stack came up."""
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import JointTrajectoryPoint

        now = self.joints()
        far = {j: q for j, q in snap.joints.items() if j in now and abs(now[j] - q) > _JOINT_TOL}
        if not far:
            return
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
