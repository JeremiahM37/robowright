"""Routes: change what the code under test receives or sends on a topic, as Playwright's
``page.route`` changes what a page receives from the network::

    with ros2.run("ros2 launch my_robot pick.launch.py", route={"/scan": rw.route.cut(at=5.0, duration=2.0)}):
        expect(robot).always.to_have_no_collisions()   # does the stack stop when its lidar goes quiet?

Every node the command starts has the topic remapped to a private name (``ros2 launch``: through
a wrapping launch file; any other node or script: ``--ros-args -r`` appended, which every ROS 2
node reads), and robowright relays between the two, applying the route to each message. A
command whose nodes ignore those arguments would read the real topic and bypass the route, so a
run where nothing subscribed to the routed side fails and says so. The direction is found from
the graph:

* the topic is published outside the command (the simulator, a driver): the route changes what
  your nodes *receive*;
* it is published by your nodes: the route changes what they *send* (their commands to the robot).

A topic published both inside and outside the command cannot be routed (the relay cannot sit
between nodes that are remapped together), and robowright says so rather than routing nothing.

The routes, all in simulated time:

* ``rw.route.abort()``: no message gets through;
* ``rw.route.drop(p)``: each message is lost with probability ``p`` (seeded from the world's seed);
* ``rw.route.delay(seconds)``: every message arrives ``seconds`` late;
* ``rw.route.cut(at, duration=None)``: the topic goes quiet from ``at`` for ``duration``
  (for the rest of the run when None);
* ``rw.route.throttle(hz)``: at most ``hz`` messages a second get through;
* any function ``fn(msg) -> msg | None``: a changed message, or None to drop it (Playwright's
  ``route.fulfill``).

Routes chain: ``rw.route.delay(0.2) | rw.route.drop(0.1)``. A route counts what it passed,
dropped and delayed (``route.stats``), and the counts are written to the trace when the run ends.
"""

from __future__ import annotations

import heapq
import itertools
import threading
from dataclasses import dataclass, field

import numpy as np

_PRIVATE = "/robowright/routed"


@dataclass
class Route:
    """What happens to one message: ``handle(msg, now, rng)`` returns ``(delay, msg)`` pairs to deliver."""

    stats: dict = field(default_factory=lambda: {"passed": 0, "dropped": 0, "delayed": 0, "changed": 0})

    def handle(self, msg, now: float, rng) -> list:
        return [(0.0, msg)]

    def describe(self) -> str:
        return type(self).__name__.lower()

    def __or__(self, other) -> Route:
        return _Chain(parts=[self, as_route(other)])

    def _count(self, out: list, msg) -> list:
        if not out:
            self.stats["dropped"] += 1
        for d, m in out:
            self.stats["passed"] += 1
            self.stats["delayed"] += d > 0
            self.stats["changed"] += m is not msg
        return out


@dataclass
class _Abort(Route):
    def handle(self, msg, now, rng):
        return []


@dataclass
class _Drop(Route):
    p: float = 0.1

    def handle(self, msg, now, rng):
        return [] if rng.random() < self.p else [(0.0, msg)]

    def describe(self):
        return f"drop({self.p})"


@dataclass
class _Delay(Route):
    seconds: float = 0.1

    def handle(self, msg, now, rng):
        return [(self.seconds, msg)]

    def describe(self):
        return f"delay({self.seconds})"


@dataclass
class _Cut(Route):
    at: float = 0.0
    duration: float | None = None

    def handle(self, msg, now, rng):
        quiet = now >= self.at and (self.duration is None or now < self.at + self.duration)
        return [] if quiet else [(0.0, msg)]

    def describe(self):
        return f"cut(at={self.at}, duration={self.duration})"


@dataclass
class _Throttle(Route):
    hz: float = 10.0
    _last: float | None = None

    def handle(self, msg, now, rng):
        if self._last is not None and now - self._last < 1.0 / self.hz - 1e-9:
            return []
        self._last = now
        return [(0.0, msg)]

    def describe(self):
        return f"throttle({self.hz})"


@dataclass
class _Fn(Route):
    fn: object = None

    def handle(self, msg, now, rng):
        out = self.fn(msg)
        return [] if out is None else [(0.0, out)]

    def describe(self):
        return getattr(self.fn, "__name__", "function")


@dataclass
class _Chain(Route):
    parts: list = field(default_factory=list)

    def handle(self, msg, now, rng):
        out = [(0.0, msg)]
        for part in self.parts:
            out = [(d + d2, m2) for d, m in out for d2, m2 in part.handle(m, now, rng)]
        return out

    def describe(self):
        return " | ".join(p.describe() for p in self.parts)

    def __or__(self, other):
        return _Chain(parts=[*self.parts, as_route(other)])


def as_route(r) -> Route:
    if isinstance(r, Route):
        return r
    if callable(r):
        return _Fn(fn=r)
    raise TypeError(f"a route is rw.route.abort/drop/delay/cut/throttle(...) or a function msg -> msg | None, not {r!r}")


class _Namespace:
    """``rw.route``: the routes a test can put on a topic (see :mod:`robowright.routes`)."""

    @staticmethod
    def abort() -> Route:
        return _Abort()

    @staticmethod
    def drop(p: float) -> Route:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"drop(p): p is a probability, got {p}")
        return _Drop(p=p)

    @staticmethod
    def delay(seconds: float) -> Route:
        if seconds < 0:
            raise ValueError(f"delay(seconds): got {seconds}")
        return _Delay(seconds=seconds)

    @staticmethod
    def cut(at: float = 0.0, duration: float | None = None) -> Route:
        return _Cut(at=at, duration=duration)

    @staticmethod
    def throttle(hz: float) -> Route:
        if hz <= 0:
            raise ValueError(f"throttle(hz): got {hz}")
        return _Throttle(hz=hz)

    @staticmethod
    def fulfill(fn) -> Route:
        return _Fn(fn=fn)


route = _Namespace()


def private_name(topic: str) -> str:
    """The name ``topic`` is remapped to inside the command."""
    return _PRIVATE + "/" + topic.strip("/")


def remap_command(args: list, topics) -> tuple[list, str | None]:
    """``args`` with every topic in ``topics`` remapped to its private name in every node it starts.

    ``ros2 launch``: the launch file is wrapped in one that applies the remaps to everything it
    includes. Anything else (``ros2 run``, a node's executable, a Python script that calls
    ``rclpy.init()``): remapping arguments are appended, which every ROS 2 node reads. Returns
    (new args, wrapper launch file text or None)."""
    pairs = [(t if t.startswith("/") else "/" + t, private_name(t)) for t in topics]
    if len(args) >= 3 and args[0] == "ros2" and args[1] == "launch":
        target, launch_args = _launch_target(args[2:])
        return ["ros2", "launch", "{WRAPPER}"], _wrapper(target, launch_args, pairs)
    return [*args, "--ros-args", *[x for a, b in pairs for x in ("-r", f"{a}:={b}")]], None


def _launch_target(rest: list) -> tuple[tuple, list]:
    """``ros2 launch`` arguments: ((package, file) or (path,), [name:=value, ...]). Options are not supported."""
    opts = [a for a in rest if a.startswith("-")]
    if opts:
        raise ValueError(f"route= with ros2 launch options is not supported: {opts}")
    pos = [a for a in rest if ":=" not in a]
    kv = [a for a in rest if ":=" in a]
    if len(pos) == 1:
        return (pos[0],), kv
    if len(pos) == 2:
        return (pos[0], pos[1]), kv
    raise ValueError(f"ros2 launch: expected a package and launch file, or a path, got {pos}")


def _wrapper(target: tuple, launch_args: list, pairs: list) -> str:
    if len(target) == 2:
        locate = (
            "from ament_index_python.packages import get_package_share_directory\n"
            "import pathlib\n"
            f"_share = pathlib.Path(get_package_share_directory({target[0]!r}))\n"
            f"_found = sorted(_share.rglob({target[1]!r}))\n"
            f"if not _found:\n    raise RuntimeError('launch file {target[1]} not found in package {target[0]}')\n"
            "_path = str(_found[0])\n"
        )
    else:
        locate = f"_path = {target[0]!r}\n"
    largs = dict(a.split(":=", 1) for a in launch_args)
    remaps = "".join(f"            SetRemap(src={a!r}, dst={b!r}),\n" for a, b in pairs)
    return (
        "# Written by robowright: the launch file under test, with topics routed through robowright.\n"
        + locate
        + "from launch import LaunchDescription\n"
        "from launch.actions import GroupAction, IncludeLaunchDescription\n"
        "from launch.launch_description_sources import AnyLaunchDescriptionSource\n"
        "from launch_ros.actions import SetRemap\n\n\n"
        "def generate_launch_description():\n"
        "    return LaunchDescription([\n"
        "        GroupAction([\n"
        f"{remaps}"
        f"            IncludeLaunchDescription(AnyLaunchDescriptionSource(_path), launch_arguments={largs!r}.items()),\n"
        "        ])\n"
        "    ])\n"
    )


class Relay:
    """Carries one routed topic between its real and private names, through a route, in sim time.

    Made before the command starts; it finds the topic's type and direction from the graph once
    someone publishes it, then relays. ``poll()`` (each world step) does that discovery and
    delivers delayed messages whose time has come."""

    def __init__(self, node, topic: str, route_: Route, clock, seed: int, callback_group=None):
        self.node, self.route, self.clock = node, route_, clock
        self.topic = topic if topic.startswith("/") else "/" + topic
        self.private = private_name(topic)
        self.direction: str | None = None  # "in" (to the code under test) or "out" (from it)
        self.type_name: str | None = None
        self._rng = np.random.default_rng([seed, abs(hash(self.topic)) % (2**31)])
        self._cb = callback_group
        self._pub = self._sub = None
        self._lock = threading.Lock()
        self._queue: list = []  # (due sim time, seq, msg)
        self._seq = itertools.count()
        self.error: str | None = None
        self.consumed = False  # something in the command subscribed to the relayed side

    def _publishers(self, name: str) -> list:
        # Called only before the relay has a publisher of its own, so every publisher found is real (the
        # bridge's own /joint_states and /clock, in the same node as the relay, included).
        return list(self.node.get_publishers_info_by_topic(name))

    def poll(self) -> None:
        if self.direction is None and self.error is None:
            self._discover()
        if self.direction == "in" and not self.consumed:
            self.consumed = self.node.count_subscribers(self.private) > 0
        now = self.clock()
        with self._lock:
            due = []
            while self._queue and self._queue[0][0] <= now + 1e-9:
                due.append(heapq.heappop(self._queue)[2])
        for m in due:
            self._pub.publish(m)

    def _discover(self) -> None:
        outside, inside = self._publishers(self.topic), self._publishers(self.private)
        if outside and inside:
            self.error = (
                f"cannot route {self.topic}: it is published both outside the command "
                f"({', '.join(sorted({p.node_name for p in outside}))}) and by its nodes "
                f"({', '.join(sorted({p.node_name for p in inside}))})"
            )
            return
        if not (outside or inside):
            return
        src, dst, pubs = (self.topic, self.private, outside) if outside else (self.private, self.topic, inside)
        types = dict(self.node.get_topic_names_and_types()).get(src)
        if not types:
            return
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.utilities import get_message

        self.type_name = types[0]
        cls = get_message(self.type_name)
        reliable = all(p.qos_profile.reliability == ReliabilityPolicy.RELIABLE for p in pubs)
        latched = all(p.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for p in pubs)
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE,
        )
        self._pub = self.node.create_publisher(cls, dst, qos)
        self._sub = self.node.create_subscription(cls, src, self._on_msg, qos, callback_group=self._cb)
        self.direction = "in" if outside else "out"

    def _on_msg(self, msg) -> None:
        now = self.clock()
        out = self.route._count(self.route.handle(msg, now, self._rng), msg)
        for d, m in out:
            if d <= 0:
                self._pub.publish(m)
            else:
                with self._lock:
                    heapq.heappush(self._queue, (now + d, next(self._seq), m))

    def summary(self) -> dict:
        return {
            "topic": self.topic,
            "route": self.route.describe(),
            "direction": {"in": "to the code under test", "out": "from the code under test"}.get(self.direction, "never published"),
            "type": self.type_name,
            **self.route.stats,
        }

    def close(self) -> None:
        for e in (self._sub, self._pub):
            if e is not None:
                if e is self._sub:
                    self.node.destroy_subscription(e)
                else:
                    self.node.destroy_publisher(e)
