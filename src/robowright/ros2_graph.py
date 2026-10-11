"""The ROS 2 graph as robowright sees it: topics a test can watch, and TF frames it can read.

``world.ros`` (on a ROS 2 world) hands out:

* ``world.ros.topic("/scan")``: a :class:`TopicHandle`, a live view of one topic. Its type comes
  from the graph; ``latest`` is the newest message, ``count`` how many have arrived, and ``hz``
  the rate over the last :data:`WINDOW` seconds of the world's clock.
* ``world.ros.frame("base_scan", "map")``: a :class:`FrameSubject`, a TF frame's pose in another
  frame, which the pose matchers in :mod:`robowright.expect` read like any other subject.

Subscriptions use the sensor-data profile (best effort), which receives from both best-effort
and reliable publishers, so a test does not need to know which a driver chose.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from rclpy.qos import qos_profile_sensor_data

from .locators import Subject

WINDOW = 2.0  # seconds of the world's clock a topic's rate is measured over


class TopicHandle(Subject):
    """One topic in the ROS graph. A query, like a locator: each read is the current value."""

    def __init__(self, graph: RosGraph, name: str, type_name: str, msg_cls):
        self._graph = graph
        self.name = name
        self.type = type_name
        self.count = 0
        self.latest = None
        self._arrivals: deque[float] = deque()  # world-clock times of recent messages
        self._lock = graph.lock
        self._sub = graph.node.create_subscription(msg_cls, name, self._on_message, qos_profile_sensor_data)

    def _on_message(self, msg) -> None:
        t = self._graph.clock()
        with self._lock:
            self.count += 1
            self.latest = msg
            self._arrivals.append(t)
            while self._arrivals and t - self._arrivals[0] > 2 * WINDOW:
                self._arrivals.popleft()

    @property
    def hz(self) -> float:
        """Messages per second over the last ``WINDOW`` s of the world's clock (0 until two arrived in it)."""
        now = self._graph.clock()
        with self._lock:
            times = [t for t in self._arrivals if now - t <= WINDOW]
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    @property
    def position(self):
        raise AttributeError(f"{self.name} is a topic, not a place: its messages are its data (see .latest)")


class FrameSubject(Subject):
    """A TF frame's pose in another frame, read live."""

    def __init__(self, graph: RosGraph, child: str, parent: str):
        self._graph = graph
        self.child = child
        self.parent = parent
        self.name = f"{child} in {parent}"

    def pose(self):
        return self._graph.backend.frame_pose(self.parent, self.child)

    @property
    def position(self) -> np.ndarray:
        return self.pose()[0]

    @property
    def quaternion(self) -> np.ndarray:
        return self.pose()[1]

    @property
    def up_axis(self) -> np.ndarray:
        w, x, y, z = self.quaternion
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    @property
    def yaw(self) -> float:
        w, x, y, z = self.quaternion
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class RosGraph:
    """What a ROS 2 backend's node can see: its topics and its TF frames."""

    def __init__(self, backend):
        self.backend = backend
        self.node = backend.node
        self.lock = backend._lock
        self._topics: dict[str, TopicHandle] = {}

    def clock(self) -> float:
        return self.backend.time

    def topics(self) -> dict[str, str]:
        """Every topic in the graph, with its type."""
        return {n: t[0] for n, t in self.node.get_topic_names_and_types() if t}

    def topic(self, name: str) -> TopicHandle:
        """The topic ``name`` (absolute, or under the ``[ros2] namespace``), subscribed on first use."""
        full = self.backend.topic_name(name)
        if full in self._topics:
            return self._topics[full]
        found = self.topics()
        if full not in found:
            raise LookupError(f"no topic {full!r} in the ROS graph; it has {sorted(found)}")
        from rosidl_runtime_py.utilities import get_message

        type_name = found[full]
        handle = TopicHandle(self, full, type_name, get_message(type_name))
        self._topics[full] = handle
        return handle

    def frame(self, child: str, parent: str) -> FrameSubject:
        """``child``'s pose in ``parent``, from TF (the frames must exist when the pose is read)."""
        return FrameSubject(self, child, parent)
