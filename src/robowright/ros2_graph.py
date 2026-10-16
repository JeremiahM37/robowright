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

    # -- acting on the graph, as Playwright's ``request`` fixture calls an API directly ------
    # A test can set the stage, or poke the stack, without rclpy boilerplate. Types come from the
    # graph, a message may be given as a dict of its fields, and every wait is in the world's time
    # (the world steps while it waits, as expect does). Each call is in the trace's timeline.

    def publish(self, topic: str, msg, *, type: str | None = None) -> None:
        """Publish ``msg`` (a message, or a dict of its fields) on ``topic``. The type is the
        message's own, else ``type``, else the one the graph has for the topic."""
        full = self.backend.topic_name(topic)
        if isinstance(msg, dict) or msg is None:
            type_name = type or self.topics().get(full)
            if type_name is None:
                raise LookupError(f"no topic {full!r} in the ROS graph to take its type from: pass type='pkg/msg/Name'")
            msg = _fill(_cls("msg", type_name), msg)
        self._take_arm(full)
        key = (full, _type_of(msg))
        pubs = self.__dict__.setdefault("_pubs", {})
        if key not in pubs:
            pubs[key] = self.node.create_publisher(msg.__class__, full, 10)
        pubs[key].publish(msg)
        self._record("publish", full, msg)

    def call(self, service: str, request=None, *, timeout: float = 10.0, type: str | None = None):
        """Call ``service`` with ``request`` (a request, a dict of its fields, or None for an empty
        one) and return the response, waiting up to ``timeout`` seconds of the world's time."""
        full = self.backend.topic_name(service)
        type_name = type or dict((n, t[0]) for n, t in self.node.get_service_names_and_types() if t).get(full)
        if type_name is None:
            raise LookupError(
                f"no service {full!r} in the ROS graph; it has {sorted(n for n, _ in self.node.get_service_names_and_types())}"
            )
        srv = _cls("srv", type_name)
        req = request if request is not None and not isinstance(request, dict) else _fill(srv.Request, request)
        clients = self.__dict__.setdefault("_clients", {})
        client = clients.get(full) or clients.setdefault(full, self.node.create_client(srv, full))
        self._wait(client.service_is_ready, timeout, f"service {full} to be ready")
        future = client.call_async(req)
        self._wait(future.done, timeout, f"a response from {full}")
        self._record("call", full, req, timeout=timeout)
        return future.result()

    def send_goal(self, action: str, goal=None, *, timeout: float = 60.0, type: str | None = None, feedback=None) -> GoalResult:
        """Send ``goal`` (a goal, a dict of its fields, or None) to ``action``, wait up to ``timeout``
        seconds of the world's time for its result, and return a :class:`GoalResult` (``.status``:
        "succeeded", "aborted", "canceled" or "rejected"; ``.result``). ``feedback(msg)`` is called
        with each feedback message. A goal still running at the timeout is cancelled."""
        from rclpy.action import ActionClient, get_action_names_and_types

        full = self.backend.topic_name(action)
        found = {n: t[0] for n, t in get_action_names_and_types(self.node) if t}
        type_name = type or found.get(full)
        if type_name is None:
            raise LookupError(f"no action {full!r} in the ROS graph; it has {sorted(found)}")
        act = _cls("action", type_name)
        g = goal if goal is not None and not isinstance(goal, dict) else _fill(act.Goal, goal)
        clients = self.__dict__.setdefault("_action_clients", {})
        client = clients.get(full) or clients.setdefault(full, ActionClient(self.node, act, full))
        self._wait(client.server_is_ready, timeout, f"action server {full} to be ready")
        self._take_arm(full)
        ev = self._record("send_goal", full, g, timeout=timeout)
        sent = client.send_goal_async(g, feedback_callback=(lambda m: feedback(m.feedback)) if feedback else None)
        self._wait(sent.done, timeout, f"{full} to accept or reject the goal")
        handle = sent.result()
        if not handle.accepted:
            return _outcome(ev, GoalResult("rejected", None))
        done = handle.get_result_async()
        if not self._wait(done.done, timeout, None):
            handle.cancel_goal_async()
            _outcome(ev, GoalResult("timed out", None))
            raise TimeoutError(f"{full}: no result within {timeout} s of simulated time (the goal was cancelled)")
        res = done.result()
        return _outcome(ev, GoalResult(_STATUS.get(res.status, f"status {res.status}"), res.result))

    def _take_arm(self, name: str) -> None:
        """A command on the channel robowright drives the arm on: robowright lets go of the arm (its hold, sent
        every step, would replace the test's goal) until it moves the arm itself again."""
        channels = getattr(self.backend, "arm_channels", None)
        if channels is not None and name in channels():
            self.backend.yield_arm()

    def _wait(self, ready, timeout: float, what: str | None) -> bool:
        world = getattr(self, "world", None)
        if world is not None:
            ok = world.run_until(ready, timeout=timeout)
        else:  # no world to step (a graph used on its own): wall time
            import time

            end = time.monotonic() + timeout
            while not ready() and time.monotonic() < end:
                time.sleep(0.01)
            ok = ready()
        if not ok and what:
            raise TimeoutError(f"waited {timeout} s of simulated time for {what}")
        return ok

    def _record(self, kind: str, name: str, msg, **extra):
        """The request in the trace: its fields (so codegen writes it again as a world.ros call) and a short text."""
        world = getattr(self, "world", None)
        if world is None or not world.trace:
            return None
        args = {"name": name, "type": _type_of(msg), "fields": _fields(msg), "message": _short(msg), **extra}
        return world.trace.event("edit", f"ros_{kind}", args)


class GoalResult:
    """What an action goal came to: ``status`` ("succeeded", "aborted", "canceled", "rejected") and ``result``."""

    def __init__(self, status: str, result):
        self.status, self.result = status, result

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"

    def __repr__(self) -> str:
        return f"GoalResult({self.status!r}, {self.result!r})"


_STATUS = {4: "succeeded", 5: "canceled", 6: "aborted"}  # action_msgs/GoalStatus


def _cls(kind: str, type_name: str):
    from rosidl_runtime_py.utilities import get_action, get_message, get_service

    return {"msg": get_message, "srv": get_service, "action": get_action}[kind](type_name)


def _fill(cls, fields: dict | None):
    msg = cls()
    if fields:
        from rosidl_runtime_py.set_message import set_message_fields

        set_message_fields(msg, fields)
    return msg


def _type_of(msg) -> str:
    mod = msg.__class__.__module__.split(".")  # e.g. std_msgs.msg._string
    return f"{mod[0]}/{mod[1]}/{msg.__class__.__name__}"


def _outcome(event, result: GoalResult) -> GoalResult:
    if event is not None:
        event.args["status"] = result.status
    return result


def _fields(msg) -> dict:
    """A message as plain data (JSON for the trace, a dict for set_message_fields), with only the fields that
    differ from a new message of its type, so a regenerated call says what was set and nothing else."""
    import json

    from rosidl_runtime_py.convert import message_to_ordereddict

    def plain(m):
        return json.loads(json.dumps(message_to_ordereddict(m), default=list))

    return _changed(plain(msg), plain(msg.__class__()))


def _changed(d: dict, default: dict) -> dict:
    out = {}
    for k, v in d.items():
        base = default.get(k)
        if isinstance(v, dict) and isinstance(base, dict):
            sub = _changed(v, base)
            if sub:
                out[k] = sub
        elif v != base:
            # Messages in a list have no default to compare with; an empty sequence in one is always its default.
            out[k] = [_drop_empty(x) if isinstance(x, dict) else x for x in v] if isinstance(v, list) else v
    return out


def _drop_empty(d: dict) -> dict:
    return {k: (_drop_empty(v) if isinstance(v, dict) else v) for k, v in d.items() if v != []}


def _short(msg, limit: int = 300) -> str:
    text = str(msg)
    return text if len(text) <= limit else text[:limit] + "..."
