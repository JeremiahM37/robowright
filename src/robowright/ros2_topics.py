"""What went over the ROS 2 graph during a run, kept in its trace: Playwright's network log, for topics.

For every topic seen while the world ran: its type, the nodes that published and subscribed to it, how many messages
it carried and when (in simulated time, counted in half-second bins), and the last message, as a dict. A test that
failed because a node never published, published too slowly, or stopped, says so in its trace.

Messages are taken raw (bytes) and only the last of each topic is decoded, when the trace is saved; a camera or point
cloud topic is counted and sized but its last message is not kept. ``/clock`` is listed but not subscribed to: it
ticks every physics step, and counting it would only slow the run (``/parameter_events`` likewise).
"""

from __future__ import annotations

import threading
import time

NOT_SUBSCRIBED = ("/clock", "/parameter_events")
BULKY = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/PointCloud2", "sensor_msgs/msg/LaserScan")
_PRIVATE = "/robowright/"  # routed topics' private names (routes.py): the real name is the one listed


class TopicLog:
    """Watches every topic a node can see (see the module). ``poll()`` (each world step) looks at the graph again
    once a second of wall time or a quarter second of simulated time, whichever comes first: a simulation faster than
    real time would otherwise run a whole short test between two looks. ``record()`` is what the trace keeps."""

    def __init__(self, node, clock, callback_group=None, bin_s: float = 0.5, max_items: int = 16):
        self.node, self.clock, self._cb = node, clock, callback_group
        self.bin_s, self.max_items = bin_s, max_items
        self._lock = threading.Lock()
        self._topics: dict[str, dict] = {}
        self._subs: dict = {}
        self._next_look = 0.0
        self._next_sim = float("-inf")
        self._own = node.get_fully_qualified_name()
        self.closed = False

    def poll(self) -> None:
        if self.closed:
            return
        now = self.clock()
        if time.monotonic() < self._next_look and now < self._next_sim:
            return
        self._next_look, self._next_sim = time.monotonic() + 1.0, now + 0.25
        try:
            names = self.node.get_topic_names_and_types()
        except Exception:  # noqa: BLE001 - the node is going away: nothing more to see
            return
        for name, types in names:
            if not types or name.startswith(_PRIVATE):
                continue
            t = self._topics.get(name)
            if t is None:
                t = self._topics[name] = {"type": types[0], "publishers": set(), "subscribers": set()}
                t.update(times=[], bytes=0, last=None)
            t["publishers"] |= self._nodes(self.node.get_publishers_info_by_topic(name))
            # this log's own subscription is not a reader of the topic: robowright is, only if its node reads it too
            t["subscribers"] |= self._nodes(self.node.get_subscriptions_info_by_topic(name), mine=1 if self._subs.get(name) else 0)
            if name not in self._subs and name not in NOT_SUBSCRIBED:
                self._subscribe(name, t)

    def _nodes(self, infos, mine: int = 0) -> set:
        """The nodes behind ``infos``, robowright's own under that name; ``mine``: entries of robowright's to leave out."""
        out = set()
        for i in infos:
            full = f"{i.node_namespace.rstrip('/')}/{i.node_name}"
            if full == self._own:
                if mine:
                    mine -= 1
                    continue
                full = "robowright"
            out.add(full)
        return out

    def _subscribe(self, name: str, t: dict) -> None:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.utilities import get_message

        try:
            cls = get_message(t["type"])
        except (AttributeError, ModuleNotFoundError, ValueError):
            self._subs[name] = None  # a type this environment cannot load: listed, not counted
            t["note"] = "type not installed here: not counted"
            return
        pubs = self.node.get_publishers_info_by_topic(name)
        latched = bool(pubs) and all(p.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for p in pubs)
        # Best effort takes from reliable and best-effort publishers alike; a latched topic needs a latched reader.
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE,
        )
        if latched:
            qos.reliability = ReliabilityPolicy.RELIABLE  # (a latched publisher is reliable; best effort would not match)

        def on_msg(raw: bytes, t=t):
            now = self.clock()
            with self._lock:
                t["times"].append(now)
                t["bytes"] += len(raw)
                t["last"] = raw

        self._subs[name] = (self.node.create_subscription(cls, name, on_msg, qos, raw=True, callback_group=self._cb), cls)

    def record(self) -> list[dict]:
        """Each topic as the trace keeps it, busiest first."""
        self.poll_now()
        out = []
        with self._lock:
            items = [(n, dict(t), list(t["times"])) for n, t in self._topics.items()]
        for name, t, times in items:
            entry = {
                "topic": name,
                "type": t["type"],
                "publishers": sorted(t["publishers"]),
                "subscribers": sorted(t["subscribers"]),
                "count": len(times),
                "bytes": t["bytes"],
            }
            if name in NOT_SUBSCRIBED:
                entry["note"] = "not counted (it ticks every step)" if name == "/clock" else "not counted"
            elif t.get("note"):
                entry["note"] = t["note"]
            if times:
                entry["first"], entry["last_time"] = round(times[0], 3), round(times[-1], 3)
                span = times[-1] - times[0]
                entry["hz"] = round((len(times) - 1) / span, 2) if span > 0 else None
                bins = [0] * (int(times[-1] // self.bin_s) + 1)
                for x in times:
                    bins[int(max(x, 0.0) // self.bin_s)] += 1
                entry["bins"], entry["bin_s"] = bins, self.bin_s
            if t["last"] is not None and t["type"] not in BULKY and self._subs.get(name):
                entry["last_message"] = self._decode(t["last"], self._subs[name][1])
            out.append(entry)
        return sorted(out, key=lambda e: (-e["count"], e["topic"]))

    def poll_now(self) -> None:
        self._next_look, self._next_sim = 0.0, float("-inf")
        self.poll()

    def _decode(self, raw: bytes, cls):
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.convert import message_to_ordereddict

        try:
            return self._trim(message_to_ordereddict(deserialize_message(raw, cls)))
        except Exception as e:  # noqa: BLE001 - a message this environment cannot decode is not a failed test
            return f"<not decoded: {type(e).__name__}>"

    def _trim(self, v):
        if isinstance(v, dict):
            return {k: self._trim(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            head = [self._trim(x) for x in v[: self.max_items]]
            return head + [f"... {len(v) - self.max_items} more"] if len(v) > self.max_items else head
        if isinstance(v, str) and len(v) > 200:
            return v[:200] + f"... ({len(v)} chars)"
        if isinstance(v, bytes):
            return f"<{len(v)} bytes>"
        if isinstance(v, float):
            return float(f"{v:.6g}")  # what a reader of the trace needs, not 17 digits
        return v

    def close(self) -> None:
        self.closed = True
        for entry in self._subs.values():
            if entry is not None:
                try:
                    self.node.destroy_subscription(entry[0])
                except Exception:  # noqa: BLE001 - its node is already gone
                    pass
        self._subs = {}


def summary_lines(topics: list[dict], limit: int = 20) -> list[str]:
    """The trace's text summary of its topics (Trace.summary)."""
    if not topics:
        return []
    lines = ["ros 2 topics:"]
    for e in topics[:limit]:
        rate = f"{e['count']} msgs" + (f", {e['hz']} Hz" if e.get("hz") else "")
        if e.get("count"):
            rate += f", {e['first']:.2f}-{e['last_time']:.2f}s"
        who = f"pub {', '.join(e['publishers']) or 'none'}; sub {', '.join(e['subscribers']) or 'none'}"
        lines.append(f"  {e['topic']} [{e['type']}] {e.get('note') or rate} ({who})")
    if len(topics) > limit:
        lines.append(f"  ... {len(topics) - limit} more")
    quiet = [e["topic"] for e in topics if e["count"] == 0 and "note" not in e and set(e["subscribers"]) - {"robowright"}]
    if quiet:
        lines.append(f"  subscribed to but never published: {', '.join(quiet)}")
    return lines
