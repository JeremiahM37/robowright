"""The ROS 2 topics of a run, kept in its trace (Playwright's network log): who published and subscribed to each
topic, how many messages it carried and when, and the last one. A stand-in for the code under test
(tests/route_node.py) listens to the robot, or commands it, while robowright serves the world over ROS 2."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

import robowright as rw
from robowright.ros2_topics import summary_lines
from robowright.trace import Trace

NODE = Path(__file__).parent / "route_node.py"
HAS_ROS = importlib.util.find_spec("rclpy") is not None
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
ros = pytest.mark.skipif(not HAS_ROS, reason="needs a ROS 2 environment (rclpy)")


def test_the_summary_names_the_busy_and_the_silent_topics():
    topics = [
        {"topic": "/joint_states", "type": "sensor_msgs/msg/JointState", "publishers": ["robowright"], "subscribers": ["/planner"],
         "count": 150, "hz": 50.0, "first": 0.0, "last_time": 2.98, "bytes": 1},
        {"topic": "/clock", "type": "rosgraph_msgs/msg/Clock", "publishers": ["robowright"], "subscribers": [], "count": 0,
         "note": "not counted (it ticks every step)", "bytes": 0},
        {"topic": "/scan", "type": "sensor_msgs/msg/LaserScan", "publishers": [], "subscribers": ["/planner", "robowright"],
         "count": 0, "bytes": 0},
    ]  # fmt: skip
    text = "\n".join(summary_lines(topics))
    assert "/joint_states [sensor_msgs/msg/JointState] 150 msgs, 50.0 Hz, 0.00-2.98s (pub robowright; sub /planner)" in text
    assert "/clock [rosgraph_msgs/msg/Clock] not counted" in text
    assert text.endswith("subscribed to but never published: /scan")


def _run(tmp_path, mode: str, seconds: float = 3.0) -> Trace:
    from robowright.ros2_bridge import Ros2Bridge

    world = rw.launch(robot="so101", backend="mujoco", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    bridge = Ros2Bridge(world)
    try:
        with bridge.run([sys.executable, str(NODE), mode, str(tmp_path / "seen.json")]):
            if mode == "listen":  # until the node is up (it writes what it heard), then as long again
                rw.expect(world).to_satisfy(lambda _w: (tmp_path / "seen.json").exists(), "the node is up", timeout=10.0)
            world.wait(seconds)
    finally:
        bridge.close()  # as the ros2 fixture does: the bridge goes first, then the world saves its trace
    return Trace(world.close())


@ros
def test_the_trace_keeps_what_the_code_under_test_heard(tmp_path):
    tr = _run(tmp_path, "listen")
    topics = {e["topic"]: e for e in tr.meta["ros_topics"]}
    js = topics["/joint_states"]
    assert js["type"] == "sensor_msgs/msg/JointState" and "robowright" in js["publishers"]
    assert any(s.endswith("/route_node") for s in js["subscribers"]), js["subscribers"]
    assert js["count"] > 50 and js["hz"] == pytest.approx(1 / tr.meta["control_dt"], rel=0.2)
    assert sum(js["bins"]) == js["count"] and js["last_time"] <= tr.meta["steps"] * tr.meta["control_dt"] + 0.1
    assert len(js["last_message"]["position"]) == len(tr.meta["joint_names"])
    assert topics["/clock"]["note"].startswith("not counted")
    text = tr.summary()
    assert "ros 2 topics:" in text and "/joint_states [sensor_msgs/msg/JointState]" in text


@ros
def test_the_trace_keeps_what_the_code_under_test_sent(tmp_path):
    tr = _run(tmp_path, "command")
    cmd = {e["topic"]: e for e in tr.meta["ros_topics"]}["/forward_position_controller/commands"]
    assert any(p.endswith("/route_node") for p in cmd["publishers"]) and "robowright" in cmd["subscribers"]
    assert cmd["count"] > 10 and len(cmd["last_message"]["data"]) > 0


@ros
def test_the_trace_keeps_the_nodes_log_lines_at_their_simulated_time(tmp_path):
    tr = _run(tmp_path, "talk", seconds=4.0)
    log = [e for e in tr.meta["ros_log"] if e["node"] == "route_node"]
    ticks = [e for e in log if e["msg"].startswith("tick ")]
    assert len(ticks) >= 4 and all(e["level"] == "INFO" for e in ticks)
    # Stamped on the simulation's clock, as the node logged them: half a second apart. (After the first: a ROS timer's
    # first period can be short, while the node's clock jumps from zero to the simulation's as /clock first arrives.)
    gaps = [b["t"] - a["t"] for a, b in zip(ticks[1:], ticks[2:])]
    assert all(abs(g - 0.5) < 0.06 for g in gaps), gaps
    (warn,) = [e for e in log if e["level"] == "WARN"]
    (err,) = [e for e in log if e["level"] == "ERROR"]
    assert "timed out" in warn["msg"] and "failed to configure" in err["msg"] and warn["t"] < err["t"]
    text = tr.summary()
    assert "ros 2 log:" in text and "ERROR [route_node] failed to configure the controller" in text, text
