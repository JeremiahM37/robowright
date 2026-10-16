"""A stand-in for the code under test in tests/test_routes.py.

route_node.py listen OUT      # writes what it receives on /joint_states (count, first position,
                              # largest lag of a stamp behind /clock) to OUT, as JSON, every 0.1 s
route_node.py command         # publishes arm commands on /forward_position_controller/commands
route_node.py deaf OUT        # as listen, but ignores its command line (so takes no remapping)
"""

import json
import sys

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

argv = remove_ros_args(sys.argv)
mode = argv[1]
rclpy.init(args=[] if mode == "deaf" else sys.argv)
node = rclpy.create_node("route_node")
state = {"count": 0, "first": None, "lag": 0.0, "lags": [], "clock": None}


def on_clock(m):
    state["clock"] = m.clock.sec + m.clock.nanosec * 1e-9


def on_states(m):
    state["count"] += 1
    if state["first"] is None and m.position:
        state["first"] = list(m.position)
    if state["clock"] is not None:
        lag = state["clock"] - (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        state["lag"] = max(state["lag"], lag)
        state["lags"] = (state["lags"] + [round(lag, 4)])[-2000:]


if mode in ("listen", "deaf"):
    out = argv[2]
    node.create_subscription(Clock, "/clock", on_clock, 10)
    node.create_subscription(JointState, "/joint_states", on_states, qos_profile_sensor_data)

    def dump():
        with open(out, "w") as f:
            json.dump({k: v for k, v in state.items() if k != "clock"}, f)

    node.create_timer(0.1, dump)
else:
    pub = node.create_publisher(Float64MultiArray, "/forward_position_controller/commands", 10)
    sent = {"q": None}

    def first_states(m):
        if sent["q"] is None:
            sent["q"] = [p + (0.3 if i == 0 else 0.0) for i, p in enumerate(m.position)]

    node.create_subscription(JointState, "/joint_states", first_states, qos_profile_sensor_data)
    node.create_timer(0.05, lambda: sent["q"] is not None and pub.publish(Float64MultiArray(data=sent["q"])))

try:
    rclpy.spin(node)
except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
    pass
