"""Waits that must end: a stalled Gazebo request helper, a ROS 2 node or action client still in use at teardown.
None needs ROS 2 or Gazebo: each is given something that never finishes."""

import subprocess
import sys
import threading
import time

import pytest

from robowright import gazebo


class _Req:
    """Just enough of a protobuf request for _Requests to send it."""

    class DESCRIPTOR:
        full_name = "gz.msgs.Empty"

    def SerializeToString(self):  # noqa: N802 - protobuf's name
        return b""


def test_a_stalled_gazebo_request_helper_times_out_and_is_replaced(monkeypatch):
    monkeypatch.setattr(gazebo, "_HELPER_SLACK", 0.5)
    r = gazebo._Requests.__new__(gazebo._Requests)
    r._lock = threading.Lock()
    r._proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    t = time.monotonic()
    with pytest.raises(TimeoutError, match="did not answer /world/w/set_pose"):
        r("/world/w/set_pose", _Req(), _Req, timeout_ms=100)
    assert time.monotonic() - t < 5
    assert r._proc.poll() is not None  # killed: _requests() starts a new one for the next request


class _Stuck:
    """A node (and its action client) whose destruction never returns, as rclpy's does while its handle is in use."""

    def __init__(self):
        self._node = self

    def get_name(self):
        return "stuck"

    def destroy_node(self):
        time.sleep(60)

    destroy = destroy_node


def test_a_node_still_in_use_does_not_hang_teardown():
    pytest.importorskip("rclpy")  # ros2_graph imports it; the gate runs this in its ROS 2 step
    from robowright.ros2_graph import destroy_action_clients, destroy_node

    t = time.monotonic()
    with pytest.warns(RuntimeWarning, match="'stuck' was still in use"):
        assert destroy_node(_Stuck(), timeout=0.2) is False
    with pytest.warns(RuntimeWarning, match="an action client of 'stuck'"):
        destroy_action_clients([_Stuck(), None], timeout=0.2)
    assert time.monotonic() - t < 5
