"""world.ros.serve: a service the test answers in place of the node that would (Playwright's route.fulfill, for ROS 2).
A stand-in for the code under test (tests/route_node.py client) calls /detect, and the test answers it."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

import robowright as rw
from robowright import expect
from robowright.trace import Trace

NODE = Path(__file__).parent / "route_node.py"
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)")


@pytest.fixture
def served(tmp_path):
    from robowright.ros2_bridge import Ros2Bridge

    world = rw.launch(robot="so101", backend="mujoco", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    bridge = Ros2Bridge(world)
    yield world, bridge
    bridge.close()
    world.close()


def _answers(path: Path) -> list:
    return json.loads(path.read_text()) if path.exists() else []


def test_the_stack_gets_the_tests_answer_and_the_call_is_kept(served, tmp_path):
    world, bridge = served
    out = tmp_path / "answers.json"
    seen = []

    def detect(request):
        seen.append(world.time)
        return {"success": True, "message": f"cube at call {len(seen)}"}

    with world.ros.serve("/detect", "std_srvs/srv/Trigger", detect) as stub:
        with bridge.run([sys.executable, str(NODE), "client", str(out)]):
            expect.poll(lambda: len(_answers(out)), name="answers the node got").to_be_greater_than(2, timeout=30)
        assert stub.count >= 3 and stub.calls[0]["response"] == {"success": True, "message": "cube at call 1"}
        assert stub.calls[0]["t"] is not None and stub.calls[0]["request"] == {}
    got = _answers(out)
    assert got[0] == {"success": True, "message": "cube at call 1"} and got[1]["message"] == "cube at call 2"
    names = [e.name for e in world.trace.events]
    assert "ros_serve" in names and "ros_serve_closed" in names


def test_a_fixed_answer_and_an_empty_one(served, tmp_path):
    world, bridge = served
    out = tmp_path / "answers.json"
    with world.ros.serve("/detect", "std_srvs/srv/Trigger", {"success": False, "message": "no cube"}):
        with bridge.run([sys.executable, str(NODE), "client", str(out)]):
            expect.poll(lambda: len(_answers(out))).to_be_greater_than(0, timeout=30)
    assert _answers(out)[0] == {"success": False, "message": "no cube"}


def test_an_error_in_the_stub_reaches_the_test(served, tmp_path):
    world, bridge = served
    out = tmp_path / "answers.json"
    with pytest.raises(RuntimeError, match=r"the stub for /detect raised .*ZeroDivisionError"):
        with world.ros.serve("/detect", "std_srvs/srv/Trigger", lambda req: 1 / 0):
            with bridge.run([sys.executable, str(NODE), "client", str(out)]):
                expect.poll(lambda: len(_answers(out))).to_be_greater_than(0, timeout=30)
    assert _answers(out)[0] == {"success": False, "message": ""}  # the caller got an empty response, not a hang


def test_a_service_already_served_is_refused(served):
    world, _ = served
    with world.ros.serve("/detect", "std_srvs/srv/Trigger"):
        with pytest.raises(RuntimeError, match="served already"):
            world.ros.serve("/detect", "std_srvs/srv/Trigger")


def test_codegen_says_the_stub_is_not_written_again(served, tmp_path):
    from robowright.codegen import generate

    world, _ = served
    with world.ros.serve("/detect", "std_srvs/srv/Trigger"):
        world.wait(0.1)
    path = world.trace.save(tmp_path / "t.zip")
    assert "the test answered /detect (std_srvs/srv/Trigger) with code of its own" in generate(Trace(path), test_name="test_x")
