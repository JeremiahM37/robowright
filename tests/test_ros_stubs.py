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
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):  # not written yet
        return []


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


def test_the_stack_gets_the_tests_action_result_and_each_goal_is_kept(served, tmp_path):
    world, bridge = served
    out = tmp_path / "outcomes.json"

    def result(goal):  # the jaws stop where they were told, and say so
        return {"position": goal.command.position, "reached_goal": True}

    with world.ros.serve_action("/grasp", "control_msgs/action/GripperCommand", result) as stub:
        with bridge.run([sys.executable, str(NODE), "grasp", str(out)]):
            expect.poll(lambda: len(_answers(out)), name="outcomes the node got").to_be_greater_than(2, timeout=30)
        assert stub.count >= 3 and stub.goals[0]["outcome"] == "succeeded"
        assert stub.goals[0]["goal"]["command"]["position"] == pytest.approx(0.01)
    got = _answers(out)
    assert got[0] == {"status": 4, "position": pytest.approx(0.01), "reached": True}  # 4: SUCCEEDED
    assert got[1]["position"] == pytest.approx(0.02)


def test_an_aborted_goal_and_an_outcome_chosen_from_the_goal(served, tmp_path):
    world, bridge = served
    out = tmp_path / "outcomes.json"
    second_fails = lambda goal: "aborted" if goal.command.position > 0.015 else "succeeded"  # noqa: E731
    with world.ros.serve_action("/grasp", "control_msgs/action/GripperCommand", outcome=second_fails):
        with bridge.run([sys.executable, str(NODE), "grasp", str(out)]):
            expect.poll(lambda: len(_answers(out))).to_be_greater_than(1, timeout=30)
    got = _answers(out)
    assert [g["status"] for g in got[:2]] == [4, 6]  # 6: ABORTED


def test_an_error_in_the_action_stub_aborts_the_goal_and_reaches_the_test(served, tmp_path):
    world, bridge = served
    out = tmp_path / "outcomes.json"
    with pytest.raises(RuntimeError, match=r"the stub for /grasp raised while answering a goal: .*KeyError"):
        with world.ros.serve_action("/grasp", "control_msgs/action/GripperCommand", lambda goal: {}["nothing"]):
            with bridge.run([sys.executable, str(NODE), "grasp", str(out)]):
                expect.poll(lambda: len(_answers(out))).to_be_greater_than(0, timeout=30)
    assert _answers(out)[0]["status"] == 6


def test_an_action_already_served_is_refused(served):
    world, _ = served
    with world.ros.serve_action("/grasp", "control_msgs/action/GripperCommand"):
        with pytest.raises(RuntimeError, match="served already"):
            world.ros.serve_action("/grasp", "control_msgs/action/GripperCommand")
    with pytest.raises(ValueError, match="'succeeded' or 'aborted'"):
        world.ros.serve_action("/grasp", "control_msgs/action/GripperCommand", outcome="maybe")


def test_a_stub_can_be_served_again_once_closed(served):
    world, _ = served
    for answer in ("first", "second"):  # the graph still lists the first server for a moment after it closes
        with world.ros.serve("/detect", "std_srvs/srv/Trigger", {"success": True, "message": answer}):
            assert world.ros.call("/detect", type="std_srvs/srv/Trigger", timeout=5).message == answer
    for _ in range(2):
        with world.ros.serve_action("/grasp", "control_msgs/action/GripperCommand"):
            pass
