"""``world.ros.publish`` / ``call`` / ``send_goal``: a test acts on the ROS graph directly, as
Playwright's ``request`` fixture calls an API, with no rclpy boilerplate: types come from the
graph, messages may be dicts, and waits are in the world's time. Against robowright's world served
over ROS 2 (the ``ros2`` fixture). Needs a ROS 2 environment; skipped elsewhere.
"""

import importlib.util
import os

import pytest

from robowright import expect

pytestmark = pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)")
if "PYTEST_XDIST_WORKER" in os.environ:  # a ROS 2 domain of this worker's own
    os.environ["ROS_DOMAIN_ID"] = str(200 + int(os.environ["PYTEST_XDIST_WORKER"].removeprefix("gw")) % 30)
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


def _goal(robot, delta=0.15, joints=None):
    target = [float(v) + delta for v in robot.qpos()[: robot.n_arm]]
    return target, {
        "trajectory": {
            "joint_names": joints or list(robot.model.arm_joints),
            "points": [{"positions": target, "time_from_start": {"sec": 1}}],
        }
    }


def test_send_goal_from_a_dict_moves_the_arm(world, robot, ros2):
    target, goal = _goal(robot)
    seen = []
    r = world.ros.send_goal("/arm_controller/follow_joint_trajectory", goal, timeout=10, feedback=seen.append)
    assert r.ok and r.status == "succeeded", r
    expect(robot).to_have_joint(robot.model.arm_joints[0], target[0], tol=0.03)


def test_a_rejected_goal_says_so(world, robot, ros2):
    _, goal = _goal(robot, joints=["no_such_joint"])
    r = world.ros.send_goal("/arm_controller/follow_joint_trajectory", goal, timeout=10)
    assert r.status == "rejected" and not r.ok


def test_publish_a_dict_on_the_controller_topic(world, robot, ros2):
    start = float(robot.qpos()[0])
    q = [float(v) for v in robot.qpos()]
    q[0] += 0.3
    # the forward controller takes arm joints then the gripper joint, in joint units
    gripper = ros2._grip
    if gripper is not None:
        _, closed, opened = gripper
        q[-1] = closed + q[-1] * (opened - closed)
    world.ros.publish("/forward_position_controller/commands", {"data": q})
    expect(robot).to_satisfy(lambda r: float(r.qpos()[0]) - start > 0.2, "arm moved", timeout=3.0)


def test_call_a_service_with_an_empty_request(world, ros2):
    resp = world.ros.call(f"/{ros2.node.get_name()}/list_parameters", timeout=5)
    assert "use_sim_time" in resp.result.names


def test_an_unknown_name_lists_what_there_is(world, ros2):
    with pytest.raises(LookupError, match=r"no action '/arm_controler/follow_joint_trajectory'.*arm_controller"):
        world.ros.send_goal("/arm_controler/follow_joint_trajectory", {}, timeout=1)
    with pytest.raises(LookupError, match="no service"):
        world.ros.call("/nope", timeout=1)


def test_each_request_is_in_the_trace(world, robot, ros2):
    assert world.trace, "the world fixture records a trace"
    _, goal = _goal(robot)
    world.ros.send_goal("/arm_controller/follow_joint_trajectory", goal, timeout=10)
    world.ros.call(f"/{ros2.node.get_name()}/list_parameters", timeout=5)
    events = {e.name: e.args for e in world.trace.events if e.name.startswith("ros_")}
    assert events["ros_send_goal"]["name"] == "/arm_controller/follow_joint_trajectory"
    assert events["ros_send_goal"]["type"] == "control_msgs/action/FollowJointTrajectory_Goal"
    assert events["ros_call"]["type"] == "rcl_interfaces/srv/ListParameters_Request"


def test_world_ros_is_gone_once_the_bridge_closes(world):
    from robowright.errors import CapabilityError
    from robowright.ros2_bridge import Ros2Bridge

    with Ros2Bridge(world):
        assert world.ros.topics()
    with pytest.raises(CapabilityError, match="ros2 fixture"):
        world.ros  # noqa: B018


def test_a_session_of_requests_becomes_a_test_that_passes(tmp_path):
    """codegen writes each world.ros request again (the same message, by its fields; a goal's outcome asserted),
    inside the bridge that served the world, and the generated test passes on its own."""
    import subprocess
    import sys

    import robowright as rw
    from robowright.codegen import generate
    from robowright.ros2_bridge import Ros2Bridge

    w = rw.launch(robot="so101", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)), name="requests")
    with Ros2Bridge(w) as bridge:
        w.ros.call(f"/{bridge.node.get_name()}/list_parameters", timeout=5)
        _, goal = _goal(w.robot)
        assert w.ros.send_goal("/arm_controller/follow_joint_trajectory", goal, timeout=10).ok
        w.ros.publish("/forward_position_controller/commands", {"data": [float(v) for v in w.robot.qpos()]})
        w.wait(0.2)
    trace = w.close(trace_path=tmp_path / "requests.zip")
    code = generate(trace, test_name="test_requests_again")
    assert "with Ros2Bridge(world" in code and "world.ros.send_goal('/arm_controller/follow_joint_trajectory'" in code
    assert "assert goal.status == 'succeeded', goal" in code
    assert "world.ros.publish('/forward_position_controller/commands'" in code
    (tmp_path / "test_requests_again.py").write_text(code)
    env = {**os.environ, "ROS_DOMAIN_ID": str(int(os.environ.get("ROS_DOMAIN_ID", "0")) + 1)}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path / "test_requests_again.py")],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        env=env,
    )
    assert run.returncode == 0, code + "\n" + run.stdout[-3000:] + run.stderr[-2000:]


def test_an_agent_reads_calls_and_sends_goals_through_the_session(tmp_path):
    """The MCP tools ros2_topic / ros2_call / ros2_send_goal / ros2_publish, through the session they call, and the
    test the session saves repeats them inside the bridge."""
    from robowright.session import Session

    s = Session(trace_dir=tmp_path)
    try:
        s.launch(robot="so101")
        s.serve_ros2()
        out = s.ros_topic("/joint_states")
        assert "sensor_msgs/msg/JointState" in out and "shoulder_pan" in out
        out = s.ros_call(f"/{s.bridge.node.get_name()}/list_parameters", {}, timeout=5)
        assert "use_sim_time" in out
        q = [float(v) + 0.1 for v in s.world.robot.qpos()[: s.world.robot.n_arm]]
        goal = {
            "trajectory": {"joint_names": list(s.world.robot.model.arm_joints), "points": [{"positions": q, "time_from_start": {"sec": 1}}]}
        }
        assert s.ros_send_goal("/arm_controller/follow_joint_trajectory", goal, timeout=10).startswith(
            "/arm_controller/follow_joint_trajectory: succeeded"
        )
        assert "rejected" in s.ros_send_goal("/arm_controller/follow_joint_trajectory", {"trajectory": {"joint_names": ["nope"]}})
        assert s.ros_publish("/forward_position_controller/commands", {"data": list(map(float, s.world.robot.qpos()))}) == (
            "published on /forward_position_controller/commands"
        )
        code = s.generate_test("test_agent_session")
        assert "with Ros2Bridge(world" in code and "assert goal.status == 'rejected', goal" in code
    finally:
        if s.world is not None:
            s.close()
