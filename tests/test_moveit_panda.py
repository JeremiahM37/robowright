"""robowright as the harness around MoveIt's own Panda demo: nothing of robowright's in the stack.

``moveit_resources_panda_moveit_config``'s ``demo.launch.py`` (``move_group`` with OMPL, ros2_control
on mock hardware, the arm and hand controllers), headless, as shipped (tests/panda_moveit_rig.py).
The code under test is pymoveit2's own examples run as published: ``ex_pose_goal.py``,
``ex_joint_goal.py`` and ``ex_gripper.py``. robowright loads the description the stack publishes,
observes the joint states over ROS 2 and asserts with ``expect(...)``; it commands nothing.

Needs a ROS 2 environment with the Panda's MoveIt and description packages and pymoveit2
(``scripts/ros2_env.sh create``, then ``scripts/ros2_env.sh pymoveit2``); skipped elsewhere.
"""

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect, robots
from robowright.errors import CapabilityError
from robowright.scene import SceneSpec

HERE = Path(__file__).parent


def _rig():
    spec = importlib.util.spec_from_file_location("panda_moveit_rig", HERE / "panda_moveit_rig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rig = _rig()
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)"),
    pytest.mark.skipif(not rig.installed(), reason="needs moveit_resources_panda_moveit_config and _description"),
    pytest.mark.skipif(not (rig.PYMOVEIT2 / "examples").is_dir(), reason="needs pymoveit2 (scripts/ros2_env.sh pymoveit2)"),
]
_worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
# This module's own ROS graph, apart from the UR module's (170-199); DDS cannot use domains above 232
os.environ["ROS_DOMAIN_ID"] = str(200 + (int(_worker.removeprefix("gw")) * 7 + os.getpid()) % 30)
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

# The stack has no /clock: mock hardware moves on the wall clock, so nothing here uses sim time.
SIM_TIME = False
# The end effector pymoveit2 plans for is the SRDF's ``arm`` chain tip, panda_link8 (the flange; the
# hand's own origin, as ``panda_hand`` is fixed to it). The stack publishes no ``panda_hand_tcp``.
BASE, TIP = "panda_link0", "panda_link8"


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    with rig.panda_moveit(workdir=tmp_path_factory.mktemp("panda")) as d:
        yield d


@pytest.fixture
def world(stack):
    # The description the stack publishes, and its TCP at the flange (the frame pymoveit2 plans for),
    # not robowright's default TCP between the fingertips, which MoveIt does not know about.
    model = robots.load(rig.description(), name="panda_ros", gripper=True, base_pos=(0.0, 0.0, 0.0), tcp=(0.0, 0.0, 0.0))
    w = rw.launch(
        SceneSpec(robot=model.name, objects=[]),
        backend="ros2",
        command=False,  # observe only: MoveIt drives the arm
        frame=BASE,
        gripper={"interface": "none"},
        use_sim_time=SIM_TIME,
        settings=rw.Settings(trace="on", trace_dir=str(stack)),
    )
    yield w
    w.close(failed=False)


class _Tool:
    """Where the robot's own TF (robot_state_publisher, from the stack's description) puts the flange."""

    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from tf2_ros import Buffer, TransformListener

        self._ctx = rclpy.Context()
        rclpy.init(context=self._ctx)
        self.node = rclpy.create_node("tool_tf", context=self._ctx, parameter_overrides=[Parameter("use_sim_time", value=SIM_TIME)])
        self.buf = Buffer()
        self._listener = TransformListener(self.buf, self.node)
        self._ex = SingleThreadedExecutor(context=self._ctx)
        self._ex.add_node(self.node)
        from robowright.ros2_bridge import _spin

        threading.Thread(target=_spin, args=(self._ex,), daemon=True).start()

    def pose(self):
        from rclpy.time import Time

        deadline = time.monotonic() + 10
        while True:
            try:
                t = self.buf.lookup_transform(BASE, TIP, Time()).transform
                q = t.rotation
                return np.array([t.translation.x, t.translation.y, t.translation.z]), np.array([q.w, q.x, q.y, q.z])
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    def close(self):
        self._ex.shutdown(timeout_sec=1.0)
        self._ctx.try_shutdown()


def _pymoveit2(example: str, *params: str) -> subprocess.Popen:
    """One of pymoveit2's examples, as published, on the wall clock."""
    cmd = [sys.executable, str(rig.PYMOVEIT2 / "examples" / example), "--ros-args"]
    for param in (*params, f"use_sim_time:={str(SIM_TIME).lower()}", "timeout_sec:=60.0"):
        cmd += ["-p", param]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(rig.PYMOVEIT2), os.environ.get("PYTHONPATH")])))
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)


def _reach(goal) -> subprocess.Popen:
    """pymoveit2's pose-goal example: ``move_group`` plans and executes the flange to ``goal``, pointing
    down (half a turn about x)."""
    xyz = ", ".join(f"{v:.4f}" for v in goal)
    return _pymoveit2("ex_pose_goal.py", f"position:=[{xyz}]", "quat_xyzw:=[1.0, 0.0, 0.0, 0.0]")


def _watch(proc: subprocess.Popen, check) -> None:
    """``check()``, with the client's own account added if it fails: a plan MoveIt could not
    find reads as that, not as an arm that never arrived."""
    try:
        check()
    except AssertionError as e:
        if proc.poll() is not None:
            e.add_note(f"the MoveIt client exited {proc.returncode}:\n{proc.stdout.read()[-1500:]}")
        raise


def test_moveit_moves_the_panda_and_robowright_sees_the_tool_arrive(world):
    r = world.robot
    assert world.model_changes == []  # robowright changes nothing in a robot it reaches over ROS 2
    with pytest.raises(CapabilityError):
        r.arm.move_to(r.tcp.position)  # observe-only: robowright never commands this arm
    tool = _Tool()
    try:
        for goal in ((0.4, 0.2, 0.4), (0.3, -0.2, 0.5), (0.45, 0.0, 0.25)):
            goal = np.array(goal)
            if np.linalg.norm(r.tcp.position - goal) < 0.01:
                continue  # already there (the stack outlives a test): no move to watch
            start = world.time
            reach = _reach(goal)
            # MoveIt's plan, executed by the arm's controller on mock hardware: robowright waits for the
            # arm, by its own kinematics of the description, to settle at the goal.
            _watch(reach, lambda g=goal: expect(r.tcp).to_be_near(g, tol=0.002, timeout=45, hold=0.5))
            out, _ = reach.communicate(timeout=30)
            assert reach.returncode == 0 and "Pose motion completed successfully" in out, out
            assert world.time - start > 0.5  # it moved there, not a jump
            # robowright's model of the arm agrees with the stack's own TF to a fraction of a millimetre
            p, _ = tool.pose()
            assert np.linalg.norm(p - r.tcp.position) < 5e-4, (p, r.tcp.position)
            down = r.kin.fk(r.true_qpos()[: r.n_arm])[:3, :3] @ r.model.derived.tool_axis
            assert down @ (0, 0, -1) > 0.999  # the tool points down, as asked
    finally:
        tool.close()


# The joint angles pymoveit2's ex_joint_goal.py is asked for (within the Panda's limits)
JOINTS = (0.3, -0.5, 0.2, -1.9, 0.1, 1.5, 0.6)


def test_moveit_reaches_the_joint_angles_it_is_asked_for(world):
    r = world.robot
    joints = list(r.model.arm_joints)
    goal = _pymoveit2("ex_joint_goal.py", f"joint_positions:=[{', '.join(map(str, JOINTS))}]")
    _watch(
        goal,
        lambda: expect(r).to_satisfy(
            lambda robot: all(abs(robot.joints[j] - v) <= 0.01 for j, v in zip(joints, JOINTS)),
            "at the joint angles asked for",
            timeout=45,
            hold=0.3,
        ),
    )
    out, _ = goal.communicate(timeout=60)
    assert goal.returncode == 0, out


def test_the_hand_opens_and_closes_as_pymoveit2_says(world):
    r = world.robot
    # The stack's hand controller is ``panda_hand_controller`` (a GripperCommand action); the example's
    # default action name is for another robot.
    # The client is a Python process that starts, finds the action server and then sends its goal:
    # the fingers began to move 0.5 to 2 s after it was launched, in 16 measured runs.
    budget = {"timeout": 20, "hold": 0.3}
    checks = (("open", lambda: expect(r.gripper).to_be_open(**budget)), ("close", lambda: expect(r.gripper).to_be_closed(**budget)))
    for action, check in checks:
        proc = _pymoveit2("ex_gripper.py", f"action:={action}", "gripper_command_action_name:=panda_hand_controller/gripper_cmd")
        _watch(proc, check)
        out, _ = proc.communicate(timeout=60)
        assert proc.returncode == 0, out


def test_a_stack_that_puts_the_tool_in_the_wrong_place_fails(world):
    """The harness has to catch a wrong outcome, not just watch: here the code under test is off by
    4 cm in height (a wrong frame or offset, say). MoveIt succeeds at what it was asked; the test of
    what was meant fails."""
    r = world.robot
    meant = np.array([0.35, 0.1, 0.3])
    if np.linalg.norm(r.tcp.position - meant) < 0.06:
        meant = np.array([0.35, -0.1, 0.3])
    reach = _reach(meant + (0.0, 0.0, 0.04))
    try:
        with pytest.raises(rw.errors.ExpectationError, match="to_be_near"):
            expect(r.tcp).to_be_near(meant, tol=0.002, timeout=15, hold=0.5)
    finally:
        out, _ = reach.communicate(timeout=60)
    assert reach.returncode == 0 and "Pose motion completed successfully" in out, out  # MoveIt did what it was told
    assert abs(np.linalg.norm(r.tcp.position - meant) - 0.04) < 0.002  # and robowright measured the miss


@pytest.fixture
def driven_world(stack):
    """robowright commanding the Panda, through the stack's own arm and hand controllers."""
    model = robots.load(rig.description(), name="panda_ros", gripper=True, base_pos=(0.0, 0.0, 0.0), tcp=(0.0, 0.0, 0.0))
    w = rw.launch(
        SceneSpec(robot=model.name, objects=[]),
        backend="ros2",
        arm={"controller": "panda_arm_controller", "interface": "trajectory"},
        gripper={"controller": "panda_hand_controller", "interface": "action"},
        frame=BASE,
        use_sim_time=SIM_TIME,
        settings=rw.Settings(trace="on", trace_dir=str(stack)),
    )
    yield w
    w.close(failed=False)


def test_robowright_moves_the_panda_through_the_stacks_own_controllers(driven_world, stack):
    """Each arm move is one goal to the stack's ``panda_arm_controller`` (as MoveIt's are), and the
    hand goes through ``panda_hand_controller``'s gripper action. The flange arrives, the fingers
    open and close, and the stack logs no error."""
    w, r = driven_world, driven_world.robot
    errors = lambda: rig._read(stack / "panda.log").count("[ERROR]")
    before, sent = errors(), w.backend.moves_sent
    goals = ((0.4, 0.2, 0.4), (0.3, -0.2, 0.5), (0.45, 0.0, 0.25))
    for goal in goals:
        r.arm.move_to(goal)
        expect(r.tcp).to_be_near(goal, tol=0.001, timeout=2, hold=0.3)
    r.gripper.close()
    expect(r.gripper).to_be_closed(timeout=5)
    r.gripper.open()
    expect(r.gripper).to_be_open(timeout=5)
    assert w.backend.moves_sent - sent == len(goals)
    assert errors() == before
