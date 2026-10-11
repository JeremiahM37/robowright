"""robowright against UFACTORY's own Gazebo simulation of an xArm 6 (``xarm_gazebo``'s launch,
``tests/xarm_gz_rig.py``), as the code under test: UFACTORY's ``xarm6_traj_controller`` answers
robowright's commands, and Gazebo's physics moves the cube. robowright connects with nothing named:
it reads the robot from ``/robot_description`` and finds the trajectory controller by itself.

Needs a ROS 2 environment with Gazebo and the checkout of ``scripts/ros2_env.sh xarm``; the gripper
tests also need its mimic-joint plugin, built by the same script. Skipped elsewhere.
"""

import importlib.util
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.scene import ObjectSpec, SceneSpec

HERE = Path(__file__).parent


def _rig():
    spec = importlib.util.spec_from_file_location("xarm_gz_rig", HERE / "xarm_gz_rig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rig = _rig()
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)"),
    pytest.mark.skipif(not rig.installed(), reason="needs xarm_ros2 and Gazebo: scripts/ros2_env.sh xarm"),
]
_worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
# This module's own ROS graph and Gazebo, as the other Gazebo tests': nothing else on the machine
# (ROS 2 domain ids stop at 232).
os.environ["ROS_DOMAIN_ID"] = str(220 + (int(_worker.removeprefix("gw")) * 7 + os.getpid()) % 12)
os.environ["GZ_PARTITION"] = f"rw-xarm-{_worker}-{os.getpid()}"
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

FRAME = "link_base"  # xArm's base link, the frame TF is read in
# The arm stands where xarm_gazebo's launch spawns it, yawed 90 degrees, and robowright's world is
# Gazebo's: the base is placed there, and its positions are the ones robowright reports.
BASE = np.array(rig.BASE_POS)
YAW = np.pi / 2
ROT = np.array([[np.cos(YAW), -np.sin(YAW), 0], [np.sin(YAW), np.cos(YAW), 0], [0, 0, 1]])
# robowright's home (where arm.home() goes) is 20 cm out and over the table by default: in the world
# that is 80 cm below this arm's base, so the arm's own home is given, in its frame, as the goals are
HOME = BASE + ROT @ np.array((0.3, 0.0, 0.3))
GOALS = tuple(BASE + ROT @ np.array(p) for p in ((0.3, 0.0, 0.35), (0.25, 0.15, 0.25), (0.3, -0.15, 0.4)))


@pytest.fixture(scope="module")
def xarm(tmp_path_factory):
    # One simulation for the module, with the gripper when its plugin is built: the arm's tests then
    # run on the arm the gripper tests need, and a second Gazebo is never started beside it.
    d = tmp_path_factory.mktemp("xarm")
    with rig.xarm_sim(workdir=d, gripper=rig.gripper_installed()) as sim:
        yield sim


def _world(xarm, *, objects=(), command=True, gripper=False, gazebo=True):
    # Nothing is named: the robot is the one /robot_description describes (SceneSpec(robot=None)), and
    # the controllers are the ones its controller manager runs. No camera: the arm's description has
    # none to render. The base is placed where Gazebo has the arm, facing the same way.
    scene = SceneSpec(robot=None, objects=list(objects), cameras=[])
    return rw.launch(
        scene,
        backend="ros2",
        command=command,
        frame=FRAME,
        use_sim_time=True,
        gazebo=gazebo,
        robot_options={"base_pos": [float(v) for v in BASE], "base_yaw": YAW, "home": [float(v) for v in HOME]},
        settings=rw.Settings(trace="on", trace_dir=str(xarm)),
    )


def _matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class _Tf:
    """TF from the arm's base link, as the stack's robot_state_publisher gives it."""

    def __init__(self, frame):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from tf2_ros import Buffer, TransformListener

        self.frame = frame
        self._ctx = rclpy.Context()
        rclpy.init(context=self._ctx)
        self.node = rclpy.create_node("xarm_tf", context=self._ctx, parameter_overrides=[Parameter("use_sim_time", value=True)])
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
                t = self.buf.lookup_transform(FRAME, self.frame, Time()).transform
                return np.array([t.translation.x, t.translation.y, t.translation.z]), _matrix(t.rotation)
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    def close(self):
        self._ex.shutdown(timeout_sec=1.0)
        self._ctx.try_shutdown()


@pytest.fixture
def observed(xarm):
    w = _world(xarm, command=False)
    yield w
    w.close(failed=False)


def test_robowright_connects_to_xarm_with_nothing_named(observed):
    """The robot and its trajectory controller are found from the stack: no name, no controller."""
    r = observed.robot
    assert observed.model_changes == []
    print(f"robot {r.name}: {r.n_arm} arm joints, base at {np.round(r.model.base_pos, 3).tolist()}")
    assert r.n_arm == 6


def test_robowright_reads_xarms_arm_where_the_stacks_tf_puts_it(observed):
    """robowright's model of the xArm 6 (from the stack's description) and the stack's TF agree on
    where the tool is: TCP = the tool frame plus the TCP offset along the tool axis."""
    r = observed.robot
    tf = _Tf("link_eef")
    try:
        errs = []
        for _ in range(5):
            p, rot = tf.pose()
            tcp_tf = BASE + ROT @ (p + rot @ r.model.derived.tcp_offset)
            errs.append(np.linalg.norm(tcp_tf - r.tcp.position))
            time.sleep(0.2)
        print(f"TCP error vs TF over {len(errs)} reads (m): {[f'{e:.6f}' for e in errs]}")
        assert max(errs) < 5e-4, errs
    finally:
        tf.close()


@pytest.fixture
def driven(xarm):
    w = _world(xarm)
    yield w
    w.close(failed=False)


def test_robowright_moves_the_arm_through_xarms_controller(driven, xarm):
    """Each move is one FollowJointTrajectory goal to UFACTORY's controller. The tool arrives within a
    millimetre of each goal, and the stack logs no error."""
    w, r = driven, driven.robot
    errors = rig._read(xarm / "control.log").count("[ERROR]")
    moves = w.backend.moves_sent
    for goal in GOALS:
        t0 = w.time
        r.arm.move_to(goal)
        expect(r.tcp).to_be_near(goal, tol=0.001, timeout=8, hold=0.3)
        print(f"goal {goal}: arrived in {w.time - t0:.2f} s (sim)")
    assert w.backend.moves_sent - moves == len(GOALS)  # one goal per move
    assert rig._read(xarm / "control.log").count("[ERROR]") == errors


def test_robowright_opens_and_closes_xarms_gripper_through_its_controller(xarm):
    """The gripper is one more joint of ``xarm_gripper_traj_controller``, found from the controller
    manager and driven by its trajectory topic; ``opening`` is read back from ``drive_joint``."""
    if not rig.gripper_installed():
        pytest.skip("needs xarm_gazebo's mimic-joint plugin: scripts/ros2_env.sh xarm")
    w = _world(xarm)
    try:
        r = w.robot
        r.gripper.close()
        expect(r.gripper).to_be_closed(timeout=8)
        closed = r.gripper.opening
        r.gripper.open()
        expect(r.gripper).to_be_open(timeout=8)
        opened = r.gripper.opening
        print(f"opening: closed {closed:.4f}, open {opened:.4f}")
        assert closed < 0.1 and opened > 0.8
    finally:
        w.close(failed=False)


# The table is Gazebo's (xarm_gazebo's world): its top is at z = 1.015, and the arm stands on its edge.
# The cube and the bin go on the table, within the arm's reach: in its own frame, as the goals are.
TABLE_TOP = 1.015
CUBE_LOCAL = (-0.45, 0.0)
BIN_LOCAL = (-0.3, -0.25)


def _on_table(local, z):
    x, y = BASE[:2] + (ROT @ np.array((*local, 0.0)))[:2]
    return (float(x), float(y), z)


SCENE = [
    ObjectSpec("cube", pos=_on_table(CUBE_LOCAL, TABLE_TOP + 0.0125)),
    ObjectSpec("bin", kind="bin", size=(0.05, 0.05, 0.02), pos=_on_table(BIN_LOCAL, TABLE_TOP + 0.01), color="blue"),
]


def test_robowright_picks_a_cube_with_xarms_gripper_in_gazebo(xarm):
    """robowright commands the arm and its gripper through UFACTORY's controllers, and Gazebo's physics
    decides whether the cube comes with the hand: the cube's pose is Gazebo's ground truth."""
    if not rig.gripper_installed():
        pytest.skip("needs xarm_gazebo's mimic-joint plugin: scripts/ros2_env.sh xarm")
    w = _world(xarm, objects=SCENE)
    try:
        r = w.robot
        assert {"cube", "bin"} <= set(w.backend.gazebo.models())
        cube, bin_ = w.scene["cube"], w.scene["bin"]
        expect(cube).to_be_near(SCENE[0].pos, tol=0.002, timeout=3)
        r.pick(cube)
        print(f"after pick: cube {np.round(cube.position, 4)}, tcp {np.round(r.tcp.position, 4)}, opening {r.gripper.opening:.3f}")
        assert cube.position[2] > TABLE_TOP + 0.0125 + 0.03
        expect(cube).to_be_near(r.tcp, tol=0.03, hold=1.0, timeout=3)
        assert 0.1 < r.gripper.opening < 0.9
        expect(r).to_be_holding(cube, timeout=2)
        r.place(on=bin_)
        expect(r.gripper).to_be_open(timeout=3)
        expect(cube).to_be_inside(bin_, timeout=5)
        print(f"after place: cube {np.round(cube.position, 4)}, bin {np.round(bin_.position, 4)}")
    finally:
        w.close(failed=False)
