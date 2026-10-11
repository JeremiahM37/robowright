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
GOALS = ((0.3, 0.0, 0.35), (0.25, 0.15, 0.25), (0.3, -0.15, 0.4))


@pytest.fixture(scope="module")
def xarm(tmp_path_factory):
    d = tmp_path_factory.mktemp("xarm")
    with rig.xarm_sim(workdir=d) as sim:
        yield sim


def _world(xarm, *, objects=(), command=True, gripper=False, gazebo=True):
    # Nothing is named: the robot is the one /robot_description describes, and the controllers are
    # the ones its controller manager runs. No camera: the arm's description has none to render.
    scene = SceneSpec(robot="robot", objects=list(objects), cameras=[]) if gazebo else SceneSpec(robot="robot", objects=list(objects))
    return rw.launch(
        scene if objects else None,
        backend="ros2",
        command=command,
        frame=FRAME,
        use_sim_time=True,
        gazebo=gazebo,
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
    print(f"robot {r.name}: {r.n_arm} arm joints")
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
            tcp_tf = p + rot @ r.model.derived.tcp_offset
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
