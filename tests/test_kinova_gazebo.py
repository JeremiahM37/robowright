"""robowright against Kinova's own Gazebo simulation of a Gen3 (7 DoF) with a Robotiq 2F-85
(``kortex_bringup``'s simulation launch, ``tests/kinova_gz_rig.py``), as the code under test: Kinova's
``joint_trajectory_controller`` and ``robotiq_gripper_controller`` answer robowright's commands, and
Gazebo's physics moves the cube. robowright loads Kinova's description (the xacro the stack expands),
reads ``/joint_states`` and TF on Gazebo's clock, and asserts.

Needs a ROS 2 environment with Gazebo, ``kortex_description``, ``gripper_controllers`` and the
checkouts of ``scripts/ros2_env.sh kortex``; skipped elsewhere.
"""

import importlib.util
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

import robowright as rw
from robowright import expect, robots
from robowright.scene import ObjectSpec, SceneSpec

HERE = Path(__file__).parent


def _rig():
    spec = importlib.util.spec_from_file_location("kinova_gz_rig", HERE / "kinova_gz_rig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rig = _rig()
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)"),
    pytest.mark.skipif(not rig.installed(), reason="needs kortex_description, Gazebo and scripts/ros2_env.sh kortex"),
]
_worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
# This module's own ROS graph and Gazebo, as test_moveit_gazebo.py's: nothing else on the machine
os.environ["ROS_DOMAIN_ID"] = str(200 + (int(_worker.removeprefix("gw")) * 7 + os.getpid()) % 30)
os.environ["GZ_PARTITION"] = f"rw-kinova-{_worker}-{os.getpid()}"
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

ARM = {"controller": "joint_trajectory_controller", "interface": "trajectory"}
GRIPPER = {"interface": "action", "controller": "robotiq_gripper_controller", "parallel": True}  # ParallelGripperCommand
# The Gen3's base stands at z = 0.3 in Gazebo (kortex's launch spawns it there), and robowright's
# world is Gazebo's, so the base is placed there too: the floor is 0.3 m below it.
BASE = rig.BASE_POS


@pytest.fixture(scope="module")
def kinova(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinova")
    with rig.ros_paths(d), rig.kinova_sim(workdir=d) as sim:
        yield sim


def _model():
    # robowright's home pose is a TCP position in its world, which is Gazebo's: 20 cm above the base
    return robots.load(rig.description(), name="kinova_gen3", base_pos=BASE, home=(0.2, 0.0, BASE[2] + 0.2))


def _world(kinova, *, objects=(), command=True, arm=None, gripper=None, gazebo=False):
    _model()
    # A scene's default camera is a Gazebo camera sensor, and robowright adds the Sensors system to
    # render it: Gazebo's OGRE then segfaults this headless server, with or without DISPLAY (measured).
    # The scene here has no camera; the arm's wrist camera, from the description, is not rendered.
    scene = (
        SceneSpec(robot="kinova_gen3", objects=list(objects), cameras=[])
        if gazebo
        else SceneSpec(robot="kinova_gen3", objects=list(objects))
    )
    return rw.launch(
        scene,
        backend="ros2",
        command=command,
        frame="base_link",
        gripper=gripper or {"interface": "none"},
        use_sim_time=True,
        gazebo=gazebo,
        **({"arm": arm} if arm else {}),
        settings=rw.Settings(trace="on", trace_dir=str(kinova)),
    )


def _matrix(q) -> np.ndarray:
    """A rotation matrix from TF's quaternion (x, y, z, w)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class _Tf:
    """TF from base_link, as the stack's robot_state_publisher (from Kinova's description) gives it."""

    def __init__(self, frame):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from tf2_ros import Buffer, TransformListener

        self.frame = frame
        self._ctx = rclpy.Context()
        rclpy.init(context=self._ctx)
        self.node = rclpy.create_node("kinova_tf", context=self._ctx, parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.buf = Buffer()
        self._listener = TransformListener(self.buf, self.node)
        self._ex = SingleThreadedExecutor(context=self._ctx)
        self._ex.add_node(self.node)
        from robowright.ros2_bridge import _spin

        threading.Thread(target=_spin, args=(self._ex,), daemon=True).start()

    def pose(self):
        """The frame's position and rotation matrix in base_link."""
        from rclpy.time import Time

        deadline = time.monotonic() + 10
        while True:
            try:
                t = self.buf.lookup_transform("base_link", self.frame, Time()).transform
                q = t.rotation
                return np.array([t.translation.x, t.translation.y, t.translation.z]), _matrix(q)
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    def close(self):
        self._ex.shutdown(timeout_sec=1.0)
        self._ctx.try_shutdown()


@pytest.fixture
def observed(kinova):
    w = _world(kinova, command=False)
    yield w
    w.close(failed=False)


def test_robowright_reads_kinovas_arm_where_the_stacks_tf_puts_it(observed):
    """robowright's model of Kinova's Gen3 (from its description) and the stack's TF agree on where
    the tool is: the TCP is the hand frame (``robotiq_85_base_link``) plus the gripper's TCP offset
    along the tool axis, and the tool points where the hand frame's z axis does."""
    w, r = observed, observed.robot
    assert w.model_changes == []  # robowright changes nothing in a robot it reaches over ROS 2
    tf = _Tf("robotiq_85_base_link")
    try:
        errs = []
        for _ in range(5):
            p, rot = tf.pose()
            # TF is in base_link, robowright's positions in Gazebo's world: the base stands at BASE
            tcp_tf = BASE + p + rot @ r.model.derived.tcp_offset
            errs.append(np.linalg.norm(tcp_tf - r.tcp.position))
            down = rot @ r.model.derived.tool_axis
            fk_axis = r.kin.fk(r.true_qpos()[: r.n_arm])[:3, :3] @ r.model.derived.tool_axis
            assert np.dot(down, fk_axis) > 0.999  # the tool axis agrees with the stack's frame
            time.sleep(0.2)
        print(f"TCP error vs TF over {len(errs)} reads (m): {[f'{e:.6f}' for e in errs]}")
        assert max(errs) < 5e-4, errs
    finally:
        tf.close()


def _record_moves(backend):
    """Each whole move robowright sends as the targets it carries: for the speed the controller saw."""
    sent = []
    real = backend.start_trajectory

    def start(targets):
        ok = real(targets)
        if ok:
            sent.append(np.array(targets, float))
        return ok

    backend.start_trajectory = start
    return sent


@pytest.fixture
def driven(kinova):
    w = _world(kinova, arm=ARM, gripper=GRIPPER)
    yield w
    w.close(failed=False)


GOALS = ((0.4, 0.2, 0.7), (0.3, -0.2, 0.8), (0.45, 0.0, 0.55))


def test_robowright_moves_the_arm_through_kinovas_controller(driven, kinova):
    """Each move is one FollowJointTrajectory goal to Kinova's ``joint_trajectory_controller``. The
    tool arrives within a millimetre of each goal, and the stack logs no error."""
    w, r = driven, driven.robot
    sent = _record_moves(w.backend)
    errors = rig._read(kinova / "control.log").count("[ERROR]")
    moves = w.backend.moves_sent
    for goal in GOALS:
        t0 = w.time
        r.arm.move_to(goal)
        expect(r.tcp).to_be_near(goal, tol=0.001, timeout=5, hold=0.3)
        print(f"goal {goal}: arrived in {w.time - t0:.2f} s (sim)")
    assert w.backend.moves_sent - moves == len(GOALS)  # one goal per move
    # robowright plans within each joint's <limit velocity> in the Gen3's URDF (1.22 to 1.40 rad/s).
    # Kinova's joint_limits.yaml, which the stack's own controllers are configured with, gives 0.8727
    # for every joint: measured, and reported, not asserted.
    limit = np.asarray(r._speed_limits, float)
    stack = np.array(yaml.safe_load(rig.JOINT_LIMITS.read_text())["maximum_velocities"])
    peak = np.max([np.abs(np.diff(s[:, : r.n_arm], axis=0)).max(axis=0) / w.backend.control_dt for s in sent], axis=0)
    print(f"peak joint speed sent (rad/s): {np.round(peak, 3).tolist()}")
    print(f"URDF velocity limits: {np.round(limit, 4).tolist()}; joint_limits.yaml: {stack.tolist()}")
    assert (peak <= limit * 1.01).all(), (peak, limit)  # targets one control step apart: discretisation
    assert rig._read(kinova / "control.log").count("[ERROR]") == errors


def test_robowright_opens_and_closes_kinovas_robotiq_through_its_controller(driven):
    """The Robotiq is driven through ``robotiq_gripper_controller``'s ParallelGripperCommand action (what
    ``ros2 action list`` shows it serving); ``opening`` is read back from the knuckle in /joint_states."""
    r = driven.robot
    r.gripper.close()
    expect(r.gripper).to_be_closed(timeout=5)
    closed = r.gripper.opening
    r.gripper.open()
    expect(r.gripper).to_be_open(timeout=5)
    opened = r.gripper.opening
    print(f"opening: closed {closed:.3f}, open {opened:.3f}")
    assert closed < 0.1 and opened > 0.8


SCENE = [
    ObjectSpec("cube", pos=(0.4, 0.1, None)),
    ObjectSpec("bin", kind="bin", size=(0.05, 0.05, 0.02), pos=(0.4, -0.25, None), color="blue"),
]


@pytest.fixture
def scene_world(kinova):
    w = _world(kinova, objects=SCENE, arm=ARM, gripper=GRIPPER, gazebo=True)
    yield w
    w.close(failed=False)


def test_robowright_picks_a_cube_with_kinovas_gripper_in_gazebo(scene_world):
    """robowright commands the Gen3 and its Robotiq through Kinova's controllers, and Gazebo's physics
    decides whether the cube comes with the hand: the cube's pose is Gazebo's ground truth."""
    w, r = scene_world, scene_world.robot
    assert {"cube", "bin"} <= set(w.backend.gazebo.models())
    cube, bin_ = w.scene["cube"], w.scene["bin"]
    expect(cube).to_be_near((0.4, 0.1, 0.0125), tol=0.002, timeout=3)
    r.pick(cube)
    print(f"after pick: cube {np.round(cube.position, 4)}, tcp {np.round(r.tcp.position, 4)}, opening {r.gripper.opening:.3f}")
    # Lifted off the floor, with the hand: Gazebo has the cube at the TCP, and the fingers stopped
    # on it (an opening between the two ends) rather than closing on nothing
    assert cube.position[2] > 0.0125 + 0.03
    # In the hand, and staying there. Where between the Robotiq's pads the cube ends up varies with
    # Gazebo's run: 5 to 16 mm from the TCP in 7 runs, always above it (the pads close in an arc and
    # the cube rides up them), so this asks for "within the pads' length", not for a point.
    expect(cube).to_be_near(r.tcp, tol=0.03, hold=1.0, timeout=3)
    assert 0.1 < r.gripper.opening < 0.9
    # The ROS 2 backend reports no contacts, so to_be_holding takes robowright's fallback for hardware:
    # the jaws were told to close and stopped short, and the cube is within reach of the TCP by
    # Gazebo's pose. That is weaker than a contact, and is what this asserts.
    expect(r).to_be_holding(cube, timeout=2)
    r.place(on=bin_)
    expect(cube).to_be_inside(bin_, timeout=5)
    print(f"after place: cube {np.round(cube.position, 4)}, bin {np.round(bin_.position, 4)}")
