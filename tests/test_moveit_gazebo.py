"""robowright as the harness around someone else's whole stack: Universal Robots' Gazebo
simulation of a UR5e (``ur_simulation_gz``, ros2_control, UR's controllers), MoveIt
(``ur_moveit_config``), and a MoveIt client standing in for the code under test
(``examples/moveit_ur/reach.py``). robowright simulates nothing here and commands nothing: it
loads UR's own description, reads ``/joint_states`` on Gazebo's clock, and asserts.

Needs a ROS 2 environment with ``ros-jazzy-ur-simulation-gz`` and ``ros-jazzy-ur-moveit-config``
(``scripts/ros2_env.sh``); skipped elsewhere.
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
from robowright.scene import ObjectSpec, SceneSpec

HERE = Path(__file__).parent
REACH = HERE.parent / "examples" / "moveit_ur" / "reach.py"


def _rig():
    spec = importlib.util.spec_from_file_location("ur_moveit_rig", HERE / "ur_moveit_rig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rig = _rig()
pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="needs a ROS 2 environment (rclpy)"),
    pytest.mark.skipif(not rig.installed(), reason="needs ur_simulation_gz and ur_moveit_config"),
]
_worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ["ROS_DOMAIN_ID"] = str(200 + int(_worker.removeprefix("gw")) % 30)
os.environ["GZ_PARTITION"] = f"rw-ur-{_worker}-{os.getpid()}"  # this module's Gazebo, and no other
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


@pytest.fixture(scope="module")
def ur_sim(tmp_path_factory):
    with rig.ur_moveit("ur5e", workdir=tmp_path_factory.mktemp("ur")) as d:
        yield d


@pytest.fixture
def world(ur_sim):
    model = robots.load(rig.description("ur5e"), name="ur5e_ros", gripper=False, base_pos=(0.0, 0.0, 0.0))
    w = rw.launch(
        SceneSpec(robot=model.name, objects=[]),
        backend="ros2",
        command=False,  # observe only: MoveIt drives the arm
        frame="base_link",
        gripper={"interface": "none"},
        use_sim_time=True,
        settings=rw.Settings(trace="on", trace_dir=str(ur_sim)),
    )
    yield w
    w.close(failed=False)


class _Tool:
    """Where the robot's own TF (robot_state_publisher, from UR's description) puts tool0."""

    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from tf2_ros import Buffer, TransformListener

        self._ctx = rclpy.Context()
        rclpy.init(context=self._ctx)
        self.node = rclpy.create_node("tool_tf", context=self._ctx, parameter_overrides=[Parameter("use_sim_time", value=True)])
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
                t = self.buf.lookup_transform("base_link", "tool0", Time()).transform
                q = t.rotation
                return np.array([t.translation.x, t.translation.y, t.translation.z]), np.array([q.w, q.x, q.y, q.z])
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    def close(self):
        self._ex.shutdown(timeout_sec=1.0)
        self._ctx.try_shutdown()


def _reach(goal) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(REACH), *map(str, goal)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _watch(proc: subprocess.Popen, check) -> None:
    """``check()``, with the client's own account added if it fails: a plan MoveIt could not
    find reads as that, not as an arm that never arrived."""
    try:
        check()
    except AssertionError as e:
        if proc.poll() is not None:
            e.add_note(f"the MoveIt client exited {proc.returncode}:\n{proc.stdout.read()[-1500:]}")
        raise


def test_moveit_moves_a_ur5e_in_gazebo_and_robowright_sees_it_arrive(world):
    r = world.robot
    assert world.model_changes == []  # robowright changes nothing in a robot it reaches over ROS 2
    with pytest.raises(CapabilityError):
        r.arm.move_to(r.tcp.position)  # observe-only: robowright never commands this arm
    tool = _Tool()
    try:
        for goal in ((0.4, 0.2, 0.4), (0.3, -0.2, 0.5), (0.45, 0.0, 0.25)):
            goal = np.array(goal)
            if np.linalg.norm(r.tcp.position - goal) < 0.01:
                continue  # already there (the simulation outlives a test): no move to watch
            start = world.time
            reach = _reach(goal)
            # MoveIt's plan, executed by UR's controller in Gazebo: robowright waits on Gazebo's
            # clock for the arm, by its own kinematics of UR's description, to settle at the goal.
            _watch(reach, lambda g=goal: expect(r.tcp).to_be_near(g, tol=0.002, timeout=45, hold=0.5))
            out, _ = reach.communicate(timeout=30)
            assert reach.returncode == 0, out
            assert world.time - start > 0.5  # it moved there, on sim time (not a clock that jumped)
            # robowright's model of the arm agrees with the robot's own TF to a fraction of a millimetre
            p, _ = tool.pose()
            assert np.linalg.norm(p - r.tcp.position) < 5e-4, (p, r.tcp.position)
            down = r.kin.fk(r.true_qpos()[: r.n_arm])[:3, :3] @ r.model.derived.tool_axis
            assert down @ (0, 0, -1) > 0.999  # the tool points down, as asked
    finally:
        tool.close()


def test_a_stack_that_puts_the_tool_in_the_wrong_place_fails(world):
    """The harness has to catch a wrong outcome, not just watch: here the code under test is
    off by 4 cm in height (a wrong frame or offset, say). MoveIt succeeds at what it was asked;
    the test of what was meant fails."""
    r = world.robot
    meant = np.array([0.35, 0.1, 0.3])
    if np.linalg.norm(r.tcp.position - meant) < 0.06:
        meant = np.array([0.35, -0.1, 0.3])
    reach = _reach(meant + (0.0, 0.0, 0.04))
    try:
        with pytest.raises(rw.errors.ExpectationError, match="to_be_near"):
            expect(r.tcp).to_be_near(meant, tol=0.002, timeout=15, hold=0.5)
    finally:
        out, _ = reach.communicate(timeout=30)
    assert reach.returncode == 0, out  # MoveIt did what it was told
    assert abs(np.linalg.norm(r.tcp.position - meant) - 0.04) < 0.002  # and robowright measured the miss


SCENE = [
    ObjectSpec("cube", pos=(0.4, 0.1, None)),
    ObjectSpec("bin", kind="bin", size=(0.05, 0.05, 0.02), pos=(0.4, -0.25, None), color="blue"),
]


@pytest.fixture
def scene_world(ur_sim):
    model = robots.load(rig.description("ur5e"), name="ur5e_ros", gripper=False, base_pos=(0.0, 0.0, 0.0))
    w = rw.launch(
        SceneSpec(robot=model.name, objects=SCENE),
        backend="ros2",
        command=False,
        frame="base_link",
        gripper={"interface": "none"},
        use_sim_time=True,
        gazebo=True,  # robowright drives the simulator as well as watching the robot
        settings=rw.Settings(trace="on", trace_dir=str(ur_sim)),
    )
    yield w
    w.close(failed=False)


def test_robowright_places_its_scene_in_ur_gazebo_world_and_reads_the_truth(scene_world):
    w = scene_world
    gz = w.backend.gazebo
    assert {"cube", "bin"} <= set(gz.models())  # spawned into UR's world (the robot's world file has neither)
    assert w.model_changes == []  # still: robowright changed nothing in the robot
    expect(w.scene["cube"]).to_be_near((0.4, 0.1, 0.0125), tol=0.002, timeout=2)  # Gazebo's own pose of it
    expect(w.scene["cube"]).to_be_at_rest(timeout=2)
    w.move_object("cube", (0.35, 0.2, 0.0125))
    expect(w.scene["cube"]).to_be_near((0.35, 0.2, 0.0125), tol=0.002, hold=0.5)  # and it stays: Gazebo's physics holds it there
    w.move_object("cube", (0.25, 0.0, 0.3))  # dropped from 30 cm: Gazebo's physics, not robowright's
    expect(w.scene["cube"]).to_be_near((0.25, 0.0, 0.0125), tol=0.003, timeout=3)


def test_robowright_pauses_and_steps_gazebo_exactly(scene_world):
    gz = scene_world.backend.gazebo
    gz.pause()
    try:
        assert gz.paused
        i0, t0 = gz.iterations, gz.time
        q0 = scene_world.robot.true_qpos()
        gz.step(250)
        assert gz.iterations - i0 == 250
        assert gz.time - t0 == pytest.approx(0.25, abs=1e-6)
        assert np.allclose(scene_world.robot.true_qpos(), q0, atol=1e-3)  # the arm holds, paused or stepped
    finally:
        gz.play()
    assert not gz.paused


def test_trials_randomize_the_scene_in_gazebo(scene_world):
    w = scene_world
    placed = []
    for _ in range(2):
        moved = w.faults.randomize_scene()
        assert set(moved) == {"cube"}  # the bin is static; the cube moves
        expect(w.scene["cube"]).to_be_near(moved["cube"], tol=0.003, timeout=2)
        placed.append(np.asarray(w.scene["cube"].position[:2]))
    assert np.linalg.norm(placed[0] - placed[1]) > 1e-3  # two trials, two different scenes, in Gazebo


def test_an_agent_session_drives_gazebo_and_sees_what_it_renders(ur_sim):
    """What an agent does over MCP (the tools are thin wrappers of these Session calls): launch
    against the robot in Gazebo, look, pause, step, add and remove things, take a screenshot."""
    import io

    from PIL import Image

    from robowright.session import Session

    s = Session(trace_dir=ur_sim)
    try:
        snap = s.launch(
            robot=rig.description("ur5e"),
            backend="ros2",
            objects=[{"name": "cube", "kind": "box", "pos": [0.45, -0.1]}],
            ros2={"frame": "base_link", "gripper": {"interface": "none"}, "use_sim_time": True, "command": False, "gazebo": True},
            robot_options={"gripper": False, "base_pos": [0.0, 0.0, 0.0]},
        )
        assert "cube: red box 25x25x25 mm, at [0.450, -0.100, 0.01" in snap and "at rest" in snap  # where Gazebo has it
        assert np.allclose(s.world.scene["cube"].position, (0.45, -0.1, 0.0125), atol=1e-3)
        assert "rw_camera_front" in s.sim_state()
        s.sim_pause()
        before = s._gazebo().iterations
        state = s.sim_step(100)
        assert s._gazebo().iterations - before == 100 and "paused" in state
        s.sim_play()
        snap = s.sim_spawn({"name": "ball", "kind": "sphere", "size": [0.02], "pos": [0.35, 0.15, 0.3], "color": "green"})
        assert "ball: green sphere" in snap
        png = s.screenshot("front")
        img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), float)
        assert img.std() > 10  # a picture of something (Gazebo's rendering), not a blank frame
        s.sim_remove("ball")
        assert "ball" not in s.sim_state()
        assert "ball" not in s.snapshot()
    finally:
        if s.world is not None:
            s.close()
