"""Robots for the ROS 2 backend's tests, started as ROS 2 processes.

* :func:`mock_hardware`: the real ``ros2_control`` stack (``controller_manager``,
  ``joint_state_broadcaster``, the stock position, trajectory and gripper controllers) on
  ``mock_components/GenericSystem`` hardware, for any robowright robot: its URDF export with a
  ``<ros2_control>`` block. The joints go where they are told, so the contract's arm tests run
  on it exactly as through a real driver.
* :func:`physics`: a robowright scene simulated in MuJoCo behind the same topics a
  ``ros2_control`` driver has (``/joint_states``, ``/forward_position_controller/commands``),
  with the objects as TF frames and a camera as an image topic. A pick through it exercises
  everything a robot behind ROS 2 is tested with: perception, grasps without contact sensing,
  images for policies.

Run as ``python tests/ros2_rig.py physics ROBOT`` to start the second on its own.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROS = Path(sys.prefix)  # the ROS 2 environment the tests run in (RoboStack, or a sourced /opt/ros)


def _exe(pkg: str, name: str) -> str:
    for root in (ROS / "lib" / pkg, *(Path(p) / "lib" / pkg for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p)):
        if (root / name).exists():
            return str(root / name)
    raise FileNotFoundError(f"{pkg}/{name} is not installed in this ROS 2 environment")


def _urdf(robot: str, initial: dict, rename: dict) -> str:
    """The robot's URDF, with a ros2_control block on mock hardware, and its joints renamed as
    ``rename`` says (a driver's names are rarely robowright's)."""
    from robowright import robots
    from robowright.robots import urdf

    m = robots.get(robot)
    path, _ = urdf.load(m)
    text = path.read_text()
    for old, new in rename.items():
        text = text.replace(f'<joint name="{old}"', f'<joint name="{new}"').replace(f'joint="{old}"', f'joint="{new}"')
    joints = [rename.get(j, j) for j in [*m.arm_joints, *list(m.derived.gripper_joints)[:1]]]
    block = ['<ros2_control name="robowright_mock" type="system">', "<hardware><plugin>mock_components/GenericSystem</plugin></hardware>"]
    for j in joints:
        block += [
            f'<joint name="{j}"><command_interface name="position"/>',
            f'<state_interface name="position"><param name="initial_value">{initial.get(j, 0.0):.6f}</param></state_interface>',
            '<state_interface name="velocity"/></joint>',
        ]
    block.append("</ros2_control>")
    return text.replace("</robot>", "\n".join(block) + "\n</robot>")


def _controllers(robot: str, mode: str, rename: dict) -> tuple[str, list[str]]:
    from robowright import robots

    m = robots.get(robot)
    grip = rename.get(g := list(m.derived.gripper_joints)[0], g)
    arm = [rename.get(j, j) for j in m.arm_joints]
    if mode == "position":  # one forward controller for the arm and gripper
        cfg = {
            "forward_position_controller": ("position_controllers/JointGroupPositionController", {"joints": [*arm, grip]}),
        }
    else:  # a trajectory controller for the arm, a gripper action server for the jaws
        cfg = {
            "arm_controller": (
                "joint_trajectory_controller/JointTrajectoryController",
                {
                    "joints": arm,
                    "command_interfaces": ["position"],
                    "state_interfaces": ["position", "velocity"],
                    "allow_nonzero_velocity_at_trajectory_end": True,
                },
            ),
            "gripper_controller": ("position_controllers/GripperActionController", {"joint": grip, "allow_stalling": True}),
        }
    lines = ["controller_manager:", "  ros__parameters:", "    update_rate: 250"]
    lines += ["    joint_state_broadcaster:", "      type: joint_state_broadcaster/JointStateBroadcaster"]
    for name, (typ, _) in cfg.items():
        lines += [f"    {name}:", f"      type: {typ}"]
    for name, (_, params) in cfg.items():
        lines += [f"{name}:", "  ros__parameters:"]
        for k, v in params.items():
            lines.append(f"    {k}: {_yaml(v)}")
    return "\n".join(lines) + "\n", ["joint_state_broadcaster", *cfg]


def _yaml(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(_yaml(x) for x in v) + "]"
    return str(v)


def _start(cmd, log: Path) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True)


def _stop(procs):
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGINT)
    for p in procs:
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()


@contextlib.contextmanager
def mock_hardware(robot: str, mode: str = "position", workdir: Path | None = None, rename: dict | None = None):
    """``ros2_control`` on mock hardware for ``robot``; ``mode`` "position" (one forward
    controller for arm and gripper) or "trajectory" (a trajectory controller and a gripper
    action server), with joints renamed as ``rename`` says. Yields the directory holding its
    URDF, settings and logs."""
    rename = rename or {}
    from robowright import robots
    from robowright.robot import home_q

    d = Path(workdir or tempfile.mkdtemp(prefix="rw-ros2-"))
    m = robots.get(robot)
    initial = {rename.get(j, j): q for j, q in zip(m.arm_joints, home_q(m.name))}
    grip, (_, opened) = next(iter(m.derived.gripper_joints.items()))
    initial[rename.get(grip, grip)] = opened
    (d / "robot.urdf").write_text(_urdf(robot, initial, rename))
    yaml, names = _controllers(robot, mode, rename)
    (d / "controllers.yaml").write_text(yaml)
    procs = [
        _start(
            [
                _exe("robot_state_publisher", "robot_state_publisher"),
                "--ros-args",
                "-p",
                f"robot_description:={(d / 'robot.urdf').read_text()}",
            ],
            d / "rsp.log",
        ),
        _start(
            [
                _exe("controller_manager", "ros2_control_node"),
                "--ros-args",
                "--params-file",
                str(d / "controllers.yaml"),
                "-r",
                "~/robot_description:=/robot_description",
            ],
            d / "control.log",
        ),
    ]
    try:
        spawn = subprocess.run(
            [sys.executable, _exe("controller_manager", "spawner"), *names, "--controller-manager-timeout", "30"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if spawn.returncode != 0:
            raise RuntimeError(
                f"spawning {names} failed:\n{spawn.stdout[-2000:]}{spawn.stderr[-2000:]}\n{(d / 'control.log').read_text()[-3000:]}"
            )
        yield d
    finally:
        _stop(procs)


@contextlib.contextmanager
def physics(robot: str = "so101", workdir: Path | None = None, cwd: Path | None = None):
    """A robowright scene in MuJoCo behind ``ros2_control``'s topics (see :func:`serve`), run
    from ``cwd`` (where a robowright.toml naming the robot is found)."""
    d = Path(workdir or tempfile.mkdtemp(prefix="rw-ros2-"))
    p = subprocess.Popen(
        [sys.executable, __file__, "physics", robot],
        stdout=(d / "physics.log").open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=cwd,
    )
    try:
        deadline = time.monotonic() + 60
        while "serving" not in (d / "physics.log").read_text():
            if p.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"the physics node did not start:\n{(d / 'physics.log').read_text()[-3000:]}")
            time.sleep(0.1)
        yield d
    finally:
        _stop([p])


def serve(robot: str):
    """Simulate ``robot``'s default scene in real time, as a ROS 2 robot: joint states and the
    objects' TF frames out, forward-controller position commands in, the front camera as an
    image topic."""
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64MultiArray
    from tf2_ros import TransformBroadcaster

    import robowright as rw
    from robowright.scene import default_scene

    rclpy.init()
    node = rclpy.create_node("robowright_physics")
    w = rw.launch(scene=default_scene(robot), settings=rw.Settings(trace="off"))
    b, m = w.backend, w.backend.robot_model
    w.robot.reset_to()
    grip, (closed, opened) = next(iter(m.derived.gripper_joints.items()))
    names = [*m.arm_joints, grip]
    command = {"q": None}

    def on_command(msg):
        if len(msg.data) == len(names):
            q = np.array(msg.data, float)
            q[-1] = (q[-1] - closed) / (opened - closed)
            command["q"] = q

    node.create_subscription(Float64MultiArray, "/forward_position_controller/commands", on_command, 10)
    states = node.create_publisher(JointState, "/joint_states", 10)
    images = node.create_publisher(Image, "/camera/image_raw", 2)
    tf = TransformBroadcaster(node)
    frame = {"n": 0}

    def tick():
        if command["q"] is not None:
            b.set_ctrl(command["q"])
        b.step()
        q, v = b.qpos(), b.qvel()
        now = node.get_clock().now().to_msg()
        msg = JointState(name=names, position=[*q[:-1], closed + q[-1] * (opened - closed)], velocity=[*v[:-1], v[-1] * (opened - closed)])
        msg.header.stamp = now
        states.publish(msg)
        out = []
        for name in w.object_names:
            p, quat = b.object_pose(name)
            # Relative to the robot's base frame, as a robot's own perception reports it.
            c, s = np.cos(-m.base_yaw), np.sin(-m.base_yaw)
            p = p - np.asarray(m.base_pos, float)
            p = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
            quat = _turn(-m.base_yaw, quat)
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = now, "robowright_base", name
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, p)
            t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = map(float, quat)
            out.append(t)
        tf.sendTransform(out)
        frame["n"] += 1
        if frame["n"] % 5 == 0:  # 10 frames a second
            img = b.render("front", 160, 120)
            im = Image(height=img.shape[0], width=img.shape[1], encoding="rgb8", step=img.shape[1] * 3, data=img.tobytes())
            im.header.stamp = now
            images.publish(im)

    node.create_timer(b.control_dt, tick)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    print("serving", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        ex.spin()


def _turn(yaw: float, q):
    """``q`` (w, x, y, z) turned by ``yaw`` about the vertical."""
    w1, z1 = np.cos(yaw / 2), np.sin(yaw / 2)
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2, w1 * y2 + z1 * x2, w1 * z2 + z1 * w2])


if __name__ == "__main__":
    if sys.argv[1] == "physics":
        serve(sys.argv[2] if len(sys.argv) > 2 else "so101")
