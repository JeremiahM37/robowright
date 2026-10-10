"""A robot in Gazebo (gz sim), driven through ``gz_ros2_control``: a third-party simulator that
robowright only talks to over ROS 2, as it would a real arm.

* the robot: robowright's URDF export of a model, with a ``<ros2_control>`` block on
  ``gz_ros2_control/GazeboSimSystem`` hardware, welded to the world at its base pose;
* the controllers: ``ros2_control``'s own (joint state broadcaster, forward position controller);
* the scene: the default scene's cube and bin as SDF models, their poses bridged to TF;
* the clock: Gazebo's, bridged to ``/clock`` (run with ``use_sim_time``).

Run as ``python tests/gazebo_rig.py ROBOT`` to start it on its own (Ctrl-C stops it).
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROS = Path(sys.prefix)


def _exe(pkg: str, name: str) -> str:
    for root in (ROS / "lib" / pkg, *(Path(p) / "lib" / pkg for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p)):
        if (root / name).exists():
            return str(root / name)
    raise FileNotFoundError(f"{pkg}/{name} is not installed in this ROS 2 environment")


def robot_urdf(robot: str, controllers: Path, initial: dict) -> str:
    from robowright import robots
    from robowright.robots import urdf

    m = robots.get(robot)
    path, meta = urdf.load(m)
    text = path.read_text()
    # Absolute mesh paths: Gazebo resolves meshes itself, not relative to the URDF. Its DART physics
    # drops an OBJ submesh with no vertex normals (and then crashes on the empty shape), so the
    # meshes are copied with normals.
    meshes = controllers.parent / "meshes"
    meshes.mkdir(exist_ok=True)
    for f in (path.parent / "meshes").glob("*.obj"):
        _with_normals(f, meshes / f.name)
    text = re.sub(r'filename="meshes/', f'filename="file://{meshes}/', text)
    # URDF 1.0 has no capsule: a cylinder as long as the capsule's body plus its caps.
    text = re.sub(
        r'<capsule radius="([^"]+)" length="([^"]+)"\s*/>',
        lambda mm: f'<cylinder radius="{mm[1]}" length="{float(mm[2]) + 2 * float(mm[1]):.9g}"/>',
        text,
    )
    # SDF wants link and joint names distinct (the SO-101 has a link and a joint called "gripper");
    # URDF does not. Links are renamed, so the joints keep the names the controllers use.
    joints_named = set(re.findall(r'<joint name="([^"]+)"', text))
    for link in set(re.findall(r'<link name="([^"]+)"', text)) & joints_named:
        text = text.replace(f'<link name="{link}"', f'<link name="{link}_link"')
        text = text.replace(f'link="{link}"', f'link="{link}_link"')
    root = meta["root"]
    x, y, z = m.base_pos
    weld = (
        '<link name="world"/>\n'
        f'<joint name="world__weld" type="fixed"><parent link="world"/><child link="{root}"/>'
        f'<origin xyz="{x} {y} {z}" rpy="0 0 {m.base_yaw}"/></joint>\n'
    )
    text = text.replace(f'<link name="{root}">', weld + f'<link name="{root}">', 1)
    joints = [*m.arm_joints, next(iter(m.derived.gripper_joints))]
    block = ['<ros2_control name="gazebo" type="system">', "<hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>"]
    for j in joints:
        block += [
            f'<joint name="{j}"><command_interface name="position"/>',
            f'<state_interface name="position"><param name="initial_value">{initial.get(j, 0.0):.6f}</param></state_interface>',
            '<state_interface name="velocity"/></joint>',
        ]
    block.append("</ros2_control>")
    block.append(
        '<gazebo><plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">'
        f"<parameters>{controllers}</parameters></plugin></gazebo>"
    )
    return text.replace("</robot>", "\n".join(block) + "\n</robot>")


def _with_normals(src: Path, dst: Path) -> None:
    """src (an OBJ of vertices and triangles) written to dst with area-weighted vertex normals."""
    v, f = [], []
    for line in src.read_text().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    v, f = np.array(v), np.array(f, int)
    n = np.zeros_like(v)
    face = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], face)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    out = [f"v {x:.7g} {y:.7g} {z:.7g}" for x, y, z in v] + [f"vn {x:.6g} {y:.6g} {z:.6g}" for x, y, z in n]
    out += [f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}" for a, b, c in f]
    dst.write_text("\n".join(out) + "\n")


def controllers_yaml(robot: str) -> str:
    from robowright import robots

    m = robots.get(robot)
    joints = [*m.arm_joints, next(iter(m.derived.gripper_joints))]
    return (
        "controller_manager:\n  ros__parameters:\n    update_rate: 250\n    use_sim_time: true\n"
        "    joint_state_broadcaster:\n      type: joint_state_broadcaster/JointStateBroadcaster\n"
        "    forward_position_controller:\n      type: position_controllers/JointGroupPositionController\n"
        "forward_position_controller:\n  ros__parameters:\n"
        f"    joints: [{', '.join(joints)}]\n"
    )


# Each object publishes its own pose with its name in the header, which ros_gz_bridge turns into a
# TF frame. (The world's pose topics carry names only where the bridge does not look.)
POSE_PLUGIN = (
    '<plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">'
    "<publish_link_pose>false</publish_link_pose><publish_model_pose>true</publish_model_pose>"
    "<use_pose_vector_msg>true</use_pose_vector_msg><update_frequency>100</update_frequency></plugin>"
)


def world_sdf(robot: str) -> str:
    from robowright.gazebo import model
    from robowright.scene import default_scene

    models = [model(o, POSE_PLUGIN, pose=True) for o in default_scene(robot).objects]
    return f"""<?xml version="1.0"?>
<sdf version="1.9"><world name="robowright">
  <physics name="1ms" type="dart"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor>
    <dart><collision_detector>bullet</collision_detector></dart></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <light type="directional" name="sun"><direction>-0.5 0.1 -0.9</direction></light>
  <model name="ground"><static>true</static><link name="link">
    <collision name="c"><geometry><plane><normal>0 0 1</normal></plane></geometry></collision>
    <visual name="v"><geometry><plane><normal>0 0 1</normal><size>4 4</size></plane></geometry></visual>
  </link></model>
  {"".join(models)}
</world></sdf>
"""


def _start(cmd, log: Path, env=None) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True, env=env)


def _stop(procs):
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGINT)
    for p in procs:
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()


@contextlib.contextmanager
def gazebo(robot: str = "so101", workdir: Path | None = None):
    """``robot`` in Gazebo behind ros2_control, its default scene's objects as TF frames."""
    from robowright import robots
    from robowright.robot import home_q
    from robowright.scene import default_scene

    d = Path(workdir or tempfile.mkdtemp(prefix="rw-gz-"))
    m = robots.get(robot)
    initial = dict(zip(m.arm_joints, home_q(m.name)))
    grip, (_, opened) = next(iter(m.derived.gripper_joints.items()))
    initial[grip] = opened
    (d / "controllers.yaml").write_text(controllers_yaml(robot))
    (d / "robot.urdf").write_text(robot_urdf(robot, d / "controllers.yaml", initial))
    (d / "world.sdf").write_text(world_sdf(robot))
    env = {**os.environ, "GZ_SIM_RESOURCE_PATH": str(d)}
    # Headless rendering (EGL): with DISPLAY set to an X server without GLX, Gazebo's renderer
    # fails to make a window and the server crashes
    procs = [_start(["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "2", str(d / "world.sdf")], d / "gz.log", env)]
    try:
        procs.append(
            _start(
                [
                    _exe("robot_state_publisher", "robot_state_publisher"),
                    "--ros-args",
                    "-p",
                    "use_sim_time:=true",
                    "-p",
                    f"robot_description:={(d / 'robot.urdf').read_text()}",
                ],
                d / "rsp.log",
            )
        )
        objects = [o.name for o in default_scene(robot).objects]
        poses = [f"/model/{n}/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V" for n in objects]
        bridge = ["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock", *poses]
        procs.append(_start([_exe("ros_gz_bridge", "parameter_bridge"), *bridge], d / "bridge.log"))
        procs.append(_start([sys.executable, __file__, "relay", ",".join(objects)], d / "relay.log"))
        create = subprocess.run(
            [_exe("ros_gz_sim", "create"), "-world", "robowright", "-name", m.name, "-topic", "robot_description"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if create.returncode != 0:
            raise RuntimeError(f"spawning the robot failed:\n{create.stdout[-2000:]}{create.stderr[-2000:]}")
        spawn = subprocess.run(
            [
                sys.executable,
                _exe("controller_manager", "spawner"),
                "joint_state_broadcaster",
                "forward_position_controller",
                "--controller-manager-timeout",
                "60",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if spawn.returncode != 0:
            log = (d / "gz.log").read_text()[-3000:]
            raise RuntimeError(f"spawning controllers failed:\n{spawn.stdout[-2000:]}{spawn.stderr[-2000:]}\n{log}")
        yield d
    finally:
        _stop(procs[::-1])


def relay(names: list[str]) -> None:
    """Republish Gazebo's poses of the scene objects as TF frames under the robot's ``world``."""
    import rclpy
    from tf2_msgs.msg import TFMessage
    from tf2_ros import TransformBroadcaster

    rclpy.init()
    node = rclpy.create_node("gazebo_object_tf", parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
    tf = TransformBroadcaster(node)
    wanted = set(names)

    def on_poses(msg):
        out = [t for t in msg.transforms if t.child_frame_id in wanted]
        for t in out:
            t.header.frame_id = "world"
        if out:
            tf.sendTransform(out)

    for name in names:
        node.create_subscription(TFMessage, f"/model/{name}/pose", on_poses, 10)
    with contextlib.suppress(KeyboardInterrupt):
        rclpy.spin(node)


if __name__ == "__main__" and sys.argv[1:2] == ["relay"]:
    relay(sys.argv[2].split(","))
elif __name__ == "__main__":
    with gazebo(sys.argv[1] if len(sys.argv) > 1 else "so101") as d:
        print("serving", d, flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(1)
