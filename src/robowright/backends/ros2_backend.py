"""ROS 2: any robot with a ROS 2 driver, real or simulated, through its topics.

The robot is read from ``sensor_msgs/JointState`` and driven through the standard
``ros2_control`` controllers, so this works with any robot that has a ``ros2_control``
driver (most arms do), and with any simulator that speaks ROS 2 (Gazebo, Isaac Sim's bridge):

* **arm**: a ``JointGroupPositionController`` (``forward_command_controller``), sent each
  control step's targets on ``/<controller>/commands``; or a ``joint_trajectory_controller``,
  sent a one-point trajectory reaching them one control period later.
* **gripper**: in the same forward controller as the arm, its own forward controller, a
  ``GripperCommand`` action (``gripper_action_controller``, ``parallel_gripper_action_controller``),
  or not at all.
* **objects**: TF frames, which become the world's ``perception`` sources, so object
  assertions and privileged policies read them as they read a simulator's ground truth.
* **cameras**: ``sensor_msgs/Image`` topics, which policies and traces read through ``render``.

A ROS 2 robot claims none of the simulator capabilities (no ground truth, contacts, state
or forces): the contract skips what needs them, and the rest of a test is the same test as
in simulation. Time is the ROS clock (``use_sim_time`` follows ``/clock``), and each control
step waits for its period to pass. ``robot.reset_to`` drives to the pose, as a real arm
cannot be teleported.

Configure it in the project's ``robowright.toml`` (or ``[tool.robowright.ros2]``)::

    [ros2]
    namespace = "/my_arm"                         # prefixes every topic below
    arm = { controller = "forward_position_controller", interface = "position" }
    gripper = { controller = "gripper_controller", interface = "action" }
    joints = { shoulder_pan = "joint1" }          # robowright name -> ROS name, where they differ
    frame = "base_link"                           # the robot's base frame, for TF lookups
    objects = { cube = "cube" }                   # object -> TF frame
    cameras = { front = "/camera/image_raw" }

or pass the same keys as keywords: ``rw.launch(..., backend="ros2", namespace="/my_arm")``.
rclpy comes with a ROS 2 install: source its ``setup.bash`` before running the tests.
"""

from __future__ import annotations

import os
import threading
import time as _time

import numpy as np

from .base import RENDER, Backend, register

_DEFAULTS = {
    "namespace": "",
    "joint_states": "joint_states",
    "joints": {},
    "arm": {"controller": "forward_position_controller", "interface": "position"},
    "gripper": {"interface": "with_arm"},
    "frame": "base_link",
    "objects": {},
    "cameras": {},
    "use_sim_time": False,
    "timeout": 10.0,  # seconds to wait for the robot's first joint states, and for reset_to
    "reset_speed": 0.5,  # rad/s (or m/s) that reset_to drives at
    # False: observe only. robowright never publishes a command or drives to a reset pose; your own
    # stack drives the robot and the test reads joint states and TF. An action that would move it
    # fails at once with a CapabilityError, rather than waiting for a move that never comes.
    "command": True,
    "reset_tolerance": 0.02,
    "node_name": None,
}
_INTERFACES = {"arm": ("position", "trajectory"), "gripper": ("with_arm", "position", "trajectory", "action", "none")}


def settings(options: dict | None = None) -> dict:
    """The ROS 2 settings: the defaults, the project's ``[ros2]`` table, then ``options``."""
    from ..plugins import find_config

    found = find_config()
    project = dict(found[1].get("ros2") or {}) if found else {}
    out = {**_DEFAULTS}
    for src in (project, options or {}):
        for k, v in src.items():
            if k not in _DEFAULTS:
                raise ValueError(f"unknown ros2 setting {k!r}; settings: {', '.join(_DEFAULTS)}")
            out[k] = {**out[k], **v} if k in ("arm", "gripper") and isinstance(v, dict) else v
    for part, allowed in _INTERFACES.items():
        if out[part].get("interface") not in allowed:
            raise ValueError(f"ros2 {part} interface is one of {', '.join(allowed)}, not {out[part].get('interface')!r}")
    return out


def _topic(ns: str, name: str) -> str:
    if name.startswith("/"):
        return name
    return f"{ns.rstrip('/')}/{name}" if ns else f"/{name}"


@register("ros2")
class Ros2Backend(Backend):
    def __init__(self, spec, seed: int = 0, **options):
        super().__init__(spec, seed)
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from sensor_msgs.msg import JointState

        self.cfg = cfg = settings(options)
        if not rclpy.ok():
            rclpy.init()
        m = self.robot_model
        self.ros_names = [cfg["joints"].get(j, j) for j in m.arm_joints]
        self._grip_joint = self._grip_range = None
        if self.has_gripper:
            name, (closed, opened) = next(iter(m.derived.gripper_joints.items()))
            self._grip_joint = cfg["joints"].get("gripper", name)
            self._grip_range = (float(closed), float(opened))

        self.node = rclpy.create_node(cfg["node_name"] or f"robowright_{os.getpid()}_{id(self) & 0xFFFF:x}")
        if cfg["use_sim_time"]:
            self.node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self._lock = threading.Lock()
        self._pos: dict[str, float] = {}
        self._vel: dict[str, float] = {}
        self._fresh = threading.Event()
        ns = cfg["namespace"]
        self.node.create_subscription(JointState, _topic(ns, cfg["joint_states"]), self._on_joints, 50)
        self._publishers()
        self._cameras = {}
        if cfg["cameras"]:
            self._subscribe_cameras()
            self.capabilities = frozenset({RENDER})
        self._tf = None
        if cfg["objects"]:
            from tf2_ros import Buffer, TransformListener

            self._tf = Buffer()
            self._tf_listener = TransformListener(self._tf, self.node)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True, name="robowright-ros2")
        self._spin.start()

        if not self._fresh.wait(cfg["timeout"]) or not self._have_all():
            seen = sorted(self._pos)
            self.close()
            want = [*self.ros_names, *([self._grip_joint] if self.has_gripper else [])]
            raise TimeoutError(
                f"no joint states for {want} on {_topic(ns, cfg['joint_states'])} within {cfg['timeout']} s "
                f"(heard {seen or 'nothing'}): is the robot's driver running, and are its joint names mapped "
                f"in [ros2] joints?"
            )
        self._t0 = self._now()
        self._tick = self._t0
        q = self.qpos()
        self._ctrl = q.copy()
        self._last_q, self._last_t = q, self._t0

    # --- wiring ----------------------------------------------------------------------------
    def _publishers(self):
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import JointTrajectory

        cfg, ns = self.cfg, self.cfg["namespace"]
        arm, grip = cfg["arm"], cfg["gripper"]
        msg = {"position": (Float64MultiArray, "commands"), "trajectory": (JointTrajectory, "joint_trajectory")}
        typ, suffix = msg[arm["interface"]]
        self._arm_pub = self.node.create_publisher(typ, _topic(ns, arm.get("topic") or f"{arm['controller']}/{suffix}"), 10)
        self._grip_pub = self._grip_action = None
        gi = grip["interface"]
        if not self.has_gripper or gi in ("with_arm", "none"):
            return
        if gi == "action":
            from control_msgs.action import GripperCommand
            from rclpy.action import ActionClient

            name = grip.get("action") or f"{grip.get('controller', 'gripper_controller')}/gripper_cmd"
            self._grip_action = ActionClient(self.node, GripperCommand, _topic(ns, name))
            self._grip_goal, self._grip_sent = None, None
        else:
            typ, suffix = msg[gi]
            self._grip_pub = self.node.create_publisher(typ, _topic(ns, grip.get("topic") or f"{grip['controller']}/{suffix}"), 10)

    def _subscribe_cameras(self):
        from sensor_msgs.msg import Image

        for cam, topic in self.cfg["cameras"].items():
            self.node.create_subscription(Image, _topic(self.cfg["namespace"], topic), lambda msg, c=cam: self._on_image(c, msg), 2)

    def _on_joints(self, msg):
        with self._lock:
            for i, n in enumerate(msg.name):
                if i < len(msg.position):
                    self._pos[n] = msg.position[i]
                if i < len(msg.velocity):
                    self._vel[n] = msg.velocity[i]
        if self._have_all():
            self._fresh.set()

    def _have_all(self) -> bool:
        want = [*self.ros_names, *([self._grip_joint] if self.has_gripper else [])]
        return all(n in self._pos for n in want)

    def _on_image(self, cam, msg):
        img = np.frombuffer(bytes(msg.data), np.uint8)
        enc = msg.encoding.lower()
        ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(enc)
        if ch is None:
            return  # an encoding robowright does not read (depth, Bayer): nothing to show
        img = img.reshape(msg.height, msg.step)[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
        if enc.startswith("bgr"):
            img = img[..., [2, 1, 0]]
        elif ch == 1:
            img = np.repeat(img, 3, axis=2)
        with self._lock:
            self._cameras[cam] = img[..., :3]

    # --- the robot -------------------------------------------------------------------------
    def _opening(self, q: float) -> float:
        closed, opened = self._grip_range
        return (q - closed) / (opened - closed)

    def _grip_q(self, opening: float) -> float:
        closed, opened = self._grip_range
        return closed + float(np.clip(opening, 0.0, 1.0)) * (opened - closed)

    def qpos(self) -> np.ndarray:
        with self._lock:
            q = [self._pos[n] for n in self.ros_names]
            if self.has_gripper:
                q.append(self._opening(self._pos[self._grip_joint]))
        return np.array(q, float)

    def qvel(self) -> np.ndarray:
        with self._lock:
            have = all(n in self._vel for n in self.ros_names)
            v = [self._vel[n] for n in self.ros_names] if have else None
            if v is not None and self.has_gripper:
                closed, opened = self._grip_range
                v.append(self._vel.get(self._grip_joint, 0.0) / (opened - closed))
        if v is not None:
            return np.array(v, float)
        # A driver that reports no velocities: differentiate the positions.
        q, t = self.qpos(), self._now()
        dt = max(t - self._last_t, 1e-6)
        out = (q - self._last_q) / dt
        self._last_q, self._last_t = q, t
        return out

    @property
    def commands(self) -> bool:
        """Whether robowright sends this robot commands (``[ros2] command``)."""
        return bool(self.cfg["command"])

    def set_ctrl(self, target) -> None:
        self._ctrl = np.array(target, float)

    def ctrl(self) -> np.ndarray:
        return self._ctrl.copy()

    def _send(self, target):
        from builtin_interfaces.msg import Duration
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        arm = list(map(float, target[: self.n_arm]))
        grip = self._grip_q(target[-1]) if self.has_gripper else None
        gi = self.cfg["gripper"]["interface"]  # "none": the gripper is read, never driven
        with_arm = self.has_gripper and gi == "with_arm"
        names = [*self.ros_names, *([self._grip_joint] if with_arm else [])]
        values = [*arm, *([grip] if with_arm else [])]
        reach = Duration(sec=0, nanosec=int(self.control_dt * 1e9))

        def trajectory(joints, positions):
            msg = JointTrajectory(joint_names=joints)
            msg.points = [JointTrajectoryPoint(positions=positions, time_from_start=reach)]
            return msg

        if self.cfg["arm"]["interface"] == "position":
            self._arm_pub.publish(Float64MultiArray(data=values))
        else:
            self._arm_pub.publish(trajectory(names, values))
        if self._grip_pub is not None:
            self._grip_pub.publish(Float64MultiArray(data=[grip]) if gi == "position" else trajectory([self._grip_joint], [grip]))
        elif self._grip_action is not None and (self._grip_sent is None or abs(grip - self._grip_sent) > 1e-4):
            self._send_grip_goal(grip)

    def _send_grip_goal(self, q: float):
        from control_msgs.action import GripperCommand

        if not self._grip_action.server_is_ready():
            return  # sent on a later step, once the controller's action server is up
        goal = GripperCommand.Goal()
        goal.command.position = q
        goal.command.max_effort = float(self.cfg["gripper"].get("max_effort", 0.0))
        self._grip_action.send_goal_async(goal)  # a new goal preempts the last
        self._grip_sent = q

    # --- time ------------------------------------------------------------------------------
    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    @property
    def time(self) -> float:
        return self._now() - self._t0

    def step(self) -> None:
        """Send this step's targets, then wait out the control period (on the ROS clock)."""
        if self.commands:
            self._send(self._ctrl)
        self._tick += self.control_dt
        if self.cfg["use_sim_time"]:
            while self._now() < self._tick - 1e-9:
                _time.sleep(min(self.control_dt / 10, 0.001))
            return
        late = self._now() - self._tick
        if late > 0:
            # Running behind (a slow policy, a debugger): carry on from now rather than
            # stepping without waiting until caught up, which would send targets in a burst.
            self._tick = self._now() if late > self.control_dt else self._tick
            return
        _time.sleep(self._tick - self._now())

    def set_joint_positions(self, q) -> None:
        """Drive to ``q`` (a real arm cannot be teleported): at ``reset_speed``, then until it
        is within ``reset_tolerance``, or ``TimeoutError`` after ``timeout`` seconds."""
        q = np.asarray(q, float)
        if not self.commands:
            return  # observe-only: the robot stays wherever its own stack has it
        start = self.qpos()
        span = np.abs(q - start)[: self.n_arm]
        n = max(1, int(np.ceil(float(span.max(initial=0.0)) / self.cfg["reset_speed"] / self.control_dt)))
        for i in range(1, n + 1):
            self._ctrl = start + (q - start) * i / n
            self.step()
        deadline = _time.monotonic() + self.cfg["timeout"]
        while np.abs(self.qpos()[: self.n_arm] - q[: self.n_arm]).max(initial=0.0) > self.cfg["reset_tolerance"]:
            if _time.monotonic() > deadline:
                raise TimeoutError(f"the robot did not reach {np.round(q, 3).tolist()} (at {np.round(self.qpos(), 3).tolist()})")
            self.step()

    # --- the world -------------------------------------------------------------------------
    def perception(self) -> dict:
        """A pose source per object with a TF frame: ``world.perception`` reads them."""
        return {name: (lambda f=frame: self._tf_pose(f)) for name, frame in self.cfg["objects"].items()}

    def _tf_pose(self, frame: str):
        from rclpy.duration import Duration
        from rclpy.time import Time

        # The first lookup of a frame waits as long as start-up may take: a TF tree comes up a piece
        # at a time (robot_state_publisher, a perception node), and asked a moment too early the
        # base frame "does not exist". Once seen, a frame that stops arriving fails within a second.
        seen = self.__dict__.setdefault("_tf_seen", set())
        wait = 1.0 if frame in seen else float(self.cfg["timeout"])
        t = self._tf.lookup_transform(self.cfg["frame"], frame, Time(), timeout=Duration(seconds=wait)).transform
        seen.add(frame)
        p = np.array([t.translation.x, t.translation.y, t.translation.z])
        quat = np.array([t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z])
        # Relative to the robot's base, which robowright mounts at base_pos, turned base_yaw.
        m = self.robot_model
        c, s = np.cos(m.base_yaw), np.sin(m.base_yaw)
        p = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]) + np.asarray(m.base_pos, float)
        yaw = np.array([np.cos(m.base_yaw / 2), 0.0, 0.0, np.sin(m.base_yaw / 2)])
        return p, _qmul(yaw, quat)

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        if camera not in self.cfg["cameras"]:
            raise KeyError(f"no ROS camera {camera!r}; [ros2] cameras has {sorted(self.cfg['cameras'])}")
        deadline = _time.monotonic() + self.cfg["timeout"]
        while True:
            with self._lock:
                img = self._cameras.get(camera)
            if img is not None:
                break
            if _time.monotonic() > deadline:
                raise TimeoutError(f"no image from {self.cfg['cameras'][camera]} within {self.cfg['timeout']} s")
            _time.sleep(0.01)
        if img.shape[:2] != (height, width):  # nearest-neighbour: what a policy asked for, cheaply
            ys = (np.arange(height) * img.shape[0] / height).astype(int)
            xs = (np.arange(width) * img.shape[1] / width).astype(int)
            img = img[ys][:, xs]
        return np.ascontiguousarray(img)

    def close(self) -> None:
        ex = getattr(self, "_executor", None)
        if ex is not None:
            ex.shutdown()
            self._spin.join(timeout=2)
            self._executor = None
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.node = None


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
