"""ROS 2: any robot with a ROS 2 driver, real or simulated, through its topics.

The robot is read from ``sensor_msgs/JointState`` and driven through the standard
``ros2_control`` controllers, so this works with any robot that has a ``ros2_control``
driver (most arms do), and with any simulator that speaks ROS 2 (Gazebo, Isaac Sim's bridge):

* **arm**: a ``JointGroupPositionController`` (``forward_command_controller``), sent each
  control step's targets on ``/<controller>/commands``; or a ``joint_trajectory_controller``.
  A move robowright plans (``arm.move_to``, ``pick``) goes to that one as its clients send
  theirs: the whole trajectory as one ``FollowJointTrajectory`` goal, which the controller
  interpolates, checks against its own tolerances and answers (a move it aborts raises
  ``ActionError`` with its reason). Targets that come one at a time (a policy's) go as
  one-point trajectories reaching them one control period later.
* **gripper**: in the same forward controller as the arm, its own forward controller, a
  ``GripperCommand`` action (``gripper_action_controller``), a ``ParallelGripperCommand`` action
  (``parallel_gripper_action_controller``, with ``parallel = true``), or not at all.
* **objects**: TF frames, which become the world's ``perception`` sources, so object
  assertions and privileged policies read them as they read a simulator's ground truth.
* **cameras**: ``sensor_msgs/Image`` topics, which policies and traces read through ``render``.

A ROS 2 robot claims none of the simulator capabilities (no ground truth, contacts, state
or forces): the contract skips what needs them, and the rest of a test is the same test as
in simulation. A robot in Gazebo can have more (``gazebo = true``): robowright then also drives
the simulator itself (:mod:`robowright.gazebo`), for the objects' true poses, and to place,
spawn and randomize them (the scene's objects missing from the Gazebo world are spawned into
it); robowright's world frame is then Gazebo's, and the robot's ``base_pos`` is where it stands
in it. Time is the ROS clock (``use_sim_time`` follows ``/clock``), and each control
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

from .base import GROUND_TRUTH, RENDER, Backend, register

_DEFAULTS = {
    "namespace": "",
    "joint_states": "joint_states",
    "joints": {},
    # The controllers robowright drives the robot through. What is left out is read from the
    # robot's own controller manager (robowright.ros2_discover): the active controller that claims
    # the arm's joints, the gripper's, and how each is driven.
    "arm": {"controller": None, "interface": None},
    "gripper": {"interface": None},
    "frame": "base_link",
    "objects": {},
    "cameras": {},
    "use_sim_time": False,
    "clock_timeout": 30.0,  # wall seconds a simulation clock may stand still before a step gives up
    "timeout": 10.0,  # seconds to wait for the robot's first joint states, and for reset_to
    "reset_speed": 0.5,  # rad/s (or m/s) that reset_to drives at
    # False: observe only. robowright never publishes a command or drives to a reset pose; your own
    # stack drives the robot and the test reads joint states and TF. An action that would move it
    # fails at once with a CapabilityError, rather than waiting for a move that never comes.
    "command": True,
    "reset_tolerance": 0.02,
    "node_name": None,
    # The robot is in Gazebo, and robowright may drive Gazebo too: true, or {world = "name"} (and
    # model = "name" for the robot's model, where more than one could be it: default, its own name).
    "gazebo": False,
    # A mobile robot's base. {frame = "base_footprint", in = "map"} reads where it is from TF (``in``
    # -> ``frame``), the estimate a localizer such as AMCL publishes. cmd_vel and navigate name its
    # velocity command topic and its navigation action when the ROS graph's own are not the names
    # (found from the graph by default: a topic cmd_vel with a subscriber, and a NavigateToPose server).
    "base": None,
    # The topic a robot's own description is on (std_msgs/String, transient local), read when
    # the robot is not named.
    "description": "robot_description",
}
_INTERFACES = {"arm": ("position", "trajectory"), "gripper": ("with_arm", "position", "trajectory", "action", "none")}


def settings(options: dict | None = None) -> dict:
    """The ROS 2 settings: the defaults, the project's ``[ros2]`` table, then ``options``."""
    from ..plugins import find_config

    found = find_config()
    project = dict(found[1].get("ros2") or {}) if found else {}
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in _DEFAULTS.items()}  # (its tables are filled in later)
    for src in (project, options or {}):
        for k, v in src.items():
            if k not in _DEFAULTS:
                raise ValueError(f"unknown ros2 setting {k!r}; settings: {', '.join(_DEFAULTS)}")
            if k in ("arm", "gripper") and not isinstance(v, dict):
                raise ValueError(f"ros2 {k} is a table, e.g. {k} = {{controller = 'name', interface = '{_INTERFACES[k][1]}'}}, not {v!r}")
            out[k] = {**out[k], **v} if k in ("arm", "gripper") else v
    if out["base"] is not None and set(out["base"]) - {"frame", "in", "cmd_vel", "navigate"}:
        raise ValueError(
            "ros2 base takes frame and in (where it is), and cmd_vel and navigate (its command topic and navigation action), "
            "e.g. base = {frame = 'base_footprint', in = 'map'}"
        )
    for part, allowed in _INTERFACES.items():
        if out[part].get("interface") not in (None, *allowed):
            raise ValueError(f"ros2 {part} interface is one of {', '.join(allowed)}, not {out[part].get('interface')!r}")
    return out


def robot_description(options: dict | None = None) -> str:
    """The URDF the stack publishes on its description topic (``[ros2] description``): the robot as
    the stack runs it, with the frames and joints its own state publishers use. Read once, at start."""
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    cfg = settings(options)
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(f"robowright_description_{os.getpid()}")
    got: list[str] = []
    # Latched, as a robot_state_publisher publishes it: a subscriber that starts late still gets it.
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    topic = _topic(cfg["namespace"], cfg["description"])
    node.create_subscription(String, topic, lambda msg: got.append(msg.data), latched)
    deadline = _time.monotonic() + cfg["timeout"]
    try:
        while not got and _time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
    if not got:
        raise TimeoutError(f"no robot description on {topic} within {cfg['timeout']} s: is the robot's stack running?")
    return got[0]


def _topic(ns: str, name: str) -> str:
    if name.startswith("/"):
        return name
    return f"{ns.rstrip('/')}/{name}" if ns else f"/{name}"


@register("ros2")
class Ros2Backend(Backend):
    builds = False  # the robot is there already: robowright changes nothing in it

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
        self.joints_stamp_ns = 0  # the header stamp of the joint state the readings are from (0: unstamped)
        self._vel: dict[str, float] = {}
        self._fresh = threading.Event()
        ns = cfg["namespace"]
        self.node.create_subscription(JointState, _topic(ns, cfg["joint_states"]), self._on_joints, 50)
        self._cameras = {}
        if cfg["cameras"]:
            self._subscribe_cameras()
            self.capabilities = frozenset({RENDER})
        self._tf = None  # made on first use (see _tf_buffer)
        self._graph = None
        self._base_pub = self._base_msg = None  # the base's velocity command, found and made on first drive (_cmd_vel_publisher)
        self._base_cmd = None  # (linear, angular, lateral) every control step publishes while a drive runs
        self._nav = None  # the navigation action client, made on first go_to
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
        if cfg["use_sim_time"]:
            # A node on sim time reads 0 until its first /clock message; started from that, the
            # world's clock would jump to the simulator's (Gazebo's minutes in) at once, and
            # every timeout running would expire before anything had moved.
            deadline = _time.monotonic() + cfg["timeout"]
            while self._now() == 0.0:
                if _time.monotonic() > deadline:
                    self.close()
                    raise TimeoutError(f"no /clock within {cfg['timeout']} s: use_sim_time is set, so the simulator must publish it")
                _time.sleep(0.005)
        try:
            self._controllers()
            self._publishers()
        except Exception:
            self.close()
            raise
        self._t0 = self._now()
        self._tick = self._t0
        self.moves_sent = 0  # whole moves sent as one goal (start_trajectory)
        self.gazebo = None
        if cfg["gazebo"]:
            self._open_gazebo(spec, cfg["gazebo"] if isinstance(cfg["gazebo"], dict) else {})
        # The schedule starts with the world: opening Gazebo above moved the clock on, and a schedule left
        # behind it would step through that time without waiting for the simulator (no motion, no time).
        self._tick = self._now()
        q = self.qpos()
        self._ctrl = q.copy()
        self._last_q, self._last_t = q, self._t0

    # --- wiring ----------------------------------------------------------------------------
    def _controllers(self) -> None:
        """Fill in what ``[ros2] arm`` and ``gripper`` leave out from the robot's controller
        manager, and refuse settings nothing in the ROS graph answers to: a command sent to a
        topic no controller reads moves nothing and says nothing."""
        from .. import ros2_discover as discover

        cfg, ns = self.cfg, self.cfg["namespace"]
        arm, grip = cfg["arm"], cfg["gripper"]
        self._grip_pub = self._grip_action = self._traj_action = self._traj = self._arm_pub = None
        if not self.commands or self.robot_model.family == "mobile":
            arm["interface"] = arm["interface"] or "position"
            grip["interface"] = grip["interface"] or "none"
            return
        if arm["controller"] and arm["interface"] and grip["interface"]:
            return self._heard(arm)
        graph = discover.Graph(namespace=ns)
        deadline = _time.monotonic() + cfg["timeout"]
        while _time.monotonic() < deadline and not graph.controllers:
            graph.manager = graph.manager or discover._manager(self.node, ns)
            if graph.manager:
                graph.controllers = discover._controllers(self.node, graph.manager, deadline)
            _time.sleep(0.05)
        found = {c.name: c for c in graph.controllers}
        if not graph.controllers:
            # No controller manager (a driver of its own, or robowright's simulation served over
            # ROS 2): the command topics something subscribes to say as much.
            heard = self._command_topics()
            name = arm["controller"] or next(iter(heard), None)
            if name in heard:
                arm.update(controller=name, interface=arm["interface"] or heard[name])
        if arm["controller"] is None:
            arm.update(discover.suggest(graph, self.ros_names).get("arm") or {})
            if arm["controller"] is None:
                seen = (
                    discover.report(graph)
                    if graph.controllers
                    else "no controller manager answered, and nothing subscribes to a command topic"
                )
                raise RuntimeError(
                    "robowright found no active controller it can drive this robot's arm through (a joint trajectory "
                    f"controller or a forward position controller claiming {self.ros_names}).\n{seen}\n"
                    "Name one with [ros2] arm = {controller = ..., interface = ...}, or set command = false to observe only."
                )
        elif arm["interface"] is None:
            known = found.get(arm["controller"])
            if known is None or not known.interface or known.interface[0] != "arm":
                have = ", ".join(f"{c.name} ({c.type}, {c.state})" for c in graph.controllers) or "none found"
                raise RuntimeError(
                    f"[ros2] arm.controller {arm['controller']!r} is not a controller robowright can drive; controllers: {have}"
                )
            arm["interface"] = known.interface[1]
        if grip["interface"] is None:
            claimed = set(found[arm["controller"]].joints) if arm["controller"] in found else set()
            hands = [c for c in graph.controllers if c.active and c.interface and c.interface[0] == "gripper"]
            if not self.has_gripper:
                grip["interface"] = "none"
            elif self._grip_joint in claimed or not graph.controllers:
                grip["interface"] = "with_arm"
            elif hands:
                grip.update(controller=hands[0].name, interface=hands[0].interface[1])
                if "ParallelGripper" in hands[0].type:
                    grip["parallel"] = True
            elif (jtc := discover.gripper_trajectory(graph, arm["controller"], self._grip_joint)) is not None:
                grip.update(controller=jtc.name, interface="trajectory")
            else:
                grip["interface"] = "none"  # read, never driven: nothing in the stack drives it
        self._heard(arm)

    def _command_topics(self) -> dict:
        """``{controller: interface}`` for the command topics in the namespace that have a subscriber."""
        kinds = {
            "/joint_trajectory": ("trajectory_msgs/msg/JointTrajectory", "trajectory"),
            "/commands": ("std_msgs/msg/Float64MultiArray", "position"),
        }
        prefix = self.cfg["namespace"].rstrip("/") + "/"
        out = {}
        for topic, types in self.node.get_topic_names_and_types():
            for suffix, (typ, interface) in kinds.items():
                if topic.startswith(prefix) and topic.endswith(suffix) and typ in types and self.node.count_subscribers(topic):
                    out[topic[len(prefix) : -len(suffix)]] = interface
        return out

    def _heard(self, arm: dict) -> None:
        """Raise unless something subscribes to the arm's command topic (or serves its action)."""
        suffix = {"position": "commands", "trajectory": "joint_trajectory"}[arm["interface"]]
        topic = _topic(self.cfg["namespace"], arm.get("topic") or f"{arm['controller']}/{suffix}")
        deadline = _time.monotonic() + self.cfg["timeout"]
        while self.node.count_subscribers(topic) == 0:
            if _time.monotonic() > deadline:
                raise RuntimeError(
                    f"nothing subscribes to {topic}: is [ros2] arm.controller {arm['controller']!r} active, and is "
                    f"{arm['interface']!r} how it is driven? (`robowright ros2 inspect` lists the robot's controllers)"
                )
            _time.sleep(0.05)

    def _publishers(self):
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import JointTrajectory

        cfg, ns = self.cfg, self.cfg["namespace"]
        arm, grip = cfg["arm"], cfg["gripper"]
        if arm["controller"] is None:  # observe-only, with no controller named: nothing to publish to
            return
        msg = {"position": (Float64MultiArray, "commands"), "trajectory": (JointTrajectory, "joint_trajectory")}
        typ, suffix = msg[arm["interface"]]
        self._arm_pub = self.node.create_publisher(typ, _topic(ns, arm.get("topic") or f"{arm['controller']}/{suffix}"), 10)
        self._traj_action = self._traj = None
        if arm["interface"] == "trajectory" and arm.get("whole_moves", True):
            from control_msgs.action import FollowJointTrajectory
            from rclpy.action import ActionClient

            self._traj_name = _topic(ns, arm.get("action") or f"{arm['controller']}/follow_joint_trajectory")
            self._traj_action = ActionClient(self.node, FollowJointTrajectory, self._traj_name)
        self._grip_pub = self._grip_action = None
        gi = grip["interface"]
        if not self.has_gripper or gi in ("with_arm", "none"):
            return
        if gi == "action":
            from control_msgs.action import GripperCommand
            from rclpy.action import ActionClient

            name = grip.get("action") or f"{grip.get('controller', 'gripper_controller')}/gripper_cmd"
            # A parallel gripper controller serves its own action, whose goal is a JointState
            if grip.get("parallel"):
                from control_msgs.action import ParallelGripperCommand

                action = ParallelGripperCommand
            else:
                action = GripperCommand
            self._grip_action = ActionClient(self.node, action, _topic(ns, name))
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
            self.joints_stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
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

        if self._arm_pub is None:
            return  # a mobile base: its commands are its own (drive_base), not joint targets
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

        if self._traj is not None:
            pass  # the controller has the whole move (start_trajectory): a target now would replace it
        elif self.cfg["arm"]["interface"] == "position":
            self._arm_pub.publish(Float64MultiArray(data=values))
        else:
            self._arm_pub.publish(trajectory(names, values))
        if self._grip_pub is not None:
            self._grip_pub.publish(Float64MultiArray(data=[grip]) if gi == "position" else trajectory([self._grip_joint], [grip]))
        elif self._grip_action is not None and (self._grip_sent is None or abs(grip - self._grip_sent) > 1e-4):
            self._send_grip_goal(grip)

    # --- a whole move, as one goal -----------------------------------------------------------
    @property
    def plans_moves(self) -> bool:
        """Whether a planned move is sent whole (a trajectory controller's action)."""
        return self.commands and self._traj_action is not None

    def start_trajectory(self, targets) -> bool:
        """Send ``targets`` (one per control step from now) as one ``FollowJointTrajectory`` goal.
        False, and nothing sent, where they move nothing the controller drives (a gripper with a
        controller of its own)."""
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint

        targets = np.asarray(targets, float)
        with_arm = self.has_gripper and self.cfg["gripper"]["interface"] == "with_arm"
        rows = [[*map(float, t[: self.n_arm]), *([self._grip_q(t[-1])] if with_arm else [])] for t in targets]
        if np.ptp(np.array([self._last_sent(with_arm), *rows]), axis=0).max(initial=0.0) < 1e-9:
            return False
        if not self._traj_action.wait_for_server(timeout_sec=self.cfg["timeout"]):
            raise TimeoutError(
                f"no action server {self._traj_name} within {self.cfg['timeout']} s: is the arm's controller active, and is "
                "[ros2] arm.controller its name? (arm.whole_moves = false sends targets on its topic instead)"
            )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [*self.ros_names, *([self._grip_joint] if with_arm else [])]
        for i, row in enumerate(rows, 1):
            sec, nanosec = divmod(round(i * self.control_dt * 1e9), 10**9)
            goal.trajectory.points.append(JointTrajectoryPoint(positions=row, time_from_start=Duration(sec=sec, nanosec=nanosec)))
        self._traj = move = {"handle": None, "done": None}
        self.moves_sent += 1

        def answered(future):
            result = future.result()
            move["done"] = (result.result.error_code, result.result.error_string)

        def accepted(future):
            move["handle"] = handle = future.result()
            if handle.accepted:
                handle.get_result_async().add_done_callback(answered)
            else:
                move["done"] = (None, "the controller rejected the goal")

        self._traj_action.send_goal_async(goal).add_done_callback(accepted)
        return True

    def _last_sent(self, with_arm: bool) -> list:
        return [*map(float, self._ctrl[: self.n_arm]), *([self._grip_q(self._ctrl[-1])] if with_arm else [])]

    def trajectory_done(self) -> bool:
        return self._traj is None or self._traj["done"] is not None

    def end_trajectory(self) -> None:
        """Close the move sent by ``start_trajectory``: cancelled if the controller still has it,
        ``ActionError`` if the controller refused it or gave it up."""
        from ..errors import ActionError

        move, self._traj = self._traj, None
        if move is None:
            return
        if move["done"] is None:
            if move["handle"] is not None and move["handle"].accepted:
                move["handle"].cancel_goal_async()
            return
        code, text = move["done"]
        if code != 0:
            names = {-1: "INVALID_GOAL", -2: "INVALID_JOINTS", -3: "OLD_HEADER_TIMESTAMP", -4: "PATH_TOLERANCE_VIOLATED"}
            names[-5] = "GOAL_TOLERANCE_VIOLATED"
            what = f"{names.get(code, code)}: {text}" if code is not None else text
            raise ActionError(f"{self.cfg['arm']['controller']} did not carry out the move ({what})")

    def _send_grip_goal(self, q: float):
        from control_msgs.action import GripperCommand, ParallelGripperCommand

        if not self._grip_action.server_is_ready():
            return  # sent on a later step, once the controller's action server is up
        max_effort = float(self.cfg["gripper"].get("max_effort", 0.0))
        if self.cfg["gripper"].get("parallel"):
            from sensor_msgs.msg import JointState

            goal = ParallelGripperCommand.Goal()
            goal.command = JointState(name=[self._grip_joint], position=[q], effort=[max_effort] if max_effort else [])
        else:
            goal = GripperCommand.Goal()
            goal.command.position = q
            goal.command.max_effort = max_effort
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
            if self._base_cmd is not None:
                self._publish_twist(*self._base_cmd)
        if self.cfg["use_sim_time"]:
            # Between two calls the simulator keeps running, so the clock can be well past this schedule. Stepping on
            # from the old schedule would wait for nothing and send the targets in a burst the simulator never runs
            # (measured: a drive after 6 s of idle took 0 s of sim time and moved the base 0 m, and so did every drive
            # after it). Carry on from the clock instead, as the wall-clock branch below does when it runs behind.
            if self._now() > self._tick + self.control_dt:
                self._tick = self._now()
        self._tick += self.control_dt
        if self.cfg["use_sim_time"]:
            seen, since = self._now(), _time.monotonic()
            while (now := self._now()) < self._tick - 1e-9:
                if now != seen:
                    seen, since = now, _time.monotonic()
                elif _time.monotonic() - since > self.cfg["clock_timeout"]:
                    # A simulator that crashed, hung or was paused publishes no more /clock: waiting for the
                    # next step would never end (an xArm pick hung a suite for half an hour here).
                    from ..errors import RobowrightError

                    raise RobowrightError(
                        f"the simulation clock (/clock) has not moved for {self.cfg['clock_timeout']:g} s of wall time, "
                        f"at t={now - self._t0:.3f} s: the simulator has stopped, crashed or is paused"
                    )
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
    def _open_gazebo(self, spec, opts: dict) -> None:
        from ..gazebo import Gazebo

        try:
            self.gazebo = gz = Gazebo(world=opts.get("world"), timeout=self.cfg["timeout"])
        except BaseException:
            self.close()
            raise
        self.capabilities = self.capabilities | {GROUND_TRUTH}
        self._object_names = {o.name for o in spec.objects}
        mine = _SPAWNED.setdefault(gz.world, set())
        for o in spec.objects:
            if gz.has(o.name) and o.name not in mine:
                continue  # the world's own model: it stays where the world has it
            pos = (o.pos[0], o.pos[1], o.half_height if o.kind != "bin" else 0.0) if o.pos[2] is None else o.pos
            quat = (np.cos(o.yaw / 2), 0.0, 0.0, np.sin(o.yaw / 2))
            if gz.has(o.name):
                gz.set_pose(o.name, pos, quat)  # spawned for an earlier test: back where this scene has it
            else:
                gz.spawn_object(o)
                mine.add(o.name)
        # The scene's cameras, other than those [ros2] cameras maps to a robot's own image topics,
        # are Gazebo cameras: render() is then what Gazebo draws, not a picture robowright makes.
        # The world's fixed geometry (its table, floor and walls), as the planner must see it: read once,
        # after the objects are placed, and kept apart from the robot and the objects (see fixtures).
        self._scene_models = gz.scene()
        self._gz_cameras = [c for c in spec.cameras if c.name not in self.cfg["cameras"]]
        for c in self._gz_cameras:
            gz.add_camera(c.name, c.pos, c.lookat, c.fovy, c.width, c.height)
        if self._gz_cameras:
            self.capabilities = self.capabilities | {RENDER}

    @property
    def fixtures(self) -> tuple[list, list]:
        """The world's fixed obstacles (robowright.obstacles.Obstacle) and the notes on what was skipped:
        Gazebo's models other than the robot and the scene's objects. Empty without ``gazebo``."""
        if self.gazebo is None or not hasattr(self, "_scene_models"):
            return [], []
        if getattr(self, "_fixtures", None) is None:
            import mujoco

            from ..obstacles import from_scene
            from ..robots import PREFIX

            s = mujoco.MjSpec()
            self.robot_model.add_to(s)
            m = s.compile()
            links = {m.body(b).name[len(PREFIX) :] for b in range(m.nbody) if m.body(b).name.startswith(PREFIX)}
            self._fixtures = from_scene(self._scene_models, robot=links, exclude=self._object_names)
        return self._fixtures

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if self.gazebo is None:
            return super().object_pose(name)
        return self.gazebo.pose(name)

    def object_velocity(self, name: str) -> np.ndarray:
        if self.gazebo is None:
            return super().object_velocity(name)
        return self.gazebo.velocity(name)

    def set_object_pose(self, name: str, pos, quat=None) -> None:
        if self.gazebo is None:
            return super().set_object_pose(name, pos, quat)
        self.gazebo.set_pose(name, pos, quat)

    def perception(self) -> dict:
        """A pose source per object with a TF frame: ``world.perception`` reads them."""
        return {name: (lambda f=frame: self._tf_pose(f)) for name, frame in self.cfg["objects"].items()}

    def _tf_pose(self, frame: str):

        p, quat = self._lookup(self.cfg["frame"], frame)
        # Relative to the robot's base, which robowright mounts at base_pos, turned base_yaw.
        m = self.robot_model
        c, s = np.cos(m.base_yaw), np.sin(m.base_yaw)
        p = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]) + np.asarray(m.base_pos, float)
        yaw = np.array([np.cos(m.base_yaw / 2), 0.0, 0.0, np.sin(m.base_yaw / 2)])
        return p, _qmul(yaw, quat)

    def _tf_buffer(self):
        if self._tf is None:
            from tf2_ros import Buffer, TransformListener

            self._tf = Buffer()
            self._tf_listener = TransformListener(self._tf, self.node)
        return self._tf

    def _lookup(self, parent: str, child: str) -> tuple[np.ndarray, np.ndarray]:
        """``child``'s pose in ``parent`` from TF, as (position, (w, x, y, z))."""
        from rclpy.duration import Duration
        from rclpy.time import Time

        # The first lookup of a frame waits as long as start-up may take: a TF tree comes up a piece
        # at a time (robot_state_publisher, a perception node), and asked a moment too early the
        # base frame "does not exist". Once seen, a frame that stops arriving fails within a second.
        seen = self.__dict__.setdefault("_tf_seen", set())
        wait = 1.0 if (parent, child) in seen else float(self.cfg["timeout"])
        t = self._tf_buffer().lookup_transform(parent, child, Time(), timeout=Duration(seconds=wait)).transform
        seen.add((parent, child))
        p = np.array([t.translation.x, t.translation.y, t.translation.z])
        return p, np.array([t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z])

    def frame_pose(self, parent: str, child: str) -> tuple[np.ndarray, np.ndarray]:
        """Any TF frame's pose in another frame (for ``world.ros.frame``)."""
        return self._lookup(parent, child)

    # --- the robot's base (a mobile robot) -------------------------------------------------
    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the base IS: Gazebo's pose of the robot's model when Gazebo is on (ground truth),
        else the TF estimate. The pose is in Gazebo's world frame, or the ``base`` frame's."""
        if self.gazebo is None:
            return self.base_estimate()
        p, q = self.gazebo.pose(self._gazebo_model())
        return p, q

    def base_estimate(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the robot THINKS its base is: TF from ``base.in`` to ``base.frame``."""
        base = self.cfg["base"] or {}
        if "frame" not in base or "in" not in base:
            raise ValueError("a mobile robot's base needs [ros2] base = {frame = ..., in = ...}, e.g. base_footprint in map")
        return self._lookup(base["in"], base["frame"])

    # --- a mobile base's commands ----------------------------------------------------------
    def drive_base(self, linear: float, angular: float, lateral: float = 0.0) -> None:
        """Command the base's velocity (m/s forward and sideways, rad/s) from now until :meth:`stop_base`:
        every control step publishes it, so the stack's own timeouts on the command never see it go stale."""
        self._base_cmd = (float(linear), float(angular), float(lateral))
        self._publish_twist(*self._base_cmd)

    def stop_base(self) -> None:
        """Zero the base's velocity, once, and stop commanding it: whatever drives it after is its stack."""
        self._base_cmd = None
        if self._base_pub is not None:
            self._publish_twist(0.0, 0.0)

    def _publish_twist(self, linear: float, angular: float, lateral: float = 0.0) -> None:
        self._cmd_vel_publisher()
        msg = self._base_msg()
        twist = msg
        if hasattr(msg, "twist"):  # TwistStamped: stamped in the base's frame
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = (self.cfg["base"] or {}).get("frame", "base_link")
            twist = msg.twist
        twist.linear.x, twist.linear.y, twist.angular.z = linear, lateral, angular
        self._base_pub.publish(msg)

    def _cmd_vel_publisher(self):
        """The publisher on the base's command topic: found in the graph (a subscriber and a Twist
        or TwistStamped type), made the first time a drive is asked for, and refused if nothing is
        there to receive it."""
        if self._base_pub is not None:
            return self._base_pub
        from ..errors import CapabilityError
        from ..ros2_discover import find_cmd_vel

        name = self.topic_name((self.cfg["base"] or {}).get("cmd_vel", "cmd_vel"))
        deadline = _time.monotonic() + self.cfg["timeout"]
        found = None
        while found is None:
            topics = {n: list(t) for n, t in self.node.get_topic_names_and_types()}
            found = find_cmd_vel(topics, {name: self.node.count_subscribers(name)}, name)
            if found is None:
                if _time.monotonic() > deadline:
                    shown = sorted(n for n, t in topics.items() if any(x.startswith("geometry_msgs/msg/Twist") for x in t))
                    raise CapabilityError(
                        f"nothing subscribes to {name} (a Twist or TwistStamped command topic) within {self.cfg['timeout']} s, so "
                        f"the base cannot be driven. Command topics in the graph: {shown or 'none'}. Name the base's topic with "
                        "[ros2] base = {cmd_vel = ...}, or set command = false to observe only."
                    )
                _time.sleep(0.05)
        from rosidl_runtime_py.utilities import get_message

        self._base_msg = get_message(found[1])
        self._base_pub = self.node.create_publisher(self._base_msg, name, 10)
        return self._base_pub

    # --- a mobile base's navigation (Nav2's NavigateToPose) --------------------------------
    def start_navigation(self, x: float, y: float, yaw: float) -> None:
        """Send a goal to the robot's own navigation stack: the pose (x, y, yaw) in the frame the
        stack plans in (``[ros2] base.in``, else ``map``). Nothing is planned here: Nav2 does that,
        and :meth:`navigation_done` and :meth:`end_navigation` say how it went."""
        import rclpy.action as rclpy_action
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient

        from ..errors import CapabilityError
        from ..ros2_discover import find_navigate

        if self._nav is None:
            name = self.topic_name((self.cfg["base"] or {}).get("navigate", "navigate_to_pose"))
            # As the command topic above: a server that is running can still be missing from this node's view of the
            # graph for a moment (one look, under load, missed Nav2's and refused a goal it would have taken).
            deadline = _time.monotonic() + self.cfg["timeout"]
            while True:
                actions = {n: list(t) for n, t in rclpy_action.get_action_names_and_types(self.node)}
                if find_navigate(actions, name) is not None or _time.monotonic() > deadline:
                    break
                _time.sleep(0.05)
            if find_navigate(actions, name) is None:
                raise CapabilityError(
                    f"no navigation action {name} (nav2_msgs/action/NavigateToPose) in the ROS graph within "
                    f"{self.cfg['timeout']} s, so the base cannot be sent anywhere: the robot's navigation stack is not running. "
                    "Name it with [ros2] base = {navigate = ...}, or drive it with robot.base.drive."
                )
            self._nav = (ActionClient(self.node, NavigateToPose, name), name)
        client, name = self._nav
        if not client.wait_for_server(timeout_sec=self.cfg["timeout"]):
            raise TimeoutError(f"no action server {name} within {self.cfg['timeout']} s")
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = (self.cfg["base"] or {}).get("in", "map")
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
        self._nav_move = move = {"handle": None, "done": None}

        def answered(future):
            result = future.result()
            move["done"] = (result.status, result.result.error_code, result.result.error_msg)

        def accepted(future):
            move["handle"] = handle = future.result()
            if handle.accepted:
                handle.get_result_async().add_done_callback(answered)
            else:
                move["done"] = (None, None, "the navigation stack rejected the goal")

        client.send_goal_async(goal).add_done_callback(accepted)

    def navigation_done(self) -> bool:
        return self._nav_move["done"] is not None

    def end_navigation(self) -> None:
        """Close the goal sent by :meth:`start_navigation`: ``ActionError`` with Nav2's own reason
        unless it reached the goal."""
        from action_msgs.msg import GoalStatus

        from ..errors import ActionError

        move = self._nav_move
        if move["done"] is None:
            if move["handle"] is not None and move["handle"].accepted:
                move["handle"].cancel_goal_async()
            return
        status, code, text = move["done"]
        if status == GoalStatus.STATUS_SUCCEEDED:
            return
        if status is None:
            raise ActionError(f"navigate_to_pose: {text}")
        word = {GoalStatus.STATUS_ABORTED: "aborted", GoalStatus.STATUS_CANCELED: "canceled"}.get(status, f"ended with status {status}")
        reason = [NAV2_ERRORS.get(code, f"error_code {code}")] if code else []
        raise ActionError(f"navigate_to_pose {word}: {', '.join([*reason, text] if text else reason) or 'Nav2 gave no reason'}")

    def cancel_navigation(self) -> None:
        move = self._nav_move
        if move["handle"] is not None and move["handle"].accepted:
            move["handle"].cancel_goal_async()

    def base_velocity(self) -> np.ndarray:
        """Linear then angular velocity of the base. With Gazebo, its own; without, the change in
        the base's pose between reads of it, over the world's clock."""
        if self.gazebo is not None:
            return self.gazebo.velocity(self._gazebo_model())
        t = self.time
        p, q = self.base_pose()
        last = self.__dict__.get("_base_last")
        self._base_last = (t, p, q)
        if last is None or t - last[0] <= 1e-9:
            return np.zeros(6)
        dt = t - last[0]
        dq = q if np.dot(q, last[2]) >= 0 else -q
        ang = 2 * np.arccos(np.clip(abs(float(np.dot(dq, last[2]))), 0.0, 1.0)) / dt
        return np.concatenate([(p - last[1]) / dt, [ang, 0.0, 0.0]])

    def _gazebo_model(self) -> str:
        """The Gazebo model that is the robot: ``[ros2] gazebo = {model = ...}``, else the one named
        as the robot is (its URDF name, which is the spawned name in the stack's own launch)."""
        if "_gazebo_name" in self.__dict__:
            return self._gazebo_name
        opts = self.cfg["gazebo"] if isinstance(self.cfg["gazebo"], dict) else {}
        models = self.gazebo.models()
        name = opts.get("model") or self.robot_model.name
        if name not in models:
            raise KeyError(
                f"Gazebo world {self.gazebo.world!r} has no model {name!r} for the robot; "
                f"set [ros2] gazebo = {{model = ...}} to one of {models}"
            )
        self._gazebo_name = name
        return name

    def graph(self):
        """The ROS graph this backend's node sees (``world.ros``)."""
        if self._graph is None:
            from ..ros2_graph import RosGraph

            self._graph = RosGraph(self)
        return self._graph

    def topic_name(self, name: str) -> str:
        return _topic(self.cfg["namespace"], name)

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        if self.gazebo is not None and camera not in self.cfg["cameras"] and any(c.name == camera for c in self._gz_cameras):
            return _resize(self.gazebo.image(camera, timeout=self.cfg["timeout"]), width, height)
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
        return _resize(img, width, height)

    def close(self) -> None:
        if getattr(self, "gazebo", None) is not None:
            self.gazebo.close()
            self.gazebo = None
        ex = getattr(self, "_executor", None)
        if ex is not None:
            ex.shutdown()
            self._spin.join(timeout=2)
            self._executor = None
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.node = None


def _resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    if img.shape[:2] != (height, width):  # nearest-neighbour: what a policy asked for, cheaply
        ys = (np.arange(height) * img.shape[0] / height).astype(int)
        xs = (np.arange(width) * img.shape[1] / width).astype(int)
        img = img[ys][:, xs]
    return np.ascontiguousarray(img)


# nav2_msgs/action/ComputePathToPose.action: the planner's error codes, which the navigation action's
# result carries too (its own result has the code and no message of its own, as measured on Nav2 Jazzy)
NAV2_ERRORS = {
    200: "UNKNOWN",
    201: "INVALID_PLANNER",
    202: "TF_ERROR",
    203: "START_OUTSIDE_MAP",
    204: "GOAL_OUTSIDE_MAP",
    205: "START_OCCUPIED",
    206: "GOAL_OCCUPIED",
    207: "TIMEOUT",
    208: "NO_VALID_PATH",
}

_SPAWNED: dict[str, set[str]] = {}  # Gazebo world -> the models robowright spawned into it


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
