"""What is running behind ROS 2: the robot's description, its joints, its controllers and
how each is driven, read from the ROS graph itself.

A test author (or an agent) who meets someone else's stack should not have to know its
controller names: ``inspect()`` asks ``controller_manager`` what it has loaded, reads which joints
each controller claims, and ``suggest()`` turns that into the ``[ros2]`` settings robowright
connects with. ``robowright ros2 inspect`` prints both::

    $ robowright ros2 inspect
    robot description: /robot_description (panda, 9 joints)
    joint states: /joint_states (9 joints, 100 Hz)
    controllers (/controller_manager):
      panda_arm_controller   joint_trajectory_controller/JointTrajectoryController   active   7 joints
      panda_hand_controller  position_controllers/GripperActionController            active   panda_finger_joint1
    ...
    [ros2]
    arm = { controller = "panda_arm_controller", interface = "trajectory" }
    gripper = { controller = "panda_hand_controller", interface = "action" }
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
from dataclasses import dataclass, field

# How robowright drives each kind of controller (see robowright.backends.ros2_backend)
_ARM = (("JointTrajectoryController", "trajectory"), ("JointGroupPositionController", "position"))
_GRIPPER = (("GripperActionController", "action"), ("ParallelGripper", "action"))
# A mobile base's velocity command, and the action Nav2 serves its goals on (see robowright.backends.ros2_backend)
CMD_VEL_TYPES = ("geometry_msgs/msg/Twist", "geometry_msgs/msg/TwistStamped")
NAVIGATE_TYPE = "nav2_msgs/action/NavigateToPose"


@dataclass
class Controller:
    name: str
    type: str
    state: str
    joints: list[str]  # the joints whose command interfaces it has claimed (or would claim)

    @property
    def active(self) -> bool:
        return self.state == "active"

    @property
    def interface(self) -> tuple[str, str] | None:
        """``("arm" | "gripper", interface)`` for a controller robowright can drive, else None."""
        for part, kinds in (("gripper", _GRIPPER), ("arm", _ARM)):
            for needle, interface in kinds:
                if needle in self.type:
                    return part, interface
        return None


def gripper_trajectory(g: Graph, arm: str | None, joint: str | None) -> Controller | None:
    """The active joint trajectory controller that drives the gripper's joint, when the stack has no
    gripper controller of its own: a gripper that is one more joint of a trajectory controller (xArm's
    is ``xarm_gripper_traj_controller``, claiming ``drive_joint``) is driven through that. ``arm`` is the
    arm's controller, which never counts as the gripper."""
    if not joint:
        return None
    for c in g.controllers:
        if c.active and c.name != arm and c.interface and c.interface[0] == "arm" and joint in c.joints:
            return c
    return None


@dataclass
class Graph:
    """What ``inspect`` found."""

    namespace: str = ""
    description: str | None = None  # the URDF text, from the description topic
    description_topic: str | None = None
    robot: str | None = None  # the URDF's robot name
    joint_states: str | None = None
    joints: list[str] = field(default_factory=list)
    joint_rate: float | None = None  # Hz
    manager: str | None = None  # the controller manager's node name
    controllers: list[Controller] = field(default_factory=list)
    sim_time: bool = False  # something publishes /clock
    frames: list[str] = field(default_factory=list)  # TF frames seen
    topics: dict[str, list[str]] = field(default_factory=dict)  # every topic and its types
    gazebo: list[str] = field(default_factory=list)  # Gazebo worlds on this machine's gz transport
    cmd_vel: tuple[str, str, int] | None = None  # the base's velocity command: (topic, type, subscribers)
    navigate: str | None = None  # the navigation action's name, when a NavigateToPose server is there
    notes: list[str] = field(default_factory=list)


def find_cmd_vel(topics: dict[str, list[str]], subscribers: dict[str, int], name: str) -> tuple[str, str] | None:
    """``(topic, type)`` for the velocity command ``name`` when something subscribes to it and its
    type is one robowright publishes (Jazzy's stacks differ: Twist or TwistStamped), else None."""
    types = [t for t in topics.get(name, []) if t in CMD_VEL_TYPES]
    if not types or not subscribers.get(name):
        return None
    return name, types[0]


def find_navigate(actions: dict[str, list[str]], name: str) -> str | None:
    """``name`` when a server for ``nav2_msgs/action/NavigateToPose`` is on it, else None."""
    return name if NAVIGATE_TYPE in actions.get(name, []) else None


def _topic(ns: str, name: str) -> str:
    return name if name.startswith("/") else (f"{ns.rstrip('/')}/{name}" if ns else f"/{name}")


def inspect(namespace: str = "", timeout: float = 5.0, gazebo: bool = True) -> Graph:
    """Look at the ROS graph under ``namespace`` for up to ``timeout`` seconds (wall clock)."""
    import rclpy
    import rclpy.action as rclpy_action
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    ctx = rclpy.Context()
    rclpy.init(context=ctx)
    node = rclpy.create_node(f"robowright_inspect_{int(time.time() * 1e3) % 10**6}", context=ctx)
    ex = SingleThreadedExecutor(context=ctx)
    ex.add_node(node)

    def spin():
        from rclpy.executors import ExternalShutdownException

        with contextlib.suppress(ExternalShutdownException):
            ex.spin()

    threading.Thread(target=spin, daemon=True).start()
    g = Graph(namespace=namespace)
    stamps: list[float] = []
    try:
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        g.description_topic = _topic(namespace, "robot_description")
        g.joint_states = _topic(namespace, "joint_states")

        def on_description(msg):
            g.description = msg.data

        def on_joints(msg):
            stamps.append(time.monotonic())
            for name in msg.name:
                if name not in g.joints:
                    g.joints.append(name)

        node.create_subscription(String, g.description_topic, on_description, latched)
        node.create_subscription(JointState, g.joint_states, on_joints, 50)
        frames: set[str] = set()
        cmd_name = _topic(namespace, "cmd_vel")
        nav_name = _topic(namespace, "navigate_to_pose")
        with contextlib.suppress(ImportError):
            from tf2_msgs.msg import TFMessage

            def on_tf(msg):
                for t in msg.transforms:
                    frames.update((t.header.frame_id.lstrip("/"), t.child_frame_id.lstrip("/")))

            node.create_subscription(TFMessage, "/tf", on_tf, 50)
            node.create_subscription(TFMessage, "/tf_static", on_tf, latched)
        deadline = time.monotonic() + timeout
        # Discovery takes a moment; controllers are asked for once their manager is seen
        settled = time.monotonic()
        while time.monotonic() < deadline:
            topics = {name: list(types) for name, types in node.get_topic_names_and_types()}
            if topics.keys() != g.topics.keys():
                settled = time.monotonic()
            g.topics = topics
            g.navigate = find_navigate({n: list(t) for n, t in rclpy_action.get_action_names_and_types(node)}, nav_name)
            if g.manager is None:
                g.manager = _manager(node, namespace)
            if g.manager and not g.controllers:
                g.controllers = _controllers(node, g.manager, deadline)
            # Each process in the stack is discovered on its own: a graph still growing (the bridge that
            # publishes /clock, seen after the controller manager) is not yet the graph to report
            quiet = time.monotonic() - settled > 0.5
            if g.description and len(stamps) > 20 and quiet and (g.controllers or time.monotonic() > deadline - timeout / 2):
                break
            time.sleep(0.1)
        g.sim_time = node.count_publishers("/clock") > 0
        subs = node.count_subscribers(cmd_name)
        if (found := find_cmd_vel(g.topics, {cmd_name: subs}, cmd_name)) is not None:
            g.cmd_vel = (*found, subs)
        g.frames = sorted(frames - {""})
        if len(stamps) > 1:
            g.joint_rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        if g.description:
            found = re.search(r"<robot[^>]*\bname\s*=\s*[\"']([^\"']+)", g.description)
            g.robot = found.group(1) if found else None
        else:
            g.description_topic = None
            g.notes.append(f"no robot description on {_topic(namespace, 'robot_description')}: give robowright the robot's model file")
        if not g.joints:
            g.joint_states = None
            g.notes.append(f"no joint states on {_topic(namespace, 'joint_states')}: is the robot's driver running, in this ROS domain?")
        if g.manager is None:
            g.notes.append("no controller manager found: robowright can observe this robot, and drive it only through topics you name")
    finally:
        ex.shutdown(timeout_sec=1.0)
        node.destroy_node()
        ctx.try_shutdown()
    if gazebo:
        g.gazebo = _gazebo_worlds()
    return g


def _manager(node, namespace: str) -> str | None:
    """The controller manager's service prefix: the node offering ``list_controllers``."""
    for name, types in node.get_service_names_and_types():
        if name.endswith("/list_controllers") and "controller_manager_msgs/srv/ListControllers" in types:
            prefix = name[: -len("/list_controllers")]
            if prefix.startswith(namespace.rstrip("/") or "/"):
                return prefix
    return None


def _controllers(node, manager: str, deadline: float) -> list[Controller]:
    try:
        from controller_manager_msgs.srv import ListControllers
    except ImportError:
        return []
    client = node.create_client(ListControllers, f"{manager}/list_controllers")
    try:
        if not client.wait_for_service(timeout_sec=max(0.1, deadline - time.monotonic())):
            return []
        future = client.call_async(ListControllers.Request())
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            return []
        out = []
        for c in future.result().controller:
            claimed = list(c.claimed_interfaces) or list(c.required_command_interfaces)
            joints = list(dict.fromkeys(i.rsplit("/", 1)[0] for i in claimed))
            out.append(Controller(c.name, c.type, c.state, joints))
        return out
    finally:
        node.destroy_client(client)


def _gazebo_worlds() -> list[str]:
    try:
        from .gazebo import worlds
    except ImportError:
        return []
    with contextlib.suppress(Exception):
        return list(worlds(timeout=1.0))
    return []


def suggest(g: Graph, arm_joints: list[str] | None = None) -> dict:
    """The ``[ros2]`` settings to connect to what ``inspect`` found: the active controller that
    drives the arm's joints (``arm_joints`` if the robot's model is known, else the one driving
    the most joints) and the gripper's. Nothing is guessed: a part with no controller robowright
    can drive is left out, and ``g.notes`` says why."""
    out: dict = {}
    if g.namespace:
        out["namespace"] = g.namespace
    if g.sim_time:
        out["use_sim_time"] = True
    arms = [c for c in g.controllers if c.active and c.interface and c.interface[0] == "arm"]
    if arm_joints:
        want = set(arm_joints)
        arms = [c for c in arms if want <= set(c.joints)] or [c for c in arms if want & set(c.joints)]
    arms.sort(key=lambda c: -len(c.joints))
    if arms:
        arm = arms[0]
        out["arm"] = {"controller": arm.name, "interface": arm.interface[1]}
        if len(arms) > 1:
            names = ", ".join(c.name for c in arms)
            g.notes.append(f"several controllers drive the arm ({names}): took {arm.name}, which claims the most joints")
    else:
        idle = [c.name for c in g.controllers if not c.active and c.interface and c.interface[0] == "arm"]
        why = f"{', '.join(idle)} not active" if idle else "none robowright can drive is loaded"
        # A mobile base has no arm controller by design: it is driven through its command topic or navigation action
        if not (g.cmd_vel or g.navigate):
            if g.controllers:
                g.notes.append(f"no active arm controller ({why}): robowright can observe (command = false)")
            out["command"] = False
    base = {}
    if g.cmd_vel:
        base["cmd_vel"] = g.cmd_vel[0]
    if g.navigate:
        base["navigate"] = g.navigate
    if base:
        # Without the frames the base's pose is not read, and the belief (where the stack thinks it is) is
        # not there either: a base's frame and the frame it localizes in are found in TF when TF has them.
        frame = next((f for f in ("base_footprint", "base_link") if f in g.frames), None)
        if frame is not None:
            base["frame"] = frame
            if "map" in g.frames:
                base["in"] = "map"
            elif "odom" in g.frames:
                base["in"] = "odom"
        out["base"] = base
    grips = [c for c in g.controllers if c.active and c.interface and c.interface[0] == "gripper"]
    if grips:
        out["gripper"] = {"controller": grips[0].name, "interface": grips[0].interface[1]}
        if "ParallelGripper" in grips[0].type:  # its action is ParallelGripperCommand, not GripperCommand
            out["gripper"]["parallel"] = True
    elif "arm" in out:
        arm_set = set(arms[0].joints)
        out["gripper"] = {"interface": "with_arm" if arm_joints and len(arm_set) > len(arm_joints) else "none"}
    if g.gazebo:
        out["gazebo"] = True if len(g.gazebo) == 1 else {"world": g.gazebo[0]}
    return out


def report(g: Graph, settings: dict | None = None) -> str:
    """``inspect``'s findings and the settings to use, as text."""
    lines = []
    if g.description_topic:
        lines.append(f"robot description: {g.description_topic} ({g.robot or 'unnamed'})")
    if g.joint_states:
        rate = f", {g.joint_rate:.0f} Hz" if g.joint_rate else ""
        lines.append(f"joint states: {g.joint_states} ({len(g.joints)} joints{rate}): {', '.join(g.joints)}")
    if g.controllers:
        lines.append(f"controllers ({g.manager}):")
        w = max(len(c.name) for c in g.controllers)
        t = max(len(c.type) for c in g.controllers)
        for c in g.controllers:
            joints = f"{len(c.joints)} joints" if len(c.joints) > 3 else ", ".join(c.joints)
            lines.append(f"  {c.name:<{w}}  {c.type:<{t}}  {c.state:<8}  {joints}".rstrip())
    lines.append(f"clock: {'simulation time (/clock is published)' if g.sim_time else 'wall clock (no /clock)'}")
    if g.gazebo:
        lines.append(f"Gazebo: world {', '.join(g.gazebo)}")
    if g.cmd_vel:
        topic, typ, subs = g.cmd_vel
        lines.append(f"base command: {topic} ({typ}, {subs} subscriber{'s' if subs != 1 else ''})")
    if g.navigate:
        lines.append(f"navigation: {g.navigate} (nav2_msgs/action/NavigateToPose)")
    if g.frames:
        shown = ", ".join(g.frames[:12]) + (f", ... ({len(g.frames)} in all)" if len(g.frames) > 12 else "")
        lines.append(f"TF frames: {shown}")
    lines += [f"note: {n}" for n in g.notes]
    if settings is not None:
        lines += ["", "[ros2]", *(f"{k} = {_toml(v)}" for k, v in settings.items())]
    return "\n".join(lines)


def _toml(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml(x)}" for k, x in v.items()) + " }"
    return f'"{v}"' if isinstance(v, str) else repr(v)
