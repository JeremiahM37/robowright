"""Serve a simulated world as a ROS 2 robot, so the code under test is yours.

A test that calls ``robot.pick(cube)`` tests robowright's grasp. To test *your* robot software
(a ROS 2 stack, MoveIt, a policy node), let it drive the simulated robot exactly as it drives
the real one, and have the test only watch and assert::

    def test_my_stack_puts_the_cube_in_the_bin(world, scene, ros2):
        with ros2.run("ros2 launch my_robot pick.launch.py use_sim_time:=true"):
            expect(scene["cube"]).to_be_inside(scene["bin"], timeout=60)

The bridge speaks the interfaces a ``ros2_control`` robot has, so nothing in your stack changes
between this simulation, another simulator, and the real arm:

* ``/joint_states`` (``sensor_msgs/JointState``) and ``/clock`` (sim time) out, every control step;
* the objects as TF frames relative to the robot's base frame, and each camera as
  ``/camera/<name>/image_raw``;
* ``/<arm_controller>/follow_joint_trajectory`` (``control_msgs/FollowJointTrajectory``, what
  MoveIt and most stacks send) in, interpolated in sim time as ``joint_trajectory_controller`` does;
* ``/<gripper_controller>/gripper_cmd`` (``control_msgs/GripperCommand``) in;
* ``/forward_position_controller/commands`` (``std_msgs/Float64MultiArray``, arm joints then
  the gripper joint) in, as ``forward_command_controller`` takes them.

The world stays robowright's: ground truth, contacts, faults and the trace all work as in any
other test, and the trace's timeline shows each goal your stack sent. The world advances in
real time while the bridge serves it (``expect`` and ``world.wait`` step it), so your nodes'
timers and timeouts behave as they would on hardware. Run your nodes with
``use_sim_time:=true`` so their clocks follow the simulation's.

Needs ``rclpy`` (a ROS 2 install, or ``scripts/ros2_env.sh``).
"""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import tempfile
import threading
import time as _time
from pathlib import Path

import numpy as np


class Ros2Bridge:
    """``world`` served as a ROS 2 robot (see the module). Close it, or use it as a context manager."""

    def __init__(
        self,
        world,
        namespace: str = "",
        arm_controller: str = "arm_controller",
        gripper_controller: str = "gripper_controller",
        forward_controller: str = "forward_position_controller",
        frame: str = "base_link",
        joint_names: dict | None = None,
        camera_hz: float = 10.0,
        realtime: bool = True,
        node_name: str = "robowright_sim",
    ):
        import rclpy
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Float64MultiArray
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self.world = w = world
        self._msgs = {"Clock": Clock, "JointState": JointState, "Image": Image}
        m = w.backend.robot_model
        self._m = m
        rename = dict(joint_names or {})
        self.arm_names = [rename.get(j, j) for j in m.arm_joints]
        self._grip = None
        if m.has_gripper:
            grip, (closed, opened) = next(iter(m.derived.gripper_joints.items()))
            self._grip = (rename.get(grip, grip), closed, opened)
        self.frame = frame
        self.realtime = realtime
        self._camera_every = max(1, round(1.0 / (camera_hz * w.dt))) if camera_hz else 0
        self._lock = threading.Lock()
        self._trajectory = None  # (goal handle, start sim time, times, positions, done event)
        self._gripper_goal = None  # (target opening, done event, result holder)
        self._forward = None
        self._wall0 = self._sim0 = None
        self.goals: list[str] = []  # what the stack under test asked for, in order (also in the trace)

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False
        ns = namespace.rstrip("/")
        self.node = node = rclpy.create_node(node_name, namespace=ns or None)
        cb = ReentrantCallbackGroup()
        self._clock = node.create_publisher(Clock, "/clock", 10)
        self._states = node.create_publisher(JointState, "joint_states", 10)
        self._images = {c.name: node.create_publisher(Image, f"camera/{c.name}/image_raw", 2) for c in w.spec.cameras}
        self._tf = TransformBroadcaster(node)
        static = StaticTransformBroadcaster(node)
        static.sendTransform(self._base_transform())
        node.create_subscription(Float64MultiArray, f"{forward_controller}/commands", self._on_forward, 10)

        def goal_ok(goal):
            missing = set(goal.trajectory.joint_names) - set(self.arm_names)
            if missing:
                node.get_logger().error(f"trajectory names joints this arm does not have: {sorted(missing)}")
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT

        self._jtc = ActionServer(
            node,
            FollowJointTrajectory,
            f"{arm_controller}/follow_joint_trajectory",
            execute_callback=self._execute_trajectory,
            goal_callback=goal_ok,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=cb,
        )
        self._FJT, self._GC = FollowJointTrajectory, GripperCommand
        if self._grip is not None:
            self._gripper_server = ActionServer(
                node, GripperCommand, f"{gripper_controller}/gripper_cmd", execute_callback=self._execute_gripper, callback_group=cb
            )
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(node)
        self._spin = threading.Thread(target=_spin, args=(self._executor,), name="robowright-ros2", daemon=True)
        self._spin.start()
        w._step_hooks.append(self._on_step)
        if w.trace:
            w.trace.event("edit", "ros2_bridge", {"namespace": ns, "arm": arm_controller, "gripper": gripper_controller})
        self._publish()

    # -- the world's side (its own thread) ---------------------------------------------
    def _on_step(self, w):
        r = w.robot
        with self._lock:
            traj, grip, fwd = self._trajectory, self._gripper_goal, self._forward
            self._forward = None
        n = len(self.arm_names)
        if fwd is not None:
            r._target[:n] = fwd[:n]
            if self._grip is not None and len(fwd) > n:
                r._target[-1] = self._opening(fwd[n])
        if traj is not None:
            handle, t0, times, points, done, idx = traj
            q = _interpolate(w.time - t0, times, points)
            r._target[idx] = q
            if w.time - t0 >= times[-1]:
                done.set()
        if grip is not None:
            target, done, out = grip
            r._target[-1] = target
            opening = float(r.true_qpos()[-1])
            moving = abs(float(w.backend.qvel()[-1])) > 1e-3
            if abs(opening - target) < 0.02 or (not moving and out.setdefault("still", 0) > int(0.2 / w.dt)):
                out["opening"] = opening
                done.set()
            out["still"] = out.get("still", 0) + (not moving)
        self._publish()
        if self.realtime:
            self._pace(w)

    def _pace(self, w):
        now = _time.perf_counter()
        if self._wall0 is None:
            self._wall0, self._sim0 = now, w.time
        ahead = (w.time - self._sim0) - (now - self._wall0)
        if ahead > 0:
            _time.sleep(ahead)
        elif ahead < -0.25:  # fell behind (a slow step): resume pacing from here rather than race
            self._wall0, self._sim0 = now, w.time

    def _publish(self):
        w, node = self.world, self.node
        from builtin_interfaces.msg import Time

        t = w.time
        stamp = Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)) % 1_000_000_000)
        self._clock.publish(self._msgs["Clock"](clock=stamp))
        q, v = w.robot.qpos(), w.backend.qvel()  # readings as the robot reports them (sensor faults apply)
        names, pos, vel = list(self.arm_names), list(map(float, q[: len(self.arm_names)])), list(map(float, v[: len(self.arm_names)]))
        if self._grip is not None:
            name, closed, opened = self._grip
            names.append(name)
            pos.append(closed + float(q[-1]) * (opened - closed))
            vel.append(float(v[-1]) * (opened - closed))
        msg = self._msgs["JointState"](name=names, position=pos, velocity=vel)
        msg.header.stamp = stamp
        self._states.publish(msg)
        self._tf.sendTransform([self._object_transform(n, stamp) for n in w.object_names])
        if self._camera_every and w.step_count % self._camera_every == 0 and _RENDER in w.backend.capabilities:
            for c in w.spec.cameras:
                img = w.faults.filter_image(w.backend.render(c.name, *w.settings.image_size))
                h, wd = img.shape[:2]
                im = self._msgs["Image"](height=h, width=wd, encoding="rgb8", step=wd * 3, data=img.tobytes())
                im.header.stamp, im.header.frame_id = stamp, c.name
                self._images[c.name].publish(im)
        del node

    def _base_transform(self):
        from geometry_msgs.msg import TransformStamped

        m = self._m
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = "world", self.frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, m.base_pos)
        t.transform.rotation.w, t.transform.rotation.z = float(np.cos(m.base_yaw / 2)), float(np.sin(m.base_yaw / 2))
        return t

    def _object_transform(self, name, stamp):
        from geometry_msgs.msg import TransformStamped

        m = self._m
        p, quat = self.world.backend.object_pose(name)
        c, s = np.cos(-m.base_yaw), np.sin(-m.base_yaw)
        p = np.asarray(p, float) - np.asarray(m.base_pos, float)
        p = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
        quat = _turn(-m.base_yaw, quat)
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, self.frame, name
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, p)
        t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = map(float, quat)
        return t

    def _opening(self, joint_position: float) -> float:
        _, closed, opened = self._grip
        return float(np.clip((joint_position - closed) / (opened - closed), 0.0, 1.0))

    def _note(self, text: str):
        self.goals.append(text)
        if self.world.trace:
            self.world.trace.event("edit", "ros2_goal", {"goal": text})

    # -- ROS's side (executor threads) --------------------------------------------------
    def _on_forward(self, msg):
        with self._lock:
            self._forward = np.array(msg.data, float)

    def _execute_trajectory(self, handle):
        tr = handle.request.trajectory
        idx = [self.arm_names.index(n) for n in tr.joint_names]
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in tr.points], float)
        points = np.array([list(p.positions) for p in tr.points], float)
        result = self._FJT.Result()
        if not len(points):
            handle.succeed()
            return result
        done = threading.Event()
        w = self.world
        with self._lock:
            start = np.asarray(w.robot._target[idx], float)
            # Start from where the arm is commanded now, as joint_trajectory_controller does.
            times, points = (np.insert(times, 0, 0.0), np.vstack([start, points])) if times[0] > 0 else (times, points)
            self._trajectory = (handle, w.time, times, points, done, idx)
        self._note(f"FollowJointTrajectory: {len(tr.points)} point(s) over {times[-1]:.2f} s for {', '.join(tr.joint_names)}")
        while not done.wait(0.05):
            if handle.is_cancel_requested:
                with self._lock:
                    self._trajectory = None
                handle.canceled()
                result.error_code = -1
                return result
            if not self._spin.is_alive():
                break
        with self._lock:
            if self._trajectory is not None and self._trajectory[0] is handle:
                self._trajectory = None
        handle.succeed()
        result.error_code = self._FJT.Result.SUCCESSFUL
        return result

    def _execute_gripper(self, handle):
        target = self._opening(handle.request.command.position)
        done, out = threading.Event(), {}
        with self._lock:
            self._gripper_goal = (target, done, out)
        self._note(f"GripperCommand: position {handle.request.command.position:.4f} (opening {target:.2f})")
        done.wait()
        with self._lock:
            if self._gripper_goal is not None and self._gripper_goal[1] is done:
                self._gripper_goal = None
        _, closed, opened = self._grip
        result = self._GC.Result()
        result.position = closed + out["opening"] * (opened - closed)
        result.reached_goal = abs(out["opening"] - target) < 0.02
        result.stalled = not result.reached_goal
        handle.succeed()
        return result

    # -- running the code under test ----------------------------------------------------
    @contextlib.contextmanager
    def run(self, command: str | list, env: dict | None = None, cwd: str | None = None, startup: float = 0.0):
        """Start ``command`` (your launch file, node or script) for the duration of the block, then
        stop it. Its output is kept in a log file whose path is in the trace and in the failure.
        ``startup``: seconds of simulated time to let pass before the block (the world steps,
        at real-time pace, while your nodes come up)."""
        log = Path(tempfile.mkstemp(prefix="robowright-stack-", suffix=".log")[1])
        args = shlex.split(command) if isinstance(command, str) else list(command)
        proc = subprocess.Popen(
            args, stdout=log.open("w"), stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}, cwd=cwd, start_new_session=True
        )
        self._note(f"started: {' '.join(args)} (log {log})")
        try:
            if startup:
                self.world.wait(startup)
            if proc.poll() is not None:
                raise RuntimeError(f"{args[0]} exited with {proc.returncode} at start:\n{log.read_text()[-3000:]}")
            yield proc
        except BaseException as e:
            tail = log.read_text()[-2000:]
            if tail and hasattr(e, "add_note"):
                e.add_note(f"output of the code under test ({log}):\n{tail}")
            raise
        finally:
            _stop(proc)

    def close(self):
        if self._on_step in self.world._step_hooks:
            self.world._step_hooks.remove(self._on_step)
        self._executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        if self._owns_rclpy:
            import rclpy

            with contextlib.suppress(Exception):
                rclpy.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_RENDER = "render"


def _spin(executor) -> None:
    """Spin until shut down (a shutdown from elsewhere ends the loop, not the thread with an error)."""
    from rclpy.executors import ExternalShutdownException

    with contextlib.suppress(ExternalShutdownException):
        executor.spin()


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(5)


def _interpolate(t: float, times: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Positions at time ``t`` along a trajectory, linearly between its points (held after the last)."""
    if t <= times[0]:
        return points[0]
    if t >= times[-1]:
        return points[-1]
    i = int(np.searchsorted(times, t)) - 1
    s = (t - times[i]) / max(times[i + 1] - times[i], 1e-9)
    return points[i] + s * (points[i + 1] - points[i])


def _turn(yaw: float, q):
    """``q`` (w, x, y, z) turned by ``yaw`` about the vertical."""
    w1, z1 = np.cos(yaw / 2), np.sin(yaw / 2)
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2, w1 * y2 + z1 * x2, w1 * z2 + z1 * w2])
