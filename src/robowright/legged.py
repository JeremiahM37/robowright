"""Legged robots: quadrupeds and humanoids on a floating base.

The API mirrors the arm's: actions stream joint targets and wait until the
robot settles, and ``robot.base`` is a live handle that ``expect`` reasons
about like any object::

    robot.stand()
    expect(robot.base).always.to_be_upright(tol_deg=20)
    world.faults.push("robot", force=(0, 60, 0), duration=0.1)
    world.wait(1.0)
    expect(robot.base).to_be_above(0.2)

There is no locomotion controller here: walking is a policy's job, run with
``robot.run_policy`` like any other policy.
"""

from __future__ import annotations

import numpy as np

from .locators import Subject
from .robot import Robot, Rollout, _minjerk, action  # noqa: F401  (Rollout re-exported)

__tracebackhide__ = True


class Base(Subject):
    """The floating base: position, orientation and velocity, live."""

    name = "base"

    def __init__(self, robot):
        self.robot = robot
        self.world = robot.world

    def pose(self):
        return self.world.backend.base_pose()

    @property
    def position(self):
        return self.pose()[0]

    @property
    def quaternion(self):
        return self.pose()[1]

    @property
    def height(self) -> float:
        return float(self.position[2])

    @property
    def up_axis(self) -> np.ndarray:
        w, x, y, z = self.quaternion
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    @property
    def yaw(self) -> float:
        w, x, y, z = self.quaternion
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @property
    def velocity(self) -> np.ndarray:
        return self.world.backend.base_velocity()

    def contacts(self) -> list[str]:
        """Everything any part of the robot touches, other than itself."""
        self.world.require("contacts", "contact queries")
        out = set()
        for c in self.world.backend.contacts():
            for me, other in ((c.a, c.b), (c.b, c.a)):
                if me.startswith("robot:") and not other.startswith("robot:"):
                    out.add(other)
        return sorted(out)


class LeggedRobot(Robot):
    """A quadruped or humanoid: joint-space actions plus a floating base."""

    _label = "robot"
    name = "robot"

    def __init__(self, world):
        self.world = world
        self.model = world.backend.robot_model
        self.n_arm = self.model.n_arm
        self._base = Base(self)
        self._target = world.backend.qpos().copy()
        self._home_q = np.asarray(self.model.stand_q(), float)

    @property
    def base(self) -> Base:
        return self._base

    def _servo_target(self) -> np.ndarray:
        return self._target.copy()  # joints that stand on the ground get their targets as given

    @property
    def home_q(self) -> np.ndarray:
        return self._home_q

    @property
    def total_mass(self) -> float:
        return self.model.total_mass

    def reset_to(self, q=None, base_pos=None, yaw: float | None = None):
        """Teleport to a joint configuration, standing at ``base_pos`` (setup only).

        By default the robot stands in its standing pose with its feet on the floor.
        """
        m = self.model
        q = self.home_q if q is None else np.asarray(q, float)
        b = self.world.backend
        x, y = (base_pos or m.base_pos)[:2]
        z = m.stand_height if base_pos is None or len(base_pos) < 3 else base_pos[2]
        yaw = m.base_yaw if yaw is None else yaw
        b.set_base_pose((x, y, z), (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
        b.set_joint_positions(q)
        self._target = q.copy()
        if self.world.trace:
            args = {"q": [float(v) for v in q], "base_pos": [float(x), float(y), float(z)], "yaw": float(yaw)}
            if q is self.home_q and base_pos is None and yaw == m.base_yaw:
                args["default"] = True  # codegen writes robot.reset_to(): same pose, readable
            self.world.trace.event("edit", "reset_to", args)

    @action
    def move_joints(self, q, speed: float | None = None, timeout: float | None = None):
        """Move every joint to ``q`` (radians) along a smooth path and wait until settled."""
        goal = np.asarray(q, float)
        start = self._target.copy()
        speed = speed or self.world.settings.max_joint_speed
        duration = max(float(np.max(np.abs(goal - start))) / speed, self.world.dt)
        self._stream(lambda s: self._set_arm(start + (goal - start) * _minjerk(s)), duration)
        # Legs carry the robot's weight, so compliant servos settle off target by design;
        # "settled" means it has stopped moving reasonably close to where it was sent.
        self._settle(goal, timeout, tol=0.25)

    @action
    def stand(self, timeout: float | None = None):
        """Return to the standing pose."""
        self.move_joints.__wrapped__(self, self.home_q, timeout=timeout)

    @action
    def crouch(self, depth: float = 0.5, timeout: float | None = None):
        """Bend toward the joints' folded limits by ``depth`` (0 = standing, 1 = fully folded)."""
        self.move_joints.__wrapped__(self, self.crouch_q(depth), timeout=timeout)

    def crouch_q(self, depth: float) -> np.ndarray:
        """Knees and hips bent further in the direction they already bend when standing."""
        if self.model.crouch is not None:
            return self.home_q + depth * (np.asarray(self.model.crouch, float) - self.home_q)
        q = self.home_q.copy()
        return q + depth * 0.6 * np.sign(q) * (np.abs(q) > 0.2)

    def observe(self, cameras=(), privileged: bool = False, task: str | None = None, image_size: tuple | None = None) -> dict:
        obs = super().observe(cameras, False, task, image_size)
        b = self.world.backend
        pos, quat = b.base_pose()
        v = b.base_velocity()
        # What an IMU and leg odometry give a real robot: orientation and velocities.
        obs.update(base_quat=quat, base_lin_vel=v[:3], base_ang_vel=v[3:], qvel=b.qvel())
        if privileged:
            obs["base_pos"] = pos
            w = self.world
            seen = w.object_names if w.has_ground_truth else [n for n in w.object_names if n in w.perception]
            obs["objects"] = {n: w.scene[n].pose() for n in seen}
        return obs

    def _set_gripper(self, a):
        raise AttributeError(f"{self.model.title} has no gripper")

    _ARM_ONLY = frozenset({"arm", "gripper", "tcp", "pick", "place", "kin", "grip_yaw", "min_grasp_z"})

    def __getattribute__(self, name):
        if name in LeggedRobot._ARM_ONLY:
            title = object.__getattribute__(self, "model").title
            raise AttributeError(
                f"{title} is a legged robot: it has no {name!r}. Legged robots have robot.base, "
                "robot.stand(), robot.crouch(), robot.move_joints() and robot.run_policy()"
            )
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        # Only reached for names the robot lacks: point pose queries (and matchers) at the base.
        if name in ("up_axis", "position", "quaternion", "pose", "height", "yaw", "velocity", "top", "bounds"):
            raise AttributeError(f"the robot has no {name!r}: its pose is the base's - use robot.base (e.g. expect(robot.base))")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
