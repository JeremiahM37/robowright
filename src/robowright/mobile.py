"""Mobile robots: a base that drives, and no arm or gripper.

A wheeled robot (a TurtleBot, a differential-drive base) has a base, which is where it is, and
wheels, which are its joints. ``robot.base`` is the base as :mod:`.legged` gives it, and
``robot.base.estimate`` is where the robot thinks its base is (the TF a localizer publishes)::

    expect(robot.base).to_be_near((2.0, 0.5, 0.0), tol=0.25, timeout=90, hold=1.0)
    expect(robot.base.estimate).to_be_near(robot.base, tol=0.1)   # the localizer is right

An action sends the base a command and waits for the answer, as an arm's does::

    robot.base.drive(0.2, 0.0, 1.0)                 # 0.2 m/s straight ahead for 1 s, then stop
    robot.base.go_to((1.0, 0.5), yaw=0.0, timeout=60)   # Nav2 plans and drives; ActionError if it cannot

Nothing here plans or steers: ``drive`` publishes velocities on the base's command topic, and
``go_to`` sends a goal to the robot's navigation stack (Nav2's NavigateToPose), which does the rest.
Without a command topic or a navigation stack the action says so; the robot is observed all the same.

On robowright's own engines (MuJoCo) there is no navigation stack, so ``go_to`` is a turn-drive-turn
controller on the ground-truth pose: it turns to face the goal, drives straight at it, and turns to
the final heading. It has no path planning and no obstacle avoidance; a wall in the way makes it time
out with ActionTimeoutError, as an unreachable goal does.
"""

from __future__ import annotations

import numpy as np

from .errors import ActionTimeoutError
from .legged import Base
from .robot import Robot, action

__tracebackhide__ = True


class Estimate(Base):
    """Where the robot believes its base is: the base's pose as its own localization has it."""

    name = "base.estimate"

    def pose(self):
        return self.world.backend.base_estimate()


class MobileBase(Base):
    """The base: where it is, ``estimate`` (where the robot thinks it is), and the actions that move it."""

    _label = "base"

    def __init__(self, robot):
        super().__init__(robot)
        self.estimate = Estimate(robot)

    @action
    def drive(self, linear: float, angular: float, duration: float):
        """Command the base's velocity (m/s forward, rad/s turning) for ``duration`` seconds, then stop."""
        linear, angular, duration = float(linear), float(angular), float(duration)
        if not np.isfinite([linear, angular, duration]).all() or duration <= 0:
            raise ValueError(f"drive takes finite velocities and a duration above zero, not {linear}, {angular}, {duration}")
        b = self.world.backend
        try:
            b.drive_base(linear, angular)
            self.world.step(max(1, round(duration / self.world.dt)))  # not world.wait: the trace has this action, not a wait
        finally:
            b.stop_base()

    @action
    def go_to(self, xy, yaw: float | None = None, timeout: float | None = None):
        """Send the base to ``xy`` (metres, in the map frame) with heading ``yaw`` (radians, 0 by default),
        and wait until it is there. Behind ROS 2 the robot's navigation stack does this and reports
        its own reason for a goal it cannot reach (ActionError). On robowright's own engines a
        turn-drive-turn controller does it on the ground-truth pose: no planning, no obstacle
        avoidance. Either way, a goal not reached within ``timeout`` raises ActionTimeoutError."""
        x, y = (float(v) for v in xy)
        timeout = timeout or self.world.settings.action_timeout
        b = self.world.backend
        if not hasattr(b, "start_navigation"):  # no navigation stack here: steer to the goal ourselves
            self._turn_drive_turn(x, y, yaw, timeout)
            return
        b.start_navigation(x, y, 0.0 if yaw is None else float(yaw))
        for _ in range(max(1, round(timeout / self.world.dt))):
            if b.navigation_done():
                break
            self.world.step()
        if not b.navigation_done():
            b.cancel_navigation()
            # Where the base is when the stack gives up is the answer to "did it stall, or is it still coming":
            # Nav2 counts a goal reached within its own xy_goal_tolerance, which is wider than an expectation's.
            pos, _ = b.base_pose()
            left = float(np.hypot(x - float(pos[0]), y - float(pos[1])))
            raise ActionTimeoutError(
                f"the base did not reach ({x}, {y}) within {timeout} s of the world's clock: it is at "
                f"({float(pos[0]):.3f}, {float(pos[1]):.3f}), {left:.3f} m from the goal"
            )
        b.end_navigation()

    # Arrival tolerances: a goal counts as reached within these, in metres and radians.
    ARRIVE_XY = 0.03
    ARRIVE_YAW = 0.03

    def _turn_drive_turn(self, x: float, y: float, yaw: float | None, timeout: float):
        """Turn to face the goal, drive at it (correcting heading as it goes), then turn to ``yaw``.
        Steered on the backend's ground-truth pose, one control step at a time; the wheels are
        stopped whether it arrives or not."""
        w, b = self.world, self.world.backend
        deadline = w.time + timeout
        k_turn, k_drive, max_w, max_v = 3.0, 1.0, 1.0, 0.3
        try:
            while True:
                pos, quat = b.base_pose()
                th = _heading(quat)
                dx, dy = x - float(pos[0]), y - float(pos[1])
                dist = float(np.hypot(dx, dy))
                if dist > self.ARRIVE_XY:
                    err = _wrap(float(np.arctan2(dy, dx)) - th)
                    if abs(err) > 0.1:  # too far off to drive yet: turn in place first
                        v = 0.0
                    else:
                        v = min(max_v, k_drive * dist)
                    omega = float(np.clip(k_turn * err, -max_w, max_w))
                elif yaw is not None and abs(_wrap(float(yaw) - th)) > self.ARRIVE_YAW:
                    v, omega = 0.0, float(np.clip(k_turn * _wrap(float(yaw) - th), -max_w, max_w))
                else:
                    return
                if w.time >= deadline - 1e-9:
                    raise ActionTimeoutError(f"the base did not reach ({x}, {y}) within {timeout} s of the world's clock")
                b.drive_base(v, omega)
                w.step()
        finally:
            b.stop_base()


def _heading(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class MobileRobot(Robot):
    """A wheeled robot: its base, and its wheels as joints. No arm, no gripper."""

    _label = "robot"
    name = "robot"

    def __init__(self, world):
        self.world = world
        self.model = world.backend.robot_model
        self.n_arm = self.model.n_arm
        self._base = MobileBase(self)

    @property
    def base(self) -> MobileBase:
        return self._base

    @property
    def joints(self) -> dict[str, float]:
        """The wheels' angles (rad), by the robot's joint names, as its joint states report them."""
        return dict(zip(self.model.arm_joints, map(float, self.world.backend.qpos())))

    def _servo_target(self):
        # The wheels' setpoints: set by base.drive, and zero otherwise (or the robot's own stack drives it).
        return self.world.backend.wheel_command()

    _ARM_ONLY = frozenset(
        {"arm", "gripper", "tcp", "pick", "place", "kin", "grip_yaw", "min_grasp_z", "stand", "crouch", "move_joints", "reset_to", "home_q"}
    )

    def _no(self, name: str) -> AttributeError:
        title = object.__getattribute__(self, "model").title
        return AttributeError(
            f"{title} is a mobile robot: it has no {name!r}. Mobile robots have robot.base (and robot.base.estimate) and robot.joints"
        )

    def __getattribute__(self, name):
        if name in MobileRobot._ARM_ONLY:
            raise object.__getattribute__(self, "_no")(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        # Only reached for names the robot lacks: arm-only names say what a mobile robot has, and
        # pose queries point at the base.
        if name in MobileRobot._ARM_ONLY:
            raise self._no(name)
        if name in ("up_axis", "position", "quaternion", "pose", "height", "yaw", "velocity", "top", "bounds"):
            raise AttributeError(f"the robot has no {name!r}: its pose is the base's - use robot.base (e.g. expect(robot.base))")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
