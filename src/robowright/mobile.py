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
        and wait until its navigation stack reports it there. A goal the stack cannot reach raises
        ActionError with the stack's own reason."""
        x, y = (float(v) for v in xy)
        timeout = timeout or self.world.settings.action_timeout
        b = self.world.backend
        b.start_navigation(x, y, 0.0 if yaw is None else float(yaw))
        for _ in range(max(1, round(timeout / self.world.dt))):
            if b.navigation_done():
                break
            self.world.step()
        if not b.navigation_done():
            b.cancel_navigation()
            raise ActionTimeoutError(f"the base did not reach ({x}, {y}) within {timeout} s of the world's clock")
        b.end_navigation()


class MobileRobot(Robot):
    """A wheeled robot: its base, and its wheels as joints. No arm, no gripper."""

    _label = "robot"
    name = "robot"

    def __init__(self, world):
        self.world = world
        self.model = world.backend.robot_model
        self.n_arm = self.model.n_arm
        self._base = MobileBase(self)
        self._target = world.backend.qpos().copy()

    @property
    def base(self) -> MobileBase:
        return self._base

    @property
    def joints(self) -> dict[str, float]:
        """The wheels' angles (rad), by the robot's joint names, as its joint states report them."""
        return dict(zip(self.model.arm_joints, map(float, self.world.backend.qpos())))

    def _servo_target(self):
        return self._target.copy()  # nothing is commanded: the robot's own stack drives it

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
