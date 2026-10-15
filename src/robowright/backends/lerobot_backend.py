"""A real arm through LeRobot's own driver: ``rw.launch(backend="lerobot", robot="so101", lerobot=arm)``.

``arm`` is a LeRobot robot, made and calibrated as LeRobot's own scripts make it (an SO-100/101,
Koch, OMX or any other with LeRobot's ``Robot`` interface: ``connect``, ``get_observation``,
``send_action``). robowright never configures the hardware: it reads ``<motor>.pos`` from the
robot's observation and sends ``<motor>.pos`` targets, at the control rate, in wall-clock time.

Units: LeRobot reports joints in degrees (``use_degrees=True``, its default) and a gripper on its
0..100 range. ``calibration`` maps a joint to (sign, offset in radians) where LeRobot's zero or
direction differs from the robot model's: robowright angle = sign * LeRobot angle + offset.

Safety: this is always hardware, so ``limits`` (as ``[ros2] limits``: ``joint_speed``, ``positions``)
default to 1 rad/s on every arm joint. robowright's own moves are planned inside them; a command it did
not plan that would break them is refused before it is sent (robowright.limits). Nothing is moved when
the world opens: ``robot.reset_to()`` drives to the home pose, slowly, only when the test asks.
"""

from __future__ import annotations

import time as _time

import numpy as np

from .base import Backend, register

_DEFAULT_LIMITS = {"joint_speed": 1.0}


@register("lerobot")
class LeRobotBackend(Backend):
    builds = False  # the robot is there already, and calibrated by LeRobot: robowright changes nothing in it

    def __init__(
        self,
        spec,
        seed: int = 0,
        lerobot=None,
        joints: dict | None = None,
        units: str = "degrees",
        calibration: dict | None = None,
        gripper: tuple = (0.0, 100.0),
        control_hz: float = 30.0,
        limits: dict | None = None,
        reset_speed: float = 0.5,
        reset_tolerance: float = 0.03,
        timeout: float = 10.0,
    ):
        super().__init__(spec, seed)
        if lerobot is None:
            raise ValueError(
                'backend="lerobot" needs the robot: rw.launch(backend="lerobot", lerobot=arm), with arm a LeRobot robot '
                "(e.g. lerobot.robots.so_follower.SOFollower(config)), made and calibrated as LeRobot's scripts make it"
            )
        if units not in ("degrees", "radians"):
            raise ValueError(f"units is degrees (LeRobot's use_degrees=True) or radians, not {units!r}")
        self.arm = lerobot
        self._connected_here = False
        if not getattr(lerobot, "is_connected", False):
            lerobot.connect()
            self._connected_here = True
        m = self.robot_model
        self.joint_map = {j: (joints or {}).get(j, j) for j in m.arm_joints}  # robowright name -> LeRobot motor
        self.grip_motor = (joints or {}).get("gripper", "gripper") if self.has_gripper else None
        self.units = units
        self.calibration = {j: tuple(v) for j, v in (calibration or {}).items()}
        self.grip_range = tuple(map(float, gripper))
        self._dt = 1.0 / float(control_hz)
        self.cfg = {
            "limits": _DEFAULT_LIMITS if limits is None else limits,
            "reset_speed": float(reset_speed),
            "reset_tolerance": float(reset_tolerance),
            "timeout": float(timeout),
        }
        self.ros_names = list(m.arm_joints)  # the names limits.positions and its messages use
        features = getattr(lerobot, "action_features", None) or {}
        missing = [f"{n}.pos" for n in [*self.joint_map.values(), *([self.grip_motor] if self.grip_motor else [])]]
        missing = [k for k in missing if features and k not in features]
        if missing:
            raise ValueError(f"the LeRobot robot has no {', '.join(missing)}; it has {', '.join(features)} (map names with joints=)")
        self._t0 = _time.monotonic()
        self._tick = self._t0
        self._obs = None
        q = self.qpos()
        self._ctrl = q.copy()
        self._arm_sent = None
        self._last_q, self._last_t = q, self._t0

    # --- units -----------------------------------------------------------------------------
    def _to_model(self, joint: str, v: float) -> float:
        a = np.radians(v) if self.units == "degrees" else float(v)
        sign, offset = self.calibration.get(joint, (1.0, 0.0))
        return sign * a + offset

    def _to_lerobot(self, joint: str, q: float) -> float:
        sign, offset = self.calibration.get(joint, (1.0, 0.0))
        a = (q - offset) / sign
        return float(np.degrees(a)) if self.units == "degrees" else float(a)

    # --- the robot -------------------------------------------------------------------------
    def _read(self) -> dict:
        if self._obs is None:
            self._obs = self.arm.get_observation()
        return self._obs

    def qpos(self) -> np.ndarray:
        obs = self._read()
        q = [self._to_model(j, obs[f"{motor}.pos"]) for j, motor in self.joint_map.items()]
        if self.has_gripper:
            closed, opened = self.grip_range
            q.append((float(obs[f"{self.grip_motor}.pos"]) - closed) / (opened - closed))
        return np.array(q, float)

    def qvel(self) -> np.ndarray:
        q, t = self.qpos(), _time.monotonic()
        out = (q - self._last_q) / max(t - self._last_t, 1e-6)
        self._last_q, self._last_t = q, t
        return out

    def set_ctrl(self, target) -> None:
        self._ctrl = np.array(target, float)

    def ctrl(self) -> np.ndarray:
        return self._ctrl.copy()

    def _send(self, target) -> None:
        from .. import limits

        arm = [float(v) for v in target[: self.n_arm]]
        start = self._arm_sent if self._arm_sent is not None else self.qpos()[: self.n_arm]
        limits.check(self.cfg["limits"], self.ros_names, start, [arm], self._dt)  # before anything is sent
        action = {f"{motor}.pos": self._to_lerobot(j, q) for (j, motor), q in zip(self.joint_map.items(), arm)}
        if self.has_gripper:
            closed, opened = self.grip_range
            action[f"{self.grip_motor}.pos"] = closed + float(np.clip(target[-1], 0.0, 1.0)) * (opened - closed)
        self.arm.send_action(action)
        self._arm_sent = np.array(arm)

    # --- time ------------------------------------------------------------------------------
    @property
    def control_dt(self) -> float:
        return self._dt

    @property
    def time(self) -> float:
        return _time.monotonic() - self._t0

    def step(self) -> None:
        """Send this step's targets, then wait out the control period (wall clock)."""
        self._send(self._ctrl)
        self._tick += self._dt
        late = _time.monotonic() - self._tick
        if late > self._dt:
            self._tick = _time.monotonic()  # behind (a slow policy): carry on from now, not in a burst
        elif late < 0:
            _time.sleep(-late)
        self._obs = None  # the next reading is this step's

    def set_joint_positions(self, q) -> None:
        """Drive to ``q`` (a real arm cannot be teleported): slowly, inside the limits, then until it is
        within ``reset_tolerance``, or ``TimeoutError`` after ``timeout`` seconds."""
        q = np.asarray(q, float)
        start = self.qpos()
        self._arm_sent = None
        speed = self.cfg["reset_speed"]
        cap = (self.cfg["limits"] or {}).get("joint_speed")
        if cap:
            speed = min(speed, 0.9 * cap)
        span = float(np.abs(q - start)[: self.n_arm].max(initial=0.0))
        n = max(1, int(np.ceil(span / speed / self._dt)))
        for i in range(1, n + 1):
            self._ctrl = start + (q - start) * i / n
            self.step()
        deadline = _time.monotonic() + self.cfg["timeout"]
        while np.abs(self.qpos()[: self.n_arm] - q[: self.n_arm]).max(initial=0.0) > self.cfg["reset_tolerance"]:
            if _time.monotonic() > deadline:
                raise TimeoutError(f"the robot did not reach {np.round(q, 3).tolist()} (at {np.round(self.qpos(), 3).tolist()})")
            self.step()

    def close(self) -> None:
        if self._connected_here and getattr(self.arm, "is_connected", False):
            self.arm.disconnect()  # only what robowright connected: a robot handed over connected stays so
