"""A wheeled base on any engine: its drive measured once, from the robot's MuJoCo model, and its wheels driven as
the model's velocity servos.

The measurement (each wheel's radius, where it sits on the base, which way it turns the base forward; for an omni
or mecanum base, the matrix from the base's twist to its wheel speeds) is MuJoCo's backend's own, run on the same
scene at rest, so every engine drives a robot from the same numbers. A wheel's motor is the model's velocity
actuator: torque kv * (target - speed), limited to its force range, with the target limited to its control range.
Each engine runs it with its own implicit velocity control, as MuJoCo does (Genesis at the model's kv and torque
limit; Bullet's motor at the torque limit, holding speed more stiffly than kv). The servo law applied as explicit
torque each physics step chattered between its torque limits on light wheels, so no engine does that.
"""

from __future__ import annotations

import numpy as np

from ..errors import CapabilityError


def measure(spec, seed: int = 0) -> tuple[dict, dict]:
    """``(wheels, servo)`` for the scene's mobile robot: the drive as MuJoCo's backend measures it, and each wheel's
    velocity servo (``kv``, ``force``: the torque limit, ``ctrl``: the speed limits) from the model's actuators."""
    from .mujoco_backend import MujocoBackend

    b = MujocoBackend(spec, seed)
    try:
        m, act = b.model, b._act
        gain = m.actuator_gainprm[act, 0]
        bias = m.actuator_biasprm[act, 2]
        if not (np.allclose(gain, -bias) and (gain > 0).all()):
            title = b.robot_model.title
            raise CapabilityError(f"{title}: its wheels are not velocity servos (kv), so another engine cannot drive them as modelled")
        force = np.where(m.actuator_forcelimited[act].astype(bool), m.actuator_forcerange[act, 1], np.inf)
        lo, hi = np.where(m.actuator_ctrllimited[act].astype(bool)[:, None], m.actuator_ctrlrange[act], [[-np.inf, np.inf]]).T
        return b.wheels, {"kv": gain.copy(), "force": force, "ctrl": (lo, hi)}
    finally:
        b.close()


def speeds(wheels: dict, title: str, linear: float, angular: float, lateral: float = 0.0) -> np.ndarray:
    """Wheel speeds (rad/s) for the base moving at ``linear`` m/s forward, ``lateral`` m/s sideways, turning at
    ``angular`` rad/s. A differential base has two wheels and no sideways motion."""
    if "matrix" in wheels:
        return wheels["matrix"] @ np.array([linear, lateral, angular])
    if lateral:
        raise CapabilityError(f"{title}: a differential-drive base cannot move sideways (lateral {lateral} m/s)")
    return np.array([wh["sign"] * (linear - angular * wh["y"]) / wheels["radius"] for wh in wheels["wheels"]])


class WheeledDrive:
    """``drive_base``/``stop_base``/``wheel_command`` for a backend that keeps ``self.wheels``, ``self._servo`` and
    ``self._wheel_u`` (the targets, rad/s) and sends those targets to its wheel motors each physics step."""

    def drive_base(self, linear: float, angular: float, lateral: float = 0.0) -> None:
        if not hasattr(self, "_servo"):  # not a wheeled robot
            return super().drive_base(linear, angular, lateral)
        self.set_wheel_speeds(speeds(self.wheels, self.robot_model.title, linear, angular, lateral))

    def stop_base(self) -> None:
        if not hasattr(self, "_servo"):
            return super().stop_base()
        self.set_wheel_speeds(np.zeros(self.robot_model.n_arm))

    def set_wheel_speeds(self, target) -> None:
        lo, hi = self._servo["ctrl"]
        self._wheel_u = np.clip(np.asarray(target, float)[: self.robot_model.n_arm], lo, hi)

    def wheel_command(self) -> np.ndarray:
        if not hasattr(self, "_servo"):
            return super().wheel_command()
        return self._wheel_u.copy()
