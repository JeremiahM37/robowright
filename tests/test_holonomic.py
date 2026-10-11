"""Holonomic bases: an omni-wheeled robot (Menagerie's robot_soccer_kit, three omni wheels) driven
sideways as well as forward and turning, on the MuJoCo backend. The model is loaded as its makers ship it.

What is in the model: each wheel is an actuated hinge (a velocity servo, kv 5, ctrlrange +-16 rad/s,
no force limit) with 20 passive roller hinges on bodies beneath it, each roller a cylinder with its own
collision. The rollers are real rollers, not a friction trick, and the rollers' axes are tangent to the
rim (a 90-degree omni wheel), so each wheel drives the floor along its rim and rolls freely along its axle.

Nothing here is typed in. The wheel's radius (0.0345 m: the rollers' centres 0.0285 m from the axle plus
their 6 mm radius) and each wheel's drive direction are read from the model at rest, and the base's
velocity maps to wheel speeds through those (robots/MuJoCo backend ``_measure_rollers``).

Mecanum wheels are not tested: no published mecanum model with modelled rollers is on this machine (the
Gazebo worlds fetch theirs from the Fuel server). The same measurement applies to 45-degree rollers, but
that is not shown here.

Skipped when the model file is absent. Expected at ROBOWRIGHT_SOCCER_KIT, default the Menagerie cache.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright.assets import diffdrive_mjcf
from robowright.errors import CapabilityError
from robowright.robots.detect import build

SOCCER = Path(
    os.environ.get(
        "ROBOWRIGHT_SOCCER_KIT",
        str(Path.home() / ".cache/robot_descriptions/mujoco_menagerie/robot_soccer_kit/robot_soccer_kit.xml"),
    )
)


@pytest.fixture(scope="module")
def path():
    if not SOCCER.exists():
        pytest.skip(f"{SOCCER} is not installed: fetch Menagerie's robot_soccer_kit (robot_descriptions) to run these")
    return str(SOCCER)


def _settled(path):
    w = rw.launch(robot=path, backend="mujoco")
    w.wait(1.0)  # the base settles on its rollers before any command
    return w


def _in_base_frame(b, dp):
    c, s = np.cos(b.yaw), np.sin(b.yaw)
    return np.array([c * dp[0] + s * dp[1], -s * dp[0] + c * dp[1]])


def test_loads_as_a_holonomic_mobile_robot_with_three_wheels(path):
    model = build(path).model
    assert model.family == "mobile"
    assert model.floating and len(model.arm_joints) == 3
    assert model.arm_joints == ("wheel1_speed", "wheel2_speed", "wheel3_speed")


def test_wheel_radius_is_measured_from_the_rollers(path):
    """Roller centres at 28.5 mm from the axle and 6 mm roller radius: a 34.5 mm rolling radius."""
    w = rw.launch(robot=path, backend="mujoco")
    assert w.backend.wheels["radius"] == pytest.approx(0.0345, abs=5e-4)
    assert w.backend.wheels["matrix"].shape == (3, 3)


def test_wheel_speeds_are_the_rolling_constraint_for_the_commanded_twist(path):
    """The servos reach the speeds the measured geometry gives for a forward command (within 5 %)."""
    w = _settled(path)
    b = w.backend
    b.drive_base(0.1, 0.0, 0.0)
    w.step(40)
    expected = b.wheels["matrix"] @ np.array([0.1, 0.0, 0.0])
    assert np.allclose(b.wheel_command(), expected, atol=1e-9)
    assert np.allclose(b.qvel(), expected, rtol=0.05)
    b.stop_base()


def test_a_differential_base_refuses_sideways_motion():
    w = rw.launch(robot=str(diffdrive_mjcf()), backend="mujoco")
    with pytest.raises(CapabilityError, match="sideways"):
        w.robot.base.drive(0.0, 0.0, 0.5, lateral=0.1)


# The three drive accuracy tests below are EXPECTED TO FAIL, and are marked strict so that a change which
# makes them pass is noticed. Measured on this model (2 s, settled rest, MuJoCo, the same every run):
#   forward 0.1 m/s: 57 % of the distance, yaw drifts -0.17 rad, sideways drift -0.014 m
#   sideways 0.1 m/s: 42 % of the distance, forward drift 0.013 m
#   pivot 1.0 rad/s: 80 % of the angle
# The wheels turn at the speeds the geometry gives (within 2 %), and the rollers under the floor do not hold:
# the contact points slide at about 0.06 m/s where the base should be still, and the rollers' own spin stays
# near zero. Changing the solver (noslip iterations, integrator), the floor's friction (x3) or the rollers'
# frictionloss (0) did not bring the travel up (57 %, 58 %, 38 %, 58 %). The cause is not isolated.
_UNRESOLVED = "the model's rollers slip under the floor: forward 57 %, sideways 42 %, pivot 80 % (see the comment above)"


@pytest.mark.xfail(strict=True, reason=_UNRESOLVED)
def test_forward_drive_goes_forward_without_drift(path):
    w = _settled(path)
    b = w.robot.base
    p0 = b.position.copy()
    b.drive(0.1, 0.0, 2.0)
    fwd, side = _in_base_frame(b, b.position - p0)
    assert 0.90 * 0.1 * 2.0 <= fwd <= 1.05 * 0.1 * 2.0
    assert abs(side) < 0.02


@pytest.mark.xfail(strict=True, reason=_UNRESOLVED)
def test_sideways_drive_goes_sideways_without_drift(path):
    w = _settled(path)
    b = w.robot.base
    p0 = b.position.copy()
    b.drive(0.0, 0.0, 2.0, lateral=0.1)
    fwd, side = _in_base_frame(b, b.position - p0)
    assert 0.90 * 0.1 * 2.0 <= side <= 1.05 * 0.1 * 2.0
    assert abs(fwd) < 0.02


@pytest.mark.xfail(strict=True, reason=_UNRESOLVED)
def test_pivot_turns_on_the_spot_the_right_way(path):
    w = _settled(path)
    b = w.robot.base
    p0, y0 = b.position.copy(), b.yaw
    b.drive(0.0, 1.0, 2.0)
    turned = (b.yaw - y0 + np.pi) % (2 * np.pi) - np.pi
    assert 0.90 * 2.0 <= turned <= 1.05 * 2.0
    assert float(np.linalg.norm((b.position - p0)[:2])) < 0.02
