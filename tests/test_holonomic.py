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


def _implied_twist(w):
    """The base twist that the wheels' measured spins imply under no slip along each wheel's rolling
    direction (A x z, at the roller contact radius), solved in the base frame from the model's own geometry."""
    b, m, d = w.backend, w.backend.model, w.backend.data
    R = d.xmat[b._base].reshape(3, 3)
    z = np.array([0.0, 0.0, 1.0])
    rows, rhs = [], []
    for j, spin in zip(b._dadr, b.qvel()):
        jid = int(m.dof_jntid[j])
        A = d.xmat[m.jnt_bodyid[jid]].reshape(3, 3) @ m.jnt_axis[jid]
        p = R.T @ (d.xanchor[jid] - d.xpos[b._base])
        p[2] = 0.0
        dvec = R.T @ np.cross(A, z)
        dvec[2] = 0.0
        dvec /= np.linalg.norm(dvec)
        rows.append([dvec[0], dvec[1], float(dvec @ np.cross(z, p))])
        rhs.append(spin * b.wheels["radius"])
    return np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]


@pytest.mark.parametrize("cmd", [(0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 1.0)])
def test_the_wheels_imply_the_commanded_twist(path, cmd):
    """No assumption about the map: the spins the servos reach, under no slip along each rolling direction,
    give the commanded base twist (forward, sideways, pivot): 5 % of the linear command, 0.02 rad/s of turn."""
    w = _settled(path)
    w.backend.drive_base(cmd[0], cmd[2], cmd[1])
    w.step(40)
    implied = _implied_twist(w)
    # 5 % of the commanded speed for the linear parts, 0.02 rad/s for the turn (the spins carry about 0.007 rad/s of yaw)
    assert np.all(np.abs(implied - np.array(cmd)) <= np.array([0.005, 0.005, 0.02]))
    w.backend.stop_base()


# The three drive accuracy tests below are EXPECTED TO FAIL (strict xfail). The wheels do what the map says:
# the test above shows the spins imply the commanded twist. The base does not move with them, because the
# rollers slip against the floor. Measured per wheel (wheel 1, 2, 3; slip along each rolling direction, m/s):
#   forward 0.1 m/s: spins imply vx 0.099; base moves vx 0.060 and yaws -0.10 rad/s; slip -0.025, -0.023, +0.034
#   sideways 0.1 m/s: spins imply vy 0.099; base moves vy 0.043 and yaws -0.11 rad/s; slip -0.053, +0.044, -0.006
#   pivot 1.0 rad/s: spins imply wz 0.985; base yaws 0.80 rad/s; slip -0.008, -0.010, -0.008
# A slip of this size in the published model is what limits it; the cause within MuJoCo's contacts is not isolated
# (noslip iterations, integrator, floor friction x3 and roller frictionloss 0 did not close it).
_UNRESOLVED = (
    "the rollers slip against the floor: forward 60 %, sideways 43 %, pivot 80 % of the commanded motion, "
    "with per-wheel slip of 0.02-0.05 m/s (see the comment above)"
)


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
