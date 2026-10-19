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


def _settled(path, engine="mujoco"):
    w = rw.launch(robot=path, backend=engine)
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


def _floor_roller(m, d, wheel):
    """Where the lowest roller of ``wheel`` (a joint id) is: the wheel's contact with the floor."""
    body = int(m.jnt_bodyid[wheel])

    def on_wheel(b):
        while b > 0 and b != body:
            b = int(m.body_parentid[b])
        return b == body

    rollers = [k for k in range(m.njnt) if k != wheel and int(m.jnt_bodyid[k]) != body and on_wheel(int(m.jnt_bodyid[k]))]
    return d.xanchor[min(rollers, key=lambda k: d.xanchor[k][2])]


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
        p = R.T @ (_floor_roller(m, d, jid) - d.xpos[b._base])  # where the wheel meets the floor
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


# As published, on MuJoCo the forward and sideways tests below are EXPECTED TO FAIL (strict xfail). The wheels do what the map says:
# the test above shows the spins imply the commanded twist. The base does not move with them, because the
# rollers slip against the floor. Measured per wheel (wheel 1, 2, 3; slip along each rolling direction, m/s):
#   forward 0.1 m/s: spins imply vx 0.099; base moves vx 0.060 and yaws -0.10 rad/s; slip -0.025, -0.023, +0.034
#   sideways 0.1 m/s: spins imply vy 0.099; base moves vy 0.043 and yaws -0.11 rad/s; slip -0.053, +0.044, -0.006
# The cause is the model's roller armature: each 3 g roller carries 0.005 of motor inertia (10^5 times its own), so it
# cannot spin up as it rolls onto the floor. Genesis and Drake do not apply armature to it the same way and drive within
# 3 %; adjusted fidelity takes it off ("free rollers") and MuJoCo then drives 98 % forward, 96 % sideways (tested below).
_UNRESOLVED = (
    "as published, each roller carries the wheel motors' armature (10^5 times its own inertia), cannot spin up and slips: "
    "forward 58 %, sideways 42 % of the commanded motion (adjusted fidelity frees them; see the comment above)"
)


# The pivot turned 80-82 % on every engine until the map's turn term took each wheel's lever from where its floor roller
# is (57.2 mm from the centre) rather than its joint's anchor (47.2 mm); it now turns 97 % (MuJoCo), 99 % (Genesis) and
# 100 % (Drake). On Genesis and Drake forward and sideways land within 3 %.
def _installed(*engines):
    """Each engine as a parameter, skipped where it is not installed (CI's unit jobs have MuJoCo and PyBullet)."""
    from robowright.backends.base import available

    have = available()
    return [pytest.param(e, marks=pytest.mark.skipif(e not in have, reason=f"{e} is not installed")) for e in engines]


ENGINES = _installed("mujoco", "genesis", "drake")


def _expect(request, engine, xfail_on):
    if engine in xfail_on:
        request.applymarker(pytest.mark.xfail(strict=True, reason=xfail_on[engine]))


@pytest.mark.parametrize("engine", ENGINES)
def test_forward_drive_goes_forward_without_drift(path, engine, request):
    _expect(request, engine, {"mujoco": _UNRESOLVED})
    w = _settled(path, engine)
    b = w.robot.base
    p0 = b.position.copy()
    b.drive(0.1, 0.0, 2.0)
    fwd, side = _in_base_frame(b, b.position - p0)
    assert 0.90 * 0.1 * 2.0 <= fwd <= 1.05 * 0.1 * 2.0
    assert abs(side) < 0.02


@pytest.mark.parametrize("engine", ENGINES)
def test_sideways_drive_goes_sideways_without_drift(path, engine, request):
    _expect(request, engine, {"mujoco": _UNRESOLVED})
    w = _settled(path, engine)
    b = w.robot.base
    p0 = b.position.copy()
    b.drive(0.0, 0.0, 2.0, lateral=0.1)
    fwd, side = _in_base_frame(b, b.position - p0)
    assert 0.90 * 0.1 * 2.0 <= side <= 1.05 * 0.1 * 2.0
    assert abs(fwd) < 0.02


@pytest.mark.parametrize("engine", ENGINES)
def test_pivot_turns_on_the_spot_the_right_way(path, engine):
    w = _settled(path, engine)
    b = w.robot.base
    p0, y0 = b.position.copy(), b.yaw
    b.drive(0.0, 1.0, 2.0)
    turned = (b.yaw - y0 + np.pi) % (2 * np.pi) - np.pi
    assert 0.90 * 2.0 <= turned <= 1.05 * 2.0
    assert float(np.linalg.norm((b.position - p0)[:2])) < 0.02


@pytest.mark.parametrize("cmd", [(0.1, 0.0, 0.0), (0.0, 0.1, 0.0)], ids=["forward", "sideways"])
def test_adjusted_fidelity_frees_the_rollers_and_mujoco_drives_as_commanded(path, cmd):
    w = rw.launch(robot=path, backend="mujoco", settings=rw.Settings(fidelity="adjusted"))
    assert any(c.what == "free rollers" and c.kind == "adjusted" for c in w.model_changes)
    w.wait(1.0)
    b = w.robot.base
    p0 = b.position.copy()
    b.drive(cmd[0], 0.0, 2.0, lateral=cmd[1])
    fwd, side = _in_base_frame(b, b.position - p0)
    want, other = (fwd, side) if cmd[0] else (side, fwd)
    assert 0.90 * 0.2 <= want <= 1.05 * 0.2
    assert abs(other) < 0.02


def test_published_fidelity_keeps_the_rollers_as_modelled(path):
    w = rw.launch(robot=path, backend="mujoco", settings=rw.Settings(fidelity="published"))
    assert all(c.what != "free rollers" for c in w.model_changes)
    m = w.backend.model
    arm = [float(m.dof_armature[m.jnt_dofadr[k]]) for k in range(m.njnt) if "passive" in m.joint(k).name]
    assert arm and min(arm) == pytest.approx(0.005)  # as the file has it
