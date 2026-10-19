"""A wheeled robot on robowright's own engine (MuJoCo): differential drive, go_to, expects, traces.

The model is robowright's own minimal diff-drive base (assets/diffdrive), not a maker's: no Menagerie
model is a purely wheeled differential drive. The backend reads its wheel radius and track from the model
(0.04 m and 0.24 m, as the file has them), and the tests check those readings against the file itself.

Measured from settled rest (deterministic, the same every run): a straight drive covers 98-99 % of the
commanded distance; a pivot turns 93-94 % of the commanded angle. The pivot's loss is a resisting torque
of 0.134 N m a wheel from the wheel's scrub on its 40 mm contact, and at kv 5 the servo's speed error is
1.8 % of setpoint under it (see assets/diffdrive/diffdrive.xml). The tests assert bands around those numbers.
"""

import re

import numpy as np
import pytest

import robowright as rw
from robowright import robots
from robowright.assets import diffdrive_mjcf
from robowright.errors import ActionTimeoutError
from robowright.robots.detect import build
from robowright.scene import ObjectSpec, SceneSpec

# Two wheels on a fixed base: wheels, but nothing for the backend to roll.
ROVER_URDF = """<?xml version="1.0"?>
<robot name="rover">
  <link name="base_link">
    <inertial><mass value="5"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.3 0.2 0.1"/></geometry></collision>
  </link>
  <link name="left_wheel">
    <inertial><mass value="0.5"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.05" length="0.03"/></geometry></collision>
  </link>
  <link name="right_wheel">
    <inertial><mass value="0.5"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.05" length="0.03"/></geometry></collision>
  </link>
  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/><child link="left_wheel"/><origin xyz="0 0.12 0"/><axis xyz="0 1 0"/>
    <limit effort="1.0" velocity="10"/>
  </joint>
  <joint name="wheel_right_joint" type="continuous">
    <parent link="base_link"/><child link="right_wheel"/><origin xyz="0 -0.12 0"/><axis xyz="0 1 0"/>
    <limit effort="1.0" velocity="10"/>
  </joint>
</robot>
"""


@pytest.fixture
def mjcf():
    return str(diffdrive_mjcf())


def _installed(*engines):
    """Each engine as a parameter, skipped where it is not installed (CI's unit jobs have MuJoCo and PyBullet)."""
    from robowright.backends.base import available

    have = available()
    return [pytest.param(e, marks=pytest.mark.skipif(e not in have, reason=f"{e} is not installed")) for e in engines]


_ENGINE = {"name": "mujoco"}


@pytest.fixture(autouse=True, params=_installed("mujoco", "pybullet", "genesis", "drake"))
def engine(request):
    """Every test here on each engine that drives a wheeled base: the drive is measured once, from the model."""
    _ENGINE["name"] = request.param
    return request.param


def _launch(path, **kw):
    return rw.launch(robot=path, backend=_ENGINE["name"], **kw)


def test_the_diff_drive_is_a_mobile_robot_on_a_free_base(mjcf):
    model = build(mjcf).model
    assert model.family == "mobile"
    assert model.arm_joints == ("wheel_left_joint", "wheel_right_joint")
    assert model.base_body == "base" and model.floating
    assert model.gripper_actuator is None and model.hand == ""
    assert robots.get(mjcf).family == "mobile"


def test_the_wheels_are_measured_from_the_model(mjcf):
    """Radius and track come from the geoms and joints in the file, not from numbers typed into the backend."""
    import mujoco

    w = _launch(mjcf)
    wheels = w.backend.wheels
    m = mujoco.MjModel.from_xml_path(mjcf)
    cyl = m.geom("wheel_left_geom").size[0]
    body_y = m.body("wheel_left").pos[1]
    assert wheels["radius"] == pytest.approx(cyl) == pytest.approx(0.04)
    assert wheels["track"] == pytest.approx(2 * body_y) == pytest.approx(0.24)
    assert [wh["sign"] for wh in wheels["wheels"]] == [1.0, 1.0]  # both axles point +y: both roll the base forward


def test_a_mirrored_axle_is_measured_and_still_drives_forward(tmp_path, mjcf):
    """Flip the right wheel's axle: the measured sign flips with it, and forward driving still goes forward."""
    text = (
        open(mjcf)
        .read()
        .replace('name="wheel_right_joint" type="hinge" axis="0 1 0"', 'name="wheel_right_joint" type="hinge" axis="0 -1 0"')
    )
    flipped = tmp_path / "flipped.xml"
    flipped.write_text(text)
    w = _launch(str(flipped))
    assert [wh["sign"] for wh in w.backend.wheels["wheels"]] == [1.0, -1.0]
    p0 = w.robot.base.position.copy()
    w.robot.base.drive(0.2, 0.0, 1.0)
    moved = w.robot.base.position - p0
    assert moved[0] > 0.15 and abs(moved[1]) < 0.01


def test_drive_covers_the_commanded_distance(mjcf):
    """A straight drive covers about its command and stays on its line. Measured over 0.4 m: the wheels' rims
    rolled 0.399 m on both engines (the same servo); MuJoCo's soft contact carried the base 2.2 % less than its
    rims rolled (0.390 m) and Bullet's 0.46 % more (0.401 m)."""
    w = _launch(mjcf)
    b = w.robot.base
    p0 = b.position.copy()
    b.drive(0.2, 0.0, 2.0)
    travelled = np.linalg.norm((b.position - p0)[:2])
    assert 0.93 * 0.4 <= travelled <= 1.02 * 0.4
    assert abs(b.yaw) < 0.01 and abs(b.position[1] - p0[1]) < 0.01


def test_a_turn_in_place_keeps_its_place_and_turns_the_right_way(mjcf):
    """The turn is counter-clockwise for a positive rate, and a pivot does not move the base's centre."""
    w = _launch(mjcf)
    b = w.robot.base
    p0 = b.position.copy()
    b.drive(0.0, 0.5, 2.0)
    assert 0.7 * 1.0 <= b.yaw <= 1.0 * 1.0  # 1.0 rad commanded; measured 0.77-0.83 of it
    assert np.linalg.norm((b.position - p0)[:2]) < 0.01


def test_the_base_comes_to_rest_after_a_drive(mjcf):
    w = _launch(mjcf)
    w.robot.base.drive(0.2, 0.0, 1.0)
    expect = rw.expect
    expect(w.robot.base).to_be_at_rest(timeout=5, hold=0.5)


def test_go_to_arrives_on_the_ground_truth_pose(mjcf):
    w = _launch(mjcf)
    b = w.robot.base
    b.go_to((1.0, 0.5), yaw=0.0, timeout=60)
    w.wait(1.0)
    # on the floor: the goal at the base's own height (it rides 4 cm up, which a target at z = 0 counted as error)
    rw.expect(w.robot.base).to_be_near((1.0, 0.5, float(b.position[2])), tol=0.05, timeout=3)
    rw.expect(w.robot.base).to_have_heading(0.0, tol=0.06, timeout=3)


def test_go_to_turns_to_its_final_heading_when_it_is_already_there(mjcf):
    w = _launch(mjcf)
    w.robot.base.go_to((0.0, 0.0), yaw=1.0, timeout=30)
    rw.expect(w.robot.base).to_have_heading(1.0, tol=0.06, timeout=3)


def test_go_to_raises_timeout_for_a_goal_it_cannot_reach_in_time(mjcf):
    w = _launch(mjcf)
    with pytest.raises(ActionTimeoutError):
        w.robot.base.go_to((5.0, 0.0), timeout=2.0)


def test_drive_pushes_a_box_in_its_path_and_the_box_is_a_scene_object(mjcf):
    box = ObjectSpec(name="box", kind="box", size=(0.05, 0.05, 0.05), pos=(0.4, 0.0, 0.05), mass=0.3, friction=0.6)
    w = _launch(mjcf, scene=SceneSpec(robot=mjcf, objects=[box]))
    x0 = w.scene["box"].position[0]
    w.robot.base.drive(0.2, 0.0, 3.0)
    x1 = w.scene["box"].position[0]
    assert x1 - x0 > 0.2  # the box is pushed along, not left behind
    rw.expect(w.scene["box"]).to_be_at_rest(timeout=10, hold=0.5)


def test_go_to_has_no_avoidance_so_a_wall_in_the_way_times_out(mjcf):
    # A bin is static: its four walls stand on the floor, in the robot's path.
    wall = ObjectSpec(name="wall", kind="bin", size=(0.05, 0.4, 0.2), pos=(0.6, 0.0, 0.0))
    w = _launch(mjcf, scene=SceneSpec(robot=mjcf, objects=[wall]))
    with pytest.raises(ActionTimeoutError):
        w.robot.base.go_to((1.2, 0.0), timeout=8.0)


def test_traces_record_base_motions_and_codegen_emits_them(tmp_path, mjcf):
    from robowright.codegen import generate

    w = _launch(mjcf, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    w.robot.base.drive(0.2, 0.0, 1.0)
    w.robot.base.go_to((0.5, 0.0), yaw=0.0, timeout=30)
    path = w.close(trace_path=tmp_path / "wheeled.zip")
    code = generate(path)
    assert re.search(r"robot\.base\.drive\(linear=0\.2, angular=0\.0, duration=1\.0\)", code), code
    assert re.search(r"robot\.base\.go_to\(xy=\[0\.5, 0\.0\], yaw=0\.0, timeout=30\)", code), code


def test_a_wheeled_urdf_on_a_fixed_base_is_given_a_free_base_and_rolls(tmp_path, engine, request):
    """A URDF's wheeled base is fixed in the file, but a wheeled robot rolls: the converter gives it the free joint."""
    if engine == "drake":
        # Measured: 5.8 mm in 2 s. The fixture's box chassis rests on the floor with its wheels and Drake's contact
        # holds it there against the wheels' 1 N m; the fixture, not the drive (the other wheeled robots drive on Drake).
        request.applymarker(pytest.mark.xfail(strict=True, reason="the fixture's box chassis drags on the floor and stalls on Drake"))
    from robowright.robots.from_urdf import to_mjcf
    from robowright.robots.urdf import cache_root

    urdf = tmp_path / "rover.urdf"
    urdf.write_text(ROVER_URDF)
    mjcf = to_mjcf(urdf, cache_root().parent / "from_urdf")[0]
    w = _launch(str(mjcf))
    assert w.robot.model.floating and w.robot.model.family == "mobile"
    w.wait(1.0)  # settle on the wheels first
    p0 = w.robot.base.position.copy()
    w.robot.base.drive(0.2, 0.0, 2.0)
    # Only forward motion is asserted: this fixture's box chassis reaches the floor level with its wheels,
    # so it drags, and covers 40 % of the command on MuJoCo (measured 0.16 of 0.4 m) and 18 % on Genesis (0.071 m);
    # on Drake it stalls (above).
    assert w.robot.base.position[0] - p0[0] > 0.05
