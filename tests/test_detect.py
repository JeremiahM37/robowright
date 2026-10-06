"""Any robot from its model file: robowright works out what each built-in robot's entry says by hand."""

import warnings

import pytest

import robowright as rw
from robowright import robots
from robowright.robots.detect import DetectionError, build, detect
from robowright.scene import default_scene


def _detect(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MuJoCo's notes on attaching a gripper
        return detect(path)


@pytest.mark.parametrize("name", robots.names())
def test_detects_what_each_built_in_robot_is_told(name, monkeypatch):
    """The parts a person names for each built-in robot, worked out from its bare model file.

    Only robots whose models are already downloaded: fetching all of Menagerie's is ~600 MB.
    """
    monkeypatch.setattr(robots.menagerie, "_fetch", lambda directory: pytest.skip(f"{directory} not downloaded"))
    m = robots.get(name)
    f = _detect(m.mjcf()).fields
    assert f.get("family", "arm") == m.family
    assert set(f["arm_joints"]) == set(m.arm_joints)
    if m.family == "legged":
        assert f["base_body"] == m.base_body
        return
    assert f["hand"] == m.hand
    assert {*f["left_finger"], *f["right_finger"]} == {*m.left_finger, *m.right_finger}
    assert f["gripper_actuator"] == m.gripper_actuator
    # Open is the same end of the range (the SO-101's entry opens it less than fully).
    assert (f["gripper_open"] > f["gripper_closed"]) == (m.gripper_open > m.gripper_closed)


def _arm(tmp_path, gripper: str):
    """A five-joint arm with two slide fingers, its gripper driven the way ``gripper`` says."""
    act = {
        "tendon": '<position name="grip" joint="fl" kp="200" ctrlrange="0 0.03"/>',
        "two motors": '<position name="grip_l" joint="fl" kp="200" ctrlrange="0 0.03"/>'
        '<position name="grip_r" joint="fr" kp="200" ctrlrange="0 0.03"/>',
        "force motor": '<motor name="grip" joint="fl" ctrlrange="-5 5"/>',
    }[gripper]
    couple = "" if gripper == "two motors" else '<equality><joint joint1="fr" joint2="fl"/></equality>'
    link = 'type="capsule" size="0.02" mass="0.3"'
    xml = f"""
<mujoco model="toy">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default><joint damping="2" armature="0.05" range="-3 3"/><position kp="300" kv="20"/></default>
  <worldbody>
    <body name="base"><geom type="cylinder" size="0.05 0.02" pos="0 0 0.02"/>
      <body name="l1" pos="0 0 0.04"><joint name="j1" axis="0 0 1"/><geom {link} fromto="0 0 0 0 0 0.1"/>
        <body name="l2" pos="0 0 0.1"><joint name="j2" axis="0 1 0"/><geom {link} fromto="0 0 0 0.2 0 0"/>
          <body name="l3" pos="0.2 0 0"><joint name="j3" axis="0 1 0"/><geom {link} fromto="0 0 0 0.18 0 0"/>
            <body name="l4" pos="0.18 0 0"><joint name="j4" axis="0 1 0"/><geom {link} fromto="0 0 0 0.04 0 0"/>
              <body name="hand" pos="0.04 0 0"><joint name="j5" axis="1 0 0"/>
                <geom type="box" size="0.01 0.04 0.015" mass="0.2"/>
                <body name="left_finger" pos="0.035 0.01 0"><joint name="fl" type="slide" axis="0 1 0" range="0 0.03"/>
                  <geom type="box" size="0.025 0.004 0.01" pos="0 0 0" mass="0.02" friction="1.5"/></body>
                <body name="right_finger" pos="0.035 -0.01 0"><joint name="fr" type="slide" axis="0 -1 0" range="0 0.03"/>
                  <geom type="box" size="0.025 0.004 0.01" pos="0 0 0" mass="0.02" friction="1.5"/></body>
              </body></body></body></body></body></body>
  </worldbody>
  {couple}
  <actuator>
    <position joint="j1" ctrlrange="-3 3"/><position joint="j2" ctrlrange="-3 3"/><position joint="j3" ctrlrange="-3 3"/>
    <position joint="j4" ctrlrange="-3 3"/><position joint="j5" ctrlrange="-3 3"/>{act}
  </actuator>
</mujoco>"""
    path = tmp_path / f"toy_{gripper.replace(' ', '_')}.xml"
    path.write_text(xml)
    return path


@pytest.mark.parametrize("gripper", ["tendon", "two motors", "force motor"])
def test_a_robot_it_has_never_seen_picks_and_places(tmp_path, gripper):
    """A made-up arm, from nothing but its model file: one gripper actuator, a finger motor
    each, or a force motor; all three are driven as one position-controlled gripper."""
    path = _arm(tmp_path, gripper)
    built = build(path)
    f = built.model
    assert f.arm_joints == ("j1", "j2", "j3", "j4", "j5") and f.hand == "hand"
    assert {*f.left_finger, *f.right_finger} == {"left_finger", "right_finger"}
    assert bool(f.gripper_mirrors) == (gripper == "two motors")
    assert bool(f.gripper_servo) == (gripper == "force motor")
    with rw.launch(scene=default_scene(str(path)), settings=rw.Settings(trace="off")) as w:
        cube, bin = w.scene["cube"], w.scene["bin"]
        w.robot.pick(cube)
        rw.expect(w.robot.gripper).to_be_holding(cube)
        w.robot.place(on=bin)
        rw.expect(cube).to_be_inside(bin)


def test_a_mobile_manipulators_arm_is_tested_with_its_base_held(tmp_path):
    """The same arm on a free-floating base: the base is held where it stands (on the floor,
    however high the model put it), and the arm picks and places as on a fixed one."""
    path = _arm(tmp_path, "tendon")
    path.write_text(path.read_text().replace('<body name="base">', '<body name="base" pos="0 0 0.3"><freejoint/>'))
    built = build(path)
    assert built.model.family == "arm" and built.model.arm_joints == ("j1", "j2", "j3", "j4", "j5")
    assert any("held where it stands" in n for n in built.notes)
    with rw.launch(scene=default_scene(str(path)), settings=rw.Settings(trace="off")) as w:
        assert w.robot.model.mjcf().read_text().count("freejoint") == 0
        cube, bin = w.scene["cube"], w.scene["bin"]
        w.robot.pick(cube)
        w.robot.place(on=bin)
        rw.expect(cube).to_be_inside(bin)


def test_inspect_explains_every_decision(capsys):
    from robowright.cli import main

    assert main(["robots", "--inspect", str(robots.get("so101").mjcf())]) == 0
    out = capsys.readouterr().out
    for field in ("gripper_actuator", "hand", "fingers", "base_pos", "home", "--rw-robot"):
        assert field in out


def _urdf_arm(tmp_path):
    """The same five-joint arm as a ROS package: a URDF with no actuators, a ``package://`` mesh,
    and a ``<mimic>`` finger."""
    pkg = tmp_path / "toy_description"
    (pkg / "meshes").mkdir(parents=True)
    (pkg / "urdf").mkdir()
    (pkg / "package.xml").write_text("<package><name>toy_description</name></package>")
    # The hand, as an OBJ box 2 x 8 x 3 cm.
    v = [(x, y, z) for x in (-0.01, 0.01) for y in (-0.04, 0.04) for z in (-0.015, 0.015)]
    f = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]
    obj = "".join(f"v {a} {b} {c}\n" for a, b, c in v) + "".join(f"f {' '.join(map(str, q))}\n" for q in f)
    (pkg / "meshes" / "hand.obj").write_text(obj)

    def link(name, mass, geom="", at="0 0 0"):
        shape = "".join(f'<{k}><origin xyz="{at}"/><geometry>{geom}</geometry></{k}>' for k in ("visual", "collision")) if geom else ""
        i = mass * 1e-3
        inertia = f'<inertia ixx="{i}" iyy="{i}" izz="{i}" ixy="0" ixz="0" iyz="0"/>'
        return f'<link name="{name}"><inertial><origin xyz="{at}"/><mass value="{mass}"/>{inertia}</inertial>{shape}</link>'

    def joint(name, parent, child, xyz, axis, kind="revolute", lo=-3, hi=3, effort=30, extra=""):
        return (
            f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/><origin xyz="{xyz}"/>'
            f'<axis xyz="{axis}"/><limit lower="{lo}" upper="{hi}" effort="{effort}" velocity="2"/>{extra}</joint>'
        )

    finger = '<box size="0.05 0.008 0.02"/>'
    (pkg / "urdf" / "toy.urdf").write_text(
        f"""<robot name="toy">
  {link("base", 1.0, '<cylinder radius="0.05" length="0.04"/>', "0 0 0.02")}
  {link("l1", 0.3, '<cylinder radius="0.02" length="0.1"/>', "0 0 0.05")}
  {link("l2", 0.3, '<box size="0.2 0.04 0.04"/>', "0.1 0 0")}
  {link("l3", 0.3, '<box size="0.18 0.04 0.04"/>', "0.09 0 0")}
  {link("l4", 0.1, '<box size="0.04 0.04 0.04"/>', "0.02 0 0")}
  {link("hand", 0.2, '<mesh filename="package://toy_description/meshes/hand.obj"/>')}
  {link("left_finger", 0.02, finger)}
  {link("right_finger", 0.02, finger)}
  {joint("j1", "base", "l1", "0 0 0.04", "0 0 1")}
  {joint("j2", "l1", "l2", "0 0 0.1", "0 1 0")}
  {joint("j3", "l2", "l3", "0.2 0 0", "0 1 0")}
  {joint("j4", "l3", "l4", "0.18 0 0", "0 1 0")}
  {joint("j5", "l4", "hand", "0.06 0 0", "1 0 0")}
  {joint("fl", "hand", "left_finger", "0.035 0.01 0", "0 1 0", "prismatic", 0, 0.03, 20)}
  {joint("fr", "hand", "right_finger", "0.035 -0.01 0", "0 -1 0", "prismatic", 0, 0.03, 20, '<mimic joint="fl"/>')}
</robot>"""
    )
    return pkg / "urdf" / "toy.urdf"


def test_a_urdf_robot_picks_and_places(tmp_path):
    """A URDF names no motors, and finds its meshes through ROS packages: robowright adds a
    servo per joint, resolves ``package://``, and the arm picks and places like any other."""
    path = _urdf_arm(tmp_path)
    built = build(path)
    f = built.model
    assert f.arm_joints == ("j1", "j2", "j3", "j4", "j5") and f.hand == "hand"
    assert {*f.left_finger, *f.right_finger} == {"left_finger", "right_finger"}
    assert f.gripper_actuator == "fl"  # the <mimic> finger follows it
    assert any("URDF" in n for n in built.notes) and any("position servo on each of 6 joints" in n for n in built.notes)
    with rw.launch(scene=default_scene(str(path)), settings=rw.Settings(trace="off")) as w:
        cube, bin = w.scene["cube"], w.scene["bin"]
        w.robot.pick(cube)
        rw.expect(w.robot.gripper).to_be_holding(cube)
        w.robot.place(on=bin)
        rw.expect(cube).to_be_inside(bin)


def test_a_urdf_with_a_mesh_it_cannot_find_says_where_it_looked(tmp_path):
    path = _urdf_arm(tmp_path)
    path.write_text(path.read_text().replace("package://toy_description/meshes/hand.obj", "package://missing_pkg/hand.obj"))
    with pytest.raises(DetectionError, match="'package://missing_pkg/hand.obj' not found"):
        build(path)


def test_a_xacro_template_is_refused_with_how_to_expand_it(tmp_path):
    path = tmp_path / "arm.urdf.xacro"
    path.write_text('<robot name="a" xmlns:xacro="http://www.ros.org/wiki/xacro"><xacro:property name="l" value="1"/></robot>')
    with pytest.raises(DetectionError, match="xacro"):
        build(path)
