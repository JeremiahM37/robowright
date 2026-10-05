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


def test_a_mobile_manipulator_is_refused_with_a_reason(tmp_path):
    path = _arm(tmp_path, "tendon")
    path.write_text(path.read_text().replace('<body name="base">', '<body name="base"><freejoint/>'))
    with pytest.raises(DetectionError, match="mobile manipulator"):
        _detect(path)


def test_inspect_explains_every_decision(capsys):
    from robowright.cli import main

    assert main(["robots", "--inspect", str(robots.get("so101").mjcf())]) == 0
    out = capsys.readouterr().out
    for field in ("gripper_actuator", "hand", "fingers", "base_pos", "home", "--rw-robot"):
        assert field in out
