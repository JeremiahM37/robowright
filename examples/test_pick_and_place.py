"""Examples: what robot tests look like with robowright.

Run with:  pytest examples/            (MuJoCo)
           pytest examples/ --rw-backend mujoco,pybullet
"""

import pytest

from robowright import UnreachableError, condition, expect
from robowright.policies import ScriptedPickPlace
from robowright.robot import widest_gap
from robowright.scene import ObjectSpec, tabletop


def test_pick_and_place(robot, scene):
    cube, bin = scene["cube"], scene["bin"]

    robot.pick(cube)
    expect(robot.gripper).to_be_holding(cube)
    expect(cube).to_be_above(0.04)

    robot.place(on=bin)
    expect(cube).to_be_inside(bin)
    expect(cube).to_be_at_rest()


def test_arm_never_hits_anything_while_picking(robot, scene):
    # Checked after every control step for the rest of the test.
    expect(robot).always.to_have_no_collisions()
    robot.pick(scene["cube"])
    robot.arm.home()


def test_selectors(scene):
    red = scene.get(color="red")
    assert red.name == "cube"
    assert scene.nearest(to=(0.2, 0.12, 0.0)).name == "cube"  # bins are not pickable
    assert [h.name for h in scene.all(kind="bin")] == ["bin"]


@pytest.mark.trials(20)
def test_policy_with_randomized_cube(world, robot, scene):
    world.faults.jitter("cube", xy_std=0.02, yaw_std=0.5)
    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    rollout = robot.run_policy(ScriptedPickPlace(), until=done, hold=1.0, timeout=15, privileged=True)
    assert rollout.success, rollout
    expect(scene["cube"]).to_be_at_rest()


@pytest.mark.trials(20)
def test_policy_with_sensor_noise_and_latency(world, robot, scene):
    world.faults.jitter("cube", xy_std=0.02)
    world.faults.joint_noise(std=0.02)
    world.faults.action_delay(steps=3)
    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    assert robot.run_policy(ScriptedPickPlace(), until=done, hold=1.0, timeout=20, privileged=True)


def test_cube_survives_a_shove_when_held(world, robot, scene):
    cube = scene["cube"]
    robot.pick(cube)
    world.faults.push("cube", force=(0.0, 1.5, 0.0), duration=0.1)
    world.wait(0.5)
    expect(robot.gripper).to_be_holding(cube)


TALL_CAN = tabletop(  # a 10 cm can, and a bin deep enough to hold it (7 cm)
    ObjectSpec("can", "cylinder", (0.02, 0.05), (0.22, -0.06, None), color="red"),
    ObjectSpec("bin", "bin", (0.05, 0.05, 0.035), (0.2, 0.12, 0.0), color="blue", mass=0.0),
)


@pytest.mark.scene(TALL_CAN)
def test_picks_a_tall_can_from_the_side(robot, scene):
    # A side grasp plans its way round the table and the objects, the hand turned to the side.
    can, bin = scene["can"], scene["bin"]
    if robot.model.derived.apertures and widest_gap(robot.model) < 0.04 + 0.01:
        pytest.skip("this gripper opens less than 1 cm wider than the can: it cannot slide in from the side")
    try:
        robot.pick(can, approach="side")
    except UnreachableError as e:  # decided before the arm moves
        pytest.skip(f"this arm cannot hold its hand level beside the can from where it is mounted: {e}")
    expect(robot.gripper).to_be_holding(can)
    try:
        robot.place(on=bin)
    except UnreachableError as e:  # decided before the arm moves
        pytest.skip(f"this arm cannot hold the can level above the bin from where it is mounted: {e}")
    expect(can).to_be_inside(bin)
