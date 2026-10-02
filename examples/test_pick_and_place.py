"""Examples: what robot tests look like with robowright.

Run with:  pytest examples/            (MuJoCo)
           pytest examples/ --rw-backend mujoco,pybullet
"""

import pytest

from robowright import condition, expect
from robowright.policies import ScriptedPickPlace


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


@pytest.mark.trials(20, min_success=0.9)
def test_policy_with_randomized_cube(world, robot, scene):
    world.faults.jitter("cube", xy_std=0.02, yaw_std=0.5)
    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    rollout = robot.run_policy(ScriptedPickPlace(), until=done, timeout=15, privileged=True)
    assert rollout.success, rollout
    expect(scene["cube"]).to_be_at_rest()


@pytest.mark.trials(10, min_success=0.7)
def test_policy_with_sensor_noise_and_latency(world, robot, scene):
    world.faults.jitter("cube", xy_std=0.02)
    world.faults.joint_noise(std=0.02)
    world.faults.action_delay(steps=3)
    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    assert robot.run_policy(ScriptedPickPlace(), until=done, timeout=20, privileged=True)


def test_cube_survives_a_shove_when_held(world, robot, scene):
    cube = scene["cube"]
    robot.pick(cube)
    world.faults.push("cube", force=(0.0, 1.5, 0.0), duration=0.1)
    world.wait(0.5)
    expect(robot.gripper).to_be_holding(cube)
