"""A synthetic suite for the parallel-scaling benchmark: 64 randomised pick-and-place tests."""

import pytest

from robowright import expect


@pytest.mark.parametrize("variant", range(64))
def test_randomised_pick_and_place(world, robot, scene, variant):
    world.faults.jitter("cube", xy_std=0.02, yaw_std=0.5)
    world.rng.random(variant)  # vary the stream per test
    robot.pick(scene["cube"])
    expect(robot.gripper).to_be_holding(scene["cube"])
    robot.place(on=scene["bin"])
    expect(scene["cube"]).to_be_inside(scene["bin"])
