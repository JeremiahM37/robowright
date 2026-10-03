"""The README's code, run as written (with the surrounding test scaffolding a user would add)."""

import numpy as np
import pytest

from robowright import condition, expect, robots


@pytest.fixture(autouse=True)
def _family(request, rw_robot):
    fam = robots.get(rw_robot).family
    want = "legged" if "legged" in request.node.name else "arm"
    if fam != want:
        pytest.skip(f"{want} snippet")


# block 1
def test_pick_and_place(robot, scene):
    cube, bin = scene["cube"], scene["bin"]

    robot.pick(cube)
    expect(robot.gripper).to_be_holding(cube)

    robot.place(on=bin)
    expect(cube).to_be_inside(bin)
    expect(cube).to_be_at_rest()


# block 2
def test_legged_recovers_from_a_shove(world, robot):
    expect(robot.base).always.to_be_upright(tol_deg=30)  # invariant: never tips over
    world.faults.push("robot", force=(0, 0.3 * robot.total_mass * 9.81, 0), duration=0.1)
    world.wait(1.5)
    expect(robot.base).to_be_upright(tol_deg=10)
    expect(robot.base).to_be_above(0.8 * robot.model.stand_height)


# block 3 + 4
def test_actions_and_matchers(world, robot, scene):
    cube, bin = scene["cube"], scene["bin"]
    expect(robot).always.to_have_no_collisions()  # invariant for the rest of the test
    robot.arm.move_to((0.22, 0.05, 0.04))
    robot.gripper.open()
    above = cube.position + [0, 0, 0.04]
    robot.arm.move_to(above)
    robot.arm.move_to(cube, linear=True, speed=0.05)
    robot.gripper.close()
    expect(robot.gripper).to_be_holding(cube)  # both jaws in contact
    robot.place(on=bin)
    expect(cube).to_be_inside(bin)
    expect(cube).to_be_at_rest(hold=0.3)
    expect(cube).not_.to_be_touching("floor")
    expect.soft(cube).to_be_near(cube.position, tol=0.01)


# block 5
def test_policy_with_camera(world, robot, scene):
    seen = {}

    def my_policy(obs):
        seen.update(obs)
        return robot.home_q.tolist() + [1.0]

    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    rollout = robot.run_policy(my_policy, until=done, timeout=0.5, cameras=("front",))
    assert not rollout.success  # a policy that does nothing never gets the cube in the bin
    if "images" in seen:
        assert seen["images"]["front"].shape[2] == 3


# block 6
@pytest.mark.trials(5, min_success=0.8)
def test_trials_randomized(world, robot, scene):
    world.faults.jitter("cube", xy_std=0.02, yaw_std=0.5)
    robot.pick(scene["cube"])
    robot.place(on=scene["bin"])
    expect(scene["cube"]).to_be_inside(scene["bin"])


# block 7 (joint name made robot-neutral: the README uses the SO-101's shoulder_lift)
def test_faults_api(world, robot, scene):
    world.faults.joint_noise(std=0.02)
    world.faults.action_delay(steps=3)
    world.faults.weak_joint(robot.joint_names[1], 0.3)
    world.faults.push("cube", force=(0, 1.5, 0), duration=0.1)
    world.faults.camera_dropout(p=0.1)
    world.faults.jitter("cube", xy_std=0.02)
    world.wait(0.5)
    assert np.all(np.isfinite(robot.qpos()))
