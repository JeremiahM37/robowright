"""A trained model, tested like any other policy.

``policies/so101_pick.onnx`` is a small network trained by behaviour cloning
(``scripts/train_pick_policy.py``). It speaks its own conventions, as any checkpoint does:
joints in degrees, the gripper from 0 to 100, normalised inputs and outputs, 10-step action
chunks. ``LearnedPolicy`` translates, and the test is the same as for the scripted policy.

A learned policy succeeds some of the time, so the test runs it 20 times and states the rate
it must reach. Measured, it succeeded on 31 of 40 seeds on MuJoCo and 37 of 40 on PyBullet
(the two it was trained on), 17 of 20 on Drake, 12 of 20 on Genesis and 8 of 20 on Isaac Sim
(a known divergence, in conftest.py). Trained on MuJoCo alone it succeeded on 2 of 40 PyBullet
seeds: the kind of gap running every engine finds.
"""

from pathlib import Path

import pytest

from robowright import condition, expect
from robowright.learned import LearnedPolicy

pytest.importorskip("onnxruntime")

HERE = Path(__file__).parent / "policies"


def so101_pick() -> LearnedPolicy:
    return LearnedPolicy(
        str(HERE / "so101_pick.onnx"),
        state=("qpos", "objects.cube", "objects.cube.quat", "objects.bin"),  # what it was trained on
        units="deg",
        gripper=(0, 100),
        normalize=str(HERE / "so101_pick.json"),
    )


@pytest.mark.robots("so101")  # trained on the SO-101 only
@pytest.mark.trials(20, min_success=0.5)
def test_learned_policy_puts_the_cube_in_the_bin(world, robot, scene):
    world.faults.jitter("cube", xy_std=0.02, yaw_std=0.5)
    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    rollout = robot.run_policy(so101_pick(), until=done, hold=1.0, timeout=15)
    assert rollout.success, rollout
    expect(scene["cube"]).to_be_at_rest()
