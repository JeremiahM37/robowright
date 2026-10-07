"""A trained model, tested like any other policy.

``policies/so101_pick.onnx`` is a small network trained by imitation
(``scripts/train_pick_policy.py``). It speaks its own conventions, as any checkpoint does:
joints in degrees, the gripper from 0 to 100, normalised inputs and outputs, 10-step action
chunks. ``LearnedPolicy`` translates, and the test is the same as for the scripted policy.

A learned policy that succeeds most of the time is a flaky test, so this one runs 20 seeds and
requires every one. It has to earn that: cloned from demonstrations it placed the cube 31 of 40
times on MuJoCo and 8 of 20 on Isaac Sim. Trained with DAgger on four engines (see the
script), it succeeded on 140 of 140 seeds on MuJoCo, PyBullet, Drake and Genesis, and 40 of 40
on Isaac Sim, which it never saw.
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
        state=("qpos", "target", "objects.cube", "objects.cube.quat", "objects.bin"),  # what it was trained on
        units="deg",
        gripper=(0, 100),
        normalize=str(HERE / "so101_pick.json"),
    )


@pytest.mark.robots("so101")  # trained on the SO-101 only
@pytest.mark.trials(20)
def test_learned_policy_puts_the_cube_in_the_bin(world, robot, scene):
    world.faults.jitter("cube", xy_std=0.02, yaw_std=0.5)
    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    rollout = robot.run_policy(so101_pick(), until=done, hold=1.0, timeout=15)
    assert rollout.success, rollout
    expect(scene["cube"]).to_be_at_rest()
