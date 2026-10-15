"""rw.attach, out of the box, on every robosuite environment (with Panda; two Pandas for the two-arm tasks) and on
Lift with each of robosuite's other single arms: made as robosuite's README makes them, stepped by the user's own
loop, read, asserted on, and reset again (robosuite rebuilds its model on every reset, and robowright follows). A
cold trial (2026-10-15) passed every case robosuite itself can build (21 of 23: robosuite refuses Baxter, two arms,
for Lift, and its Sawyer model fails this MuJoCo's mesh check before robowright is involved). Skipped where
robosuite is absent."""

import numpy as np
import pytest

import robowright as rw
from robowright import expect

suite = pytest.importorskip("robosuite")

ENVIRONMENTS = [
    "Door",
    "Lift",
    "NutAssembly",
    "NutAssemblyRound",
    "NutAssemblySingle",
    "NutAssemblySquare",
    "PickPlace",
    "PickPlaceBread",
    "PickPlaceCan",
    "PickPlaceCereal",
    "PickPlaceMilk",
    "PickPlaceSingle",
    "Stack",
    "TwoArmHandover",
    "TwoArmLift",
    "TwoArmPegInHole",
    "Wipe",
]
ARMS = ["IIWA", "Jaco", "Kinova3", "UR5e"]
CASES = [(e, None) for e in ENVIRONMENTS] + [("Lift", arm) for arm in ARMS]


@pytest.mark.parametrize(("env_name", "arm"), CASES, ids=[f"{e}-{a or 'Panda'}" for e, a in CASES])
def test_attach_to_every_robosuite_environment(env_name, arm):
    robots = arm or (["Panda", "Panda"] if env_name.startswith("TwoArm") else "Panda")
    env = suite.make(
        env_name, robots=robots, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, control_freq=20, horizon=200
    )
    w = rw.attach(env, settings=rw.Settings(trace="off"), name=f"{env_name}-{arm or 'Panda'}")
    try:
        env.reset()
        lo, hi = env.action_spec
        rng = np.random.default_rng(0)
        for _ in range(20):
            env.step(rng.uniform(lo, hi))
        bodies = w.bodies()
        assert len(bodies) > 10  # the arm(s), the gripper(s), the table and the task's objects, found by itself
        expect(w.body(bodies[-1])).to_be_above(-100.0, timeout=0)
        env.reset()  # a new model: robowright reads it, not the one before
        env.step(rng.uniform(lo, hi))
        assert len(w.bodies()) == len(bodies)
        assert np.all(np.isfinite(w.body(bodies[-1]).position))
    finally:
        w.close()
        env.close()
