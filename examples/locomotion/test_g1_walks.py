"""Unitree's published walking policies, run on their published models with their published gains.

The checkpoints, PD gains, default poses and observation scales all come from unitree_rl_gym at a
pinned commit (fetched on first use, or taken from ``ROBOWRIGHT_UNITREE_RL_GYM``). Each robot is
commanded 0.5 m/s forward for 10 s from the start pose its deploy script uses. The criteria were
written before the first run and are not tuned:

* upright within 30 degrees at every physics step (a fall is what this catches);
* at least 1 m walked forward in x;
* base height within 25 % of its start height at every control step.

A shove in the middle of the walk must not topple the robot. ``test_g1_walks_one_metre_upright``
also carries the G1; the H1 and H1-2 run the same code with only their YAML.
"""

import numpy as np
import pytest

pytest.importorskip("torch", reason="the published policies are TorchScript checkpoints: install torch")

import unitree_rl_gym as ug  # noqa: E402  (this directory)

import robowright as rw  # noqa: E402
from robowright import expect  # noqa: E402

WALK_SECONDS = 10.0
SHOVE_AT = 5.0  # seconds into the walk
ROBOTS = ("g1.yaml", "h1.yaml", "h1_2.yaml")


@pytest.fixture(scope="module", params=ROBOTS, ids=lambda c: c.rsplit(".", 1)[0])
def published(request):
    try:
        pub = ug.load(request.param)
    except Exception as e:  # no network, or git missing: say so instead of failing the suite
        pytest.skip(f"cannot fetch unitree_rl_gym (set ROBOWRIGHT_UNITREE_RL_GYM to a checkout): {type(e).__name__}: {e}")
    return pub, ug.register(pub)


def _start(robot, pub):
    robot.reset_to(q=np.zeros(pub.config["num_actions"]), base_pos=(0.0, 0.0, ug.start_height(pub)))


def _walk(name, pub, shove=None, seed=0):
    """Walk from the deploy start pose; return the robot and the base's track, step by step."""
    world = rw.launch(robot=name, backend="mujoco", seed=seed)
    robot = world.robot
    _start(robot, pub)
    policy = ug.LocomotionPolicy(pub, cmd=(0.5, 0.0, 0.0))
    log = {"x": [], "z": [], "tilt": []}
    shove_call = round(SHOVE_AT / world.dt)

    def step(obs):
        n = len(log["x"])
        log["x"].append(float(robot.base.position[0]))
        log["z"].append(robot.base.height)
        log["tilt"].append(float(np.degrees(np.arccos(np.clip(robot.base.up_axis[2], -1, 1)))))
        if shove is not None and n == shove_call:
            world.faults.push("robot", force=shove, duration=0.1)
        return policy(obs)

    robot.run_policy(step, timeout=WALK_SECONDS)
    return robot, {k: np.array(v) for k, v in log.items()}


def test_walks_one_metre_upright(published):
    pub, name = published
    h0 = ug.start_height(pub)
    world = rw.launch(robot=name, backend="mujoco", seed=0)
    robot = world.robot
    _start(robot, pub)
    expect(robot.base).always.to_be_upright(tol_deg=30)
    expect(robot.base).always.to_be_above(0.75 * h0)
    x0 = float(robot.base.position[0])
    robot.run_policy(ug.LocomotionPolicy(pub, cmd=(0.5, 0.0, 0.0)), timeout=WALK_SECONDS)
    walked = float(robot.base.position[0]) - x0
    assert walked >= 1.0, f"walked {walked:.3f} m in {WALK_SECONDS:g} s"
    assert 0.75 * h0 <= robot.base.height <= 1.25 * h0, f"base at {robot.base.height:.3f} m (start {h0:.3f} m)"


def test_recovers_from_a_shove_mid_walk(published):
    pub, name = published
    weight = rw.launch(robot=name, backend="mujoco", seed=0).robot.total_mass * 9.81
    _, log = _walk(name, pub, shove=(0.0, 0.3 * weight, 0.0))
    assert log["tilt"].max() < 30.0, f"fell: max tilt {log['tilt'].max():.1f} deg"
    assert log["tilt"][-1] < 30.0, f"ended tilted {log['tilt'][-1]:.1f} deg"


def test_walk_is_deterministic(published):
    pub, name = published
    _, a = _walk(name, pub)
    _, b = _walk(name, pub)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["z"], b["z"])
