"""gym-pybullet-drones' HoverAviary, with robowright attached. Its controller is the published one.

HoverAviary runs with ``ActionType.PID``: the action is a target position, and the env's own published
``DSLPIDControl`` turns it into motor speeds each control step. The test only chooses the target, which is
the env's own hover target. The env has no success predicate for hovering (its termination is an exact hit
within 1e-4 m, which a PID controller does not reach), so the outcome compared here is the hover position:
robowright's reading of the drone body against the env's own state vector, and the same 0.05 m hover
threshold PandaReach uses, applied to both readings.

This file needs gym-pybullet-drones, which needs Python 3.12; it skips elsewhere.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright.trace import Trace

pytest.importorskip("pybullet")
drones_env = pytest.importorskip("gym_pybullet_drones.envs.HoverAviary")
enums = pytest.importorskip("gym_pybullet_drones.utils.enums")

SEEDS = [0, 1, 2]
HOVER_THRESHOLD = 0.05  # metres, the same threshold PandaReach uses
STEPS = 150  # 5 s of control at 30 Hz


def _trace_dir(tmp_path_factory) -> Path:
    d = os.environ.get("GR_TRACE_DIR")
    return Path(d) if d else tmp_path_factory.mktemp("gr_traces")


@pytest.mark.parametrize("seed", SEEDS)
def test_hover_aviary_drone_hovers_where_the_env_says(seed, tmp_path_factory):
    """Driver: the env's own target and its own PID controller. Outcome: the drone within 0.05 m of the target,
    by robowright's reading and by the env's state vector."""
    env = drones_env.HoverAviary(gui=False, record=False, act=enums.ActionType.PID)
    world = rw.attach(env, name=f"hover_seed{seed}", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset(seed=seed)
        drone = world.body("cf2")
        target = np.asarray(env.TARGET_POS, float)
        for _ in range(STEPS):
            _, _, terminated, truncated, _ = env.step(target.reshape(1, 3))
            if terminated or truncated:
                break
        env_pos = env._getDroneStateVector(0)[0:3]
        assert np.allclose(drone.position, env_pos, atol=1e-3), "robowright's drone is the env's drone"
        env_hover = bool(np.linalg.norm(env_pos - target) < HOVER_THRESHOLD)
        robowright = bool(np.linalg.norm(drone.position - target) < HOVER_THRESHOLD)
        print(
            f"\nhover seed={seed} env_hover={env_hover} robowright={robowright} "
            f"distance={np.linalg.norm(drone.position - target) * 1000:.1f} mm time={world.backend.time:.2f} s"
        )
        assert robowright == env_hover
        assert world.backend.time == pytest.approx(STEPS / env.CTRL_FREQ, rel=1e-6)
    finally:
        world.close(failed=False)


def test_hover_aviary_trace_states_and_redraw_refusal(tmp_path_factory):
    env = drones_env.HoverAviary(gui=False, record=False, act=enums.ActionType.PID)
    world = rw.attach(env, name="hover_trace", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset(seed=0)
        for _ in range(5):
            env.step(np.asarray(env.TARGET_POS, float).reshape(1, 3))
        path = world.close(failed=False)
        tr = Trace(path)
        assert tr.meta["attached"]["engine"] == "pybullet"
        assert len(tr.arrays["t"]) == 6  # the start and five control steps
    except BaseException:
        world.close(failed=True)
        raise
