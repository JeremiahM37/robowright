"""rw.attach on an Isaac Lab environment made the way Isaac Lab's own scripts make it (gym.make with parse_env_cfg),
several copies at once on the GPU: robowright watches one of them. Needs Isaac Lab (isaaclab, isaaclab_tasks) and an
NVIDIA GPU; skipped elsewhere. Isaac Sim starts once for the module (AppLauncher, headless).
"""

import importlib.util

import numpy as np
import pytest

import robowright as rw
from robowright import expect

pytestmark = pytest.mark.skipif(importlib.util.find_spec("isaaclab_tasks") is None, reason="needs Isaac Lab (isaaclab_tasks)")

TASK = "Isaac-Lift-Cube-Franka-v0"
N = 4


@pytest.fixture(scope="module")
def env():
    from isaaclab.app import AppLauncher

    AppLauncher(headless=True)  # Isaac Sim, once for the module: the environments need it running
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers Isaac Lab's tasks with gymnasium)
    from isaaclab_tasks.utils import parse_env_cfg

    e = gym.make(TASK, cfg=parse_env_cfg(TASK, num_envs=N))
    yield e
    e.close()


def _zeros(env):
    import torch

    return torch.zeros(env.action_space.shape, device=env.unwrapped.device)


def test_attaches_to_one_copy_and_finds_the_scene(env):
    w = rw.attach(env, env_index=1, settings=rw.Settings(trace="off"), name="isaaclab")
    try:
        env.reset(seed=0)
        for _ in range(5):
            env.step(_zeros(env))
        bodies = w.bodies()
        assert "object" in bodies and "robot/panda_hand" in bodies, bodies
        assert "robot/panda_joint1" in w.backend.joint_names
        # where the watched copy's cube is, in that copy's own frame: Isaac Lab's world position less its origin
        u = env.unwrapped
        world_pos = u.scene.rigid_objects["object"].data.root_pos_w[1].cpu().numpy()
        local = world_pos - u.scene.env_origins[1].cpu().numpy()
        assert np.allclose(w.body("object").position, local, atol=1e-5)
        assert w.body("panda_hand").name  # a link name only one articulation has answers alone
        expect(w.body("object")).to_be_above(0.0, timeout=0)
    finally:
        w.close()


def test_an_invariant_holds_through_the_users_own_loop(env):
    w = rw.attach(env, env_index=0, settings=rw.Settings(trace="off"))
    try:
        env.reset(seed=0)
        expect(w.body("object")).always.to_be_above(-0.05)  # the cube never falls through the table
        for _ in range(20):
            env.step(_zeros(env))
        assert w.step_count == 20
    finally:
        w.close()


def test_a_push_moves_the_watched_copys_cube_and_no_other(env):
    w = rw.attach(env, env_index=2, settings=rw.Settings(trace="off"))
    try:
        env.reset(seed=0)
        for _ in range(10):
            env.step(_zeros(env))  # settle
        data = env.unwrapped.scene.rigid_objects["object"].data
        before = data.root_pos_w.cpu().numpy().copy()
        w.faults.push("object", force=(0.0, 3.0, 0.0), duration=0.2)
        for _ in range(15):
            env.step(_zeros(env))
        after = data.root_pos_w.cpu().numpy()
        moved = np.linalg.norm(after[:, :2] - before[:, :2], axis=1)
        assert moved[2] > 0.01, moved  # the watched copy's cube was pushed
        assert np.all(np.delete(moved, 2) < 0.005), moved  # the others were not
    finally:
        w.close()


def test_a_free_body_is_moved_where_the_test_says(env):
    w = rw.attach(env, env_index=3, settings=rw.Settings(trace="off"))
    try:
        env.reset(seed=0)
        env.step(_zeros(env))
        w.move_object("object", (0.6, 0.1, 0.2))
        assert np.allclose(w.body("object").position, (0.6, 0.1, 0.2), atol=1e-4)
        for _ in range(25):
            env.step(_zeros(env))
        expect(w.body("object")).to_be_at_rest(timeout=0)
        assert w.body("object").position[2] < 0.1  # it fell back onto the table
    finally:
        w.close()


def test_an_index_out_of_range_is_refused(env):
    with pytest.raises(ValueError, match=f"not one of this environment's {N} copies"):
        rw.attach(env, env_index=N)


def test_a_run_keeps_its_trace_and_replay_says_it_cannot_rebuild_it(env, tmp_path):
    from robowright.replay import replay
    from robowright.trace import Trace

    w = rw.attach(env, env_index=0, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)), name="isaaclab-trace")
    env.reset(seed=0)
    for _ in range(10):
        env.step(_zeros(env))
    expect(w.body("object")).to_be_above(0.0, timeout=0)
    path = w.close(trace_path=tmp_path / "isaaclab.zip")
    tr = Trace(path)
    assert tr.meta["attached"]["engine"] == "isaaclab"
    assert tr.meta["attached"]["env_index"] == 0 and tr.meta["attached"]["num_envs"] == N
    assert any("to_be_above" in str(e) for e in tr.events)
    with pytest.raises(ValueError, match="an Isaac Lab or Isaac Sim run cannot be replayed"):
        replay(path)
