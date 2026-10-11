"""Gymnasium-Robotics' MuJoCo robots, with robowright attached. None of these ships a published policy.

FetchReach is the one task here with an outcome that can be read: the gripper within the env's own
distance threshold (0.05 m) of the goal. Its driver is a proportional reach written in this file,
so it is the test's driver and not a published policy; it is used only to put the robot where the
outcome can be checked. The other tasks (Fetch pick-and-place, Franka Kitchen, the Shadow hand)
get attach checks with random actions: subjects, joints by name, contacts, an invariant, a fault
where there is a free body, and a trace redrawn to a PNG. No outcome is asserted for them.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.trace import Trace

gym = pytest.importorskip("gymnasium")
gymnasium_robotics = pytest.importorskip("gymnasium_robotics")
gym.register_envs(gymnasium_robotics)

SEEDS = [0, 1, 2, 3, 4]
REACH_THRESHOLD = 0.05  # FetchReach's distance_threshold, as the env defines it
REACH_STEPS = 50  # FetchReach's own episode length


def _trace_dir(tmp_path_factory) -> Path:
    d = os.environ.get("GR_TRACE_DIR")
    return Path(d) if d else tmp_path_factory.mktemp("gr_traces")


def _redraw(path, out_dir: Path, name: str, size=(320, 240)) -> Path:
    from PIL import Image

    from robowright.attached import AttachedRenderer
    from robowright.trace import Trace

    r = AttachedRenderer(Trace(path))
    try:
        img = r.frame(r.steps - 1, size=size)
    finally:
        r.close()
    assert img.shape == (size[1], size[0], 3) and img.any()
    out = out_dir / f"{name}.png"
    Image.fromarray(img).save(out)
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_fetch_reach_gripper_at_goal_matches_the_env_success(seed, tmp_path_factory):
    """Driver: a proportional reach written here (not a published policy). Outcome: gripper within 0.05 m of the goal.

    The goal is the env's ``desired_goal``. The sim has no goal to read: the ``target0`` marker site stays at its
    XML position (0.8, 0.75, 0.5) across resets, render mode or not, so only the env knows where the goal is.
    """
    env = gym.make("FetchReach-v4", max_episode_steps=REACH_STEPS)
    world = rw.attach(env, name=f"fetch_reach_seed{seed}", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        obs, _ = env.reset(seed=seed)
        grip = world.body("robot0:grip")
        expect(world.body("robot0:gripper_link")).always.not_.to_be_touching("floor0")
        env_success = False
        for _ in range(REACH_STEPS):
            action = np.concatenate([np.clip(10.0 * (obs["desired_goal"] - obs["achieved_goal"]), -1, 1), [0.0]])
            obs, _, terminated, truncated, info = env.step(action.astype(np.float32))
            env_success = bool(info["is_success"])
            if terminated or truncated:
                break
        goal = obs["desired_goal"]
        d = float(np.linalg.norm(grip.position - goal))
        robowright = d < REACH_THRESHOLD
        print(f"\nfetch-reach seed={seed} env_success={env_success} robowright={robowright} distance={d * 1000:.1f} mm")
        assert robowright == env_success, "robowright's reading disagrees with the environment's is_success"
    finally:
        world.close(failed=False)


def test_fetch_pick_and_place_attach_checks_and_puck_fault(tmp_path_factory):
    """Random actions: the driver is noise, not a policy. Checks the object is a subject, contacts and joints
    read live, an invariant holds, and a moved object is recorded in the trace."""
    env = gym.make("FetchPickAndPlace-v4", max_episode_steps=30)
    world = rw.attach(env, name="fetch_pick_seed0", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset(seed=0)
        obj = world.body("object0")
        expect(obj).always.to_satisfy(lambda s: s.position[2] > 0.3, "object0 not fallen below the table top (z > 0.3)")
        z0 = float(obj.position[2])
        world.move_object("object0", obj.position + np.array([0.0, 0.10, 0.0]))  # the fault: moved after reset
        moved_y = float(world.body("object0").position[1])
        rng = np.random.default_rng(0)
        for _ in range(30):
            env.step(rng.uniform(-1, 1, size=4).astype(np.float32))
        contacts = world.body("object0").contacts()
        print(f"\nfetch-pick object z0={z0:.3f} y_after_move={moved_y:.3f} z_end={float(obj.position[2]):.3f} contacts={contacts}")
        assert world.step_count == 30
        path = world.close(failed=False)
        edits = [e for e in Trace(path).events if e.get("type") == "edit"]
        assert [e["name"] for e in edits] == ["move_object"], "the fault is recorded in the trace"
    except BaseException:
        world.close(failed=True)
        raise


def test_franka_kitchen_attach_checks(tmp_path_factory):
    """Random actions: attach checks only. Joints by name are read from the live state and the trace is redrawn."""
    env = gym.make("FrankaKitchen-v1", max_episode_steps=20)
    world = rw.attach(env, name="franka_kitchen", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset(seed=0)
        joint = world.joint("robot:panda0_joint1")
        before = joint.value
        rng = np.random.default_rng(0)
        for _ in range(20):
            env.step(rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32))
        after = joint.value
        print(f"\nfranka joint1 {before:.3f} -> {after:.3f}, end-effector {world.body('end_effector').position.round(3)}")
        assert after != before, "the joint value is read from the live state, so random actions move it"
        path = world.close(failed=False)
        out = _redraw(path, _trace_dir(tmp_path_factory), "franka_kitchen-final")
        print(f"franka redrawn: {out}")
    except BaseException:
        world.close(failed=True)
        raise


def test_shadow_hand_attach_checks(tmp_path_factory):
    """Random actions: attach checks only, on the Shadow hand's fingertips and palm."""
    env = gym.make("HandReach-v3", max_episode_steps=20)
    world = rw.attach(env, name="shadow_hand", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset(seed=0)
        tip = world.body("robot0:S_fftip")
        before = tip.position.copy()
        expect(world.body("robot0:palm")).always.not_.to_be_touching("floor0")
        rng = np.random.default_rng(0)
        for _ in range(20):
            env.step(rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32))
        print(f"\nshadow fftip {before.round(3)} -> {tip.position.round(3)}")
        assert np.linalg.norm(tip.position - before) > 1e-4, "the fingertip moves under the random actions"
        path = world.close(failed=False)
        out = _redraw(path, _trace_dir(tmp_path_factory), "shadow_hand-final")
        print(f"shadow redrawn: {out}")
    except BaseException:
        world.close(failed=True)
        raise
