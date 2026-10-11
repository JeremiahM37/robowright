"""panda-gym's PyBullet tasks, with robowright attached. panda-gym ships no policy.

PandaReach is the task here with an outcome that can be read: the hand within the env's own distance
threshold (0.05 m) of the target. Its driver is a proportional reach written in this file, so it is the
test's driver and not a published policy; it is used only to put the hand where the outcome can be
checked. PandaPickAndPlace gets attach checks: the free object, joints by name, a fault on the object,
a push, and a trace that refuses frame redraw with a clear reason, because PyBullet's frames come from
its own renderer.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.errors import CapabilityError
from robowright.render import renderer_for
from robowright.trace import Trace

gym = pytest.importorskip("gymnasium")
pytest.importorskip("pybullet")
panda_gym = pytest.importorskip("panda_gym")

SEEDS = [0, 1, 2, 3, 4]
REACH_THRESHOLD = 0.05  # PandaReach's distance_threshold, as the env defines it
REACH_STEPS = 50
GAIN = 0.05  # metres of hand motion per unit action: panda-gym's own end-effector step


def _trace_dir(tmp_path_factory) -> Path:
    d = os.environ.get("GR_TRACE_DIR")
    return Path(d) if d else tmp_path_factory.mktemp("gr_traces")


def _make(task: str, **kw):
    return gym.make(task, render_mode="rgb_array", renderer="Tiny", **kw)


@pytest.mark.parametrize("seed", SEEDS)
def test_panda_reach_hand_at_goal_matches_the_env_success(seed, tmp_path_factory):
    """Driver: a proportional reach written here (not a published policy). Outcome: hand within 0.05 m of the target.

    The target is a body in the simulation, so robowright reads it directly; the env's ``desired_goal`` is
    checked against it as a sanity check on the naming.
    """
    env = _make("PandaReach-v3")
    world = rw.attach(env, name=f"panda_reach_seed{seed}", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        obs, _ = env.reset(seed=seed)
        target = world.body("target")
        hand = world.body("panda_grasptarget")  # the env's own end-effector: its link 11, the grasp target
        assert np.allclose(target.position, obs["desired_goal"], atol=1e-6), "robowright's target is the env's goal"
        env_success = False
        for _ in range(REACH_STEPS):
            action = np.clip((target.position - hand.position) / GAIN, -1, 1)
            obs, _, terminated, truncated, info = env.step(action.astype(np.float32))
            env_success = bool(info["is_success"])
            if terminated or truncated:
                break
        d = float(np.linalg.norm(hand.position - target.position))
        robowright = d < REACH_THRESHOLD
        print(f"\npanda-reach seed={seed} env_success={env_success} robowright={robowright} distance={d * 1000:.1f} mm")
        assert robowright == env_success, "robowright's reading disagrees with the environment's is_success"
    finally:
        world.close(failed=False)


def test_panda_reach_trace_has_states_and_refuses_frame_redraw(tmp_path_factory):
    """Random actions: the trace records the joints and the summary reads; redraw is not offered, and says why."""
    env = _make("PandaReach-v3")
    world = rw.attach(env, name="panda_reach_trace", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset(seed=0)
        joint = world.joint("panda_joint1")
        before = joint.value
        rng = np.random.default_rng(0)
        for _ in range(10):
            env.step(rng.uniform(-1, 1, size=3).astype(np.float32))
        assert joint.value != before, "the joint value is read from the live state"
        path = world.close(failed=False)
        tr = Trace(path)
        assert tr.arrays["qpos"].shape == (11, len(world.backend.joint_names))
        assert "panda_joint1" in tr.meta["joint_names"]
        assert tr.meta["attached"]["engine"] == "pybullet"
        with pytest.raises(CapabilityError, match="frame redraw is not available for the pybullet engine"):
            renderer_for(tr)
    except BaseException:
        world.close(failed=True)
        raise


def test_panda_pick_and_place_free_object_fault_and_push(tmp_path_factory):
    """Random actions: attach checks only. The object is the one free body; a moved and a pushed object are recorded."""
    env = _make("PandaPickAndPlace-v3")
    world = rw.attach(env, name="panda_pick", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset(seed=0)
        assert "object" in world.backend.free_bodies
        obj = world.body("object")
        expect(obj).always.to_satisfy(lambda s: s.position[2] > 0.0, "object above the floor (z > 0)")
        world.move_object("object", obj.position + np.array([0.0, 0.10, 0.0]))
        y_moved = float(world.body("object").position[1])
        world.push("object", force=[0.0, 2.0, 0.0], duration=0.1)
        rng = np.random.default_rng(0)
        for _ in range(10):
            env.step(rng.uniform(-1, 1, size=4).astype(np.float32))
        y_end = float(world.body("object").position[1])
        print(f"\npanda-pick object y_after_move={y_moved:.3f} y_end={y_end:.3f}")
        assert y_end > y_moved, "the pushed object moved in the push's direction"
        path = world.close(failed=False)
        edits = [e["name"] for e in Trace(path).events if e.get("type") == "edit"]
        assert "move_object" in edits, "the fault is recorded in the trace"
    except BaseException:
        world.close(failed=True)
        raise


def test_bare_pybullet_client_with_the_tests_own_loop(tmp_path):
    """A client with no environment around it: the test steps it, robowright reads it. A free ball falls under gravity."""
    import pybullet as p

    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.8, physicsClientId=cid)
        p.setTimeStep(1 / 240, physicsClientId=cid)
        p.createMultiBody(
            0.1,
            p.createCollisionShape(p.GEOM_SPHERE, radius=0.02, physicsClientId=cid),
            basePosition=[0, 0, 1],
            physicsClientId=cid,
        )
        w = rw.attach(
            engine="pybullet",
            client=cid,
            step=lambda: p.stepSimulation(physicsClientId=cid),
            dt=1 / 240,
            settings=rw.Settings(trace_dir=str(tmp_path)),
        )
        w.step(24)
        (name,) = w.bodies()  # a body created with no name is named by PyBullet's own base-link name
        z = float(w.body(name).position[2])
        expect(w.body(name)).to_be_above(0.9)
        assert w.backend.time == pytest.approx(0.1)
        assert z == pytest.approx(1 - 0.5 * 9.8 * 0.1**2, abs=2e-3)  # free fall, to the solver's accuracy
        w.close(failed=False)
    finally:
        p.disconnect(physicsClientId=cid)
