"""dm_control's suite, with robowright attached through ``physics.model.ptr``. No suite task ships a policy.

Reacher is the one task with an outcome to read: the finger touches the target when their spheres
overlap, which is what the env's reward of 1 means (``get_reward`` is ``tolerance(dist, (0, r_target + r_finger))``).
The driver is a Jacobian reach written in this file, so it is the test's driver and not a published
policy. Walker is an attach check only: random actions, joints by name, contacts, an invariant and a trace.
"""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

import robowright as rw
from robowright import expect

suite = pytest.importorskip("dm_control.suite")

SEEDS = [0, 1, 2, 3, 4]
REACH_STEPS = 200


def _trace_dir(tmp_path_factory) -> Path:
    d = os.environ.get("DM_TRACE_DIR")
    return Path(d) if d else tmp_path_factory.mktemp("dm_traces")


def _reach_action(physics, gain: float = 20.0, kd: float = 0.5) -> np.ndarray:
    """Joint torques that move the finger toward the target: a damped least-squares step, then a PD on the joints."""
    m, d = physics.model.ptr, physics.data.ptr
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "finger")
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, None, body)
    J = jacp[:2, :2]
    target = np.asarray(d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target")][:2])
    finger = np.asarray(d.xpos[body][:2])
    dq = np.linalg.solve(J.T @ J + 1e-3 * np.eye(2), J.T @ (target - finger))  # damped least squares
    q_des = d.qpos[:2] + dq
    return np.clip(gain * (q_des - d.qpos[:2]) - kd * d.qvel[:2], -1.0, 1.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_reacher_finger_on_target_matches_the_env_reward(seed, tmp_path_factory):
    env = suite.load("reacher", "easy", task_kwargs={"random": seed})
    world = rw.attach(env, name=f"reacher_seed{seed}", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        ts = env.reset()
        finger, target = world.body("finger"), world.body("target")
        r_sum = float(world.model.geom_size[mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, "target")][0]) + float(
            world.model.geom_size[mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, "finger")][0]
        )
        expect(world.body("arm")).always.not_.to_be_touching("ground")
        reward = 0.0
        for _ in range(REACH_STEPS):
            ts = env.step(_reach_action(env.physics))
            reward = float(ts.reward or 0.0)
        d = float(np.linalg.norm(finger.position - target.position))
        robowright = d <= r_sum
        print(
            f"\nreacher seed={seed} env_reward={reward:.3f} robowright={robowright} distance={d * 1000:.1f} mm radii={r_sum * 1000:.0f} mm"
        )
        assert robowright == (reward == 1.0), "robowright's reading disagrees with the env's reward"
    finally:
        world.close(failed=False)


def test_reacher_joints_by_name_and_a_redrawn_trace(tmp_path_factory):
    env = suite.load("reacher", "easy", task_kwargs={"random": 0})
    world = rw.attach(env, name="reacher_trace", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset()
        shoulder = world.joint("shoulder")
        before = shoulder.value
        for _ in range(50):
            env.step(_reach_action(env.physics))
        after = shoulder.value
        assert after != before
        path = world.close(failed=False)
        from PIL import Image

        from robowright.attached import AttachedRenderer
        from robowright.trace import Trace

        r = AttachedRenderer(Trace(path))
        try:
            img = r.frame(r.steps - 1, size=(320, 240))
        finally:
            r.close()
        assert img.shape == (240, 320, 3) and img.any()
        out = _trace_dir(tmp_path_factory) / "reacher-final.png"
        Image.fromarray(img).save(out)
        print(f"\nreacher shoulder {before:.3f} -> {after:.3f}, redrawn {out} cameras={r.cameras}")
    except BaseException:
        world.close(failed=True)
        raise


def test_walker_attach_checks_with_random_actions(tmp_path_factory):
    """Random actions: attach checks only (no outcome asserted, and no policy is driving)."""
    env = suite.load("walker", "walk", task_kwargs={"random": 0})
    world = rw.attach(env, name="walker_random", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset()
        expect(world.body("torso")).always.to_satisfy(lambda s: s.position[2] > 0.0, "torso above the floor (z > 0)")
        hip = world.joint("right_hip")
        rng = np.random.default_rng(0)
        spec = env.action_spec()
        before = hip.value
        for _ in range(100):
            env.step(rng.uniform(spec.minimum, spec.maximum))
        contacts = world.body("right_foot").contacts()
        torso_z = world.body("torso").position[2]
        print(f"\nwalker right_hip {before:.3f} -> {hip.value:.3f}, torso z={torso_z:.3f}, right_foot contacts={contacts}")
        assert hip.value != before
        path = world.close(failed=False)
        from PIL import Image

        from robowright.attached import AttachedRenderer
        from robowright.trace import Trace

        r = AttachedRenderer(Trace(path))
        try:
            img = r.frame(r.steps - 1, size=(320, 240))
        finally:
            r.close()
        assert img.any()
        Image.fromarray(img).save(_trace_dir(tmp_path_factory) / "walker-final.png")
    except BaseException:
        world.close(failed=True)
        raise


def test_bring_ball_attach_checks_fault_and_redraw(tmp_path_factory):
    """Manipulator bring_ball, random actions. The ball rides three joints (two slides and a hinge), not a free
    joint, so it is moved as a fault by setting its slide joint. No outcome is asserted: no policy drives it."""
    env = suite.load("manipulator", "bring_ball", task_kwargs={"random": 0})
    world = rw.attach(env, name="bring_ball", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset()
        ball = world.body("ball")
        expect(ball).always.to_satisfy(lambda s: s.position[2] > 0.0, "ball above the floor (z > 0)")
        z0 = float(ball.position[2])
        world.set_joint("ball_x", world.joint("ball_x").value + 0.05)  # the fault: moved after reset
        x_fault = float(ball.position[0])
        rng = np.random.default_rng(0)
        spec = env.action_spec()
        for _ in range(100):
            ts = env.step(rng.uniform(spec.minimum, spec.maximum))
        print(
            f"\nbring_ball z0={z0:.3f} x_after_fault={x_fault:.3f} x_end={float(ball.position[0]):.3f} "
            f"z_end={float(ball.position[2]):.3f} env_reward={float(ts.reward or 0.0):.3f} contacts={ball.contacts()}"
        )
        path = world.close(failed=False)
        from PIL import Image

        from robowright.attached import AttachedRenderer
        from robowright.trace import Trace

        r = AttachedRenderer(Trace(path))
        try:
            img = r.frame(r.steps - 1, size=(320, 240))
        finally:
            r.close()
        assert img.any()
        Image.fromarray(img).save(_trace_dir(tmp_path_factory) / "bring_ball-final.png")
    except BaseException:
        world.close(failed=True)
        raise
