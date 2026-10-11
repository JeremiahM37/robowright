"""Meta-World's own scripted expert policies, run in their own environments, with robowright attached.

Meta-World (Farama) ships a scripted policy per task in ``metaworld.policies``. Each policy is run
as published, against the environment's own observation, action and step. robowright attaches to
the environment: every ``env.step`` is observed, the test reads the task's physical outcome from
bodies, sites and joints, and the environment's own ``info["success"]`` is the reference.

Meta-World's success flag is per step. An episode runs to its 500-step limit without terminating,
so the verdict compared here is the flag at the final step; the flag ever raised is printed too.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import condition, expect

metaworld = pytest.importorskip("metaworld")
policies = pytest.importorskip("metaworld.policies")

MAX_STEPS = 500  # metaworld's max_path_length for these tasks
SEEDS = [0, 1, 2, 3, 4]
TABLE = "tablelink"
ARM_LINKS = ["right_l1", "right_l2", "right_l3", "right_l4", "right_l5", "right_l6", "right_l1_2", "right_l2_2", "right_l4_2"]

# The physical outcome per task, as robowright reads it at the final step. The thresholds are the
# ones written before the first run, except where a note says otherwise.
#  - pick-place: the puck is within 0.07 m of the goal site, the radius of the env's own success line.
#  - door-open: the hinge has swung at least 0.3 rad from closed (closed is 0).
#  - drawer-open: the slide is at least 0.10 m out.
#  - button-press: the button box is pressed at least 0.03 m in.
# The slide and hinge bounds are one-sided. Each joint has a declared range, but the simulation lets
# the slides go past it under the arm's force (drawer reached -0.20 m against a declared -0.16 m, and
# the button box -0.09 m against -0.06 m), so the declared range is not used as a bound.
OUTCOMES = {
    "pick-place-v3": lambda w: condition(w.body("obj"), "to_be_near", w.body("goal"), tol=0.07)(),
    "door-open-v3": lambda w: condition(w.joint("doorjoint"), "to_be_between", -np.inf, -0.3)(),
    "drawer-open-v3": lambda w: condition(w.joint("goal_slidey"), "to_be_between", -np.inf, -0.10)(),
    "button-press-topdown-v3": lambda w: condition(w.joint("btnbox_joint"), "to_be_between", -np.inf, -0.03)(),
}


def rollout(task: str, policy_cls, seed: int, trace_dir: Path, tag: str = "run", move_puck=None) -> dict:
    """One episode with the published policy driving; robowright checks its invariants on every step.

    ``move_puck`` is an offset applied to the puck right after reset; the policy's first observation
    is then taken from the environment's own observation function, so it sees the moved puck.
    """
    warnings.filterwarnings("ignore", message="Constant\\(s\\) may be too high")
    env = metaworld.make_mt_envs(task, seed=seed)
    world = rw.attach(env, name=f"mw_{task}_{tag}_seed{seed}", settings=rw.Settings(trace_dir=str(trace_dir)))
    try:
        obs, _ = env.reset(seed=seed)
        if task == "pick-place-v3":
            expect(world.body("obj")).always.to_satisfy(lambda s: s.position[2] > 0.0, "puck above the table (z > 0)")
        for arm in ARM_LINKS:  # no arm link touches the table, in any step of the published policy
            expect(world.body(arm)).always.not_.to_be_touching(TABLE)
        if move_puck is not None:
            puck = world.body("obj")
            world.move_object("obj", puck.position + np.asarray(move_puck))
            obs = env.unwrapped._get_obs()
        policy = policy_cls()
        env_last, env_ever, steps = False, False, 0
        for steps in range(1, MAX_STEPS + 1):  # noqa: B007 - steps is read after the loop
            obs, _, terminated, truncated, info = env.step(policy.get_action(obs))
            env_last = bool(info["success"])
            env_ever = env_ever or env_last
            if terminated or truncated:
                break
        robowright = bool(OUTCOMES[task](world))
        return dict(seed=seed, steps=steps, env_success=env_last, env_ever=env_ever, robowright=robowright, _world=world)
    except BaseException:
        world.close(failed=True)
        raise


TASKS = [
    pytest.param("pick-place-v3", policies.SawyerPickPlaceV3Policy, id="pick-place"),
    pytest.param("door-open-v3", policies.SawyerDoorOpenV3Policy, id="door-open"),
    pytest.param("drawer-open-v3", policies.SawyerDrawerOpenV3Policy, id="drawer-open"),
    pytest.param("button-press-topdown-v3", policies.SawyerButtonPressTopdownV3Policy, id="button-press-topdown"),
]


def _trace_dir(tmp_path_factory) -> Path:
    d = os.environ.get("MW_TRACE_DIR")
    return Path(d) if d else tmp_path_factory.mktemp("metaworld_traces")


DOOR_XFAIL = (
    "the env's flag is |handle x - goal x| <= 0.08 at each step, and the policy swings the hinge to -1.85 rad with the handle "
    "0.09 m past goal x (seed 0, measured), so the final flag is False while the door is open; "
    "the flag was raised at some step in 5 of 5 seeds"
)


@pytest.mark.parametrize(
    ("task", "policy", "seed"),
    [pytest.param("pick-place-v3", policies.SawyerPickPlaceV3Policy, s, id=f"pick-place-{s}") for s in SEEDS]
    + [
        pytest.param(
            "door-open-v3", policies.SawyerDoorOpenV3Policy, s, id=f"door-open-{s}", marks=pytest.mark.xfail(strict=True, reason=DOOR_XFAIL)
        )
        for s in SEEDS
    ]
    + [pytest.param("drawer-open-v3", policies.SawyerDrawerOpenV3Policy, s, id=f"drawer-open-{s}") for s in SEEDS]
    + [pytest.param("button-press-topdown-v3", policies.SawyerButtonPressTopdownV3Policy, s, id=f"button-press-{s}") for s in SEEDS],
)
def test_published_policy_outcome_matches_the_environment(task, policy, seed, tmp_path_factory):
    res = rollout(task, policy, seed, _trace_dir(tmp_path_factory))
    world = res.pop("_world")
    print(f"\n{task} {res}")
    failure = None
    try:
        assert res["robowright"] == res["env_success"], "robowright's verdict disagrees with the environment's final success flag"
    except AssertionError as e:
        failure = e
    world.close(failed=failure is not None)
    if failure is not None:
        raise failure


@pytest.mark.parametrize(("task", "policy"), TASKS)
def test_trace_of_the_published_policy_redraws_to_a_png(task, policy, tmp_path_factory):
    from PIL import Image

    from robowright.attached import AttachedRenderer
    from robowright.trace import Trace

    res = rollout(task, policy, 0, _trace_dir(tmp_path_factory), tag="redraw")
    world = res.pop("_world")
    path = world.close(failed=False)
    r = AttachedRenderer(Trace(path))
    try:
        img = r.frame(r.steps - 1, size=(320, 240))
    finally:
        r.close()
    out = _trace_dir(tmp_path_factory) / f"{task}-final.png"
    Image.fromarray(img).save(out)
    assert img.shape == (240, 320, 3) and img.any()
    print(f"\n{task} redrawn: {out} cameras={r.cameras}")


def test_pick_place_puck_moved_3cm_after_reset_printed_not_asserted(tmp_path_factory):
    """A fault: the puck moved 3 cm in y after reset, before the policy first sees it. Printed, not asserted."""
    rows = []
    for seed in SEEDS:
        res = rollout(
            "pick-place-v3", policies.SawyerPickPlaceV3Policy, seed, _trace_dir(tmp_path_factory), tag="moved", move_puck=(0.0, 0.03, 0.0)
        )
        world = res.pop("_world")
        world.close(failed=False)
        rows.append(res)
    lines = [f"fault seed={r['seed']} env_success={r['env_success']} env_ever={r['env_ever']} robowright={r['robowright']}" for r in rows]
    print("\n" + "\n".join(lines))
