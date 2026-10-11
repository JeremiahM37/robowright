"""ManiSkill 3's own motion-planning solutions, run in their own environments, with robowright attached.

The code under test is ``mani_skill.examples.motionplanning.panda.solutions``, as published: ``solvePickCube``,
``solveStackCube`` and ``solvePushCube``. Each one resets the environment itself, plans with mplib and steps
it through ``env.step``, so robowright sees every control step. The verdict the test compares against is the
environment's own ``success`` from ``evaluate()``, which is what its ``info["success"]`` reports.

Machines without a GPU Vulkan driver (SAPIEN needs one even with rendering off, and this one does not
render): point the loader at software Vulkan (lavapipe) before Python starts::

    export VK_ICD_FILENAMES=/mnt/bulk/rw-ext/vk/share/vulkan/icd.d/lvp_icd.x86_64.json
    export LD_LIBRARY_PATH=/mnt/bulk/rw-ext/vk/lib

Needs mani_skill (3.0.1 here), SAPIEN (3.0.3) and mplib; the module is skipped when one is missing.

mplib must be 0.1.1, the version ManiSkill 3.0.1 pins, and it must run on numpy 1.x: with numpy 2 its native
``ArticulatedModel`` segfaults on any URDF, which looks like a driver fault and is not one. mplib 0.2 changed
``set_base_pose`` to take its own Pose, which ManiSkill's solver does not pass, so it cannot be used either.

The physical outcome, per task, as robowright reads it from actor poses and contacts (constants from the
environment's own code, read from the environment, not copied):

  - PickCube: the cube within ``goal_thresh`` of the goal site.
  - StackCube: cubeA on cubeB (xy offset within the env's tolerance, z offset within 5 mm), at rest, and not
    touching both fingers.
  - PushCube: the object's xy within ``goal_radius`` of the goal region, and not lifted off the table.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill")
pytest.importorskip("sapien")
pytest.importorskip("mplib")
solutions = pytest.importorskip("mani_skill.examples.motionplanning.panda.solutions")
import mani_skill.envs  # noqa: E402, F401 - registers the task ids

SEEDS = [0, 1, 2, 3, 4]
FINGERS = ["panda_leftfinger", "panda_rightfinger"]  # the link names the fingers' contacts are reported under
FAULT_OFFSET = np.array([0.0, 0.03, 0.0])  # the fault: the object moved 3 cm in y right after reset


def _trace_dir(tmp_path_factory) -> Path:
    d = os.environ.get("MS_TRACE_DIR")
    return Path(d) if d else tmp_path_factory.mktemp("maniskill_traces")


def _make(task: str):
    return gym.make(task, obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu", render_mode=None)


def _float(x) -> float:
    return float(np.asarray(x.detach().cpu() if hasattr(x, "detach") else x).reshape(-1)[0])


def _env_success(env) -> bool:
    return bool(_float(env.unwrapped.evaluate()["success"]))


def _move_after_reset(env, world, name: str, record: dict) -> None:
    """Wrap the environment's reset: the object is moved once the reset is done, before the solution plans.

    The positions before and after the move go into ``record``, so the test can check the fault happened.
    """
    reset = env.reset

    def moved_reset(*args, **kwargs):
        out = reset(*args, **kwargs)
        before = world.body(name).position
        world.move_object(name, before + FAULT_OFFSET)
        record.update(before=before, after=world.body(name).position)
        return out

    env.reset = moved_reset


def _pick(env, world) -> dict:
    core = env.unwrapped
    cube, goal = world.body("cube"), world.body("goal_site")
    dist = float(np.linalg.norm(cube.position - goal.position))
    return dict(robowright=dist <= float(core.goal_thresh), distance_mm=dist * 1000)


def _stack(env, world) -> dict:
    core = env.unwrapped
    a, b = world.body("cubeA"), world.body("cubeB")
    half = np.asarray(core.cube_half_size.cpu(), dtype=float).reshape(-1)
    offset = a.position - b.position
    on_top = np.linalg.norm(offset[:2]) <= np.linalg.norm(half[:2]) + 0.005 and abs(offset[2] - 2 * half[2]) <= 0.005
    at_rest = np.linalg.norm(a.velocity[:3]) <= 1e-2 and np.linalg.norm(a.velocity[3:]) <= 0.5
    touching = set(a.contacts())
    held = all(f in touching for f in FINGERS)
    return dict(robowright=bool(on_top and at_rest and not held), held=held, on_top=bool(on_top), at_rest=bool(at_rest))


def _push(env, world) -> dict:
    core = env.unwrapped
    obj, goal = world.body("cube"), world.body("goal_region")  # ManiSkill names PushCube's object "cube"
    xy = float(np.linalg.norm(obj.position[:2] - goal.position[:2]))
    lifted = obj.position[2] >= float(core.cube_half_size) + 5e-3  # a float here, a tensor in StackCube
    return dict(robowright=xy < float(core.goal_radius) and not lifted, distance_mm=xy * 1000)


# task, solver, the robowright reading, the invariant (checked on every step), the object the fault moves
TASKS = {
    "PickCube-v1": dict(solve=solutions.solvePickCube, read=_pick, obj="cube"),
    "StackCube-v1": dict(solve=solutions.solveStackCube, read=_stack, obj="cubeA"),
    "PushCube-v1": dict(solve=solutions.solvePushCube, read=_push, obj="cube"),
}


def rollout(task: str, seed: int, trace_dir: Path, tag: str = "run", fault: bool = False) -> dict:
    spec = TASKS[task]
    env = _make(task)
    world = rw.attach(env, name=f"ms_{task}_{tag}_seed{seed}", settings=rw.Settings(trace_dir=str(trace_dir)))
    moved: dict = {}
    try:
        if fault:
            _move_after_reset(env, world, spec["obj"], moved)
        expect(world.body(spec["obj"])).always.to_satisfy(lambda s: s.position[2] > 0.0, "the object stays above the table (z > 0)")
        spec["solve"](env, seed=seed, debug=False, vis=False)
        env_success = _env_success(env)
        reading = spec["read"](env, world)
    except BaseException:
        world.close(failed=True)
        raise
    return dict(task=task, seed=seed, env_success=env_success, world=world, moved=moved, **reading)


@pytest.mark.parametrize(("task", "seed"), [(t, s) for t in TASKS for s in SEEDS], ids=lambda v: str(v))
def test_solution_outcome_matches_the_environment(task, seed, tmp_path_factory):
    res = rollout(task, seed, _trace_dir(tmp_path_factory))
    world = res.pop("world")
    res.pop("task")
    res.pop("moved")
    print(f"\n{task} seed={seed} {res}")
    failure = None
    try:
        assert res["robowright"] == res["env_success"], "robowright's reading disagrees with the environment's success"
    except AssertionError as e:
        failure = e
    world.close(failed=failure is not None)
    if failure is not None:
        raise failure


@pytest.mark.parametrize("task", list(TASKS))
def test_object_moved_after_reset_is_printed_not_asserted(task, tmp_path_factory):
    """A fault: the object moved 3 cm in y right after reset. The solutions read the object's pose when they plan,
    so whether they still succeed is a finding, printed per seed. Only the agreement of the two readings is asserted."""
    for seed in SEEDS:
        res = rollout(task, seed, _trace_dir(tmp_path_factory), tag="moved", fault=True)
        world = res.pop("world")
        moved = res.pop("moved")
        shift = moved["after"] - moved["before"]
        assert np.allclose(shift, FAULT_OFFSET, atol=1e-5), "the fault moved the object by the offset"
        print(f"\nfault {task} seed={seed} env_success={res['env_success']} robowright={res['robowright']}")
        world.close(failed=False)
