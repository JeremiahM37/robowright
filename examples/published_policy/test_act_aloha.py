"""A published policy, tested in the environment it was published for, with robowright attached.

LeRobot's ACT checkpoint ``lerobot/act_aloha_sim_transfer_cube_human`` is run in gym-aloha's
``AlohaTransferCube-v0``, exactly as LeRobot's own pipeline runs it: the checkpoint's normaliser
statistics go into LeRobot's pre- and post-processors, and the policy's action chunks are the
env's actions. robowright never touches the policy or the loop. It attaches to the environment:
every ``env.step`` is observed, the test's assertions read the cube's ground-truth pose and its
contacts, and the trace of each episode is saved.

"Success" is gym-aloha's own reward: the cube touches the left gripper's finger and not the table.
The test asserts the physical facts with robowright and checks that its verdict agrees with the
environment's ``is_success`` on every seed.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect

REPO = "lerobot/act_aloha_sim_transfer_cube_human"
ENV_ID = "gym_aloha/AlohaTransferCube-v0"
EPISODE_LEN = 400  # as in lerobot/envs/configs.py AlohaEnv.episode_length
LEFT_FINGER = "vx300s_left/10_left_gripper_finger"  # the geom gym-aloha's reward tests
TABLE = "table"
CUBE = "box"  # the free body whose geom is gym-aloha's red_box
ABOVE_TABLE = 0.03  # the cube's centre, in metres: resting on the table it is at 0.02
SEEDS = [0, 1, 2, 3, 4]

pytest.importorskip("lerobot")
pytest.importorskip("gym_aloha")
gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")


def _checkpoint() -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(REPO, local_files_only=True)  # no network: a cached checkpoint or a skip
    except Exception as e:  # noqa: BLE001 - any miss means the checkpoint is not here
        pytest.skip(f"checkpoint {REPO} is not in the local Hugging Face cache: {type(e).__name__}")


def _load_stats(path: str) -> dict:
    """The normaliser statistics, which the checkpoint stores as buffers LeRobot does not load into the model."""
    from safetensors import safe_open

    names = {
        "observation_images_top": "observation.images.top",
        "observation_state": "observation.state",
        "action": "action",
    }
    stats: dict = {}
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            for pre in ("normalize_inputs.buffer_", "normalize_targets.buffer_", "unnormalize_outputs.buffer_"):
                if key.startswith(pre):
                    feat = key[len(pre) :].rsplit(".", 1)[0]
                    stats.setdefault(names[feat], {})[key.split(".")[-1]] = f.get_tensor(key).clone()
                    break
    return stats


@pytest.fixture(scope="module")
def act():
    """The published policy with LeRobot's own pre- and post-processing, built once."""
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    path = _checkpoint()
    policy = ACTPolicy.from_pretrained(path, strict=False)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, dataset_stats=_load_stats(path))
    return policy, pre, post, preprocess_observation


def _trace_dir(tmp_path_factory) -> Path:
    d = os.environ.get("ACT_TRACE_DIR")
    return Path(d) if d else tmp_path_factory.mktemp("act_traces")


def rollout(act, seed: int, trace_dir: Path, fault=None, tag: str = "run") -> dict:
    """One episode, with the policy driving the environment as LeRobot drives it.

    ``fault`` is ``(body, offset)`` applied to the cube right after reset, and the policy's first
    observation is then taken from the environment's own observation function, so the policy
    sees the moved cube.
    """
    policy, pre, post, preprocess_observation = act
    env = gym.make(ENV_ID, obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=EPISODE_LEN)
    world = rw.attach(env, name=f"act_aloha_{tag}_seed{seed}", settings=rw.Settings(trace_dir=str(trace_dir)))
    try:
        obs, _ = env.reset(seed=seed)
        cube = world.body(CUBE)
        if fault is not None:
            before = cube.position.copy()
            world.move_object(CUBE, before + np.asarray(fault))
            assert np.allclose(cube.position, before + np.asarray(fault), atol=1e-9)
            core = env.unwrapped
            raw = core._env.task.get_observation(core._env.physics)  # the environment's own observation
            obs = core._format_raw_obs(raw)
        start = cube.position.copy()
        policy.reset()
        steps, env_success, done = 0, False, False
        while not done:
            batch = pre(preprocess_observation(obs))
            with torch.inference_mode():
                action = post(policy.select_action(batch))
            obs, _, terminated, truncated, info = env.step(action[0].cpu().numpy())
            steps += 1
            env_success = env_success or bool(info.get("is_success", False))
            done = terminated or truncated or steps >= EPISODE_LEN
        touching = set(cube.contacts())
        verdict = LEFT_FINGER in touching and TABLE not in touching  # gym-aloha's own reward, read from contacts
        result = dict(
            seed=seed,
            env_success=env_success,
            verdict=verdict,
            steps=steps,
            start=start.round(3).tolist(),
            end=cube.position.round(3).tolist(),
            touching=sorted(touching),
        )
        result["_world"] = world
        result["_cube"] = cube
        return result
    except BaseException:
        world.close(failed=True)
        raise


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(0, id="seed0"),
        *[pytest.param(s, id=f"seed{s}") for s in SEEDS[1:]],
    ],
)
def test_act_holds_the_cube_above_the_table(act, tmp_path_factory, seed):
    res = rollout(act, seed, _trace_dir(tmp_path_factory), tag="hold")
    world, cube = res.pop("_world"), res.pop("_cube")
    print(f"\nseed={seed} robowright={res}")
    failure = None
    try:
        assert res["verdict"] == res["env_success"], "robowright's verdict disagrees with the environment's is_success"
        expect(cube).to_be_touching(LEFT_FINGER)
        expect(cube).not_.to_be_touching(TABLE)
        expect(cube).to_satisfy(lambda s: s.position[2] > ABOVE_TABLE, f"above the table (z > {ABOVE_TABLE})")
    except BaseException as e:
        failure = e
    path = world.close(failed=failure is not None)
    if failure is not None:
        from robowright.trace import Trace

        pytest.fail(f"{failure}\n{Trace(path).summary(events=10)}", pytrace=False)


def test_cube_moved_3cm_after_reset(act, tmp_path_factory):
    """The fault from the brief: the cube 3 cm off after reset. The outcome is reported, not asserted."""
    rows = []
    for seed in SEEDS:
        res = rollout(act, seed, _trace_dir(tmp_path_factory), fault=(0.03, 0.0, 0.0), tag="moved3cm")
        world = res.pop("_world")
        res.pop("_cube")
        path = world.close(failed=False)
        rows.append({**res, "trace": path.name})
    print(
        "\n"
        + "\n".join(
            f"fault seed={r['seed']} env_success={r['env_success']} robowright={r['verdict']} "
            f"steps={r['steps']} start={r['start']} end={r['end']}"
            for r in rows
        )
    )
