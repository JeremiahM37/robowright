"""Running trained models as policies: robowright.learned.LearnedPolicy."""

import sys
import textwrap
import warnings

import numpy as np
import pytest

import robowright as rw
from robowright import condition, plugins
from robowright.learned import LearnedPolicy
from robowright.policies import ScriptedPickPlace

STATE = ("qpos", "objects.cube", "objects.cube.quat", "objects.bin", "objects.bin.quat")
LO, HI = np.full(20, -400.0), np.full(20, 400.0)


class InItsOwnConventions:
    """The scripted expert behind a model's interface, as a checkpoint trained on another robot
    stack would have it: joints in reverse order and in degrees, the gripper 0..100, state and
    action scaled to -1..1, a batch dimension, 10-step chunks. Its conversions are written here
    independently of LearnedPolicy's, so the pick only succeeds if both directions agree."""

    def __init__(self):
        self.expert = ScriptedPickPlace()
        self.resets = 0

    def reset(self):
        self.expert.reset()
        self.resets += 1

    def select_action(self, batch):
        s = (np.asarray(batch["obs"])[0] + 1) / 2 * (HI - LO) + LO
        joints = s[:6][::-1]  # the model's order is the gripper, then the arm from the wrist down
        q = np.append(np.radians(joints[:5]), joints[5] / 100)
        objects = {"cube": (s[6:9], s[9:13]), "bin": (s[13:16], s[16:20])}
        chunk = self.expert({"qpos": q, "objects": objects, "robot": "so101"})
        out = np.array([np.append(row[5] * 100, np.degrees(row[:5][::-1])) for row in chunk])
        return {"action": (2 * (out - LO[:6]) / (HI[:6] - LO[:6]) - 1)[None]}


def _policy(model, **kw):
    order = ["gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]
    stats = {"state": {"min": LO, "max": HI}, "action": {"min": LO[:6], "max": HI[:6]}}
    return LearnedPolicy(model, state=STATE, state_key="obs", joints=order, units="deg", gripper=(0, 100), normalize=stats, **kw)


def test_a_model_in_its_own_conventions_picks_and_places(quiet_world):
    w = quiet_world()
    model = InItsOwnConventions()
    policy = _policy(model)
    assert policy.privileged and policy.cameras == ()  # run_policy reads both off the policy
    done = condition(w.scene["cube"], "to_be_inside", w.scene["bin"])
    rollout = w.robot.run_policy(policy, until=done, hold=1.0, timeout=15)
    assert rollout.success, rollout
    assert model.resets == 1


def test_the_translation_round_trips():
    policy = _policy(InItsOwnConventions())
    policy._bind("so101")
    q = np.array([0.1, -0.4, 0.7, 1.2, -0.3, 0.25])
    m = policy.to_model(q)
    assert np.allclose(m, [25.0, *np.degrees([-0.3, 1.2, 0.7, -0.4, 0.1])])
    a = 2 * (m - LO[:6]) / (HI[:6] - LO[:6]) - 1  # as the model would return it, normalised
    assert np.allclose(policy.to_robot(a), q)


def test_the_targets_last_commanded_are_state_too(quiet_world):
    """``"target"`` is what the robot was last told, in the model's conventions like ``"qpos"``:
    while the jaws rest on a cube the readings stand still, and the command shows how far a
    close has got."""
    w = quiet_world()
    seen = []
    policy = LearnedPolicy(lambda b: seen.append(b["observation.state"][0]) or np.zeros(6), state=("qpos", "target"), units="deg")
    goal = np.array([0.2, -0.3, 0.4, 0.5, -0.1, 0.35])
    w.robot.run_policy(lambda obs: goal, timeout=0.1)  # the last command, still being reached
    w.robot.run_policy(policy, timeout=0.02)
    qpos, target = np.split(seen[0], 2)
    assert np.allclose(target, [*np.degrees(goal[:5]), goal[5]])
    assert not np.allclose(qpos, target, atol=0.5)  # the joints have not got there in 0.1 s


def test_a_torch_module_gets_tensors_images_and_the_task(quiet_world):
    torch = pytest.importorskip("torch")
    seen = {}

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))

        def forward(self, batch):
            seen.update({k: (type(v).__name__, tuple(getattr(v, "shape", ())), v) for k, v in batch.items()})
            hold = batch["observation.state"][:, None, :].repeat(1, 10, 1)  # a 10-step chunk that holds still
            return hold * self.scale

    w = quiet_world()
    policy = LearnedPolicy(Net(), images=["front"], image_size=(64, 48), steps=4)
    assert policy.cameras == ("front",) and not policy.privileged
    obs = w.robot.observe(policy.cameras, task="put the cube in the bin", image_size=policy.image_size)
    chunk = policy(obs)
    assert chunk.shape == (4, 6) and np.allclose(chunk, w.robot.qpos(), atol=1e-6)
    assert seen["observation.state"][:2] == ("Tensor", (1, 6))
    kind, shape, img = seen["observation.images.front"]
    assert (kind, shape) == ("Tensor", (1, 3, 48, 64)) and img.dtype == torch.float32 and 0.0 <= img.min() and img.max() <= 1.0
    assert seen["task"][2] == ["put the cube in the bin"]
    rollout = w.robot.run_policy(policy, task="hold still", timeout=0.5)
    assert rollout.steps == 25 and len(rollout.infer_ms) == 7  # 4 of each chunk run: 25 steps take 7 calls


def test_files_modules_and_plugins_name_a_model(tmp_path, monkeypatch):
    """A TorchScript or torch.export file, ``module:name``, and ``loader:reference`` from a robowright.policies plugin."""
    torch = pytest.importorskip("torch")

    class Hold(torch.nn.Module):
        def forward(self, state):
            return state

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # TorchScript is deprecated, and still common
        torch.jit.script(Hold()).save(str(tmp_path / "hold.pt"))
    torch.export.save(torch.export.export(Hold(), (torch.zeros(1, 6),)), str(tmp_path / "hold.pt2"))
    (tmp_path / "my_models.py").write_text(
        textwrap.dedent(
            """
            import numpy as np

            class Hold:
                def __call__(self, batch):
                    return batch["observation.state"]
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    loaded = []
    monkeypatch.setattr(plugins, "_entry_points", lambda group: [_Loader(loaded)] if group == plugins.POLICIES else [])

    q = np.array([0.1, -0.4, 0.7, 1.2, -0.3, 0.25])
    obs = {"qpos": q, "robot": "so101"}
    files = [(str(tmp_path / f), {"args": ["observation.state"]}) for f in ("hold.pt", "hold.pt2")]
    for ref, kw in [*files, ("my_models:Hold", {}), ("hub:org/some-policy", {})]:
        policy = LearnedPolicy(ref, units="deg", gripper=(0, 100), **kw)
        assert np.allclose(policy(obs), [q]), ref
    assert loaded == [("org/some-policy", None)]
    with pytest.raises(ValueError, match="no module 'nowhere' and no policy loader of that name; policy loaders installed: hub"):
        LearnedPolicy("nowhere:thing")


class _Loader:
    name, value = "hub", "fake_pkg:load"

    def __init__(self, log):
        self.log = log

    def load(self):
        def load(reference, device=None):
            self.log.append((reference, device))
            return lambda batch: batch["observation.state"]

        return load


def test_a_project_names_its_policies(tmp_path, monkeypatch):
    """robowright.toml's [policies.NAME]: the model (relative to the file) and its settings,
    which a test can still override."""
    torch = pytest.importorskip("torch")

    class Hold(torch.nn.Module):
        def forward(self, state):
            return state

    (tmp_path / "ckpt").mkdir()
    torch.export.save(torch.export.export(Hold(), (torch.zeros(1, 6),)), str(tmp_path / "ckpt" / "hold.pt2"))
    (tmp_path / "robowright.toml").write_text(
        '[policies.holder]\nmodel = "ckpt/hold.pt2"\nargs = ["observation.state"]\nunits = "deg"\nsteps = 3\n'
    )
    monkeypatch.chdir(tmp_path)
    model, settings = plugins.project_policies()["holder"]
    assert model == str(tmp_path / "ckpt" / "hold.pt2") and settings == {"args": ("observation.state",), "units": "deg", "steps": 3}
    policy = LearnedPolicy("holder", steps=1)
    assert (policy.units, policy.steps, policy.args) == ("deg", 1, ["observation.state"])
    assert policy.to_config() == {"model": "holder", "steps": 1}  # what a generated test needs: the name, and what it changed


def test_codegen_writes_the_learned_policy_into_the_test(tmp_path, monkeypatch):
    from robowright.codegen import generate

    (tmp_path / "my_models.py").write_text("class Hold:\n    def __call__(self, batch):\n        return batch['observation.state']\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with rw.launch(settings=rw.Settings(trace="on", trace_dir=str(tmp_path))) as w:
        w.robot.reset_to()
        w.robot.run_policy(LearnedPolicy("my_models:Hold", steps=2), timeout=0.2)
        path = w.close(trace_path=tmp_path / "t.zip")
    code = generate(path)
    assert "from robowright.learned import LearnedPolicy" in code
    assert "policy = LearnedPolicy(model='my_models:Hold', steps=2)" in code
    # A model handed in as an object cannot be written into a test: codegen says so.
    with rw.launch(settings=rw.Settings(trace="on", trace_dir=str(tmp_path))) as w:
        w.robot.reset_to()
        w.robot.run_policy(LearnedPolicy(lambda b: b["observation.state"]), timeout=0.1)
        path = w.close(trace_path=tmp_path / "u.zip")
    assert "policy = ...  # TODO: recreate" in generate(path)


def test_mistakes_say_what_to_do(quiet_world):
    w = quiet_world()
    obs = w.robot.observe()
    with pytest.raises(KeyError, match=r"needs obs\['objects'\]: run it with privileged=True"):
        LearnedPolicy(lambda b: None, state=("qpos", "objects.cube"))(obs)
    with pytest.raises(ValueError, match="the model's action has 3 values; with joints .* it needs 6"):
        LearnedPolicy(lambda b: np.zeros(3))(obs)
    with pytest.raises(ValueError, match=r"joints \['elbow'\] are not so101's"):
        LearnedPolicy(lambda b: None, joints=["elbow"])(obs)
    with pytest.raises(KeyError, match=r"returned \['logits'\]; set output_key"):
        LearnedPolicy(lambda b: {"logits": np.zeros(6)})(obs)
    with pytest.raises(ValueError, match='units is "rad" or "deg"'):
        LearnedPolicy(lambda b: None, units="turns")
    with pytest.raises(ValueError, match="state feature 'objects'"):
        LearnedPolicy(lambda b: None, state=("objects",))


def test_a_model_that_leaves_the_gripper_alone(quiet_world):
    w = quiet_world()
    w.robot.gripper.open(0.4)
    arm = w.robot.qpos()[:5]
    policy = LearnedPolicy(lambda b: b["observation.state"], gripper=None)
    assert policy(w.robot.observe()).shape == (1, 5)  # the arm's targets: the gripper holds its own
    w.robot.run_policy(policy, timeout=0.3)
    assert abs(w.robot.gripper.opening - 0.4) < 0.08 and np.allclose(w.robot.qpos()[:5], arm, atol=0.01)


def test_the_example_policy_needs_onnxruntime_only_to_run():
    """Importing robowright.learned pulls in no ML framework: onnxruntime and torch load on use."""
    import subprocess

    code = "import sys, robowright.learned; print(any(m in sys.modules for m in ('torch', 'onnxruntime')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
