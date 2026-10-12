"""Replay and codegen for attached runs: a trace of an environment robowright did not build is run again and turned into a test.

gymnasium is not in the project's venv, so the environment is test_attach's ToyEnv, the same stand-in those
tests use. ``gymnasium.make`` (which the generated tests call for an environment with a spec) is a stub module
written by these tests; it builds the ToyEnv and stamps ``env.spec`` as gymnasium does. The generated tests are
run in a subprocess, so they exercise the code they would have in a user's project. A real gymnasium
environment is not exercised here.
"""

import subprocess
import sys
import textwrap
import zipfile
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import mujoco
import numpy as np
import pytest
from test_attach import XML, ToyEnv

import robowright as rw
from robowright import expect
from robowright.codegen import companion, generate
from robowright.replay import replay
from robowright.trace import Trace

TESTS = Path(__file__).resolve().parent
SRC = Path(rw.__file__).resolve().parent.parent
N_STEPS = 60  # enough to exceed INLINE_ACTIONS, so the long-run companion is exercised too


def _env(spec_id: str | None = "ToyBox-v0"):
    env = ToyEnv()
    if spec_id is not None:  # what gymnasium.make leaves on an environment
        env.spec = SimpleNamespace(id=spec_id, kwargs={"variant": "box"})
    return env


def _record(out_dir: Path, *, spec_id: str | None = "ToyBox-v0", steps: int = N_STEPS, name: str = "attached-run") -> Path:
    """A run as a user's test would make it: resets, their own steps, pushes, moves, joint edits and expectations."""
    env = _env(spec_id)
    w = rw.attach(env, settings=rw.Settings(trace="on", trace_dir=str(out_dir)), name=name)
    env.reset(seed=7)
    expect(w.body("box")).always.to_be_above(0.2)  # invariant: checked on every step below
    for _ in range(steps // 2):
        env.step(np.array([0.3]))
    w.observe()  # a step the loop took below env.step
    w.set_joint("slide", 0.2)
    w.push("box", (0.0, 0.0, 2.0), duration=0.05)
    for _ in range(steps - steps // 2):
        env.step(np.array([0.0]))
    w.move_object("box", [0.05, 0.0, 0.3])
    expect(w.body("box")).to_be_above(0.2, timeout=1.0)
    path = w.close()
    assert path is not None
    return path


def _rewrite_member(path: Path, member: str, transform) -> Path:
    """The trace with one member replaced by ``transform(bytes) -> bytes``; the others copied as they are."""
    out = path.with_name(path.stem + "-edited.zip")
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == member:
                data = transform(data)
            dst.writestr(info.filename, data)
    return out


def _npz_with(data: bytes, **changes) -> bytes:
    arrays = dict(np.load(BytesIO(data)))
    arrays.update(changes)
    buf = BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _gym_stub(root: Path) -> Path:
    """A ``gymnasium`` package with the one call codegen makes, ``make(id, **kwargs)``, for a subprocess to import."""
    pkg = root / "gymnasium"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        textwrap.dedent(f'''
        """Stand-in for gymnasium in the tests: make() builds the test ToyEnv and stamps env.spec as gymnasium does."""
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, {str(TESTS)!r})
        from test_attach import ToyEnv


        def make(env_id, **kwargs):
            env = ToyEnv()
            env.spec = SimpleNamespace(id=env_id, kwargs=kwargs)
            return env
        ''')
    )
    return root


def _run_test_file(test_file: Path, stub_root: Path | None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [str(stub_root) if stub_root else "", str(SRC)] if p)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
        cwd=test_file.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_a_trace_records_the_spec_the_actions_and_the_calls(tmp_path):
    path = _record(tmp_path)
    tr = Trace(path)
    env_rec = tr.meta["env"]
    spec = env_rec["spec"]
    assert (spec["id"], spec["kwargs"], spec["reproducible"]) == ("ToyBox-v0", {"variant": "box"}, True)
    assert env_rec["actions_recorded"] is True and env_rec["note"] is None
    controls = tr.controls
    assert len(controls) == N_STEPS + 1
    assert sum(kind == "observe" for _, kind, _ in controls) == 1
    assert [s for s, _, _ in controls] == list(range(1, N_STEPS + 2))  # every physics step, numbered as the world counted them
    names = [e["name"] for e in tr.events]
    assert {"env.reset", "move_object", "set_joint", "Push"} <= set(names)
    assert names.index("env.reset") < names.index("move_object")  # the calls keep the order the test made them in
    assert tr.events[names.index("env.reset")]["args"] == {"seed": 7}


def test_a_hand_built_environment_is_recorded_without_a_spec(tmp_path):
    path = _record(tmp_path, spec_id=None, steps=4)
    assert Trace(path).meta["env"]["spec"] is None
    assert companion(path) is None  # a short run is written into the test, not beside it


def test_replay_of_the_recorded_run_matches(tmp_path):
    path = _record(tmp_path)
    r = replay(path, make_env=_env)
    assert r.matches, r.summary()
    assert r.identical, r.summary()
    assert r.steps == N_STEPS + 1
    assert any("expectations and invariants are not checked again" in n for n in r.notes)


def test_replay_reports_a_changed_action_as_a_mismatch(tmp_path):
    path = _record(tmp_path)
    tr = Trace(path)
    actions = [(step, row) for step, kind, row in tr.controls if kind == "action"]
    changed_row = 10
    changed_step = actions[changed_row][0]
    rows = np.array([row for _, row in actions], dtype=float)
    rows[changed_row] += 0.5  # the environment's own action at that step, one value different
    edited = _rewrite_member(path, "actions.npz", lambda data: _npz_with(data, action=rows))
    r = replay(edited, make_env=_env)
    assert not r.matches
    assert r.first_divergent_step == changed_step  # the first state that differs is the one the change produced
    assert r.max_qpos_error > 1e-6
    assert "MISMATCH" in r.summary()


def test_replay_uses_the_recorded_spec_when_no_make_env_is_given(tmp_path, monkeypatch):
    path = _record(tmp_path)
    fake = ModuleType("gymnasium")
    calls = []

    def make(env_id, **kwargs):
        calls.append((env_id, kwargs))
        return _env(env_id)

    fake.make = make
    monkeypatch.setitem(sys.modules, "gymnasium", fake)
    r = replay(path)
    assert calls == [("ToyBox-v0", {"variant": "box"})]
    assert r.matches and r.identical


def test_replay_refuses_a_hand_built_environment_without_make_env(tmp_path):
    path = _record(tmp_path, spec_id=None, steps=4)
    with pytest.raises(ValueError, match="make_env="):
        replay(path)


def test_a_generated_test_passes_when_run_in_a_subprocess(tmp_path):
    rec = tmp_path / "rec"
    path = _record(rec)
    out = tmp_path / "out"
    out.mkdir()
    test_file = out / "test_toy_run.py"
    test_file.write_text(generate(path, test_name="test_toy_run_regression"))
    extra = companion(path, test_name="test_toy_run_regression")
    assert extra is not None  # more than INLINE_ACTIONS steps: the actions are beside the test
    (out / extra[0]).write_bytes(extra[1])
    text = test_file.read_text()
    assert "gymnasium.make('ToyBox-v0', variant='box')" in text
    assert "rw.attach(env, seed=0" in text
    assert "world.faults.push('box'" in text and "world.move_object('box'" in text and "world.set_joint('slide'" in text
    proc = _run_test_file(test_file, _gym_stub(tmp_path / "stub"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed" in proc.stdout


def test_a_hand_built_environment_gives_a_stub_that_fails_loudly(tmp_path):
    path = _record(tmp_path / "rec", spec_id=None, steps=4)
    out = tmp_path / "out"
    out.mkdir()
    test_file = out / "test_hand_built.py"
    text = generate(path, test_name="test_hand_built_regression")
    test_file.write_text(text)
    assert "def make_env():" in text and "# Stub:" in text
    assert "import gymnasium" not in text  # nothing to import for a hand-built run
    proc = _run_test_file(test_file, None)
    assert proc.returncode != 0
    assert "NotImplementedError: make_env() is a stub" in proc.stdout + proc.stderr


def test_replay_says_when_the_rebuilt_model_differs_from_the_recorded_one(tmp_path):
    path = _record(tmp_path)

    def heavier_env():  # the same scene with a box twice as heavy: another version of the environment
        env = ToyEnv()
        env.model = mujoco.MjModel.from_xml_string(XML.replace('mass="0.05"', 'mass="0.10"'))
        env.data = mujoco.MjData(env.model)
        return env

    r = replay(path, make_env=heavier_env)
    assert any("different MuJoCo model" in n for n in r.notes)
    assert not r.matches  # the box falls differently, so the states differ too


def test_a_real_gymnasium_robotics_run_replays_and_generates_a_test_that_runs_in_a_fresh_process(tmp_path):
    """FetchPickAndPlace from gymnasium_robotics: a third-party registration, and a non-default episode limit.

    Only runs where gymnasium and gymnasium_robotics are installed (not the project venv). The recording process
    registers the environment; the generated test and the replay CLI run in fresh processes, which must find it
    from the trace alone.
    """
    gymnasium = pytest.importorskip("gymnasium")
    gymnasium_robotics = pytest.importorskip("gymnasium_robotics")
    gymnasium.register_envs(gymnasium_robotics)
    env = gymnasium.make("FetchPickAndPlace-v4", max_episode_steps=100)
    w = rw.attach(env, settings=rw.Settings(trace="on", trace_dir=str(tmp_path / "rec")), name="fetch")
    env.reset(seed=3)
    env.action_space.seed(0)
    expect(w.body("object0")).always.to_be_above(0.0)
    for _ in range(80):
        env.step(env.action_space.sample())
    path = w.close()
    assert path is not None

    tr = Trace(path)
    spec = tr.meta["env"]["spec"]
    assert spec["entry_point"].startswith("gymnasium_robotics.")
    assert spec["max_episode_steps"] == 100 and spec["registered_max_episode_steps"] == 50

    r = replay(path)  # in this process, through the same spec
    assert r.identical, r.summary()
    assert not any("different MuJoCo model" in n for n in r.notes)

    cli = subprocess.run(
        [sys.executable, "-m", "robowright.cli", "replay", str(path)],
        capture_output=True,
        text=True,
        timeout=600,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert "bit-identical" in cli.stdout

    out = tmp_path / "gen"
    out.mkdir()
    test_file = out / "test_fetch_gen.py"
    test_file.write_text(generate(path, test_name="test_fetch_regression"))
    extra = companion(path, test_name="test_fetch_regression")
    assert extra is not None
    (out / extra[0]).write_bytes(extra[1])
    text = test_file.read_text()
    assert "gymnasium.make('gymnasium_robotics:FetchPickAndPlace-v4', reward_type='sparse', max_episode_steps=100)" in text
    proc = _run_test_file(test_file, None)  # a fresh process, which never imported gymnasium_robotics
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed" in proc.stdout


def test_a_reset_before_attach_is_named_as_the_reason_the_run_cannot_be_replayed(tmp_path):
    """gymnasium needs a reset before a step. One made before rw.attach never reaches the trace: replay and
    codegen say so, instead of stepping an environment that was never reset."""
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("gymnasium.envs.mujoco.reacher_v5")
    env = gym.make("Reacher-v5")
    env.reset(seed=1)
    world = rw.attach(env)
    for _ in range(3):
        env.step(env.action_space.sample())
    path = world.close(failed=True, trace_path=tmp_path / "late.zip")
    with pytest.raises(ValueError, match=r"reset before rw.attach\(env\).*call rw.attach\(env\) first"):
        replay(path)
