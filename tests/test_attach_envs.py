"""Real third-party environments through an attached session: gymnasium-robotics (Fetch, and gymnasium's own
"module:EnvId" form) and Meta-World through its own factory. Each test is skipped where its package is not installed.

The session API is driven directly (not over MCP), so this file needs only the environment packages.
"""

import numpy as np
import pytest

from robowright.errors import RobowrightError
from robowright.session import Session
from robowright.trace import Trace


@pytest.fixture
def session(tmp_path):
    s = Session(trace_dir=tmp_path)
    yield s
    if s.world is not None:
        s.close()


def test_fetch_unknown_version_hint_then_unversioned_id(session):
    pytest.importorskip("gymnasium")
    pytest.importorskip("gymnasium_robotics")
    with pytest.raises(RobowrightError) as info:
        session.sim_attach("FetchPickAndPlace-v9", seed=3)
    msg = str(info.value)
    assert "FetchPickAndPlace-v4" in msg and "packages registered: gymnasium_robotics" in msg, msg
    snap = session.sim_attach("FetchPickAndPlace", seed=3)  # gymnasium resolves the unversioned id to the latest
    assert "attached FetchPickAndPlace-v4, MujocoFetchPickAndPlaceEnv (mujoco)" in snap
    assert "- object0: at" in snap


def test_fetch_subjects_steps_expect_push_and_trace(session, tmp_path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("gymnasium_robotics")
    session.sim_attach("FetchPickAndPlace-v4", seed=3)
    assert "object0*" in session.sim_subjects() and "robot0:grip" in session.sim_subjects()
    assert "stepped 5 of 5" in session.sim_env_step(None, 5) and "episode continues" in session.sim_env_step([0.5, 0, 0, 0], 20)
    text = session.expect("object0", "to_be_at_rest", timeout=1)
    assert text.startswith("PASS") and "checked once" in text, text  # resting, and no advance so no wait
    text = session.expect("object0", "to_be_at_rest", {"hold": 0.5})
    assert text.startswith("FAIL") and "checked once" in text, text  # a hold cannot pass without time
    before = session.world.body("object0").position.copy()
    assert "queued a push" in session.push("object0", [0.0, 0.0, 40.0], duration=0.2)
    session.sim_env_step(None, 10)
    moved = float(np.linalg.norm(session.world.body("object0").position - before))
    assert moved > 0.05, f"the pushed object moved {moved:.3f} m"
    path = session.save_trace(str(tmp_path / "fetch.zip"))
    assert len(Trace(path).arrays["t"]) == session.world.step_count + 1  # one sample per env step, and the start state


def test_gymnasium_robotics_module_colon_env_id(session):
    pytest.importorskip("gymnasium")
    pytest.importorskip("gymnasium_robotics")
    assert "attached FetchReach-v4" in session.sim_attach("gymnasium_robotics:FetchReach-v4", seed=0)


def test_metaworld_through_its_own_factory(session):
    pytest.importorskip("metaworld")
    snap = session.sim_attach(entry_point="metaworld:make_mt_envs", kwargs={"name": "pick-place-v3", "seed": 0}, seed=0)
    assert "- obj: at" in snap, snap
    assert "stepped 5 of 5" in session.sim_env_step(None, 5)
    assert "obj*" in session.sim_subjects() and "tablelink" in session.sim_subjects()


def test_an_agents_fetch_session_becomes_a_test_that_passes_by_itself(session, tmp_path):
    """What an agent did over an attached session (steps, a shove, a check) is written as a test, which is then
    run in a fresh process: the environment made from its recorded id, the reset and the actions repeated."""
    import os
    import subprocess
    import sys

    pytest.importorskip("gymnasium_robotics")
    session.sim_attach("FetchPickAndPlace", seed=3)
    session.sim_env_step([0.5, 0.0, 0.0, 0.0], 30)
    session.push("object0", [0.0, 0.0, 40.0], 0.2)
    session.sim_env_step(None, 10)
    assert session.expect("object0", "to_be_above", {"target": 0.3}).startswith("PASS")
    test = tmp_path / "test_fetch_agent.py"
    code = session.generate_test("test_fetch_agent", str(test))
    assert "gymnasium_robotics:FetchPickAndPlace-v4" in code and "env.reset(seed=3)" in code, code
    src = os.path.dirname(os.path.dirname(os.path.abspath(__import__("robowright").__file__)))
    env = {**os.environ, "PYTHONPATH": src}
    cmd = [sys.executable, "-m", "pytest", str(test), "-q", "-p", "robowright.pytest_plugin", "-p", "no:cacheprovider"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout[-2000:]
