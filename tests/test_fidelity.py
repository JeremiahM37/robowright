import os

import pytest

import robowright as rw
from robowright import fidelity
from robowright.trace import Trace


def _world(robot="so101", mode=None, **kw):
    return rw.launch(rw.tabletop(robot=robot), settings=rw.Settings(fidelity=mode, **kw))


def test_published_is_the_default(monkeypatch):
    monkeypatch.delenv(fidelity.ENV, raising=False)
    w = _world(trace="off")
    assert w.fidelity == "published"
    assert all(c.kind != "adjusted" for c in w.model_changes)
    w.close()


def test_every_change_has_a_kind_and_sourced_ones_cite_their_source():
    w = _world("panda", "published", trace="off")
    kinds = {c.kind for c in w.model_changes}
    assert kinds <= set(fidelity.KINDS)
    grip = [c for c in w.model_changes if c.what == "grip force"]
    assert grip and grip[0].kind == "sourced" and grip[0].source.startswith("https://")
    w.close()


def test_adjusted_mode_adds_robowrights_own_tuning():
    published = _world("so101", "published", trace="off")
    adjusted = _world("so101", "adjusted", trace="off")
    extra = set(adjusted.model_changes) - set(published.model_changes)
    assert extra and all(c.kind == "adjusted" for c in extra)
    assert any(c.what == "policy servo integral" for c in extra)
    published.close()
    adjusted.close()


def test_the_policy_integral_only_helps_in_adjusted_mode():
    for mode, helps in (("published", False), ("adjusted", True)):
        w = _world("so101", mode, trace="off")
        assert w.robot._helps_policies is helps
        w.close()


def test_record_keeps_only_what_ran():
    with fidelity.using("published"), fidelity.recording() as log:
        fidelity.record("adjusted", "x", "not applied here")
        fidelity.record("repair", "y", "applied")
    assert [c.what for c in log] == ["y"]
    with fidelity.using("adjusted"), fidelity.recording() as log:
        fidelity.record("adjusted", "x", "applied here")
    assert [c.what for c in log] == ["x"]


def test_a_bad_mode_is_refused(monkeypatch):
    monkeypatch.setenv(fidelity.ENV, "tuned")
    with pytest.raises(ValueError, match="published, adjusted"):
        fidelity.mode()


def test_the_trace_says_how_faithful_the_run_was(tmp_path):
    w = _world("so101", "adjusted", trace="on", trace_dir=str(tmp_path))
    w.robot.pick(w.scene["cube"])
    path = w.close(failed=False, trace_path=tmp_path / "t.zip")
    tr = Trace(path)
    assert tr.meta["fidelity"] == "adjusted"
    assert any(c["kind"] == "adjusted" for c in tr.meta["model_changes"])
    assert "fidelity: adjusted" in tr.summary()


def test_a_replay_rebuilds_in_the_traces_mode(tmp_path, monkeypatch):
    from robowright.replay import replay

    w = _world("so101", "adjusted", trace="on", trace_dir=str(tmp_path))
    w.robot.pick(w.scene["cube"])
    path = w.close(failed=False, trace_path=tmp_path / "t.zip")
    monkeypatch.setenv(fidelity.ENV, "published")
    from robowright.backends import base

    built, create = [], base.create
    monkeypatch.setattr(base, "create", lambda *a, **k: built.append(fidelity.mode()) or create(*a, **k))
    assert replay(path).identical
    assert built == ["adjusted"]  # rebuilt as it ran, not in the environment's mode


def test_rw_fidelity_option(pytester):
    pytester.makepyfile(
        test_f="""
def test_mode(world):
    assert world.fidelity == "adjusted"
"""
    )
    before = os.environ.get(fidelity.ENV)
    pytester.runpytest("-p", "no:cacheprovider", "--rw-fidelity", "adjusted").assert_outcomes(passed=1)
    assert os.environ.get(fidelity.ENV) == before  # the inner run's mode does not leak into this one


def test_a_robot_behind_ros2_runs_as_it_is():
    from robowright.backends.base import GROUND_TRUTH, Backend

    class Real(Backend):  # what a ROS 2 robot claims: no ground truth
        capabilities = frozenset()

    assert Real.model_changes == [] and GROUND_TRUTH not in Real.capabilities
