"""What a new user meets first: versions, install hints, help, and messages that say what went wrong."""

import importlib.metadata
import importlib.util
import os
import sys
import time

import pytest

import robowright as rw
from robowright import cli, condition


def test_the_version_is_the_installed_one():
    assert rw.__version__ == importlib.metadata.version("robowright")


def test_the_drake_hint_mentions_python_only_where_it_matters():
    from robowright.backends.base import _INSTALL

    assert ("3.12" in _INSTALL["drake"]) == (sys.version_info < (3, 12))


def _without(monkeypatch, *names):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name in names else real(name, *a, **k))


def test_a_missing_engine_is_one_line_before_anything_runs(pytester, monkeypatch):
    _without(monkeypatch, "pybullet")
    pytester.makepyfile("def test_x(world):\n    pass\n")
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-backend", "mujoco,pybullet")
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    r.stderr.fnmatch_lines(["*--rw-backend pybullet: the 'pybullet' engine is not installed: pip install*robowright?pybullet?*"])
    assert "Traceback" not in r.stderr.str()


def test_debug_refuses_to_wait_for_a_person_in_ci(pytester, monkeypatch):
    monkeypatch.setenv("CI", "true")
    pytester.makepyfile("def test_x(world):\n    pass\n")
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-debug")
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    r.stderr.fnmatch_lines(["*--rw-debug waits for a person*cannot run in CI*"])


def test_a_rollout_that_failed_says_why(robot, scene):
    done = condition(scene["cube"], "to_be_inside", scene["bin"])
    rollout = robot.run_policy(lambda obs: obs["target"], until=done, timeout=0.2)
    assert not rollout.success
    assert rollout.reason and "bin" in rollout.reason
    assert rollout.reason in repr(rollout)


def test_init_says_the_mcp_server_needs_its_extra(monkeypatch):
    _without(monkeypatch, "mcp")
    assert 'pip install "robowright[mcp]"' in cli._mcp_missing()
    monkeypatch.undo()
    if importlib.util.find_spec("mcp") is not None:
        assert cli._mcp_missing() == ""


def test_show_trace_finds_a_trace_by_test_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="no traces"):
        cli._find_trace(None)
    traces = tmp_path / "robowright-traces"
    traces.mkdir()
    old, new = traces / "test_pick[mujoco-panda].zip", traces / "test_pick[drake-panda].zip"
    old.write_bytes(b"")
    new.write_bytes(b"")
    now = time.time()
    os.utime(old, (now - 10, now - 10))
    assert cli._find_trace("test_pick").endswith(new.name)  # the newest
    assert cli._find_trace("mujoco").endswith(old.name)
    assert cli._find_trace(str(old)) == str(old)  # a path is itself
    with pytest.raises(SystemExit, match="no trace named like 'test_place'"):
        cli._find_trace("test_place")


def test_test_help_is_robowrights_options_not_all_of_pytests():
    text = cli._rw_options_help()
    assert text and "--rw-backend" in text and "--rw-trace" in text
    assert "--junitxml" not in text  # pytest's own
    assert len(text.splitlines()) < 200
