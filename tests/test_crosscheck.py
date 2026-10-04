"""Cross-checking a run on another engine."""

import pytest

import robowright as rw
from robowright import expect
from robowright.crosscheck import crosscheck


def _pick_and_place(tmp_path, robot):
    with rw.launch(robot=robot, name=robot, settings=rw.Settings(trace="on", trace_dir=str(tmp_path))) as w:
        w.robot.reset_to()
        w.robot.pick(w.scene["cube"])
        w.robot.place(on=w.scene["bin"])
        expect(w.scene["cube"]).to_be_inside(w.scene["bin"])
    return w.trace_path


def test_an_outcome_both_engines_share(tmp_path):
    pytest.importorskip("pybullet")
    r = crosscheck(_pick_and_place(tmp_path, "panda"), "pybullet")
    assert r.passed and r.agrees, r.summary()
    assert r.object_offsets["cube"] < 0.03


def test_an_outcome_that_depends_on_the_engine(tmp_path):
    """A policy run on the WidowX: its 2.2 N grip holds the cube in MuJoCo; PyBullet's contacts let it slip."""
    pytest.importorskip("pybullet")
    from robowright import condition
    from robowright.policies import ScriptedPickPlace

    s = rw.Settings(trace="on", trace_dir=str(tmp_path))
    with rw.launch(robot="wx250s", seed=0, name="wx", settings=s) as w:
        w.robot.reset_to()
        w.faults.jitter("cube", xy_std=0.02, yaw_std=0.5)
        done = condition(w.scene["cube"], "to_be_inside", w.scene["bin"])
        assert w.robot.run_policy(ScriptedPickPlace(), until=done, hold=1.0, timeout=15, privileged=True).success
    r = crosscheck(w.trace_path, "pybullet")
    assert not r.passed and not r.agrees
    assert "VERDICTS DIFFER" in r.summary()
