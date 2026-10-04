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
    """A shove on the WidowX's 2.2 N grip: MuJoCo lets the cube go, PyBullet's stiffer contacts keep it."""
    pytest.importorskip("pybullet")
    s = rw.Settings(trace="on", trace_dir=str(tmp_path))
    with rw.launch(robot="wx250s", name="wx", settings=s) as w:
        w.robot.reset_to()
        w.robot.pick(w.scene["cube"])
        w.faults.push("cube", force=(0.0, 1.5, 0.0), duration=0.1)
        w.wait(0.5)
        with pytest.raises(rw.ExpectationError):
            expect(w.robot.gripper).to_be_holding(w.scene["cube"])
    r = crosscheck(w.trace_path, "pybullet")
    assert r.passed and not r.agrees, r.summary()
    assert "VERDICTS DIFFER" in r.summary()
