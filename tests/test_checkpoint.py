"""world.checkpoint() and world.restore(), Playwright's storageState: set up once, start each case from that moment."""

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.codegen import generate
from robowright.errors import RobowrightError
from robowright.replay import replay
from robowright.scene import ObjectSpec, tabletop


def _where(w):
    return np.concatenate([w.backend.qpos(), *(w.backend.object_pose(n)[0] for n in w.object_names)])


def test_a_restored_world_is_the_world_it_was():
    with rw.launch() as w:
        cube, bin_ = w.scene["cube"], w.scene["bin"]
        w.robot.pick(cube)
        ready = w.checkpoint("cube in hand")
        before, t = _where(w), w.time
        w.robot.place(on=bin_)
        expect(cube).to_be_inside(bin_, timeout=2)
        w.restore(ready)
        assert np.array_equal(_where(w), before) and w.time == t
        expect(w.robot.gripper).to_be_holding(cube)  # what it held, it holds again
        w.robot.place(on=(0.25, -0.12, 0.0))  # and the robot goes on from there, not from its last target
        expect(cube).not_.to_be_inside(bin_)


def test_a_case_run_again_from_a_checkpoint_comes_out_the_same():
    with rw.launch() as w:
        w.robot.pick(w.scene["cube"])
        ready = w.checkpoint()
        runs = []
        for _ in range(2):
            w.restore(ready)
            w.robot.place(on=w.scene["bin"])
            runs.append(_where(w))
        assert np.array_equal(*runs)  # bit for bit


def test_a_traced_run_with_restores_replays_and_reads_back_as_code(tmp_path):
    with rw.launch(settings=rw.Settings(trace="on", trace_dir=str(tmp_path))) as w:
        w.robot.pick(w.scene["cube"])
        ready = w.checkpoint("cube in hand")
        w.robot.place(on=w.scene["bin"])
        w.restore(ready)
        w.robot.place(on=(0.25, -0.12, 0.0))
    (path,) = tmp_path.glob("*.zip")
    assert replay(path).identical
    code = generate(path)
    assert 'cp_cube_in_hand = world.checkpoint("cube in hand")' in code.replace("'", '"')
    assert "world.restore(cp_cube_in_hand)" in code


def test_a_checkpoint_starts_another_world_of_the_same_scene():
    with rw.launch() as a:
        a.robot.pick(a.scene["cube"])
        ready, there = a.checkpoint(), _where(a)
    with rw.launch() as b:
        b.restore(ready)
        assert np.array_equal(_where(b), there)
        expect(b.robot.gripper).to_be_holding(b.scene["cube"])


def test_a_checkpoint_of_another_world_is_refused():
    with rw.launch() as a:
        ready = a.checkpoint()
    other = tabletop(ObjectSpec("cube", "box", (0.0125,) * 3, (0.2, 0.05, None)))
    with rw.launch(other) as b, pytest.raises(RobowrightError, match="is of another world"):
        b.restore(ready)


def test_an_engine_that_cannot_save_its_state_says_so():
    pytest.importorskip("pybullet")
    with rw.launch(backend="pybullet") as w, pytest.raises(rw.CapabilityError, match="world.checkpoint needs the 'state'"):
        w.checkpoint()
