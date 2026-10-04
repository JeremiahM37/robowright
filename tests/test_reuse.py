"""A closed world's scene is kept and restored for the next world: it must run exactly like a new build."""

from importlib.util import find_spec

import numpy as np
import pytest

import robowright as rw


def _pick_and_place(w):
    """A run that touches everything a kept backend must forget: contacts, a push, a weakened joint."""
    w.robot.reset_to()
    w.faults.push("cube", (0.0, 0.3, 0.0), duration=0.04)
    w.robot.pick(w.scene["cube"])
    w.faults.weak_joint(w.robot.model.arm_joints[-1], 0.7)
    w.robot.place(on=w.scene["bin"])
    return w


def _record(w, run):
    qs = []
    w._step_hooks.append(
        lambda world: qs.append(np.concatenate([world.backend.qpos(), *(world.backend.object_pose(n)[0] for n in world.object_names)]))
    )
    run(w)
    w.close()
    return np.array(qs)


BACKENDS = ["mujoco", *(b for b, mod in (("drake", "pydrake"), ("genesis", "genesis"), ("isaac", "isaacsim")) if find_spec(mod))]


def _shove(w):
    w.faults.push("robot", (0.0, 8.0, 0.0), duration=0.1)
    w.wait(0.6)


_RUNS = {"so101": _pick_and_place, "go2": _shove}


@pytest.mark.parametrize("robot", ["so101", "go2"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_reused_scene_runs_bit_for_bit_like_a_fresh_one(monkeypatch, backend, robot):
    from robowright.backends import base

    run = _RUNS[robot]

    def dirty(w):  # leaves gains scaled, a force pending and the robot elsewhere
        w.robot.reset_to()
        if robot != "go2":
            w.faults.weak_joint(w.robot.model.arm_joints[0], 0.5)
            w.robot.pick(w.scene["cube"])
        w.backend.apply_force("cube" if robot != "go2" else "robot", (0.0, 0.0, 2.0))
        w.close()

    settings = rw.Settings(trace="off")
    monkeypatch.setenv("ROBOWRIGHT_REUSE", "0")
    fresh = _record(rw.launch(robot=robot, backend=backend, settings=settings), run)
    monkeypatch.setenv("ROBOWRIGHT_REUSE", "2")
    base.close_kept()
    first = rw.launch(robot=robot, backend=backend, settings=settings)
    built = first.backend
    dirty(first)
    again = rw.launch(robot=robot, backend=backend, seed=5, settings=settings)
    assert again.backend is built  # kept, not rebuilt
    reused = _record(again, run)
    base.close_kept()
    assert fresh.shape == reused.shape and np.array_equal(fresh, reused)


@pytest.mark.parametrize("robot", ["so101", "go2"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_tracing_does_not_change_the_run(monkeypatch, tmp_path, backend, robot):
    monkeypatch.setenv("ROBOWRIGHT_REUSE", "0")
    runs = [
        _record(rw.launch(robot=robot, backend=backend, settings=rw.Settings(trace=t, trace_dir=str(tmp_path))), _RUNS[robot])
        for t in ("off", "on")
    ]
    assert np.array_equal(*runs)
