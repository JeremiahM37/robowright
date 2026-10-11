"""Replay of a run recorded on an attached Drake simulation: the factory rebuilds the simulation, and replay re-runs it.

The scene is Drake's own table and block, as tests/test_attach_drake.py loads them. Recording and replay each build
the diagram again through ``build()``, so a replay is a fresh simulation, not the recorded one stepped again. Replay
is compared with the recording as ``replay()`` compares every other engine; these tests state what was measured.
"""

import os

import numpy as np
import pytest

pydrake = pytest.importorskip("pydrake")

from pydrake.math import RigidTransform  # noqa: E402
from pydrake.multibody.parsing import Parser  # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.analysis import Simulator  # noqa: E402
from pydrake.systems.framework import DiagramBuilder  # noqa: E402

import robowright as rw  # noqa: E402
from robowright import expect  # noqa: E402
from robowright.replay import replay  # noqa: E402
from robowright.trace import Trace  # noqa: E402

SHARE = os.path.join(os.path.dirname(pydrake.__file__), "share", "drake", "examples")
TABLE = os.path.join(SHARE, "kuka_iiwa_arm", "models", "table", "extra_heavy_duty_table.sdf")
BLOCK = os.path.join(SHARE, "kuka_iiwa_arm", "models", "objects", "block_for_pick_and_place.urdf")
DT = 0.01  # the control period: one AdvanceTo per control step
START_Z = 0.9  # the block's start height; it falls onto the table
TOLERANCE = 1e-6  # the replay default


def build(z: float = START_Z, mass: float | None = None):
    """The table and the block, as their own code would set them up: returns the Simulator and its plant."""
    b = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(b, time_step=1e-3)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)
    with open(TABLE) as f:
        parser.AddModelsFromString(f.read(), "sdf")
    with open(BLOCK) as f:
        (block,) = parser.AddModelsFromString(f.read(), "urdf")
    plant.Finalize()
    sim = Simulator(b.Build())
    pc = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
    body = plant.GetBodyByName("base_link", block)
    plant.SetFreeBodyPose(pc, body, RigidTransform(np.array([0.0, 0.0, z])))
    if mass is not None:
        body.SetMass(pc, mass)
    return sim, plant


def _record(out_dir, *, own_loop: bool = False, name: str = "drake-run"):
    """A run with their own AdvanceTo, a push and a move. ``own_loop`` steps Drake in the test's loop, observed."""
    sim, plant = build()
    ctx = sim.get_mutable_context()

    def advance():
        sim.AdvanceTo(ctx.get_time() + DT)

    w = rw.attach(
        simulator=sim,
        plant=plant,
        advance=None if own_loop else advance,
        settings=rw.Settings(trace="on", trace_dir=str(out_dir)),
        name=name,
    )
    block = w.body("base_link")
    if own_loop:
        for step in range(180):
            advance()
            w.observe()  # their loop, observed: robowright applies no push here
            if step == 90:
                w.move_object("base_link", [0.05, 0.0, START_Z])
        expect(block).to_be_above(0.0)
    else:
        w.step(120)
        w.push("base_link", (0.5, 0.0, 0.0), duration=0.3)  # a push on the first 0.3 s of the next steps
        w.step(60)
        w.move_object("base_link", [0.05, 0.0, START_Z])
        w.step(40)
        expect(block).to_be_above(0.0)
    path = w.close()
    assert path is not None
    return path


def _assert_agrees(r):
    assert r.engine == "drake"
    assert r.matches, r.summary()
    assert max(r.max_qpos_error, r.max_object_error, r.max_body_error, r.max_body_angle) <= TOLERANCE, r.summary()
    assert "within tolerance" in r.summary()


def test_a_run_driven_by_robowright_replays_and_agrees_with_the_recording(tmp_path):
    path = _record(tmp_path)
    tr = Trace(path)
    assert tr.meta["attached"]["engine"] == "drake" and tr.meta["attached"]["inputs_recorded"] is False
    assert np.any(tr.arrays["forces"]), "the push is not in the trace's forces"  # the forces the steps had
    r = replay(path, make_env=build)
    _assert_agrees(r)
    assert r.steps == 220
    assert any("make_env must rebuild the same ones" in n for n in r.notes)
    # Measured on this machine: three replays in fresh processes gave all four errors exactly 0.0 (bit-identical).


def test_a_factory_may_return_the_simulator_alone(tmp_path):
    path = _record(tmp_path)
    r = replay(path, make_env=lambda: build()[0])  # one plant in the diagram: found without being passed
    _assert_agrees(r)


def test_a_run_that_their_own_loop_stepped_replays_too(tmp_path):
    path = _record(tmp_path, own_loop=True)
    r = replay(path, make_env=build)
    _assert_agrees(r)
    assert r.steps == 180


def test_a_changed_initial_state_is_reported_as_diverged_from_the_first_step(tmp_path):
    path = _record(tmp_path)
    r = replay(path, make_env=lambda: build(z=START_Z + 0.05))
    assert not r.matches
    assert r.first_divergent_step == 0, r.summary()  # the start state itself differs
    assert "differs from the recorded run" in r.summary()
    assert r.max_body_error == pytest.approx(0.05, abs=1e-3)  # the block starts 5 cm higher


def test_a_changed_mass_is_reported_as_diverged(tmp_path):
    path = _record(tmp_path)
    r = replay(path, make_env=lambda: build(mass=2.0))
    assert not r.matches
    assert r.first_divergent_step is not None and r.first_divergent_step > 0, r.summary()
    assert "MISMATCH" in r.summary()


def test_a_drake_trace_without_a_factory_is_refused_with_the_reason(tmp_path):
    path = _record(tmp_path)
    with pytest.raises(ValueError, match="cannot rebuild this run's environment"):
        replay(path)


def test_a_factory_that_is_not_a_drake_simulator_is_refused(tmp_path):
    path = _record(tmp_path)
    gym_like = type("NotDrake", (), {})()
    with pytest.raises(ValueError, match="make_env must return its Simulator"):
        replay(path, make_env=lambda: gym_like)


def test_codegen_does_not_write_a_test_for_a_drake_run_yet(tmp_path):
    from robowright.codegen import generate

    with pytest.raises(ValueError, match="a test cannot be generated for a drake run"):
        generate(_record(tmp_path))
