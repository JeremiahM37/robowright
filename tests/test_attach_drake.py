"""robowright attached to a Drake simulation that Drake ships, loaded as the files are and not written here.

The scenes are Drake's own model files from ``pydrake/share/drake/examples``: the table
(``kuka_iiwa_arm/models/table/extra_heavy_duty_table.sdf``) and the block
(``kuka_iiwa_arm/models/objects/block_for_pick_and_place.urdf``), and the Acrobot
(``acrobot/Acrobot.sdf``). Drake's own arm examples need ``drake_models``, which this pydrake does not ship,
so the manipulator is not in these scenes. The start pose of the block is set in the context, as a Drake
example does before it simulates; the files themselves say nothing about where a body starts.
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
from robowright.backends.attached_drake import DrakeEngine  # noqa: E402
from robowright.errors import InvariantViolation  # noqa: E402
from robowright.trace import Trace  # noqa: E402

SHARE = os.path.join(os.path.dirname(pydrake.__file__), "share", "drake", "examples")
TABLE = os.path.join(SHARE, "kuka_iiwa_arm", "models", "table", "extra_heavy_duty_table.sdf")
BLOCK = os.path.join(SHARE, "kuka_iiwa_arm", "models", "objects", "block_for_pick_and_place.urdf")
ACROBOT = os.path.join(SHARE, "acrobot", "Acrobot.sdf")
DT = 0.01  # the control period: one AdvanceTo per control step
TABLE_TOP = 0.7645  # the table's surface collision: its pose z (0.736) plus half its thickness (0.0285)
BLOCK_HALF = 0.1  # the block's visual box is 0.2 m tall
BLOCK_START = (0.0, 0.0, 0.9)


def _read(path):
    with open(path) as f:
        return f.read()


class DraftScene:
    """A Drake diagram with the table, a block and a Simulator, and the handles a test needs."""

    def __init__(self, extra_blocks=0):
        b = DiagramBuilder()
        self.plant, _ = AddMultibodyPlantSceneGraph(b, time_step=1e-3)
        parser = Parser(self.plant)
        parser.SetAutoRenaming(True)
        parser.AddModelsFromString(_read(TABLE), "sdf")
        self.blocks = [parser.AddModelsFromString(_read(BLOCK), "urdf") for _ in range(1 + extra_blocks)]
        self.plant.Finalize()
        self.diagram = b.Build()
        self.sim = Simulator(self.diagram)
        self.context = self.sim.get_mutable_context()
        self.pc = self.plant.GetMyMutableContextFromRoot(self.context)
        self.block_body = self.plant.GetBodyByName("base_link", self.blocks[0][0])
        self.plant.SetFreeBodyPose(self.pc, self.block_body, RigidTransform(np.array(BLOCK_START)))

    def advance(self):
        self.sim.AdvanceTo(self.context.get_time() + DT)

    def attach(self, tmp_path, **kw):
        return rw.attach(
            simulator=self.sim,
            plant=self.plant,
            advance=kw.pop("advance", None),
            settings=rw.Settings(trace_dir=str(tmp_path)),
            **kw,
        )


def test_bodies_and_joints_are_named_and_a_shared_name_is_qualified_by_its_model(tmp_path):
    s = DraftScene(extra_blocks=1)
    w = s.attach(tmp_path)
    try:
        bodies = w.bodies()
        assert "link" in bodies  # the table's only link, unqualified: its name is unique
        qualified = [b for b in bodies if "::base_link" in b]
        assert len(qualified) == 2, bodies
        assert "base_link" not in bodies  # two blocks share it, so the bare name is not a body
        with pytest.raises(KeyError, match="no body, site or geom named 'base_link'"):
            w.body("base_link")
        assert set(w.object_names) == set(qualified)  # both blocks are free bodies
    finally:
        w.close(failed=False)


def test_pose_and_velocity_of_the_block_are_ground_truth(tmp_path):
    s = DraftScene()
    w = s.attach(tmp_path)
    try:
        block = w.body("base_link")
        assert np.allclose(block.position, BLOCK_START)
        assert np.allclose(block.quaternion, [1, 0, 0, 0])
        assert np.allclose(block.velocity, 0)
        s.advance()
        p_drake = s.plant.EvalBodyPoseInWorld(s.pc, s.block_body).translation()
        assert np.allclose(block.position, p_drake)
        assert block.velocity[2] == pytest.approx(s.plant.EvalBodySpatialVelocityInWorld(s.pc, s.block_body).translational()[2])
    finally:
        w.close(failed=False)


def test_the_block_rests_on_the_table_and_its_contact_force_is_its_weight(tmp_path):
    s = DraftScene()
    w = s.attach(tmp_path, advance=s.advance)
    try:
        w.step(200)  # two seconds: the block falls the 0.1 m to the table and settles
        block = w.body("base_link")
        expect(block).to_be_near((0.0, 0.0, TABLE_TOP + BLOCK_HALF), tol=0.005, timeout=1.0)
        expect(block).to_be_at_rest(timeout=1.0, hold=0.2)
        assert "link" in block.contacts()  # the table, by its link name
        normal = sum(f for a, b, f in w.backend.contact_pairs() if {a, b} == {"base_link", "link"})
        weight = s.block_body.default_mass() * 9.81
        assert normal == pytest.approx(weight, rel=0.02)  # the block holds still on the table only if it pushes back with its weight
    finally:
        w.close(failed=False)


def test_an_invariant_holds_on_every_step_of_their_own_loop_and_fails_at_the_first_step_that_breaks_it(tmp_path):
    s = DraftScene()
    w = s.attach(tmp_path)
    try:
        block = w.body("base_link")
        expect(block).always.to_be_above(TABLE_TOP - 0.01)  # it never goes through the table
        for _ in range(300):
            s.advance()
            w.observe()  # their loop steps Drake; robowright observes each control step
        assert w.step_count == 300
        assert block.position[2] == pytest.approx(TABLE_TOP + BLOCK_HALF, abs=0.01)
        expect(block).always.to_be_above(TABLE_TOP + 0.5)  # false from the first step: the block rests lower
        with pytest.raises(InvariantViolation, match="invariant violated"):
            s.advance()
            w.observe()
        assert w.step_count == 301
    finally:
        if w.status == "running":
            w.close(failed=False)


def test_a_push_moves_the_block_and_is_recorded_as_a_fault(tmp_path):
    s = DraftScene()
    w = s.attach(tmp_path, advance=s.advance)
    try:
        w.step(200)  # settle on the table first
        block = w.body("base_link")
        x0 = block.position[0]
        # 0.5 N for 0.3 s slides the block about 14 cm. A harder push tips it over: it is 0.2 m tall and the force
        # acts at its centre, so that is the physics, not a fault of the attach.
        w.push("base_link", (0.5, 0.0, 0.0), duration=0.3)
        w.step(100)  # the push acts on the first 0.3 s of these
        dx = block.position[0] - x0
        assert dx > 0.05, dx  # moved along +x: the force reached the block through the plant
        path = w.close(failed=False)
        assert any(e["type"] == "fault" and e["name"] == "Push" for e in Trace(path).events), "the push is not in the trace"
    finally:
        if w.status == "running":
            w.close(failed=False)


def test_the_trace_records_every_step_and_names_the_drake_engine(tmp_path):
    s = DraftScene()
    w = s.attach(tmp_path, advance=s.advance)
    try:
        w.step(25)
        path = w.close(failed=False)
    finally:
        if w.status == "running":
            w.close(failed=False)
    tr = Trace(path)
    assert tr.meta["attached"]["engine"] == "drake"
    assert tr.meta["steps"] == 25 and len(tr) == 26
    assert tr.arrays["t"][-1] == pytest.approx(25 * DT, abs=1e-9)


def test_joints_are_read_by_name_and_set_at_rest_on_the_shipped_acrobot(tmp_path):
    b = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(b, time_step=1e-3)
    Parser(plant).AddModelsFromString(_read(ACROBOT), "sdf")
    plant.Finalize()
    diagram = b.Build()
    sim = Simulator(diagram)
    ctx = sim.get_mutable_context()
    pc = plant.GetMyMutableContextFromRoot(ctx)
    plant.SetPositions(pc, np.array([0.0, 0.0]))
    w = rw.attach(
        simulator=sim, plant=plant, advance=lambda: sim.AdvanceTo(ctx.get_time() + DT), settings=rw.Settings(trace_dir=str(tmp_path))
    )
    try:
        assert w.backend.named_joints() == ["shoulder", "elbow"]
        assert {"upper_link", "lower_link"} <= set(w.bodies())
        w.set_joint("shoulder", 0.5)  # off the upright balance point (0, 0), where the Acrobot would stay put
        assert w.joint("shoulder").value == pytest.approx(0.5)
        w.step(50)  # gravity swings the shoulder: the value moves, from a joint the test named
        assert abs(w.joint("shoulder").value - 0.5) > 1e-3
        with pytest.raises(KeyError, match="no joint named 'nope'"):
            w.joint("nope")
    finally:
        w.close(failed=False)


def test_a_diagram_needs_its_context_and_a_plant_inside_it_is_found(tmp_path):
    s = DraftScene()
    w = rw.attach(s.diagram, context=s.context, settings=rw.Settings(trace_dir=str(tmp_path)))
    try:
        assert isinstance(w.backend, DrakeEngine)
        assert w.backend.plant is s.plant
        assert "link" in w.bodies()
    finally:
        w.close(failed=False)
    with pytest.raises(ValueError, match="needs its context"):
        rw.attach(s.diagram)
