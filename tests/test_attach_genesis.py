"""robowright attached to a Genesis scene built by Genesis's own example code, on the CPU backend.

The scene is copied from ``genesis/_main.py``, the ``play()`` function of the installed genesis package
(``genesis-world`` 1.4.3): a ground plane and the Franka Panda from ``assets/xml/franka_emika_panda/panda.xml``.
Two changes, both for running headless: the viewer is off (``show_viewer=False``), and the scene gets one
manipuland, the blue box that Genesis ships in ``assets/urdf/blue_box``, so there is a free body to touch and push.
The scene is built once for this file (Genesis compiles its kernels on build, which takes about half a minute).
"""

import numpy as np
import pytest

gs = pytest.importorskip("genesis")

import robowright as rw  # noqa: E402
from robowright import expect  # noqa: E402
from robowright.backends.attached_genesis import GenesisEngine  # noqa: E402
from robowright.errors import InvariantViolation  # noqa: E402
from robowright.trace import Trace  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    gs.init(backend=gs.cpu)
    # genesis/_main.py play(), with the viewer off so it runs headless
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.add_entity(
        gs.morphs.URDF(file="urdf/blue_box/model.urdf", pos=(-0.45, 0.0, 0.3)), name="cube"
    )  # behind the robot: the arm reaches forward, and lands on a block in front
    scene.build()
    return scene


BOX_LINK = "part_1"  # the blue box's one link, as its URDF names it


def _attach(scene, tmp_path, **kw):
    scene.reset()  # the start of every test is the scene as it was built
    return rw.attach(scene, settings=rw.Settings(trace_dir=str(tmp_path)), **kw)


def test_the_scene_is_attached_as_a_genesis_engine_and_its_entities_and_links_are_named(scene, tmp_path):
    w = _attach(scene, tmp_path)
    try:
        assert isinstance(w.backend, GenesisEngine) and w.backend.engine == "genesis"
        assert {"hand", "left_finger", "right_finger", "link0", BOX_LINK} <= set(w.bodies())
        assert w.body("cube").name == "cube"  # the entity's name resolves to its base link
        assert w.object_names == ["cube"]  # the blue box is the one free body; the robot and the ground are fixed
        assert {"joint1", "joint7", "finger_joint1", "finger_joint2"} <= set(w.backend.named_joints())
    finally:
        w.close(failed=False)


def test_pose_and_velocity_are_ground_truth(scene, tmp_path):
    w = _attach(scene, tmp_path)
    try:
        cube = scene.entities[2]
        assert np.allclose(w.body("cube").position, cube.get_pos().cpu().numpy())
        assert np.allclose(w.body("cube").quaternion, cube.get_quat().cpu().numpy(), atol=1e-6)
        hand = scene.entities[1].get_link(name="hand")
        assert np.allclose(w.body("hand").position, hand.get_pos().cpu().numpy())
        assert np.allclose(w.body("hand").velocity[:3], hand.get_vel().cpu().numpy())
        assert np.isclose(np.linalg.norm(w.body("cube").quaternion), 1.0, atol=1e-5)
    finally:
        w.close(failed=False)


def test_the_block_rests_on_the_ground_and_its_contact_normal_force_is_its_weight(scene, tmp_path):
    w = _attach(scene, tmp_path, advance=lambda: scene.step())
    try:
        w.step(150)  # 1.5 s: the box falls the 0.3 m and settles on the plane
        cube = w.body("cube")
        expect(cube).to_be_at_rest(timeout=2.0, hold=0.2)
        pairs = [(a, b, f) for a, b, f in w.backend.contact_pairs() if BOX_LINK in (a, b)]
        assert pairs, "the block touches nothing"
        others = {b if a == BOX_LINK else a for a, b, _ in pairs}
        assert others <= set(w.bodies()) and BOX_LINK not in others
        normal = sum(f for _, _, f in pairs)
        mass = float(scene.entities[2].get_mass())
        assert normal == pytest.approx(mass * 9.81, rel=0.02)  # at rest, the ground pushes back with the block's weight
    finally:
        w.close(failed=False)


def test_a_push_moves_the_block_for_one_step_each_and_is_recorded(scene, tmp_path):
    w = _attach(scene, tmp_path, advance=lambda: scene.step())
    try:
        w.step(150)  # settle first
        cube = w.body("cube")
        x0 = cube.position[0]
        # The block weighs 4.7 N and its friction holds it below about 3 N: measured on this scene, 1 N and 3 N for
        # 0.1 s move it 0.1 mm and 0.5 mm, and 6 N moves it 16 mm. The push is away from the robot's base.
        w.push("cube", (-6.0, 0.0, 0.0), duration=0.1)
        w.step(60)
        dx = x0 - cube.position[0]
        assert dx > 0.005, dx  # the force reached the block through the solver
        path = w.close(failed=False)
        assert any(e["type"] == "fault" and e["name"] == "Push" for e in Trace(path).events)
    finally:
        if w.status == "running":
            w.close(failed=False)


def test_a_joint_is_read_and_set_by_name_on_the_franka(scene, tmp_path):
    w = _attach(scene, tmp_path)
    try:
        w.set_joint("joint1", 0.3)
        assert w.joint("joint1").value == pytest.approx(0.3, abs=1e-5)
        assert w.backend.qpos().shape == (len(w.backend.joint_names),)
        with pytest.raises(KeyError, match="no joint named 'nope'"):
            w.joint("nope")
    finally:
        w.close(failed=False)


def test_an_invariant_and_an_expectation_wait_through_the_scene_step(scene, tmp_path):
    w = _attach(scene, tmp_path, advance=lambda: scene.step())
    try:
        expect(w.body("cube")).always.to_be_above(0.0)  # the box never goes below the ground
        t0 = w.time
        expect(w.body("cube")).to_satisfy(lambda c: c.position[2] < 0.2, "the block has fallen below 0.2 m", timeout=3.0)
        assert w.time > t0  # the wait advanced the scene's own steps, not the clock of the test
        expect(w.body("cube")).always.to_be_above(5.0)  # false from the first step
        with pytest.raises(InvariantViolation, match="invariant violated"):
            w.step(5)
    finally:
        if w.status == "running":
            w.close(failed=False)


def test_their_own_steps_are_observed_and_the_trace_names_the_engine(scene, tmp_path):
    w = _attach(scene, tmp_path)
    try:
        t0 = w.time
        for _ in range(12):
            scene.step()  # their loop calls the scene directly: robowright sees each step through the wrapper
        assert w.step_count == 12
        assert w.time == pytest.approx(t0 + 12 * scene.dt, abs=1e-4)
        path = w.close(failed=False)
    finally:
        if w.status == "running":
            w.close(failed=False)
    tr = Trace(path)
    assert tr.meta["attached"]["engine"] == "genesis"
    assert tr.meta["steps"] == 12


def test_closing_the_world_gives_the_scene_its_own_step_back(scene, tmp_path):
    w = _attach(scene, tmp_path)
    assert "step" in scene.__dict__
    w.close(failed=False)
    assert "step" not in scene.__dict__ and "reset" not in scene.__dict__
    before = w.step_count
    scene.step()
    assert w.step_count == before  # nothing observes the scene once the world is closed
