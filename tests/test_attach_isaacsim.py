"""rw.attach on an Isaac Sim world a script built with Isaac Sim's own API (isaacsim.core.api.World, without Isaac
Lab): a cube dropped beside a Franka, stepped by the script's own world.step(). Needs Isaac Sim, and Isaac Lab for its
app launcher and the Franka's model file; skipped elsewhere. Isaac Sim starts once for the module (headless).
"""

import importlib.util

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.errors import CapabilityError, InvariantViolation

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("isaacsim") is None or importlib.util.find_spec("isaaclab_assets") is None,
    reason="needs Isaac Sim and Isaac Lab (isaacsim, isaaclab_assets)",
)

CUBE_START = np.array([0.5, 0.3, 0.3])


@pytest.fixture(scope="module")
def sim():
    from isaaclab.app import AppLauncher

    AppLauncher(headless=True)
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.scene.add(DynamicCuboid(prim_path="/World/cube", name="cube", position=CUBE_START, size=0.05, mass=0.1))
    add_reference_to_stage(usd_path=FRANKA_PANDA_CFG.spawn.usd_path, prim_path="/World/franka")
    world.scene.add(SingleArticulation(prim_path="/World/franka", name="franka"))
    world.reset()
    yield world


@pytest.fixture
def world(sim, tmp_path):
    sim.reset()  # the cube back where the script put it
    w = rw.attach(sim, settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    yield w
    w.close()


def test_the_scene_is_named_and_read_as_isaac_sim_reads_it(world, sim):
    assert world.backend.engine == "isaacsim"
    bodies = world.bodies()
    assert "cube" in bodies and "franka/panda_hand" in bodies and "panda_hand" not in bodies  # (short names resolve, below)
    assert world.backend.free_bodies == ["cube"]
    pos, quat = sim.scene.get_object("cube").get_world_pose()
    assert np.allclose(world.body("cube").position, pos, atol=1e-6)
    assert np.allclose(world.body("panda_hand").position, world.body("franka/panda_hand").position)
    assert np.allclose(world.body("franka").position, world.body("franka/panda_link0").position, atol=1e-6)  # its root
    q = sim.scene.get_object("franka").get_joint_positions()
    assert world.backend.joint_value("panda_joint4") == pytest.approx(float(q[3]), abs=1e-6)


def test_the_scripts_own_steps_are_observed_and_the_cube_comes_to_rest(world, sim):
    expect(world.body("cube")).to_be_above(0.2, timeout=0)
    for _ in range(300):
        sim.step(render=False)  # the script's loop, as written for Isaac Sim: 5 s, for the dropped cube to settle
    assert world.step_count == 300 and world.time == pytest.approx(sim.current_time)
    cube = world.body("cube")
    expect(cube).to_be_at_rest(timeout=0)
    assert cube.position[2] == pytest.approx(0.025, abs=0.005)  # resting on the ground, half its size up
    assert len(world.trace.t) == 301
    # Isaac Sim's own velocity for the settled cube is the one it had as it landed; robowright's is how it moves now
    assert np.linalg.norm(sim.scene.get_object("cube").get_linear_velocity()) > 0.01
    assert np.linalg.norm(cube.velocity[:3]) < 1e-4


def test_an_invariant_fails_at_the_step_it_breaks(world, sim):
    expect(world.body("cube")).always.to_be_above(0.2)
    with pytest.raises(InvariantViolation, match="to_be_above"):
        for _ in range(120):
            sim.step(render=False)
    assert 5 < world.step_count < 120  # it falls past 0.2 m in a few steps, and the loop stopped there


def test_a_push_moves_the_cube_it_was_given_to(world, sim):
    for _ in range(60):
        sim.step(render=False)  # at rest on the ground first
    x0 = world.body("cube").position[0]
    world.push("cube", (3.0, 0.0, 0.0), duration=0.1)
    for _ in range(60):
        sim.step(render=False)
    assert world.body("cube").position[0] - x0 > 0.01, "the push did not move the cube"
    with pytest.raises(CapabilityError, match="free bodies"):
        world.backend.apply_force("franka/panda_hand", (1.0, 0.0, 0.0))


def test_a_joint_set_by_the_test_is_where_isaac_sim_says_it_is(world, sim):
    world.backend.set_joint("panda_joint1", 0.5)
    sim.step(render=False)
    assert float(sim.scene.get_object("franka").get_joint_positions()[0]) == pytest.approx(world.backend.joint_value("panda_joint1"))
    with pytest.raises(KeyError, match="no joint named"):
        world.backend.joint_value("elbow")


def test_contacts_say_why_they_are_not_read(world):
    with pytest.raises(CapabilityError, match="contacts are not read from an Isaac Sim world"):
        world.backend.contact_pairs()


def test_the_run_is_kept_but_not_replayed(world, sim):
    from robowright.replay import replay

    for _ in range(10):
        sim.step(render=False)
    path = world.close()
    assert rw.Trace(path).meta["attached"]["engine"] == "isaacsim"
    with pytest.raises(ValueError, match="cannot be replayed"):
        replay(path)
