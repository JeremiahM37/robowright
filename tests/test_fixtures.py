"""A straight joint-space move through a fixed obstacle is detected (no ROS 2 and no Gazebo: the MuJoCo
world is robowright's own, and the fixture is handed to its backend the way the ROS 2 backend hands over
what Gazebo has). A line that clears it is not reported, and the obstacle is named."""

import numpy as np
import pytest

import robowright as rw
from robowright.obstacles import Obstacle
from robowright.scene import SceneSpec


@pytest.fixture
def world():
    w = rw.launch(SceneSpec(robot="xarm7", objects=[]), backend="mujoco", seed=0)
    yield w
    w.close(failed=False)


def _line(w):
    r = w.robot
    a = np.asarray(r.home_q, float)
    b = a + np.array([0.9, -0.5, 0.4, 0.0, 0.3, 0.0, 0.0])[: r.n_arm]
    return r, a, b


def _box_at(p, half):
    return Obstacle("wall", "box", (half, half, half), tuple(p), (1.0, 0.0, 0.0, 0.0))


def test_a_straight_joint_line_through_a_box_is_detected(world):
    r, a, b = _line(world)
    mid_tcp = r.kin.tcp((a + b) / 2)
    world.backend.fixtures = ([_box_at(mid_tcp, 0.12)], [])
    planner = r.planner
    planner.sync()
    assert planner.fixture_names  # the box became a named geom of the planner's model
    assert planner.line_hits(a, b) == {"wall"}


def test_a_line_that_clears_the_fixtures_is_not_reported(world):
    r, a, b = _line(world)
    world.backend.fixtures = ([_box_at((5.0, 5.0, 5.0), 0.1)], [])
    planner = r.planner
    planner.sync()
    assert planner.line_hits(a, b) == set()


def test_no_fixtures_means_nothing_to_check(world):
    r, a, b = _line(world)
    world.backend.fixtures = ([], [])  # the MuJoCo backend object is reused across worlds: say what this one has
    r._planner = None
    assert r.planner.fixture_names == {}
    assert r.planner.line_hits(a, b) == set()
