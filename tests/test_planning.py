"""Collision-free arm paths (robowright.planning)."""

import numpy as np
import pytest

import robowright as rw
from robowright.robot import DOWN
from robowright.scene import ObjectSpec, tabletop

# A wall 20 cm tall between two points low down, one either side of it.
WALL = tabletop(
    ObjectSpec("wall", "box", (0.1, 0.01, 0.1), (0.2, 0.0, None), color="gray", mass=50.0),
    ObjectSpec("cube", "box", (0.0125, 0.0125, 0.0125), (0.2, 0.15, None), color="red"),
    robot="panda",
)
A, B = np.array([0.2, -0.12, 0.06]), np.array([0.2, 0.12, 0.06])


def _launch():
    w = rw.launch(scene=WALL, settings=rw.Settings(trace="off"))
    w.robot.reset_to(w.robot._clear_ik(A, DOWN, 0.0))
    return w


def test_a_planned_path_goes_round_what_a_direct_move_would_hit():
    with _launch() as w:
        r = w.robot
        start = r._target[: r.n_arm].copy()
        goal = r._clear_ik(B, DOWN, 0.0)
        r.planner.sync()
        assert not r.planner.edge_clear(start, np.asarray(goal))  # straight in joint space: through the wall
        path = r.planner.plan(start, goal)
        assert len(path) > 2 and all(r.planner.edge_clear(a, b) for a, b in zip(path, path[1:]))
        assert np.allclose(path[0], start) and np.allclose(path[-1], goal)


def test_the_same_scene_plans_the_same_path():
    paths = []
    for _ in range(2):
        with _launch() as w:
            r = w.robot
            goal = r._clear_ik(B, DOWN, 0.0)
            r.planner.sync()
            paths.append(np.array(r.planner.plan(r._target[: r.n_arm].copy(), goal)))
    assert paths[0].shape == paths[1].shape and np.array_equal(paths[0], paths[1])


def test_a_planned_move_gets_over_the_wall_without_touching_it():
    with _launch() as w:
        rw.expect(w.robot).always.to_have_no_collisions()
        w.robot.arm.move_to(B, plan=True)
        rw.expect(w.robot.tcp).to_be_near(B, tol=0.01)


def test_no_path_through_a_closed_box_is_reported():
    with _launch() as w:
        with pytest.raises(rw.UnreachableError):
            w.robot.arm.move_to((0.2, 0.0, 0.05), plan=True)  # inside the wall
