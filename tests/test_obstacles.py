"""Gazebo's static scene as robowright's fixed obstacles (no ROS 2 and no Gazebo needed): a scene is
built by hand the way ``gz.msgs.Scene`` has it (each pose relative to its parent), and the reader must
give the obstacles where Gazebo would put them, leave out the robot and the scene's objects, and say
what it skipped."""

import numpy as np

from robowright.obstacles import from_scene, quat_rot

IDENT = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _table():
    # xarm_gazebo's world: OpenRobotics' Table at (0, -0.84), its top a 1.5 x 0.8 x 0.03 box at z = 1.0
    legs = [(sx * 0.68, sy * 0.38, 0.5) for sx in (1, -1) for sy in (1, -1)]
    return {
        "name": "table",
        "pose": ((0.0, -0.84, 0.0), (1.0, 0.0, 0.0, 0.0)),
        "links": [
            {
                "name": "link",
                "pose": IDENT,
                "collisions": [{"name": "surface", "pose": ((0, 0, 1.0), (1.0, 0, 0, 0)), "geometry": {"box": (0.75, 0.4, 0.015)}}]
                + [
                    {"name": f"leg{i}", "pose": (p, (1.0, 0.0, 0.0, 0.0)), "geometry": {"box": (0.02, 0.02, 0.5)}}
                    for i, p in enumerate(legs)
                ],
            }
        ],
    }


def test_the_table_top_is_where_gazebo_puts_it():
    obs, notes = from_scene([_table()])
    top = next(o for o in obs if o.name == "table" and o.size == (0.75, 0.4, 0.015))
    np.testing.assert_allclose(top.pos, (0.0, -0.84, 1.0))  # the model's pose, then the collision's
    assert top.kind == "box" and top.quat == (1.0, 0.0, 0.0, 0.0)
    assert len(obs) == 5 and notes == []
    legs_at = sorted(tuple(np.round(o.pos, 3)) for o in obs if o.size == (0.02, 0.02, 0.5))
    assert legs_at == [(-0.68, -1.22, 0.5), (-0.68, -0.46, 0.5), (0.68, -1.22, 0.5), (0.68, -0.46, 0.5)]


def test_a_plane_and_the_robot_and_the_objects():
    floor = {
        "name": "ground_plane",
        "pose": IDENT,
        "links": [{"name": "link", "pose": IDENT, "collisions": [{"name": "c", "pose": IDENT, "geometry": {"plane": (50.0, 50.0)}}]}],
    }
    robot = {
        "name": "UF_ROBOT",
        "pose": ((-0.2, -0.5, 1.021), (0.7071068, 0.0, 0.0, 0.7071068)),
        "links": [{"name": "link1", "pose": IDENT, "collisions": [{"name": "c", "pose": IDENT, "geometry": {"box": (0.1, 0.1, 0.1)}}]}],
    }
    cube = {
        "name": "cube",
        "pose": ((-0.2, -0.95, 1.0275), (1.0, 0.0, 0.0, 0.0)),
        "joints": [],
        "links": [{"name": "link", "pose": IDENT, "collisions": [{"name": "c", "pose": IDENT, "geometry": {"box": (0.0125,) * 3}}]}],
    }
    obs, notes = from_scene([floor, robot, cube], robot={"link1"}, exclude={"cube"})
    assert [o.name for o in obs] == ["ground_plane"]
    assert obs[0].kind == "plane" and obs[0].size == (50.0, 50.0)


def test_a_mesh_is_skipped_and_said_so_and_a_sphere_and_cylinder_are_kept():
    model = {
        "name": "prop",
        "pose": ((1.0, 0.0, 0.5), (1.0, 0.0, 0.0, 0.0)),
        "links": [
            {
                "name": "link",
                "pose": IDENT,
                "collisions": [
                    {"name": "m", "pose": IDENT, "geometry": {"mesh": "file:///x.dae"}},
                    {"name": "s", "pose": ((0, 0, 0.2), (1.0, 0, 0, 0)), "geometry": {"sphere": 0.05}},
                    {"name": "c", "pose": IDENT, "geometry": {"cylinder": (0.03, 0.1)}},
                ],
            }
        ],
    }
    obs, notes = from_scene([model])
    assert [o.kind for o in obs] == ["sphere", "cylinder"]
    np.testing.assert_allclose(obs[0].pos, (1.0, 0.0, 0.7))
    assert len(notes) == 1 and "mesh" in notes[0] and "not an obstacle" in notes[0]


def test_a_mesh_model_with_a_bounding_box_is_its_box():
    model = {
        "name": "bin_mesh",
        "pose": ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        "bounding_box": ((-0.1, -0.2, 0.0), (0.1, 0.2, 0.3)),
        "links": [{"name": "link", "pose": IDENT, "collisions": [{"name": "m", "pose": IDENT, "geometry": {"mesh": "x.dae"}}]}],
    }
    obs, notes = from_scene([model])
    assert len(obs) == 1 and obs[0].kind == "box"
    np.testing.assert_allclose(obs[0].size, (0.1, 0.2, 0.15))
    np.testing.assert_allclose(obs[0].pos, (0.5, 0.0, 0.15))
    assert len(notes) == 1 and "bounding box" in notes[0]


def test_a_yawed_model_turns_its_collisions_about_its_own_origin():
    yaw = np.pi / 2
    q = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    model = {
        "name": "wall",
        "pose": ((0.0, 0.0, 0.0), q),
        "links": [
            {
                "name": "link",
                "pose": IDENT,
                "collisions": [{"name": "c", "pose": ((0.3, 0.0, 0.0), (1.0, 0, 0, 0)), "geometry": {"box": (0.1, 0.2, 0.1)}}],
            }
        ],
    }
    obs, _ = from_scene([model])
    np.testing.assert_allclose(obs[0].pos, (0.0, 0.3, 0.0), atol=1e-9)
    np.testing.assert_allclose(quat_rot(obs[0].quat, (1, 0, 0)), (0, 1, 0), atol=1e-9)


def test_a_nested_model_carries_its_parents_pose():
    inner = {
        "name": "inner",
        "pose": ((0.0, 0.1, 0.0), (1.0, 0.0, 0.0, 0.0)),
        "links": [{"name": "link", "pose": IDENT, "collisions": [{"name": "c", "pose": IDENT, "geometry": {"box": (0.05, 0.05, 0.05)}}]}],
    }
    outer = {"name": "outer", "pose": ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), "links": [], "models": [inner]}
    obs, _ = from_scene([outer])
    np.testing.assert_allclose(obs[0].pos, (1.0, 0.1, 0.0))


def test_visuals_stand_in_where_a_link_has_no_collisions():
    # Gazebo's scene info carries visuals, and collisions only where the world has them: the visual is the geometry.
    model = {
        "name": "shelf",
        "pose": IDENT,
        "links": [
            {
                "name": "link",
                "pose": IDENT,
                "collisions": [],
                "visuals": [{"name": "v", "pose": IDENT, "geometry": {"box": (0.2, 0.2, 0.01)}}],
            }
        ],
    }
    obs, notes = from_scene([model])
    assert [o.size for o in obs] == [(0.2, 0.2, 0.01)] and notes == []
