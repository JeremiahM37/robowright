import numpy as np
import pytest

import robowright as rw
from robowright.scene import ObjectSpec, tabletop


def test_get_is_strict(quiet_world):
    w = quiet_world(
        scene=tabletop(
            ObjectSpec("a", color="red", pos=(0.2, 0.0, None)),
            ObjectSpec("b", color="red", pos=(0.25, 0.0, None)),
            ObjectSpec("c", color="green", pos=(0.2, -0.08, None), tags=("target",)),
        )
    )
    assert w.scene.get(color="green").name == "c"
    assert w.scene.get(tag="target").name == "c"
    with pytest.raises(LookupError, match="exactly one"):
        w.scene.get(color="red")
    assert [h.name for h in w.scene.all(color="red")] == ["a", "b"]
    assert w.scene.nearest(to=(0.26, 0, 0)).name == "b"


def test_unknown_object(quiet_world):
    w = quiet_world()
    with pytest.raises(KeyError, match="no object named"):
        w.scene["nope"]


def test_handles_are_live(quiet_world):
    w = quiet_world()
    cube = w.scene["cube"]
    before = cube.position.copy()
    w.move_object("cube", [0.2, 0.0, 0.0125])
    assert not np.allclose(before, cube.position)


def test_bin_bounds_follow_yaw(quiet_world):
    w = quiet_world(scene=tabletop(ObjectSpec("bin", "bin", (0.06, 0.03, 0.02), (0.2, 0.1, 0.0), yaw=np.pi / 2)))
    lo, hi = w.scene["bin"].bounds()
    assert hi[0] - lo[0] == pytest.approx(0.06, abs=1e-6)
    assert hi[1] - lo[1] == pytest.approx(0.12, abs=1e-6)


def test_perception_hook_when_no_ground_truth(quiet_world):
    w = quiet_world()
    w.backend.capabilities = frozenset()  # pretend we are on hardware
    w.perception["cube"] = lambda: (np.array([0.1, 0.2, 0.3]), np.array([1.0, 0, 0, 0]))
    assert w.scene["cube"].position.tolist() == [0.1, 0.2, 0.3]
    with pytest.raises(rw.CapabilityError, match="perception"):
        _ = w.scene["bin"].position


def _stacked(quiet_world):
    """A red cube in the bin, a green cube on the floor and a blue one resting on the green."""
    w = quiet_world(
        scene=tabletop(
            ObjectSpec("in_bin", color="red", pos=(0.2, 0.12, 0.03)),
            ObjectSpec("floor_cube", color="green", pos=(0.22, -0.06, None)),
            ObjectSpec("stacked", color="blue", pos=(0.22, -0.06, 0.0375)),
            ObjectSpec("bin", kind="bin", size=(0.05, 0.05, 0.02), pos=(0.2, 0.12, 0.0), color="blue"),
        )
    )
    w.wait(1.0)  # everything settles where it rests
    return w


def test_relational_filters_find_objects_by_where_they_are(quiet_world):
    w = _stacked(quiet_world)
    s = w.scene
    assert s.get(inside="bin").name == "in_bin"
    assert s.count(inside="bin") == 1
    assert s.get(on="floor_cube").name == "stacked"
    assert {h.name for h in s.all(touching="floor")} >= {"floor_cube"} and "stacked" not in {h.name for h in s.all(touching="floor")}
    assert [h.name for h in s.all(kind="box", near=((0.22, -0.06, 0.0), 0.05))] == ["floor_cube", "stacked"]
    assert s.get(color="blue", kind="box").name == "stacked"
    assert s.get(kind="box", where=lambda h: h.top > 0.045).name == "stacked"  # anything else, as a function


def test_filters_are_checked_now_not_when_the_scene_was_built(quiet_world):
    w = _stacked(quiet_world)
    assert w.scene.count(inside="bin") == 1
    w.move_object("in_bin", (0.3, -0.12, 0.0125))
    w.wait(0.3)
    assert w.scene.count(inside="bin") == 0


def test_held_is_the_object_between_the_fingers(quiet_world):
    w = quiet_world(scene=tabletop(ObjectSpec("cube", pos=(0.22, -0.06, None)), ObjectSpec("other", pos=(0.25, 0.06, None), color="green")))
    assert w.scene.count(held=True) == 0
    w.robot.pick(w.scene["cube"])
    assert w.scene.get(held=True).name == "cube"
    assert w.scene.get(held=False).name == "other"


def test_a_misspelt_filter_is_an_error_not_a_match_for_everything(quiet_world):
    w = quiet_world(scene=tabletop(ObjectSpec("cube")))
    with pytest.raises(TypeError, match="unknown scene filter.*'colour'"):
        w.scene.all(colour="red")
