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
