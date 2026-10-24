"""expect(obj).to_be_visible / to_be_hidden, Playwright's toBeVisible and toBeHidden: whether an object shows in a
camera's image, in the frame and not behind something else."""

import pytest

import robowright as rw
from robowright import expect
from robowright.errors import CapabilityError, ExpectationError
from robowright.scene import ObjectSpec, tabletop

# A tall box on the table, between the front camera (0.6, -0.42, 0.42) and the cube (0.22, -0.06): its top is above
# the line of sight there (~0.21 m).
CUBE = ObjectSpec("cube", "box", (0.0125, 0.0125, 0.0125), (0.22, -0.06, None), color="red")
WALL = ObjectSpec("wall", "box", (0.06, 0.06, 0.15), (0.40, -0.23, None), color="gray", mass=2.0)


def test_the_cube_on_the_table_is_visible():
    with rw.launch() as w:
        expect(w.scene["cube"]).to_be_visible()
        with pytest.raises(ExpectationError, match="pixels of cube show in the front camera"):
            expect(w.scene["cube"]).to_be_hidden()


def test_out_of_the_frame_is_hidden():
    with rw.launch() as w:
        w.move_object("cube", (1.5, -1.5, 0.0125))  # behind the camera
        expect(w.scene["cube"]).to_be_hidden()
        expect(w.scene["cube"]).not_.to_be_visible()


def test_behind_something_is_hidden_though_in_the_frame():
    with rw.launch(tabletop(CUBE, WALL)) as w:
        expect(w.scene["cube"]).to_be_hidden()
        expect(w.scene["wall"]).to_be_visible()
        w.move_object("wall", (0.40, 0.30, 0.15))  # out of the way
        expect(w.scene["cube"]).to_be_visible(timeout=1.0)


def test_an_unknown_camera_is_named():
    with rw.launch() as w, pytest.raises(ValueError, match="unknown camera 'top'"):
        expect(w.scene["cube"]).to_be_visible(camera="top")


def test_an_engine_that_cannot_tell_says_so():
    pytest.importorskip("pybullet")
    with rw.launch(backend="pybullet") as w, pytest.raises(CapabilityError, match="to_be_visible runs on MuJoCo"):
        expect(w.scene["cube"]).to_be_visible()
