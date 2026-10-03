from importlib.util import find_spec

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.scene import ObjectSpec, SceneSpec, tabletop

pytest.importorskip("pybullet")
BACKENDS = ["mujoco", "pybullet", *(["genesis"] if find_spec("genesis") else [])]


@pytest.mark.parametrize("backend", BACKENDS)
def test_same_test_passes_on_every_backend(quiet_world, backend):
    w = quiet_world(backend=backend)
    cube, bin = w.scene["cube"], w.scene["bin"]
    w.robot.pick(cube)
    expect(w.robot.gripper).to_be_holding(cube)
    w.robot.place(on=bin)
    expect(cube).to_be_inside(bin)


@pytest.mark.parametrize("backend", BACKENDS)
def test_shapes_and_rendering(quiet_world, backend):
    scene = tabletop(
        ObjectSpec("box", "box", (0.01, 0.015, 0.012), (0.2, -0.05, None)),
        ObjectSpec("can", "cylinder", (0.012, 0.02), (0.25, 0.0, None), color="green"),
        ObjectSpec("ball", "sphere", (0.015,), (0.2, 0.05, None), color="yellow"),
    )
    w = quiet_world(backend=backend, scene=scene)
    w.wait(0.5)
    for name, z in (("box", 0.012), ("can", 0.02), ("ball", 0.015)):
        assert w.scene[name].position[2] == pytest.approx(z, abs=0.003)
    img = w.backend.render("front", 160, 120)
    assert img.shape == (120, 160, 3) and img.std() > 5


@pytest.mark.parametrize("backend", BACKENDS)
def test_contact_labels(quiet_world, backend):
    w = quiet_world(backend=backend)
    w.wait(0.1)
    pairs = {frozenset((c.a, c.b)) for c in w.backend.contacts()}
    assert frozenset(("cube", "floor")) in pairs


def test_backends_agree_on_kinematics(quiet_world):
    tcp = {}
    for b in BACKENDS:
        w = quiet_world(backend=b)
        w.robot.arm.move_to((0.24, 0.04, 0.05))
        tcp[b] = w.robot.tcp.position
    assert np.linalg.norm(tcp["mujoco"] - tcp["pybullet"]) < 0.004


def test_scene_spec_round_trip():
    s = tabletop()
    assert SceneSpec.from_dict(s.to_dict()) == s


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        rw.launch(backend="gazebo")
