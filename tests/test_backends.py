import os
from importlib.util import find_spec

import numpy as np
import pytest

import robowright as rw
from robowright import expect
from robowright.scene import ObjectSpec, SceneSpec, tabletop

pytest.importorskip("pybullet")
BACKENDS = ["mujoco", "pybullet", *(b for b, mod in (("drake", "pydrake"), ("genesis", "genesis")) if find_spec(mod))]


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


@pytest.mark.skipif(not find_spec("genesis"), reason="Genesis not installed")
def test_no_garbage_collection_while_genesis_renders(quiet_world, monkeypatch):
    """A MuJoCo renderer freed by the garbage collector mid-render (one a dropped scene left in a
    reference cycle) releases the thread's EGL context, Genesis's included, and the render fails
    with "no valid context": seen now and then in mixed test runs. Collection stays off while
    Genesis's context is current, and is back on after."""
    import gc

    from genesis.ext.pyrender import renderer

    w = quiet_world(backend="genesis")
    seen = []
    render = renderer.Renderer.render

    def watched(self, *args, **kwargs):
        seen.append(gc.isenabled())
        return render(self, *args, **kwargs)

    monkeypatch.setattr(renderer.Renderer, "render", watched)
    img = w.backend.render(w.spec.cameras[0].name, 64, 48)
    assert seen and not any(seen) and gc.isenabled()
    assert img.std() > 5


def test_a_collected_mujoco_renderer_releases_the_threads_gl_context():
    """Why the test above matters: freeing a MuJoCo renderer un-currents whatever EGL context
    the thread has, not only its own."""
    import mujoco

    pytest.importorskip("OpenGL.EGL")
    from OpenGL import EGL

    if os.environ.get("MUJOCO_GL") != "egl":
        pytest.skip("MuJoCo is not rendering through EGL")
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
    mine, other = mujoco.Renderer(m, 32, 32), mujoco.Renderer(m, 32, 32)
    other._gl_context.make_current()
    assert EGL.eglGetCurrentContext()
    mine.close()
    assert not EGL.eglGetCurrentContext()
    other.close()
