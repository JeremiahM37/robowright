import pytest

TEST = """
import pytest
from robowright import expect

def test_pass(robot, scene):
    expect(scene["cube"]).to_be_at_rest()

def test_fail(robot, scene):
    expect(scene["cube"]).to_be_inside(scene["bin"], timeout=0.2)

@pytest.mark.trials(6, min_success=0.5)
def test_flaky(world, seed):
    assert seed % 2 == 0

@pytest.mark.trials(4, min_success=1.0)
def test_too_flaky(world, seed):
    assert seed != 2

def test_soft(scene):
    expect.soft(scene["cube"]).to_be_above(1.0, timeout=0)
"""


def test_plugin_end_to_end(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider")
    r.assert_outcomes(passed=3, failed=2, errors=1)  # test_soft passes its call, then errors at teardown
    out = r.stdout.str()
    assert "robowright trials" in out
    assert "PASS test_robots.py::test_flaky[mujoco]: 3/6 passed" in out
    assert "seed 2: AssertionError" in out
    assert "robowright show-trace robowright-traces/test_robots.py__test_fail[mujoco].zip" in out
    traces = sorted(p.name for p in (pytester.path / "robowright-traces").iterdir())
    assert "test_robots.py__test_fail[mujoco].zip" in traces
    assert not any("test_pass" in t for t in traces)  # retain-on-failure


def test_backend_parametrization(pytester):
    pytest.importorskip("pybullet")
    pytester.makepyfile(
        test_b="""
import pytest
def test_any(world):
    assert world.backend.name in ("mujoco", "pybullet")

@pytest.mark.backends("mujoco")
def test_only_mujoco(world):
    assert world.backend.name == "mujoco"
"""
    )
    r = pytester.runpytest("--rw-backend", "mujoco,pybullet", "-v")
    r.assert_outcomes(passed=3)
    r.stdout.fnmatch_lines(["*test_any?mujoco?*PASSED*", "*test_any?pybullet?*PASSED*"])


def test_scene_marker_and_trace_on(pytester):
    pytester.makepyfile(
        test_s="""
import pytest
from robowright import ObjectSpec, tabletop

@pytest.mark.scene(lambda: tabletop(ObjectSpec("ball", "sphere", (0.015,), (0.2, 0.0, None), color="green")))
@pytest.mark.seed(9)
def test_ball(world, scene):
    assert world.seed == 9
    assert scene.get(color="green").name == "ball"
"""
    )
    r = pytester.runpytest("--rw-trace", "on")
    r.assert_outcomes(passed=1)
    assert (pytester.path / "robowright-traces" / "test_s.py__test_ball[mujoco].zip").exists()
