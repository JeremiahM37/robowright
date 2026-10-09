"""The run's HTML report, traces as text, and ``robowright init``."""

import subprocess
import sys

TEST = """
from robowright import expect

def test_pass(robot, scene):
    expect(scene["cube"]).to_be_at_rest()

def test_wrong_spot(robot, scene):
    robot.pick(scene["cube"])
    robot.place(on=(0.25, -0.1, 0.0))
    expect(scene["cube"]).to_be_inside(scene["bin"], timeout=0.5)
"""


def test_a_failure_reads_as_text_and_in_the_report(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-trace-text", "--rw-report", "out/report.html")
    r.assert_outcomes(passed=1, failed=1)
    out = r.stdout.str()
    # The trace as text, at the end of the run: what ran, what failed and the world then.
    assert "test_robots.py::test_wrong_spot[mujoco]: FAILED" in out
    assert "robot.place(on=[0.25, -0.1, 0])" in out
    assert "expect(cube).to_be_inside(container=bin" in out
    assert "at the failure" in out and "cube: at (0.2" in out
    page = (pytester.path / "out" / "report.html").read_text()
    assert "failed 1" in page and "passed 1" in page
    assert "test_wrong_spot[mujoco]" in page and "at the failure" in page
    viewer = pytester.path / "out" / "report-traces" / "test_robots.py__test_wrong_spot[mujoco].html"
    assert viewer.exists() and 'href="report-traces/' in page


def test_init_sets_up_a_project_whose_tests_pass(tmp_path):
    cli = [sys.executable, "-m", "robowright.cli", "init", str(tmp_path)]
    r = subprocess.run(cli, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in ("tests/test_robot.py", "robowright.toml", "pytest.ini", ".github/workflows/robowright.yml", ".mcp.json"):
        assert (tmp_path / f).exists(), f
    assert "robowright-traces/" in (tmp_path / ".gitignore").read_text()
    (tmp_path / "pytest.ini").write_text((tmp_path / "pytest.ini").read_text() + "# mine\n")
    again = subprocess.run(cli, capture_output=True, text=True)
    assert "kept     " in again.stdout and (tmp_path / "pytest.ini").read_text().endswith("# mine\n")  # never overwrites
    pytest = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    run = subprocess.run(pytest, cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-2000:]
    assert "3 passed" in run.stdout  # the controller, its 20 randomized trials, the reference controller


def test_headed_without_a_display_says_so(monkeypatch):
    """MuJoCo's viewer ends the process when it cannot open a window; robowright checks first."""
    import pytest

    import robowright as rw
    from robowright.errors import CapabilityError

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    if not sys.platform.startswith("linux"):
        pytest.skip("the display check is for Linux")
    with pytest.raises(CapabilityError, match="headed runs need a display"):
        rw.launch(settings=rw.Settings(trace="off", headed=True))
