"""The run's HTML report, traces as text, and ``robowright init``."""

import json
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


def test_init_mcp_entry_starts_the_server_with_the_interpreter_that_ran_init(tmp_path):
    cli = [sys.executable, "-m", "robowright.cli", "init", str(tmp_path)]
    assert subprocess.run(cli, capture_output=True, text=True).returncode == 0
    entry = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["robowright"]
    assert entry == {"command": sys.executable, "args": ["-m", "robowright.cli", "mcp"]}  # no venv needed on PATH


def test_test_help_says_what_robowright_test_is_before_pytests_help():
    r = subprocess.run([sys.executable, "-m", "robowright.cli", "test", "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    head, _, rest = r.stdout.partition("\n\n")
    assert head.startswith("usage: robowright test") and "pytest" in rest.split("\n", 1)[0]
    assert "positional arguments" in r.stdout  # pytest's own help follows


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


def test_init_ros2_writes_a_project_for_a_stack(tmp_path):
    """robowright init --ros2 "<launch command>": the settings name the stack, pytest runs against ROS 2, and the example
    test is one any ros2_control robot can run (run against UR's Gazebo simulation in tests/test_stack_pytest.py)."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: robowright depends on tomli there
        import tomli as tomllib

    cmd = 'ros2 launch my_robot sim.launch.py world:="a b"'
    r = subprocess.run(
        [sys.executable, "-m", "robowright.cli", "init", str(tmp_path), "--ros2", cmd, "--gazebo"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    cfg = tomllib.loads((tmp_path / "robowright.toml").read_text())
    assert cfg["ros2"]["stack"]["command"] == cmd  # quoted as TOML, quotes and all
    assert cfg["ros2"]["use_sim_time"] is True and cfg["ros2"]["gazebo"] is True
    assert "--rw-backend ros2" in (tmp_path / "pytest.ini").read_text()
    assert "ros2_control" not in (tmp_path / ".mcp.json").read_text()
    compile((tmp_path / "tests" / "test_robot.py").read_text(), "test_robot.py", "exec")
    assert not (tmp_path / ".github").exists()  # a CI runner has no ROS 2 stack: no workflow is written for one


def test_a_sharded_run_merges_into_one_report(pytester):
    """Two shards, each run as on its own CI machine (its own trace directory, deleted once its blob is written: the
    blob must carry everything), merged into one report with every test and the failure's trace."""
    import shutil

    pytester.makepyfile(test_robots=TEST)
    for shard in ("1/2", "2/2"):
        n = shard[0]
        r = pytester.runpytest(
            "-p", "no:cacheprovider", "--rw-shard", shard, "--rw-trace-dir", f"machine{n}/traces", "--rw-blob", f"blobs/shard-{n}.zip"
        )
        assert r.ret in (0, 1), r.stdout.str()
        shutil.rmtree(pytester.path / f"machine{n}", ignore_errors=True)  # (a shard that only passed made none)
    blobs = sorted(str(p) for p in (pytester.path / "blobs").glob("*.zip"))
    assert len(blobs) == 2
    cli = [sys.executable, "-m", "robowright.cli", "merge-reports", *blobs, "-o", str(pytester.path / "merged" / "report.html")]
    r = subprocess.run(cli, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    page = (pytester.path / "merged" / "report.html").read_text()
    assert "2 tests" in page and "failed 1" in page and "passed 1" in page, page[:2000]
    assert "test_wrong_spot[mujoco]" in page and "at the failure" in page  # the trace, read from the blob
    viewer = pytester.path / "merged" / "report-traces" / "test_robots.py__test_wrong_spot[mujoco].html"
    assert viewer.exists() and 'href="report-traces/' in page


def test_merge_reports_refuses_what_is_not_a_blob(tmp_path):
    bad = tmp_path / "not-a-blob.zip"
    bad.write_bytes(b"not a zip")
    cli = [sys.executable, "-m", "robowright.cli", "merge-reports", str(bad), "-o", str(tmp_path / "r.html")]
    r = subprocess.run(cli, capture_output=True, text=True)
    assert r.returncode == 2 and "robowright merge-reports:" in r.stderr
