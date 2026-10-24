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
    # What the camera saw when it failed, redrawn from the trace: the cube on the table, beside the bin.
    seen = pytester.path / "out" / "report-traces" / "test_robots.py__test_wrong_spot[mujoco].png"
    assert seen.exists() and 'src="report-traces/test_robots.py__test_wrong_spot[mujoco].png"' in page
    import numpy as np
    from PIL import Image

    img = np.asarray(Image.open(seen).convert("RGB")).astype(int)
    assert ((img[..., 0] > 150) & (img[..., 1] < 80) & (img[..., 2] < 80)).sum() > 20  # the red cube is in it
    assert "test_pass[mujoco].png" not in page  # a test that passed has none


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
    assert (pytester.path / "merged" / "report-traces" / "test_robots.py__test_wrong_spot[mujoco].png").exists()  # its picture too


def test_merge_reports_refuses_what_is_not_a_blob(tmp_path):
    bad = tmp_path / "not-a-blob.zip"
    bad.write_bytes(b"not a zip")
    cli = [sys.executable, "-m", "robowright.cli", "merge-reports", str(bad), "-o", str(tmp_path / "r.html")]
    r = subprocess.run(cli, capture_output=True, text=True)
    assert r.returncode == 2 and "robowright merge-reports:" in r.stderr


def test_the_ci_workflow_init_writes_runs_its_shards_and_merges_them(tmp_path):
    """The commands the generated workflow runs, as it runs them: each shard's pytest with its blob, then the report
    job's merge-reports, in the project init made."""
    import re

    assert subprocess.run([sys.executable, "-m", "robowright.cli", "init", str(tmp_path)], capture_output=True).returncode == 0
    wf = (tmp_path / ".github" / "workflows" / "robowright.yml").read_text()
    try:
        import yaml

        jobs = yaml.safe_load(wf)["jobs"]
        assert jobs["report"]["needs"] == "test" and jobs["test"]["strategy"]["matrix"]["shard"] == [1, 2]
    except ImportError:
        pass
    runs = re.findall(r"- run: (.+)", wf)
    shard_cmd = next(r for r in runs if r.startswith("pytest "))
    merge_cmd = next(r for r in runs if r.startswith("robowright merge-reports"))
    for n in (1, 2):
        args = shard_cmd.replace("${{ matrix.shard }}", str(n)).split()[1:]
        args = [a for a in args if a not in ("-n", "auto")]  # (xdist is optional here)
        r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args], cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:]  # a shard dealt no tests passes too (here, one of the two is)
        assert (tmp_path / f"blob-{n}.zip").exists()
    blobs = sorted(str(p) for p in tmp_path.glob("blob-*.zip"))
    out = merge_cmd.split(" -o ")[1]
    cli = [sys.executable, "-m", "robowright.cli", "merge-reports", *blobs, "-o", out]
    r = subprocess.run(cli, cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    page = (tmp_path / out).read_text()
    assert "3 tests" in page and "passed 3" in page, page[:1500]  # every test of the example project, across both shards


def test_init_agents_writes_a_planner_a_generator_and_a_healer(tmp_path):
    """As `npx playwright init-agents`: Claude Code agent definitions that use robowright's MCP tools."""
    import re
    from pathlib import Path

    cli = [sys.executable, "-m", "robowright.cli", "init-agents", str(tmp_path)]
    r = subprocess.run(cli, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    tools = set(re.findall(r"async def ((?:robot|ros2|sim)_\w+)", (Path(__file__).parents[1] / "src/robowright/mcp_server.py").read_text()))
    for name in ("robowright-planner", "robowright-generator", "robowright-healer"):
        text = (tmp_path / ".claude" / "agents" / f"{name}.md").read_text()
        head, body = text.split("\n---\n", 1)
        assert head.startswith("---\n") and f"name: {name}\n" in head and "\ndescription: " in head
        named = set(re.findall(r"\b((?:robot|ros2|sim)_[a-z_]+)\b", body))
        assert named and named <= tools, named - tools  # every tool it is told to use exists
    assert json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["robowright"]
    healer = tmp_path / ".claude" / "agents" / "robowright-healer.md"
    healer.write_text("mine\n")
    assert "kept     " in subprocess.run(cli, capture_output=True, text=True).stdout and healer.read_text() == "mine\n"


def test_the_ci_workflow_installs_robowright_the_way_it_is_installed_here(monkeypatch):
    """From PyPI at this version when it came from PyPI; from its repository otherwise (a CI machine cannot reach a
    local checkout, and an unpublished name would 404)."""
    from importlib import metadata

    from robowright import cli

    class Dist:
        version = "0.2.0"

        def __init__(self, direct):
            self.direct = direct

        def read_text(self, name):
            return self.direct if name == "direct_url.json" else None

    monkeypatch.setattr(metadata, "distribution", lambda name: Dist(None))
    assert 'pip install "robowright==0.2.0" pytest-xdist' in cli._init_ci()
    monkeypatch.setattr(metadata, "distribution", lambda name: Dist('{"url": "file:///src/robowright"}'))
    assert 'pip install "robowright @ git+https://github.com/JeremiahM37/robowright" pytest-xdist' in cli._init_ci()
    assert "__ROBOWRIGHT__" not in cli._init_ci()


TAGGED = """
import pytest

@pytest.mark.needs_gpu
@pytest.mark.annotate("issue", "https://github.com/me/robot/issues/12")
def test_grasp(rw_annotate):
    rw_annotate("found", "the left finger slips")

@pytest.mark.smoke
@pytest.mark.use(expect_timeout=3)
def test_home():
    pass

def test_plain():
    pass
"""


def test_tags_and_annotations_in_the_report(pytester):
    """A test's own markers are its tags (Playwright's @tag), and annotate() and rw_annotate notes show with it."""
    pytester.makeini("[pytest]\nmarkers =\n    needs_gpu: needs a GPU\n    smoke: quick ones\n")
    pytester.makepyfile(test_tagged=TAGGED)
    for workers in ((), ("-n", "2")):  # under xdist the notes come from the workers' reports
        r = pytester.runpytest("-p", "no:cacheprovider", *workers, "--rw-report", "r.html", "--rw-json", "r.json")
        r.assert_outcomes(passed=3)
        tests = json.loads((pytester.path / "r.json").read_text())["tests"]
        assert tests["test_tagged.py::test_grasp"]["tags"] == ["needs_gpu"]
        assert tests["test_tagged.py::test_home"]["tags"] == ["smoke"], "settings markers (use) are not tags"
        assert tests["test_tagged.py::test_plain"]["tags"] == []
        assert tests["test_tagged.py::test_grasp"]["annotations"] == [
            {"type": "issue", "description": "https://github.com/me/robot/issues/12"},
            {"type": "found", "description": "the left finger slips"},
        ]
        page = (pytester.path / "r.html").read_text()
        assert '<button data-t="needs_gpu">@needs_gpu 1</button>' in page and '<span class="tag">@smoke</span>' in page
        assert '<a href="https://github.com/me/robot/issues/12">' in page and "the left finger slips" in page


def test_junit_carries_each_tests_robot_engine_seed_and_trace(pytester):
    """pytest --junitxml: each testcase's properties say its robot, engine and seed, and a failure's trace, so a CI
    dashboard can split results by robot and link the evidence."""
    import xml.etree.ElementTree as ET

    pytester.makepyfile(test_robots=TEST)
    pytester.runpytest("-p", "no:cacheprovider", "--rw-robot", "so101,panda", "--junitxml", "j.xml")
    cases = {
        c.get("name"): {p.get("name"): p.get("value") for p in c.iter("property")}
        for c in ET.parse(pytester.path / "j.xml").iter("testcase")
    }
    assert cases["test_pass[mujoco-panda]"] == {"robowright.backend": "mujoco", "robowright.robot": "panda", "robowright.seed": "0"}
    bad = cases["test_wrong_spot[mujoco-so101]"]
    assert bad["robowright.robot"] == "so101" and bad["robowright.trace"].endswith("test_robots.py__test_wrong_spot[mujoco-so101].zip")
