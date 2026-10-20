"""Playwright's test.only with forbidOnly, globalTimeout, and expect.configure."""

import pytest

import robowright as rw
from robowright import expect
from robowright.errors import ExpectationError

FOCUS = """
import pytest

def test_a():
    pass

@pytest.mark.only
def test_b():
    pass

def test_c():
    pass
"""


def test_only_runs_the_focused_tests(pytester, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    pytester.makepyfile(test_f=FOCUS)
    r = pytester.runpytest("-p", "no:cacheprovider", "-v")
    r.assert_outcomes(passed=1)
    assert "test_b PASSED" in r.stdout.str()


@pytest.mark.parametrize("how", ["flag", "ci"])
def test_forbid_only_refuses_a_focus_left_in(pytester, monkeypatch, how):
    pytester.makepyfile(test_f=FOCUS)
    if how == "ci":
        monkeypatch.setenv("CI", "true")
        r = pytester.runpytest("-p", "no:cacheprovider")
    else:
        monkeypatch.delenv("CI", raising=False)
        r = pytester.runpytest("-p", "no:cacheprovider", "--rw-forbid-only")
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    assert "1 test(s) marked only" in r.stderr.str() and "test_f.py::test_b" in r.stderr.str()


def test_no_forbid_only_on_ci_lets_a_focus_run(pytester, monkeypatch):
    monkeypatch.setenv("CI", "true")
    pytester.makepyfile(test_f=FOCUS)
    pytester.runpytest("-p", "no:cacheprovider", "--no-rw-forbid-only").assert_outcomes(passed=1)


def test_global_timeout_stops_the_run_and_fails_it(pytester):
    pytester.makepyfile(
        test_slow="""
import time

def test_1():
    time.sleep(0.2)

def test_2():
    time.sleep(0.6)

def test_3():
    pass
"""
    )
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-global-timeout", "0.5")
    assert r.ret != 0
    assert "--rw-global-timeout: the run passed 0.5 s" in r.stdout.str()
    assert {k: v for k, v in r.parseoutcomes().items() if k in ("passed", "failed")} == {"passed": 1, "failed": 1}
    assert "ran past the run's --rw-global-timeout" in r.stdout.str()  # the second, stopped; the third never started


def test_global_timeout_stops_a_test_that_is_still_running(pytester):
    pytester.makepyfile(test_hang="import time\n\ndef test_hangs():\n    time.sleep(30)\n\ndef test_after():\n    pass\n")
    t0 = __import__("time").monotonic()
    r = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--rw-global-timeout", "1")
    assert __import__("time").monotonic() - t0 < 20, "the hanging test was stopped at the run's limit"
    assert r.ret != 0
    assert "TimeLimitError" in r.stdout.str() and "--rw-global-timeout: the run passed 1 s" in r.stdout.str()


def test_expect_configure_sets_defaults():
    w = rw.launch()
    slow = expect.configure(timeout=0.3)
    with pytest.raises(ExpectationError, match=r"failed after 0.30s"):
        slow(w.scene["cube"]).to_be_inside(w.scene["bin"])
    soft = expect.configure(soft=True, message="tidy-up check")
    soft(w.scene["cube"]).to_be_inside(w.scene["bin"], timeout=0.05)  # recorded, not raised
    assert len(w._soft_failures) == 1 and "tidy-up check" in w._soft_failures[0]
    w._soft_failures.clear()
    assert slow.configure(soft=True)(w.scene["cube"]).soft
    w.close()


SERIAL = """
import pytest

pytestmark = pytest.mark.serial

def test_1_connect():
    pass

def test_2_calibrate():
    assert False, "calibration failed"

def test_3_pick():
    pass

def test_4_place():
    pass
"""

OTHER = """
import pytest

@pytest.mark.fixme("the gripper slips on the new pads")
def test_known_broken():
    assert False

def test_independent():
    pass
"""


@pytest.mark.parametrize("workers", [[], ["-n", "3", "--dist", "loadgroup"]], ids=["serial-run", "xdist"])
def test_a_serial_module_skips_the_rest_once_one_fails(pytester, workers):
    if workers:
        pytest.importorskip("xdist")
    pytester.makepyfile(test_serial=SERIAL, test_other=OTHER)
    r = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rs", *workers)
    r.assert_outcomes(passed=2, failed=1, skipped=3)  # 1 passes, 2 fails, 3 and 4 skipped; other: 1 skip, 1 pass
    out = r.stdout.str()
    assert "an earlier test in test_serial.py failed" in out
    assert "fixme: the gripper slips on the new pads" in out


def test_json_results(pytester):
    import json

    pytester.makepyfile(test_other=OTHER)
    pytester.runpytest("-p", "no:cacheprovider", "--rw-json", "out/results.json")
    doc = json.loads((pytester.path / "out" / "results.json").read_text())
    assert doc["counts"] == {"skipped": 1, "passed": 1}
    assert doc["tests"]["test_other.py::test_known_broken"]["outcome"] == "skipped"
    assert "fixme" in doc["tests"]["test_other.py::test_known_broken"]["message"]


def test_serial_tests_under_load_distribution_are_warned_about(pytester):
    pytest.importorskip("xdist")
    pytester.makepyfile(test_serial=SERIAL)
    r = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-n", "2", "--dist", "load")
    assert "use --dist loadgroup" in r.stdout.str()


def _outcomes(r):
    return {k: v for k, v in r.parseoutcomes().items() if k not in ("warnings", "warning", "deselected")}


def test_rw_slow_triples_the_time_limit(pytester):
    pytester.makepyfile(
        test_s="""
import time
import pytest

@pytest.mark.rw_slow
def test_slow_but_within_three_times():
    time.sleep(1.2)  # past --rw-timeout 0.5, within 1.5

def test_plain_past_its_limit():
    time.sleep(1.2)
"""
    )
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-timeout", "0.5")
    assert _outcomes(r) == {"passed": 1, "failed": 1}
    assert "FAILED test_s.py::test_plain_past_its_limit - robowright.errors.TimeLimitErro" in r.stdout.str()


def test_rw_attempt_says_which_try_this_is(pytester):
    pytester.makepyfile(
        test_a="""
from pathlib import Path

def test_knows_its_attempt(rw_attempt):
    Path(f"attempt-{rw_attempt}").write_text("")
    assert rw_attempt == 1, "fails the first time"
"""
    )
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-retries", "1")
    assert _outcomes(r) == {"rerun": 1, "flaky": 1}
    assert (pytester.path / "attempt-0").exists() and (pytester.path / "attempt-1").exists()


USE = """
import time
import pytest

pytestmark = pytest.mark.use(expect_timeout=7.0, action_timeout=20.0, trace="off")

def test_module_options(world):
    assert (world.settings.expect_timeout, world.settings.action_timeout, world.settings.trace) == (7.0, 20.0, "off")

class TestSlowArm:
    pytestmark = pytest.mark.use(max_joint_speed=0.5, expect_timeout=9.0)

    def test_class_over_module(self, world):
        assert (world.settings.expect_timeout, world.settings.max_joint_speed, world.settings.action_timeout) == (9.0, 0.5, 20.0)

    @pytest.mark.use(expect_timeout=1.0, trace="on")
    def test_own_over_class(self, world):
        assert (world.settings.expect_timeout, world.settings.trace) == (1.0, "on")

@pytest.mark.use(timeout=0.5)
def test_its_time_limit():
    time.sleep(1.2)

@pytest.mark.use(retries=1)
def test_its_retries(rw_attempt):
    assert rw_attempt == 1
"""


def test_use_sets_options_per_module_class_and_test(pytester):
    """pytest.mark.use is Playwright's test.use: a module's options, a class's over them, a test's over both."""
    pytester.makepyfile(test_u=USE)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-trace", "on")
    assert _outcomes(r) == {"passed": 3, "failed": 1, "rerun": 1, "flaky": 1}, r.stdout.str()
    assert "FAILED test_u.py::test_its_time_limit - robowright.errors.TimeLimitErro" in r.stdout.str()
    traces = sorted(p.name for p in (pytester.path / "robowright-traces").glob("*.zip"))
    assert traces == ["test_u.py__TestSlowArm__test_own_over_class[mujoco].zip"], "use(trace=...) over --rw-trace"


def test_use_refuses_an_option_it_does_not_have(pytester):
    pytester.makepyfile(test_u="import pytest\n\npytestmark = pytest.mark.use(expect_timout=5)\n\ndef test_a():\n    pass\n")
    r = pytester.runpytest("-p", "no:cacheprovider")
    assert r.ret == pytest.ExitCode.USAGE_ERROR
    assert "pytest.mark.use has no option expect_timout; it has action_timeout" in r.stderr.str()


PROJECTS = """
[projects.nightly]
robot = ["so101", "panda"]
backend = "mujoco"
retries = 1
mark = "not quick"
use = { expect_timeout = 6.0 }

[projects.typo]
robots = "panda"
"""

PROJECT_TESTS = """
import pytest

def test_robot(world, rw_robot, rw_attempt):
    assert world.settings.expect_timeout == 6.0
    assert rw_attempt == 1, "fails the first time: the project's retries run it again"

@pytest.mark.quick
def test_quick():
    pass
"""


def test_a_project_runs_with_its_settings(pytester):
    """--rw-project is Playwright's --project: named settings in robowright.toml, under the command line's flags."""
    pytester.makeini("[pytest]\nmarkers =\n    quick: fast ones\n")
    (pytester.path / "robowright.toml").write_text(PROJECTS)
    pytester.makepyfile(test_p=PROJECT_TESTS)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-project", "nightly")
    assert _outcomes(r) == {"flaky": 2, "rerun": 2}, r.stdout.str()  # both robots, retried, quick deselected
    assert "robowright: project nightly: robots so101,panda, backends mujoco, -m 'not quick'" in r.stdout.str()
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-project", "nightly", "--rw-robot", "so101", "--rw-retries", "0")
    assert _outcomes(r) == {"failed": 1}, "the flags given win over the project's"


def test_an_unknown_project_or_key_is_a_usage_error(pytester):
    (pytester.path / "robowright.toml").write_text(PROJECTS)
    pytester.makepyfile(test_p="def test_a():\n    pass\n")
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-project", "weekly")
    assert r.ret == pytest.ExitCode.USAGE_ERROR and "no [projects.weekly]" in r.stderr.str() and "nightly, typo" in r.stderr.str()
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-project", "typo")
    assert r.ret == pytest.ExitCode.USAGE_ERROR and "unknown robots" in r.stderr.str()


MULTI = """
[projects.arms]
robot = ["so101", "panda"]
use = { expect_timeout = 6.0 }

[projects.quick]
backend = "pybullet"
mark = "quick"
seed = 5
"""

MULTI_TESTS = """
import pytest

def test_world(world, rw_project, rw_backend, rw_robot, seed):
    want = {"arms": (6.0, "mujoco", 0), "quick": (2.0, "pybullet", 5)}[rw_project]
    assert (world.settings.expect_timeout, rw_backend, seed) == want

@pytest.mark.quick
def test_plain(rw_project):
    assert rw_project in ("arms", "quick")
"""


def test_several_projects_run_every_test_in_each(pytester):
    """--rw-project arms,quick runs each test once per project with that project's settings, as Playwright runs every
    project; a project's mark keeps only its own tests."""
    pytester.makeini("[pytest]\nmarkers =\n    quick: fast ones\n")
    (pytester.path / "robowright.toml").write_text(MULTI)
    pytester.makepyfile(test_m=MULTI_TESTS)
    r = pytester.runpytest("-p", "no:cacheprovider", "-v", "--rw-project", "arms,quick")
    assert _outcomes(r) == {"passed": 4}, r.stdout.str()
    out = r.stdout.str()
    for test_id in ("test_world[arms-mujoco-so101]", "test_world[arms-mujoco-panda]", "test_plain[arms]", "test_plain[quick]"):
        assert f"{test_id} PASSED" in out
    assert "test_world[quick" not in out.replace("deselected", ""), "quick's mark keeps only its quick tests"
    assert _outcomes(pytester.runpytest("-p", "no:cacheprovider", "--rw-project", "all")) == {"passed": 4}


DEPENDS = """
[projects.setup]
match = "setup_*.py"

[projects.arms]
depends = ["setup"]
ignore = "setup_*.py"

[projects.nightly]
depends = "arms"
ignore = ["setup_*.py"]
"""


def test_a_project_runs_after_the_projects_it_depends_on(pytester):
    """Playwright's project dependencies: --rw-project nightly brings in arms and setup, runs setup's tests first,
    and once one of them fails, skips the projects that depend on it (and theirs)."""
    (pytester.path / "robowright.toml").write_text(DEPENDS)
    pytester.makeini("[pytest]\npython_files = test_*.py setup_*.py\n")
    pytester.makepyfile(
        setup_calibrate="import os\n\ndef test_calibrate():\n    assert not os.environ.get('BROKEN'), 'calibration failed'\n",
        test_arm="def test_reach():\n    pass\n",
    )
    r = pytester.runpytest("-p", "no:cacheprovider", "-v", "--rw-project", "nightly")
    out = r.stdout.str()
    order = [line.split(" ")[0] for line in r.stdout.lines if " PASSED" in line or " SKIPPED" in line or " FAILED" in line]
    assert order[0] == "setup_calibrate.py::test_calibrate[setup]", order
    assert _outcomes(r) == {"passed": 3}, out  # setup runs its file; arms and nightly leave it out
    assert "test_reach[setup]" not in out, "match keeps the setup project to its own files"
    import os

    os.environ["BROKEN"] = "1"
    try:
        r = pytester.runpytest("-p", "no:cacheprovider", "-rs", "--rw-project", "nightly")
    finally:
        del os.environ["BROKEN"]
    assert _outcomes(r) == {"failed": 1, "skipped": 2}, r.stdout.str()
    assert "project arms depends on setup, where a test failed" in r.stdout.str()
    assert "project nightly depends on arms, where a test failed" in r.stdout.str()


def test_a_dependency_cycle_is_a_usage_error(pytester):
    (pytester.path / "robowright.toml").write_text('[projects.a]\ndepends = ["b"]\n[projects.b]\ndepends = ["a"]\n')
    pytester.makepyfile(test_x="def test_x():\n    pass\n")
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-project", "a")
    assert r.ret == pytest.ExitCode.USAGE_ERROR and "depends on itself through a -> b -> a" in r.stderr.str()


LOGGED = """
import robowright as rw
from robowright import expect

def test_put_away(robot, scene):
    with rw.step("put it away"):
        robot.pick(scene["cube"])
    expect(scene["cube"]).to_be_inside(scene["bin"], timeout=0.2)
"""


def test_rw_log_prints_each_action_and_expectation_as_it_runs(pytester):
    """--rw-log is Playwright's DEBUG=pw:api: each step, action and expectation as it happens, with simulated and wall
    time and its line, past pytest's capture, with the trace off too."""
    pytester.makepyfile(test_logged=LOGGED)
    r = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--rw-log", "--rw-trace", "off")
    out = r.stdout.str()
    assert "rw:api --- test_logged.py::test_put_away[mujoco]" in out
    assert "> 'put it away'  test_logged.py:5" in out
    assert "> robot.pick(obj=cube)  test_logged.py:6" in out
    assert "< robot.pick: done in" in out and "s simulated" in out and "s wall" in out
    assert "expect expect(cube).to_be_inside: FAILED" in out and "test_logged.py:7" in out
    assert not (pytester.path / "robowright-traces").exists(), "--rw-trace off still saves nothing"


TEARDOWN = """
[projects.setup]
match = "setup_*.py"
teardown = "cleanup"

[projects.cleanup]
match = "cleanup_*.py"

[projects.arms]
depends = ["setup"]
ignore = ["setup_*.py", "cleanup_*.py"]
"""


def test_a_setup_projects_teardown_runs_last_whatever_happened(pytester):
    """Playwright's teardown: the cleanup project a setup project names joins the run and runs after everything else,
    when the tests before it failed too."""
    (pytester.path / "robowright.toml").write_text(TEARDOWN)
    pytester.makeini("[pytest]\npython_files = test_*.py setup_*.py cleanup_*.py\n")
    pytester.makepyfile(
        setup_calibrate="def test_calibrate():\n    pass\n",
        test_arm="def test_reach():\n    assert False, 'the arm missed'\n",
        cleanup_park="def test_park():\n    pass\n",
    )
    r = pytester.runpytest("-p", "no:cacheprovider", "-v", "--rw-project", "arms")
    ran = [line.split(" ")[0] for line in r.stdout.lines if " PASSED" in line or " FAILED" in line]
    assert ran == ["setup_calibrate.py::test_calibrate[setup]", "test_arm.py::test_reach[arms]", "cleanup_park.py::test_park[cleanup]"], ran
    assert _outcomes(r) == {"passed": 2, "failed": 1}
