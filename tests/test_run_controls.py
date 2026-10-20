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
