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
    time.sleep(0.6)

def test_2():
    time.sleep(0.6)

def test_3():
    pass
"""
    )
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-global-timeout", "0.5")
    assert r.ret != 0
    assert "--rw-global-timeout: the run passed 0.5 s" in r.stdout.str()
    assert r.parseoutcomes().get("passed") == 1  # the first finished; the rest never started


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
