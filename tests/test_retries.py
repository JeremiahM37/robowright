"""--rw-retries: a failed robot test runs again in a fresh world, and one that then passes is reported flaky,
with the failed attempt's trace kept (Playwright's retries and trace: 'on-first-retry')."""

import pytest

TEST = """
from pathlib import Path
from robowright import expect

def _attempt(name):
    f = Path(name)
    n = int(f.read_text()) if f.exists() else 0
    f.write_text(str(n + 1))
    return n

def test_fails_once(world, scene):
    first = _attempt("fails_once.count") == 0
    expect(scene["cube"]).to_be_at_rest()
    assert not first, "the first attempt fails"

def test_always_fails(world, scene):
    _attempt("always_fails.count")
    expect(scene["cube"]).to_be_inside(scene["bin"], timeout=0.1)

def test_passes(world, scene):
    expect(scene["cube"]).to_be_at_rest()
"""


def _assert_outcomes(r, **want):
    """pytest's assert_outcomes knows only its own categories, not rerun and flaky."""
    got = r.parseoutcomes()
    got = {k: v for k, v in got.items() if k not in ("warnings", "warning", "deselected")}
    assert got == want, r.stdout.str()[-3000:]


def _traces(pytester):
    d = pytester.path / "robowright-traces"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def test_a_test_that_passes_on_retry_is_flaky_not_passed(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-retries", "2", "-k", "fails_once or passes", "--rw-report", "report.html")
    _assert_outcomes(r, passed=1, rerun=1, flaky=1)
    assert r.ret == 0  # flaky is not a failure
    assert (pytester.path / "fails_once.count").read_text() == "2"  # ran twice, and stopped once it passed
    traces = _traces(pytester)
    assert "test_robots.py__test_fails_once[mujoco].zip" in traces  # the failed attempt's trace survives its retry
    assert not any("test_passes" in t for t in traces)
    page = (pytester.path / "report.html").read_text()
    assert '<tr class="flaky">' in page and "the first attempt fails" in page


def test_a_test_that_always_fails_fails_after_its_retries(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-retries", "2", "-k", "always_fails")
    _assert_outcomes(r, failed=1, rerun=2)
    assert (pytester.path / "always_fails.count").read_text() == "3"
    traces = _traces(pytester)
    for name in ("", "-retry1", "-retry2"):  # each attempt's own trace
        assert f"test_robots.py__test_always_fails[mujoco]{name}.zip" in traces


def test_on_first_retry_records_only_the_first_retry(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-retries", "2", "--rw-trace", "on-first-retry")
    _assert_outcomes(r, passed=1, failed=1, rerun=3, flaky=1)
    assert _traces(pytester) == [
        "test_robots.py__test_always_fails[mujoco]-retry1.zip",
        "test_robots.py__test_fails_once[mujoco]-retry1.zip",  # recorded on its retry, which passed
    ]
    assert "robowright show-trace robowright-traces/test_robots.py__test_fails_once[mujoco]-retry1.zip" in r.stdout.str()


def test_without_retries_nothing_changes(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider")
    _assert_outcomes(r, passed=1, failed=2)
    assert "rerun" not in r.stdout.str().split("short test summary")[-1]


def test_retries_under_xdist(pytester):
    pytest.importorskip("xdist")
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "-n", "2", "--rw-retries", "1", "-k", "fails_once or passes")
    _assert_outcomes(r, passed=1, rerun=1, flaky=1)
