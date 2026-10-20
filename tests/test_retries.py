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
    assert '<tr class="flaky" data-tags="">' in page and "the first attempt fails" in page


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


def test_on_all_retries_records_every_retry(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-retries", "2", "--rw-trace", "on-all-retries")
    _assert_outcomes(r, passed=1, failed=1, rerun=3, flaky=1)
    assert _traces(pytester) == [
        "test_robots.py__test_always_fails[mujoco]-retry1.zip",
        "test_robots.py__test_always_fails[mujoco]-retry2.zip",
        "test_robots.py__test_fails_once[mujoco]-retry1.zip",
    ]


def test_retain_on_first_failure_keeps_only_the_first_attempts_trace(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-retries", "2", "--rw-trace", "retain-on-first-failure")
    _assert_outcomes(r, passed=1, failed=1, rerun=3, flaky=1)
    assert _traces(pytester) == [
        "test_robots.py__test_always_fails[mujoco].zip",
        "test_robots.py__test_fails_once[mujoco].zip",
    ]


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


def test_a_marker_gives_one_test_its_own_retries(pytester):
    pytester.makepyfile(
        test_marked="""
import pytest
from pathlib import Path

@pytest.mark.rw_retries(1)
def test_marked(world):
    f = Path("marked.count")
    n = int(f.read_text()) if f.exists() else 0
    f.write_text(str(n + 1))
    assert n > 0, "fails the first time"

def test_unmarked_fails_once_and_for_all(world):
    assert False
"""
    )
    r = pytester.runpytest("-p", "no:cacheprovider")
    _assert_outcomes(r, failed=1, rerun=1, flaky=1)


def test_fail_on_flaky_fails_a_run_with_a_flaky_test(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-retries", "1", "--rw-fail-on-flaky", "-k", "fails_once or passes")
    _assert_outcomes(r, passed=1, rerun=1, flaky=1)
    assert r.ret == 1
    assert "FLAKY test_robots.py::test_fails_once[mujoco]" in r.stdout.str()


def test_fail_on_flaky_leaves_a_clean_run_passing(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-retries", "1", "--rw-fail-on-flaky", "-k", "passes")
    _assert_outcomes(r, passed=1)
    assert r.ret == 0


REPEAT = """
from pathlib import Path
from robowright import expect

def test_counts(world, scene, rw_repeat):
    f = Path("runs.txt")
    f.write_text((f.read_text() if f.exists() else "") + f"{rw_repeat} {world.seed}\\n")
    expect(scene["cube"]).to_be_at_rest()
    assert rw_repeat != 2, "only the second run fails"
"""


def test_repeat_each_runs_every_test_n_times_with_the_same_seed(pytester):
    pytester.makepyfile(test_robots=REPEAT)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-repeat-each", "3")
    _assert_outcomes(r, passed=2, failed=1)
    assert "FAILED test_robots.py::test_counts[mujoco-repeat2]" in r.stdout.str()
    runs = (pytester.path / "runs.txt").read_text().split()
    assert runs[0::2] == ["1", "2", "3"] and len(set(runs[1::2])) == 1  # every repeat has the same seed


def test_a_flaky_tests_failed_attempt_is_said_even_under_q(pytester):
    pytester.makepyfile(test_robots=TEST)
    r = pytester.runpytest("-q", "-p", "no:cacheprovider", "--rw-retries", "1", "-k", "fails_once")
    assert "FLAKY test_robots.py::test_fails_once[mujoco]: its failed attempt: AssertionError: the first attempt fails" in r.stdout.str()
