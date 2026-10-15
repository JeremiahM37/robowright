"""--rw-timeout, Playwright's test timeout: a test that runs too long fails, saying what was running,
whether it hangs in its body or in a fixture (where a stack that never comes up waits)."""

TEST = """
import time
import pytest
import robowright as rw

def test_quick(world):
    world.step(5)

def test_hangs(world):
    with rw.step("wait for a stack that never answers"):
        while True:
            world.step(1)

@pytest.fixture
def stack():
    time.sleep(30)  # a launch that never comes up


@pytest.fixture
def closes_for_ever():
    yield
    time.sleep(30)  # a teardown that never finishes (an executor waiting for a thread)


def test_hangs_in_teardown(closes_for_ever):
    pass

def test_hangs_in_setup(stack):
    pass

@pytest.mark.rw_timeout(0)
def test_no_limit_here(world):
    time.sleep(0.6)

@pytest.mark.trials(3)
def test_trials_stop(world):
    while True:
        world.step(1)
"""


def test_a_test_past_its_time_limit_fails_and_says_what_was_running(pytester):
    pytester.makepyfile(test_t=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-timeout", "0.5", "-k", "quick or hangs and not setup and not teardown")
    r.assert_outcomes(passed=1, failed=1)
    out = r.stdout.str()
    assert "TimeLimitError: the test ran past its 0.5 s time limit (wall clock), while step 'wait for a stack that never answers'" in out
    assert "test_t.py__test_hangs[mujoco].zip" in out  # the trace of the run that ran out of time is kept


def test_a_fixture_that_hangs_is_stopped_too(pytester):
    pytester.makepyfile(test_t=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-timeout", "0.5", "-k", "setup")
    r.assert_outcomes(errors=1)
    assert r.duration < 20


def test_the_marker_sets_one_tests_limit(pytester):
    pytester.makepyfile(test_t=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-timeout", "0.3", "-k", "no_limit")
    r.assert_outcomes(passed=1)


def test_a_trials_test_out_of_time_stops_rather_than_counting_a_failed_trial(pytester):
    pytester.makepyfile(test_t=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-timeout", "0.5", "-k", "trials")
    r.assert_outcomes(failed=1)
    assert "TimeLimitError" in r.stdout.str()
    assert r.duration < 20


def test_a_teardown_that_hangs_is_stopped_too(pytester):
    pytester.makepyfile(test_t=TEST)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-timeout", "0.5", "-k", "teardown")
    r.assert_outcomes(passed=1, errors=1)  # the test passed; its teardown ran out of time
    assert "TimeLimitError" in r.stdout.str()
    assert r.duration < 20
