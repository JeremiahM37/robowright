"""--rw-timeout, Playwright's test timeout: a test that runs too long fails, saying what was running,
whether it hangs in its body or in a fixture (where a stack that never comes up waits)."""

TEST = """
import time
import pytest
import robowright as rw

def test_quick(world):
    world.step(5)

def _for_a_long_while():  # 30 s, not for ever: if the limit never fires, this fails instead of eating memory
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30:
        yield
    raise AssertionError("the time limit never fired")


def test_hangs(world):
    with rw.step("wait for a stack that never answers"):
        for _ in _for_a_long_while():
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
    for _ in _for_a_long_while():
        world.step(1)
"""


def test_a_test_past_its_time_limit_fails_and_says_what_was_running(pytester):
    pytester.makepyfile(test_t=TEST)
    # 5 s, not less: the limit covers the quick test's setup too, and its first world took over 0.5 s under the full
    # verify's parallel load. The hanging test hangs for ever, so any limit shows it stopped.
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-timeout", "5", "-k", "quick or hangs and not setup and not teardown")
    r.assert_outcomes(passed=1, failed=1)
    out = r.stdout.str()
    assert "TimeLimitError: the test ran past its 5 s time limit (wall clock), while step 'wait for a stack that never answers'" in out
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


HUNG = """
import signal, threading, time

def test_stuck_where_python_cannot_see_the_limit():
    # SIGALRM blocked: the Python-level limit can never run, as in a main thread stuck inside a C call
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    threading.Thread(target=time.sleep, args=(60,), name="busy-helper", daemon=True).start()
    time.sleep(60)
"""


def test_a_test_stuck_in_a_c_call_ends_the_run_and_leaves_every_threads_stack(pytester, monkeypatch):
    monkeypatch.setenv("ROBOWRIGHT_HANG_GRACE", "2")
    pytester.makepyfile(test_hung=HUNG)
    t0 = __import__("time").monotonic()
    r = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--rw-timeout", "1", "--rw-trace-dir", "traces")
    assert __import__("time").monotonic() - t0 < 30  # its limit and the grace, not the test's own 60 s
    assert r.ret != 0
    (hung,) = (pytester.path / "traces").glob("hung-*.txt")
    text = hung.read_text()
    assert "test_stuck_where_python_cannot_see_the_limit" in text and text.count("Thread 0x") >= 2, text


SLOW = """
import threading, time

def test_slow_with_a_helper():
    threading.Thread(target=time.sleep, args=(30,), name="helper-thread", daemon=True).start()
    end = time.monotonic() + 30
    while time.monotonic() < end:
        time.sleep(0.01)
"""


def test_a_time_limit_says_where_the_other_threads_were(pytester):
    pytester.makepyfile(test_slow=SLOW)
    r = pytester.runpytest("-p", "no:cacheprovider", "--rw-timeout", "1", "--rw-trace-dir", "traces")
    r.assert_outcomes(failed=1)
    out = r.stdout.str()
    assert "time limit" in out and "other threads:" in out and "thread helper-thread" in out, out[-3000:]
    assert not list((pytester.path / "traces").glob("hung-*.txt"))  # stopped by the limit: no hang file left


def test_the_time_limit_hooks_wrap_each_test():
    """They start and stop the clock around setup and teardown; a hook that lost its wrapper marker never runs its
    code (pytest calls it, gets an unstarted generator back), and every limit silently stops working."""
    from robowright import pytest_plugin

    for name in ("pytest_runtest_setup", "pytest_runtest_teardown"):
        assert getattr(getattr(pytest_plugin, name), "pytest_impl", {}).get("wrapper"), name
