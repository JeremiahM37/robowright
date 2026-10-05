import pytest

from robowright.stats import TrialReport, wilson


def test_wilson_known_values():
    lo, hi = wilson(18, 20)
    assert lo == pytest.approx(0.699, abs=1e-3) and hi == pytest.approx(0.972, abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)
    assert wilson(10, 10)[1] == 1.0


def test_trial_report_threshold():
    assert TrialReport("t", 20, 18, 0.9).ok
    assert not TrialReport("t", 20, 17, 0.9).ok
    assert not TrialReport("t", 20, 20, 0.9, use_lower_bound=True).ok  # 20/20 lower bound is 84%
    assert "18/20" in TrialReport("t", 20, 18, 0.9).summary()


def test_a_verdict_is_settled_once_the_rest_cannot_change_it():
    r = TrialReport("t", 20, 18, 0.9, ran=18)
    assert r.required == 18 and r.settled and r.ok
    assert "18/18" in r.summary() and "of 20, settled after 18" in r.summary()
    r = TrialReport("t", 20, 14, 0.9, ran=17)  # 3 failures: 18 passes are out of reach
    assert r.settled and not r.ok
    assert not TrialReport("t", 20, 15, 0.9, ran=17).settled  # 2 failures: 18 still reachable
    assert TrialReport("t", 10, 7, 0.7, ran=7).settled
    # With the lower bound, 20/20 is not enough, so no number of passes settles it as a pass.
    r = TrialReport("t", 20, 1, 0.9, use_lower_bound=True, ran=1)
    assert r.required == 21 and r.settled and not r.ok
    for n in range(1, 25):  # stopping early never changes a verdict
        for ms in (0.5, 0.7, 0.9, 1.0):
            for k in range(n + 1):
                for order in ([True] * k + [False] * (n - k), [False] * (n - k) + [True] * k):
                    r = TrialReport("t", n, 0, ms)
                    for i, ok in enumerate(order):
                        r.passed += ok
                        r.ran = i + 1
                        if r.settled:
                            break
                    assert r.ok == TrialReport("t", n, k, ms).ok
