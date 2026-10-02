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
