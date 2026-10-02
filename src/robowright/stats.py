"""Statistics for tests that only pass most of the time.

Robot behaviour under randomisation is a rate, not a boolean. ``trials``
runs a test across seeds and judges the success rate with a Wilson score
interval, so "18/20" and "2/2" are not treated as equally convincing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class TrialReport:
    name: str
    n: int
    passed: int
    min_success: float
    use_lower_bound: bool = False
    failures: list = field(default_factory=list)  # (seed, message, trace path)

    @property
    def rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.passed, self.n)

    @property
    def ok(self) -> bool:
        value = self.interval[0] if self.use_lower_bound else self.rate
        return value >= self.min_success - 1e-12

    def summary(self) -> str:
        lo, hi = self.interval
        req = f"{'95% lower bound' if self.use_lower_bound else 'rate'} >= {self.min_success:.0%}"
        return f"{self.passed}/{self.n} passed ({self.rate:.0%}, 95% CI {lo:.0%}-{hi:.0%}); required {req}"
