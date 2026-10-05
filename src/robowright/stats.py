"""Statistics for tests that only pass most of the time.

Robot behaviour under randomisation is a rate, not a boolean. ``trials``
runs a test across seeds and judges the success rate with a Wilson score
interval, so "18/20" and "2/2" are not treated as equally convincing.

A run stops as soon as its verdict is settled: once 18 of 20 have passed a
``min_success=0.9`` test passes whatever the last two do, and once 3 have
failed it cannot. The verdict is the one all ``n`` trials would give.
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
    ran: int | None = None  # trials run, when the verdict was settled before all n

    @property
    def runs(self) -> int:
        return self.n if self.ran is None else self.ran

    @property
    def rate(self) -> float:
        return self.passed / self.runs if self.runs else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.passed, self.runs)

    def _passes(self, k: int) -> bool:
        """Whether ``k`` passes out of all ``n`` trials meet the requirement."""
        value = wilson(k, self.n)[0] if self.use_lower_bound else (k / self.n if self.n else 0.0)
        return value >= self.min_success - 1e-12

    @property
    def required(self) -> int:
        """The fewest passes out of ``n`` that pass (``n + 1`` if none do)."""
        return next((k for k in range(self.n + 1) if self._passes(k)), self.n + 1)

    @property
    def settled(self) -> bool:
        """Whether the remaining trials can no longer change the verdict."""
        need = self.required
        return self.passed >= need or self.runs - self.passed > self.n - need

    @property
    def ok(self) -> bool:
        return self.passed >= self.required

    def summary(self) -> str:
        lo, hi = self.interval
        req = f"{'95% lower bound' if self.use_lower_bound else 'rate'} >= {self.min_success:.0%}"
        out = f"{self.passed}/{self.runs} passed ({self.rate:.0%}, 95% CI {lo:.0%}-{hi:.0%}); required {req}"
        if self.runs < self.n:
            out += f" of {self.n}, settled after {self.runs}"
        return out
