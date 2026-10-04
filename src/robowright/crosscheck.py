"""Cross-check a run on another physics engine.

Tests and agent sessions run cheaply on MuJoCo. ``crosscheck`` takes the
trace of one, makes the same calls on another engine (Isaac Sim's PhysX,
Drake's hydroelastic contact, ...) from the same scene and seed, and says
whether the outcome holds there and how far each object ends up from
where it did originally:

    robowright crosscheck trace.zip --backend isaac

A run whose result depends on one engine's contact model shows up here
before anyone trusts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .codegen import generate
from .trace import Trace


@dataclass
class CrossCheck:
    source_backend: str
    backend: str
    source_status: str
    passed: bool
    error: str | None = None
    # object -> distance (m) between where it ended up on each engine
    object_offsets: dict[str, float] = field(default_factory=dict)

    @property
    def agrees(self) -> bool:
        """Same verdict on both engines (both passed, or both failed)."""
        return self.passed == (self.source_status == "passed")

    def summary(self) -> str:
        verdict = "passes" if self.passed else f"fails ({self.error})"
        lines = [f"{self.backend}: {verdict}; on {self.source_backend} it {self.source_status}"]
        for name, d in self.object_offsets.items():
            lines.append(f"  {name} ends {1000 * d:.1f} mm from where it did on {self.source_backend}")
        lines.append("verdicts agree" if self.agrees else "VERDICTS DIFFER: the outcome depends on the engine")
        return "\n".join(lines)


def crosscheck(trace: str | Path | Trace, backend: str) -> CrossCheck:
    import robowright as rw

    tr = trace if isinstance(trace, Trace) else Trace(trace)
    code = generate(tr, test_name="test_crosscheck", backend=backend)
    captured = {}
    real = rw.launch

    def launch(*a, **k):
        k["settings"] = _no_trace(k.get("settings"))
        w = real(*a, **k)
        captured["world"] = w
        close = w.close

        def close_and_keep(*ca, **ck):
            captured["objects"] = {n: w.backend.object_pose(n)[0].copy() for n in w.object_names}
            return close(*ca, **ck)

        w.close = close_and_keep
        return w

    namespace: dict = {}
    error = None
    rw.launch = launch
    try:
        exec(compile(code, f"<crosscheck {tr.path.name}>", "exec"), namespace)
        namespace["test_crosscheck"]()
    except Exception as e:  # the regenerated calls failing on this engine is a result, not a crash
        error = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    finally:
        rw.launch = real
    w = captured.get("world")
    if "objects" not in captured and w is not None:  # failed before the world closed
        captured["objects"] = {n: w.backend.object_pose(n)[0].copy() for n in w.object_names}
        w.close(failed=True)
    names = tr.meta["object_names"]
    final = tr.arrays["obj_pos"][-1]
    offsets = {
        n: float(np.linalg.norm(captured["objects"][n] - final[i]))
        for i, n in enumerate(names)
        if n in captured.get("objects", {}) and not tr.scene().object(n).static
    }
    return CrossCheck(tr.meta["backend"], backend, tr.meta["status"], error is None, error, offsets)


def _no_trace(settings):
    import dataclasses

    from .world import Settings

    return dataclasses.replace(settings or Settings(), trace="off")
