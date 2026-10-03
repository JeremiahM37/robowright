"""Deterministic replay of a trace.

Replay rebuilds the scene from the trace, restores the exact starting state,
and feeds the recorded motor commands and external forces back in step by
step. No IK, no policy, no test code runs - so if the replayed states match
the recorded ones, the run is reproducible, and if they diverge you learn
exactly at which step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .backends import base as backends
from .trace import Trace


@dataclass
class ReplayResult:
    steps: int
    max_qpos_error: float  # radians
    max_object_error: float  # metres
    first_divergent_step: int | None  # first step above tolerance, if any
    qpos_error: np.ndarray = field(repr=False)
    object_error: np.ndarray = field(repr=False)

    @property
    def identical(self) -> bool:
        return self.max_qpos_error == 0.0 and self.max_object_error == 0.0

    def summary(self) -> str:
        if self.identical:
            return f"replayed {self.steps} steps: bit-identical"
        div = "" if self.first_divergent_step is None else f", first divergence above tolerance at step {self.first_divergent_step}"
        return (
            f"replayed {self.steps} steps: max joint error {self.max_qpos_error:.2e} rad, "
            f"max object error {self.max_object_error:.2e} m{div}"
        )


def _is_edit(e) -> bool:
    return e["type"] == "edit" or (e["type"] == "fault" and e["name"] == "WeakJoint")


def replay(trace: str | Path | Trace, backend: str | None = None, tol: float = 1e-6) -> ReplayResult:
    tr = trace if isinstance(trace, Trace) else Trace(trace)
    meta = tr.meta
    b = backends.create(backend or meta["backend"], tr.scene(), seed=meta["seed"])
    try:
        restored = tr.state0 is not None and backends.STATE in b.capabilities
        if restored:
            b.set_state(tr.state0)
        begin = meta.get("begin_event_index", 0)
        edits: dict[int, list] = {}
        for idx, e in enumerate(tr.events):
            if not _is_edit(e):
                continue
            # Edits made before recording began are already in state0 - except
            # gain changes, which live in the model rather than the state.
            if restored and idx < begin and e["type"] == "edit":
                continue
            edits.setdefault(e["step"], []).append(e)
        a = tr.arrays
        names = meta["object_names"]
        n = len(a["t"])
        qerr = np.zeros(n)
        oerr = np.zeros(n)
        for i in range(1, n):
            _apply(b, edits.get(i - 1, []))
            b.set_ctrl(a["ctrl"][i])
            for k, name in enumerate(meta.get("force_names", names)):
                f = a["forces"][i][k]
                if np.any(f):
                    b.apply_force(name, f)
            b.step()
            qerr[i] = float(np.max(np.abs(b.qpos() - a["qpos"][i])))
            if names and backends.GROUND_TRUTH in b.capabilities:
                pos = np.array([b.object_pose(nm)[0] for nm in names])
                oerr[i] = float(np.max(np.linalg.norm(pos - a["obj_pos"][i], axis=1)))
        bad = np.nonzero((qerr > tol) | (oerr > tol))[0]
        return ReplayResult(n - 1, float(qerr.max()), float(oerr.max()), int(bad[0]) if bad.size else None, qerr, oerr)
    finally:
        b.close()


def _apply(b, events):
    for e in events:
        if e["type"] == "fault":
            b.set_gain_scale(e["args"]["joint"], e["args"]["scale"])
        elif e["name"] == "move_object":
            b.set_object_pose(e["args"]["object"], e["args"]["pos"], e["args"]["quat"])
        elif e["name"] == "reset_to":
            if "base_pos" in e["args"]:
                yaw = e["args"].get("yaw", 0.0)
                b.set_base_pose(e["args"]["base_pos"], (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)))
            b.set_joint_positions(np.array(e["args"]["q"]))
