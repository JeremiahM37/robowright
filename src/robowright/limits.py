"""Limits checked on every command before it leaves robowright for a real robot (``[ros2] limits``).

A test driving real hardware should not be able to send a move faster, or to a place further, than
the person running it allowed. ``joint_speed`` (rad/s, or m/s for a prismatic joint) caps how fast
any arm joint is commanded to move between two points of a command; ``positions`` bounds each named
joint. robowright's own moves (``move_to``, ``move_joints``, ``reset_to``) are planned inside the
speed limit, however fast they were asked to go. A command it did not plan (a policy's target) that
would break a limit is refused whole, before anything is sent, with what it asked for, and the arm
holds where it is: slowing that down instead would change what the test measures.

    [ros2.limits]
    joint_speed = 0.5
    positions = { shoulder_lift_joint = [-2.0, -0.5] }
"""

from __future__ import annotations

import numpy as np

from .errors import SafetyLimitError


def check(limits: dict | None, names: list[str], start, rows, dt: float) -> None:
    """Raise SafetyLimitError if moving from ``start`` through ``rows`` (one row of arm joint
    positions per ``dt`` seconds, in ``names`` order) breaks ``limits``."""
    if not limits:
        return
    rows = np.atleast_2d(np.asarray(rows, float))[:, : len(names)]
    for joint, (lo, hi) in (limits.get("positions") or {}).items():
        if joint not in names:
            raise SafetyLimitError(f"[ros2] limits.positions names {joint!r}, which is not an arm joint: {', '.join(names)}")
        col = rows[:, names.index(joint)]
        bad = np.flatnonzero((col < lo) | (col > hi))
        if bad.size:
            where = f"{joint} would go to {col[bad[0]]:.3f}, outside its limits [{lo}, {hi}]"
            raise SafetyLimitError(f"refused: {where} ([ros2] limits.positions); nothing was sent")
    cap = limits.get("joint_speed")
    if cap:
        path = np.vstack([np.asarray(start, float)[: len(names)], rows])
        speed = np.abs(np.diff(path, axis=0)) / dt
        i, j = np.unravel_index(int(np.argmax(speed)), speed.shape)
        if speed[i, j] > cap + 1e-9:
            raise SafetyLimitError(
                f"refused: {names[j]} would move at {speed[i, j]:.2f}/s, faster than {cap}/s ([ros2] limits.joint_speed); nothing was sent"
            )
