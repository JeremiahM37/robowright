"""Limits checked on every command before it leaves robowright for a real robot (``[ros2] limits``).

A test driving real hardware should not be able to send a move faster, or to a place further, than
the person running it allowed. ``joint_speed`` (rad/s, or m/s for a prismatic joint) caps how fast
any arm joint is commanded to move between two points of a command; ``positions`` bounds each named
joint. robowright's own moves (``move_to``, ``move_joints``, ``reset_to``) are planned inside the
speed limit, however fast they were asked to go. A command it did not plan (a policy's target) that
would break a limit is refused whole, before anything is sent, with what it asked for, and the arm
holds where it is: slowing that down instead would change what the test measures.

``workspace`` is a box the tool centre point must stay in (metres, in the frame ``robot.tcp.position``
is in), checked at every point of a command from the arm's own kinematics: above the table, short of
the wall, inside the cell's fence.

    [ros2.limits]
    joint_speed = 0.5
    positions = { shoulder_lift_joint = [-2.0, -0.5] }
    workspace = { min = [-0.6, -0.6, 0.02], max = [0.6, 0.6, 0.8] }
"""

from __future__ import annotations

import numpy as np

from .errors import SafetyLimitError


def check(limits: dict | None, names: list[str], start, rows, dt: float, tcp=None) -> None:
    """Raise SafetyLimitError if moving from ``start`` through ``rows`` (one row of arm joint
    positions per ``dt`` seconds, in ``names`` order) breaks ``limits``. ``tcp`` maps a row to where
    the tool centre point would be, for ``workspace``."""
    if not limits:
        return
    unknown = set(limits) - {"joint_speed", "positions", "workspace"}
    if unknown:
        raise SafetyLimitError(f"[ros2] limits has no {', '.join(sorted(unknown))}: it takes joint_speed, positions and workspace")
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
    box = limits.get("workspace")
    if box:
        lo, hi = _box(box)
        if tcp is None:
            raise SafetyLimitError("[ros2] limits.workspace needs the arm's kinematics, which this backend does not have; nothing was sent")
        for row in rows:
            p = np.asarray(tcp(row), float)
            out = (p < lo) | (p > hi)
            if out.any():
                axis = "xyz"[int(np.argmax(out))]
                raise SafetyLimitError(
                    f"refused: the tool would reach ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), outside the workspace "
                    f"{tuple(map(float, lo))}..{tuple(map(float, hi))} in {axis} ([ros2] limits.workspace); nothing was sent"
                )


def _box(box) -> tuple[np.ndarray, np.ndarray]:
    try:
        lo, hi = np.asarray(box["min"], float), np.asarray(box["max"], float)
    except (KeyError, TypeError, ValueError):
        raise SafetyLimitError(f"[ros2] limits.workspace is {{min = [x, y, z], max = [x, y, z]}}, not {box!r}") from None
    if lo.shape != (3,) or hi.shape != (3,) or (lo >= hi).any():
        raise SafetyLimitError(f"[ros2] limits.workspace min must be below max on each of x, y and z: {box!r}")
    return lo, hi
