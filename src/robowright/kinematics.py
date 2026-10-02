"""Backend-independent kinematics read from a URDF.

The same chain drives inverse kinematics for every backend (MuJoCo,
PyBullet, real hardware), so a test that says ``move_to((0.2, 0, 0.05))``
asks for the same joint targets no matter where it runs.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def homogeneous(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray
    lower: float
    upper: float


class Chain:
    """A serial chain from ``base`` to ``tip`` parsed from a URDF file."""

    def __init__(self, urdf_path: str | Path, base: str, tip: str, tcp_offset=(0.0, 0.0, 0.0)):
        root = ET.parse(urdf_path).getroot()
        joints: dict[str, Joint] = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
            lim = j.find("limit")
            joints[j.find("child").get("link")] = Joint(
                name=j.get("name"),
                type=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                origin=homogeneous(rpy_to_matrix(*rpy), xyz),
                axis=axis,
                lower=float(lim.get("lower", -np.pi)) if lim is not None else -np.pi,
                upper=float(lim.get("upper", np.pi)) if lim is not None else np.pi,
            )
        chain: list[Joint] = []
        link = tip
        while link != base:
            if link not in joints:
                raise ValueError(f"link {link!r} is not connected to {base!r}")
            chain.append(joints[link])
            link = joints[link].parent
        self.joints = list(reversed(chain))
        self.movable = [j for j in self.joints if j.type in ("revolute", "continuous")]
        self.names = [j.name for j in self.movable]
        self.lower = np.array([j.lower for j in self.movable])
        self.upper = np.array([j.upper for j in self.movable])
        self.tcp = homogeneous(np.eye(3), tcp_offset)

    def fk(self, q, tcp: bool = True) -> np.ndarray:
        """World-from-tip transform for joint vector ``q`` (movable joints, in order)."""
        T = np.eye(4)
        i = 0
        for j in self.joints:
            T = T @ j.origin
            if j.type in ("revolute", "continuous"):
                T = T @ homogeneous(axis_angle(j.axis, q[i]), (0, 0, 0))
                i += 1
        return T @ self.tcp if tcp else T

    def ik(
        self,
        target_pos,
        q0,
        approach=None,
        yaw: float | None = None,
        iters: int = 200,
        tol: float = 1e-4,
        damping: float = 1e-3,
    ) -> tuple[np.ndarray, float]:
        """Damped least-squares IK.

        ``approach`` is a world direction the tool's -z axis should point along
        (e.g. ``(0, 0, -1)`` for a top-down grasp). ``yaw`` pins the rotation of
        the tool x axis about world z. Returns ``(q, position_error)``.
        """
        q = np.array(q0, dtype=float).copy()
        target_pos = np.asarray(target_pos, dtype=float)
        a = None if approach is None else np.asarray(approach, float) / np.linalg.norm(approach)

        def residual(qv):
            T = self.fk(qv)
            r = [T[:3, 3] - target_pos]
            if a is not None:
                r.append(0.3 * (-T[:3, 2] - a))
            if yaw is not None:
                x = T[:3, 0]
                r.append(0.3 * np.array([np.arctan2(np.sin(np.arctan2(x[1], x[0]) - yaw), np.cos(np.arctan2(x[1], x[0]) - yaw))]))
            return np.concatenate(r)

        eps = 1e-6
        for _ in range(iters):
            r = residual(q)
            if np.linalg.norm(r[:3]) < tol and np.linalg.norm(r[3:]) < 10 * tol:
                break
            J = np.empty((r.size, q.size))
            for k in range(q.size):
                dq = q.copy()
                dq[k] += eps
                J[:, k] = (residual(dq) - r) / eps
            step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(r.size), r)
            q = np.clip(q - step, self.lower, self.upper)
        err = float(np.linalg.norm(self.fk(q)[:3, 3] - target_pos))
        return q, err
