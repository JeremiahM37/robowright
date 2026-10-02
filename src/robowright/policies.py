"""Policies: anything that maps an observation to joint targets.

A policy is a callable ``policy(obs) -> action`` where ``action`` is a
6-vector of joint position targets (5 arm joints + gripper, radians) or an
``(n, 6)`` action chunk executed one control step per row. An optional
``reset()`` is called before each rollout, and an optional ``to_config()``
lets codegen recreate the policy in a generated test.

``ScriptedPickPlace`` is a closed-loop, chunked reference policy. It stands
in for a learned policy in examples and benchmarks: it reads joint
encoders (so injected sensor noise hurts it) and ground-truth object poses.
"""

from __future__ import annotations

import numpy as np

from . import assets
from .kinematics import Chain
from .robot import DOWN, GRIPPER_CLOSED, GRIPPER_OPEN, MIN_GRASP_Z, TCP_OFFSET


class ScriptedPickPlace:
    def __init__(self, object: str = "cube", target: str = "bin", chunk: int = 10, speed: float = 1.5, tolerance: float = 0.012):
        self.object, self.target = object, target
        self.chunk, self.speed, self.tolerance = chunk, speed, tolerance
        self.chain = Chain(assets.so101_urdf(), "base_link", "gripper_link", TCP_OFFSET)
        self.reset()

    def to_config(self) -> dict:
        return {"object": self.object, "target": self.target, "chunk": self.chunk, "speed": self.speed, "tolerance": self.tolerance}

    def reset(self):
        self.phase = 0
        self.wait = 0
        self.cmd = None
        self.goal = None

    def _ik(self, p, seed, yaw):
        best = None
        for y in (yaw, yaw + np.pi / 2, yaw - np.pi / 2):
            q, err = self.chain.ik(p, seed, DOWN, y)
            score = err + 0.002 * abs(q[4])
            if best is None or score < best[1]:
                best = (q, score)
        return best[0]

    def _plan(self, objs):
        op, oq = objs[self.object]
        tp, tq = objs[self.target]
        oyaw = 2 * np.arctan2(oq[3], oq[0])
        tyaw = 2 * np.arctan2(tq[3], tq[0])
        grasp = np.array([op[0], op[1], max(op[2], MIN_GRASP_Z)])
        rim = tp[2] + 0.04
        return [  # (tcp goal, yaw, gripper, settle steps)
            (grasp + [0, 0, 0.05], oyaw, GRIPPER_OPEN, 0),
            (grasp, oyaw, GRIPPER_OPEN, 0),
            (grasp, oyaw, GRIPPER_CLOSED, 20),
            (grasp + [0, 0, 0.05], oyaw, GRIPPER_CLOSED, 0),
            (np.array([tp[0], tp[1], rim + 0.04]), tyaw, GRIPPER_CLOSED, 0),
            (np.array([tp[0], tp[1], rim + 0.015]), tyaw, GRIPPER_CLOSED, 0),
            (np.array([tp[0], tp[1], rim + 0.015]), tyaw, GRIPPER_OPEN, 15),
            (np.array([tp[0], tp[1], rim + 0.04]), tyaw, GRIPPER_OPEN, 0),
        ]

    def __call__(self, obs) -> np.ndarray:
        q = np.asarray(obs["qpos"], float)
        if self.cmd is None:
            self.cmd = q.copy()
        plan = self._plan(obs["objects"])
        if self.phase >= len(plan):
            return np.tile(self.cmd, (self.chunk, 1))
        if self.goal is None:
            # Re-plan from fresh observations at the start of every phase, then
            # hold that goal: the object moves once it is grasped.
            goal_p, yaw, grip, settle = plan[self.phase]
            self.goal = (np.concatenate([self._ik(goal_p, self.cmd[:5], yaw), [grip]]), goal_p, settle)
        goal, goal_p, settle = self.goal
        tcp = self.chain.fk(q[:5])[:3, 3]
        if np.allclose(self.cmd, goal) and np.linalg.norm(tcp - goal_p) < self.tolerance:
            self.wait += self.chunk
            if self.wait >= settle:
                self.phase, self.wait, self.goal = self.phase + 1, 0, None
            return np.tile(self.cmd, (self.chunk, 1))
        # Stream a chunk that moves the command toward the goal at a bounded joint speed.
        step = self.speed * 0.02
        out = []
        for _ in range(self.chunk):
            self.cmd = self.cmd + np.clip(goal - self.cmd, -step, step)
            out.append(self.cmd.copy())
        return np.array(out)


class Replay:
    """Plays back a fixed joint trajectory (e.g. from a recorded demo), then holds."""

    def __init__(self, actions):
        self.actions = np.asarray(actions, float)
        self.i = 0

    def reset(self):
        self.i = 0

    def __call__(self, obs):
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a
