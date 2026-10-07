"""Train the small learned policy the examples test: imitation of ScriptedPickPlace, with DAgger.

It exists to show a trained model being tested the way any checkpoint would be: exported to
ONNX, loaded through ``LearnedPolicy`` with its own conventions (joints in degrees, gripper
0..100, normalised state and action, 10-step action chunks) and run with ``run_policy``.
It reads object poses, like the scripted policy it imitates; a camera-based policy plugs in
the same way with ``images=[...]``.

    .venv/bin/python scripts/train_pick_policy.py            # writes examples/policies/

What it took to make a policy that succeeds every time (the README has the measurements):

* Demonstrations from one engine make a policy that fails on another: trained on MuJoCo alone
  it succeeded 2/40 times on PyBullet, whose servos trail their targets further (7.5 mrad
  against 2.5 at the 90th percentile) while moving. It trains on every engine it can run.
* Copying demonstrations alone (behaviour cloning) placed the cube 31/40 times on MuJoCo. The
  demonstrator waited a fixed time after closing and after opening the gripper; to a policy
  that cannot see time, "wait" and "go" look the same, it averaged them and stalled. The
  demonstrator here closes slowly instead, and the policy sees the targets it last commanded
  (``"target"``), so how far a close has got is in what it reads.
* A cloned policy drifts into states the demonstrator never visited and does not know how to
  recover from them. DAgger runs the policy, has the demonstrator say what it would do in each
  state the policy reached, and trains on that too. The demonstrator (``_coach``) has to decide
  from what it observes for this to work: one following its plan step by step sent the policy
  back to waypoints it had cut past, and the retrained policy collapsed (0 of 200).

With 400 demonstrations and three rounds of 200 it succeeded on 140 of 140 seeds per engine.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "examples" / "policies"
STATE = ("qpos", "target", "objects.cube", "objects.cube.quat", "objects.bin")
CHUNK = 10
# The conventions the model is trained in, as a real robot's dataset often has them.
UNITS, GRIPPER = "deg", (0.0, 100.0)


def _coach():
    """The demonstrator, which also says what it would do in any state a policy reaches.

    It decides from what it observes, not from a remembered step of a plan: a policy cuts
    corners (it heads down to the cube without stopping above it first), and a demonstrator
    that still waits for it at the skipped waypoint sends it back up, from beside the cube,
    where its own demonstrations closed the gripper; a policy trained on both does neither.
    It moves from the targets last commanded, toward the IK solution nearest them, so what it
    says next is always a small move from where the policy is. It closes and opens the gripper slowly rather
    than waiting a fixed time, which a policy cannot see. The waypoints and grasp come from
    ScriptedPickPlace.
    """
    from robowright.policies import ScriptedPickPlace
    from robowright.robot import GRIPPER_CLOSED, solve_ik

    class Coach(ScriptedPickPlace):
        grip_speed = 0.5  # the gripper's command, in openings per second
        line_up = 0.008  # how far off the cube (m) the tool may be to head down to it

        def reset(self):
            super().reset()
            self._goal = None

        def __call__(self, obs):
            self._bind(obs["robot"])
            cmd = np.array(obs["target"], float)
            n = len(cmd) - 1
            objs = obs["objects"]
            cube, bin_ = np.asarray(objs[self.object][0]), np.asarray(objs[self.target][0])
            tcp, at = np.asarray(obs["tcp"]), self.kin.tcp(cmd[:n])
            plan = self._plan(objs)
            above, grasp, carry, release = plan[0], plan[1], plan[4], plan[5]
            open_ = above[2]
            grip, opening = cmd[n], obs["qpos"][-1]
            near = lambda p, tol=self.tolerance: np.linalg.norm(at - p) < tol and np.linalg.norm(tcp - p) < tol

            holding = grip <= GRIPPER_CLOSED + 0.02 and opening > 0.06 and np.linalg.norm(cube - tcp) < 0.03
            in_bin = np.abs(cube[:2] - bin_[:2]).max() < 0.045
            if in_bin and not holding:  # let go and back off
                goal = (carry[0], carry[1], open_) if grip >= open_ - 0.02 else (at, carry[1], open_)
            elif holding and np.linalg.norm(tcp[:2] - bin_[:2]) < 0.015:  # over the bin: lower, open
                goal = (release[0], release[1], open_) if near(release[0]) else (release[0], release[1], grip)
            elif holding:  # lift straight up to carrying height (the bin's: not the cube's, which rises with the hand), then over the bin
                low = tcp[2] < carry[0][2] - 0.01 and np.linalg.norm(tcp[:2] - bin_[:2]) > 0.05
                goal = (np.array([*at[:2], carry[0][2]]), grasp[1], grip) if low else (carry[0], carry[1], grip)
            elif near(grasp[0]) and grip > GRIPPER_CLOSED + 0.02:  # at the cube: close
                goal = (grasp[0], grasp[1], GRIPPER_CLOSED)
            elif grip <= GRIPPER_CLOSED + 0.02 and not holding and near(grasp[0]):  # closed on nothing
                goal = (above[0], above[1], open_)
            else:  # line up over the cube, then head down to it, opening on the way
                off = np.linalg.norm(at[:2] - grasp[0][:2])
                # Once on the way down it keeps going: a joint move's path bends, and partway
                # down the tool is 1.5 cm aside, which sent it back up to line up again.
                going_down = at[2] < above[0][2] - 0.01 and off < 0.03
                lined = going_down or (off < self.line_up and np.linalg.norm(tcp[:2] - grasp[0][:2]) < 2 * self.line_up)
                goal = (grasp[0] if lined else above[0], grasp[1], open_)
            goal_p, yaw, g = goal
            # Toward an IK solution for the goal, at ScriptedPickPlace's joint and tool speeds.
            # It is solved from the joints commanded when the goal appears (or moves), then kept:
            # a five-joint arm reaches a point many ways, and solving again from each place it
            # had got to switched between them and went round in circles.
            goal_p = np.asarray(goal_p, float)
            if self._goal is None or np.linalg.norm(self._goal[0] - goal_p) > 0.003 or abs(self._goal[1] - yaw) > 0.01:
                self._goal = (goal_p, yaw, solve_ik(self.kin, goal_p, cmd[:n], self.home, yaw=yaw)[0])
            q_goal = self._goal[2]
            goal_q = np.append(q_goal, g)
            limit = np.full(n + 1, self.speed * 0.02)
            limit[n] = self.grip_speed * 0.02
            out = []
            for _ in range(self.chunk):
                delta = np.clip(goal_q - cmd, -limit, limit)
                moved = np.linalg.norm(self.kin.tcp(cmd[:n] + delta[:n]) - self.kin.tcp(cmd[:n]))
                if moved > self.tool_speed * 0.02:
                    delta[:n] *= self.tool_speed * 0.02 / moved
                cmd = cmd + delta
                out.append(cmd.copy())
            return np.array(out)

    return Coach(chunk=CHUNK)


def _episode(job):
    """One rollout: the coach acting (a demonstration), or a policy acting and the coach saying
    what it would have done. Returns the states, the coach's actions and whether it succeeded."""
    seed, backend, model = job
    import robowright as rw
    from robowright.learned import LearnedPolicy
    from robowright.scene import default_scene

    conv = LearnedPolicy(lambda b: None, state=STATE, units=UNITS, gripper=GRIPPER)  # just the translation
    coach = _coach()
    learner = None
    if model is not None:
        learner = LearnedPolicy(f"{model}/so101_pick.onnx", state=STATE, units=UNITS, gripper=GRIPPER, normalize=f"{model}/so101_pick.json")
    states, actions = [], []

    def act(obs):
        conv._bind(obs["robot"])
        chunk = coach(obs)
        states.append(conv.inputs(obs)["observation.state"][0])
        actions.append(np.array([conv.to_model(a) for a in chunk]))
        return chunk if learner is None else learner(obs)

    with rw.launch(scene=default_scene("so101"), backend=backend, seed=seed, settings=rw.Settings(trace="off")) as w:
        w.robot.reset_to()
        w.faults.jitter("cube", xy_std=0.03, yaw_std=0.7)
        done = rw.condition(w.scene["cube"], "to_be_inside", w.scene["bin"])
        ok = w.robot.run_policy(act, until=done, hold=1.0, timeout=15, privileged=True).success
    return backend, np.array(states), np.array(actions), ok


def _rollouts(pool, jobs):
    eps = pool.map(_episode, jobs)
    for be in dict.fromkeys(e[0] for e in eps):
        mine = [e for e in eps if e[0] == be]
        print(f"  {be}: {sum(e[3] for e in mine)}/{len(mine)} succeeded", flush=True)
    return eps


def _train(S, A, stats, epochs):
    import torch

    sm, ss = np.array(stats["state"]["mean"]), np.array(stats["state"]["std"])
    am, as_ = np.array(stats["action"]["mean"]), np.array(stats["action"]["std"])
    X = torch.tensor((S - sm) / ss, dtype=torch.float32)
    Y = torch.tensor((A - am) / as_, dtype=torch.float32).reshape(len(A), -1)
    print(f"  {len(X)} samples, state {X.shape[1]}, action chunk {A.shape[1]}x{A.shape[2]}", flush=True)
    torch.manual_seed(0)
    dof = A.shape[2]

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, CHUNK * dof),
            )

        def forward(self, state):
            return self.net(state.clamp(-10, 10)).reshape(-1, CHUNK, dof)

    net = Policy()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), 256):
            idx = perm[i : i + 256]
            loss = torch.nn.functional.mse_loss(net(X[idx]).reshape(len(idx), -1), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch}: loss {total / len(X):.5f}", flush=True)
    net.eval()
    return net, X.shape[1]


def _export(net, n_state, stats, out: Path):
    import torch

    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        (torch.zeros(1, n_state),),
        str(out / "so101_pick.onnx"),
        input_names=["observation.state"],
        output_names=["action"],
        dynamic_axes={"observation.state": {0: "batch"}, "action": {0: "batch"}},
        dynamo=False,
    )
    (out / "so101_pick.json").write_text(json.dumps(stats, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600, help="demonstrations")
    ap.add_argument("--dagger", type=int, default=4, help="rounds of running the policy and being corrected")
    ap.add_argument("--dagger-episodes", type=int, default=320, help="rollouts per round")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--backends", default="mujoco,pybullet,drake,genesis")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--keep", type=Path, help="keep each round's model here")
    args = ap.parse_args()
    engines = args.backends.split(",")

    def jobs(n, start, model):
        return [(start + i, engines[i % len(engines)], model) for i in range(n)]

    # A worker is replaced every few rollouts: Genesis's grow past 2.5 GB.
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool, tempfile.TemporaryDirectory() as tmp:
        print("demonstrations", flush=True)
        eps = [e for e in _rollouts(pool, jobs(args.episodes, 1000, None)) if e[3]]
        S = np.concatenate([e[1] for e in eps]).astype(np.float32)
        A = np.concatenate([e[2] for e in eps]).astype(np.float32)
        flat = A.reshape(-1, A.shape[-1])
        stats = {  # fixed from the demonstrations, so every round's model reads the same inputs
            # Floored: the bin never moves in a demonstration, and when a policy knocks it 5 cm a
            # spread of zero would make that input thousands of deviations out.
            "state": {"mean": S.mean(0).tolist(), "std": np.maximum(S.std(0), 0.01).tolist()},
            "action": {"mean": flat.mean(0).tolist(), "std": (flat.std(0) + 1e-3).tolist()},
        }
        net, n_state = _train(S, A, stats, args.epochs)
        for r in range(args.dagger):
            model = (args.keep or Path(tmp)) / f"round{r}"
            _export(net, n_state, stats, model)
            print(f"DAgger round {r + 1}: the policy acts, the coach corrects", flush=True)
            eps = _rollouts(pool, jobs(args.dagger_episodes, 100_000 * (r + 1), str(model)))
            S = np.concatenate([S, *[e[1] for e in eps]]).astype(np.float32)
            A = np.concatenate([A, *[e[2] for e in eps]]).astype(np.float32)
            net, n_state = _train(S, A, stats, args.epochs)
    _export(net, n_state, stats, args.out)
    print("wrote", args.out / "so101_pick.onnx")


if __name__ == "__main__":
    main()
