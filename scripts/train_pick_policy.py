"""Train the small learned policy the examples test: behaviour cloning of ScriptedPickPlace.

It exists to show a trained model being tested the way any checkpoint would be: exported to
ONNX, loaded through ``LearnedPolicy`` with its own conventions (joints in degrees, gripper
0..100, normalised state and action, 10-step action chunks) and run with ``run_policy``.
It reads object poses, like the scripted policy it imitates; a camera-based policy plugs in
the same way with ``images=[...]``.

    .venv/bin/python scripts/train_pick_policy.py            # writes examples/policies/
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "examples" / "policies"
STATE = ("qpos", "objects.cube", "objects.cube.quat", "objects.bin")
CHUNK = 10
# The conventions the model is trained in, as a real robot's dataset often has them.
UNITS, GRIPPER = "deg", (0.0, 100.0)


def _episode(job):
    seed, backend = job
    import robowright as rw
    from robowright.learned import LearnedPolicy
    from robowright.policies import ScriptedPickPlace
    from robowright.scene import default_scene

    conv = LearnedPolicy(lambda b: None, state=STATE, units=UNITS, gripper=GRIPPER)  # just the translation
    expert = ScriptedPickPlace(chunk=CHUNK)
    states, actions = [], []

    def recording(obs):
        conv._bind(obs["robot"])
        chunk = expert(obs)
        states.append(conv.inputs(obs)["observation.state"][0])
        actions.append(np.array([conv.to_model(a) for a in chunk]))
        return chunk

    with rw.launch(scene=default_scene("so101"), backend=backend, seed=seed, settings=rw.Settings(trace="off")) as w:
        w.robot.reset_to()
        w.faults.jitter("cube", xy_std=0.03, yaw_std=0.7)
        done = rw.condition(w.scene["cube"], "to_be_inside", w.scene["bin"])
        ok = w.robot.run_policy(recording, until=done, hold=1.0, timeout=15, privileged=True).success
    return (np.array(states), np.array(actions)) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    # Demonstrations from one engine only make a policy that fails on the other: trained on
    # MuJoCo alone it placed the cube 17/20 times there and 1/20 on PyBullet, whose servos trail
    # their targets further (7.5 mrad against 2.5 at the 90th percentile) while moving.
    ap.add_argument("--backends", default="mujoco,pybullet")
    args = ap.parse_args()

    import torch

    with mp.get_context("spawn").Pool(args.workers) as pool:
        engines = args.backends.split(",")
        jobs = [(1000 + i, engines[i % len(engines)]) for i in range(args.episodes)]
        eps = [e for e in pool.map(_episode, jobs) if e is not None]
    print(f"{len(eps)}/{args.episodes} demonstrations succeeded")
    S = np.concatenate([e[0] for e in eps]).astype(np.float32)
    A = np.concatenate([e[1] for e in eps]).astype(np.float32)
    flat = A.reshape(-1, A.shape[-1])
    stats = {
        "state": {"mean": S.mean(0).tolist(), "std": (S.std(0) + 1e-3).tolist()},
        "action": {"mean": flat.mean(0).tolist(), "std": (flat.std(0) + 1e-3).tolist()},
    }
    sm, ss = np.array(stats["state"]["mean"]), np.array(stats["state"]["std"])
    am, as_ = np.array(stats["action"]["mean"]), np.array(stats["action"]["std"])
    X = torch.tensor((S - sm) / ss, dtype=torch.float32)
    Y = torch.tensor((A - am) / as_, dtype=torch.float32).reshape(len(A), -1)
    print(f"{len(X)} samples, state {X.shape[1]}, action chunk {A.shape[1]}x{A.shape[2]}")

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
            return self.net(state).reshape(-1, CHUNK, dof)

    net = Policy()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for epoch in range(args.epochs):
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
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: loss {total / len(X):.5f}")

    OUT.mkdir(parents=True, exist_ok=True)
    net.eval()
    torch.onnx.export(
        net,
        (torch.zeros(1, X.shape[1]),),
        str(OUT / "so101_pick.onnx"),
        input_names=["observation.state"],
        output_names=["action"],
        dynamic_axes={"observation.state": {0: "batch"}, "action": {0: "batch"}},
        dynamo=False,
    )
    (OUT / "so101_pick.json").write_text(json.dumps(stats, indent=1) + "\n")
    print("wrote", OUT / "so101_pick.onnx")


if __name__ == "__main__":
    main()
