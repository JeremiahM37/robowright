"""rw.attach, out of the box, on every task Isaac Lab registers (isaaclab_tasks), each made the way Isaac Lab's own scripts
make it (gym.make with parse_env_cfg, two copies on the GPU): robowright watches copy 1 while the loop steps it, reads its
bodies and passes an assertion. A task Isaac Lab itself cannot make here is skipped with Isaac Lab's reason, after doing
what that reason asks where it can (cameras on, or the CPU for a task that supports nothing else). Each task runs in a
process of its own: an Isaac Lab process makes one environment (a second one hung). Each task first runs one step on
its own: one Isaac Lab cannot run unattached is skipped with its error, as not robowright's.

Measured 2026-10-16 on an RTX 5080 (Isaac Lab 2.3.2, Isaac Sim 5.1), 175 tasks: 142 attached and passed, none failed,
33 skipped: 23 camera tasks (Isaac Sim's RTX renderer crashed starting, with driver 615, before any task ran), 7 not
registered by importing isaaclab_tasks (the Pink IK and humanoid manipulation variants), 3 that Isaac Lab runs only on
the CPU and whose own termination check fails there.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("isaaclab_tasks") is None, reason="needs Isaac Lab (isaaclab_tasks)")


def _tasks() -> list[str]:
    """The task ids isaaclab_tasks registers, read from its source: importing it needs Isaac Sim running."""
    spec = importlib.util.find_spec("isaaclab_tasks")
    if spec is None:
        return []
    found = set()
    for root in spec.submodule_search_locations:
        for path in Path(root).rglob("*.py"):
            found.update(re.findall(r"""id\s*=\s*["'](Isaac-[A-Za-z0-9_.-]+)["']""", path.read_text(errors="replace")))
    return sorted(found)


CHILD = textwrap.dedent(
    """
    import json, sys, traceback
    task, cameras, device = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=cameras, device=device).app
    out = {}
    try:
        import gymnasium as gym, torch
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        import robowright as rw
        from robowright import expect
        try:
            env = gym.make(task, cfg=parse_env_cfg(task, device=device, num_envs=2))
        except Exception as e:  # Isaac Lab cannot make it here: its reason, not robowright's
            out["cannot_make"] = f"{type(e).__name__}: {str(e)[:300]}"
        else:
            u = env.unwrapped
            n, dev = u.num_envs, u.device

            def batch(space):  # one row per copy, as Isaac Lab's step takes it
                if space.shape and space.shape[0] == n:  # already batched
                    return torch.as_tensor(space.sample(), device=dev).reshape(n, -1)
                return torch.stack([torch.as_tensor(space.sample(), device=dev).reshape(-1) for _ in range(n)])

            def action():
                spaces = getattr(u, "action_spaces", None)  # a multi-agent task: an action per agent
                if isinstance(spaces, dict):
                    return {agent: batch(sp) for agent, sp in spaces.items()}
                return batch(env.action_space)

            try:  # Isaac Lab runs the task on its own first: an error here is the task's, not robowright's
                env.reset(seed=0)
                env.step(action())
            except Exception as e:
                out["cannot_run"] = f"{type(e).__name__}: {str(e)[:300]}"
                raise SystemExit
            w = rw.attach(env, env_index=1, settings=rw.Settings(trace="off"), name=task)
            env.reset(seed=0)
            for _ in range(5):
                env.step(action())
            bodies = w.bodies()
            assert bodies, "robowright found no bodies"
            expect(w.body(bodies[-1])).to_be_above(-100.0, timeout=0)
            out = {"ok": True, "bodies": len(bodies), "steps": w.step_count}
            w.close()
            env.close()
    except SystemExit:
        pass
    except Exception as e:
        out = {"fail": f"{type(e).__name__}: {str(e)[:400]}", "tb": traceback.format_exc()[-2500:]}
    print("RESULT " + json.dumps(out), flush=True)
    app.close()
    """
)


def _run(task: str, cameras: bool, device: str) -> dict:
    import time

    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-c", CHILD, task, "1" if cameras else "0", device]
    # The result is read as soon as it is printed: Isaac Sim's shutdown can then hang (seen on the CPU), and a timeout
    # waiting for the exit threw away a result already given.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)
    seen: list[str] = []
    deadline = time.monotonic() + 600
    try:
        for line in proc.stdout:
            seen.append(line)
            if line.startswith("RESULT "):
                return json.loads(line[7:])
            if time.monotonic() > deadline:
                return {"crashed": "no result within 600 s", "log": "".join(seen[-40:])}
        proc.wait(30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, 9)
            proc.wait()
    text = "".join(seen)
    where = " in Isaac Sim's RTX renderer, as it started" if "librtx" in text else ""
    return {"crashed": f"exit {proc.returncode}{where}", "log": text[-1500:]}


@pytest.mark.parametrize("task", _tasks())
def test_attach_to_every_isaaclab_task(task):
    cameras, device = False, "cuda:0"
    for _ in range(3):
        r = _run(task, cameras, device)
        reason = r.get("cannot_make", "")
        if "--enable_cameras" in reason and not cameras:
            cameras = True  # as Isaac Lab's message asks
            continue
        if "--device cpu" in reason and device != "cpu":
            device = "cpu"  # a task Isaac Lab supports only on the CPU
            continue
        break
    if r.get("ok"):
        return
    if "cannot_make" in r:
        pytest.skip(f"Isaac Lab cannot make {task} here: {r['cannot_make'][:200]}")
    if "cannot_run" in r:
        pytest.skip(f"Isaac Lab cannot run {task} here, before robowright is attached: {r['cannot_run'][:200]}")
    if "crashed" in r and cameras:
        pytest.skip(f"{task} needs cameras, and Isaac Sim crashed starting with them ({r['crashed']}), before any task ran")
    pytest.fail(f"{task}: {r.get('fail') or r.get('crashed')}\n{r.get('tb') or r.get('log', '')}")
