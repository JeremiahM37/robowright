"""The robot x simulator matrix.

    python bench/matrix.py                     # every available backend, every robot
    python bench/matrix.py --backends mujoco,pybullet --trials 10
    python bench/matrix.py --markdown-only     # re-render MATRIX.md from the saved JSON

For each (backend, arm) pair it runs the same randomised pick-and-place
across seeded trials and records the success rate (with a Wilson interval),
how hard a shove the grasp survives, where the cube ends up compared with
MuJoCo for the same seed, and the wall-clock cost of a control step. For
legged robots it records standing, and the largest sideways push (as a
fraction of body weight) the robot survives without falling.

Each (backend, robot) cell runs in its own process, so a crash or a hung
engine costs one cell, not the run. Results go to bench/results/matrix.json
and MATRIX.md. Nothing in MATRIX.md is written by hand.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results"
sys.path.insert(0, str(ROOT / "bench"))


def _arm_cell(backend: str, robot: str, trials: int, seed0: int) -> dict:
    import robowright as rw
    from robowright import expect
    from robowright.stats import wilson

    out = {"backend": backend, "robot": robot, "kind": "arm", "trials": []}
    steps = wall = 0.0
    for i in range(trials):
        seed = seed0 + i
        t0 = time.perf_counter()
        w = rw.launch(robot=robot, backend=backend, seed=seed, settings=rw.Settings(trace="off"))
        build = time.perf_counter() - t0
        rec = {"seed": seed, "build_s": round(build, 3)}
        t1, s1 = time.perf_counter(), w.step_count
        try:
            w.robot.reset_to()
            w.faults.jitter("cube", xy_std=0.015, yaw_std=0.6)
            cube, bin_ = w.scene["cube"], w.scene["bin"]
            w.robot.pick(cube)
            w.robot.place(on=bin_)
            expect(cube).to_be_inside(bin_)
            expect(cube).to_be_at_rest()
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        steps += w.step_count - s1
        wall += time.perf_counter() - t1
        try:
            rec["cube"] = [round(float(x), 4) for x in w.scene["cube"].position]
        except Exception:
            pass
        w.close()
        out["trials"].append(rec)
    ok = sum(t["ok"] for t in out["trials"])
    lo, hi = wilson(ok, trials)
    out.update(passed=ok, n=trials, ci=[round(lo, 3), round(hi, 3)], ms_per_step=round(1000 * wall / max(steps, 1), 3))
    out["grip"] = _grip_margin(backend, robot)
    return out


def _grip_margin(backend: str, robot: str) -> dict:
    """Largest sideways shove (N, 0.1 s) a held cube survives, by bisection."""
    import robowright as rw

    def survives(force: float) -> bool | None:
        w = rw.launch(robot=robot, backend=backend, settings=rw.Settings(trace="off"))
        try:
            w.robot.reset_to()
            cube = w.scene["cube"]
            w.robot.pick(cube)
            if w.robot.gripper.holding() != "cube":
                return None
            w.faults.push("cube", force=(force, force, 0.0), duration=0.1)
            w.wait(0.5)
            return w.robot.gripper.holding() == "cube"
        except Exception:
            return None
        finally:
            w.close()

    if survives(0.0) is not True:
        return {"held": False}
    lo, hi = 0.0, 20.0
    if survives(hi):
        return {"held": True, "max_shove_n": f">{hi:.0f}"}
    for _ in range(6):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if survives(mid) else (lo, mid)
    return {"held": True, "max_shove_n": round(lo * np.sqrt(2), 2)}  # force was (f, f): magnitude f*sqrt(2)


def _legged_cell(backend: str, robot: str, trials: int, seed0: int) -> dict:
    import robowright as rw

    out = {"backend": backend, "robot": robot, "kind": "legged"}

    def run(push_fraction: float) -> tuple[bool, float]:
        w = rw.launch(robot=robot, backend=backend, settings=rw.Settings(trace="off"))
        try:
            r = w.robot
            r.reset_to()
            w.wait(0.5)
            weight = r.total_mass * 9.81
            if push_fraction:
                w.faults.push("robot", force=(0.0, push_fraction * weight, 0.0), duration=0.1)
            t0, s0 = time.perf_counter(), w.step_count
            w.wait(2.0)
            ms = 1000 * (time.perf_counter() - t0) / (w.step_count - s0)
            up = r.base.up_axis[2] > np.cos(np.radians(15))
            return bool(up and r.base.height > 0.6 * r.model.stand_height), ms
        finally:
            w.close()

    stands, ms = run(0.0)
    out.update(stands=stands, ms_per_step=round(ms, 3))
    if stands:
        lo, hi = 0.0, 4.0
        for _ in range(7):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if run(mid)[0] else (lo, mid)
        out["max_push_bw"] = round(lo, 2)
    return out


def _cell(args) -> dict:
    backend, robot, kind, trials, seed0 = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    t = time.time()
    try:
        fn = _arm_cell if kind == "arm" else _legged_cell
        res = fn(backend, robot, trials, seed0)
    except Exception as e:
        res = {"backend": backend, "robot": robot, "kind": kind, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
    res["cell_seconds"] = round(time.time() - t, 1)
    print(f"  {backend:9s} {robot:9s} {_summary(res)}  ({res['cell_seconds']}s)", flush=True)
    return res


def _summary(r: dict) -> str:
    if "error" in r:
        return "ERROR " + r["error"][:80]
    if r["kind"] == "arm":
        return f"{r['passed']}/{r['n']} pick-and-place, grip {r['grip'].get('max_shove_n', '-')} N, {r['ms_per_step']} ms/step"
    return f"stands={r['stands']} push {r.get('max_push_bw', '-')} x weight, {r['ms_per_step']} ms/step"


def agreement(results: list[dict]) -> dict:
    """Per robot and backend: median distance (mm) from MuJoCo's final cube position, same seed."""
    ref = {
        (r["robot"], t["seed"]): t.get("cube")
        for r in results
        if r.get("backend") == "mujoco" and r["kind"] == "arm"
        for t in r.get("trials", [])
    }
    out = {}
    for r in results:
        if r["kind"] != "arm" or r.get("backend") == "mujoco" or "trials" not in r:
            continue
        d = [
            1000 * float(np.linalg.norm(np.subtract(t["cube"], ref[(r["robot"], t["seed"])])))
            for t in r["trials"]
            if t.get("cube") and ref.get((r["robot"], t["seed"]))
        ]
        if d:
            out[f"{r['robot']}/{r['backend']}"] = round(float(np.median(d)), 1)
    return out


def markdown(data: dict) -> str:
    from robowright import robots

    res = data["results"]
    backends = data["backends"]
    by = {(r["backend"], r["robot"]): r for r in res}
    env = data["env"]
    lines = [
        "# robowright robot x simulator matrix",
        "",
        f"Generated by `python bench/matrix.py` on {env['date']} ({env['cpu']}, {env['cores']} threads, Python {env['python']}).",
        *[
            f"The {b} column was measured on {e['cpu']} ({e['cores']} threads, Python {e['python']}) and merged with `--merge`."
            for b, e in data.get("env_extra", {}).items()
        ],
        "Every cell is measured; nothing here is written by hand.",
        "",
        "## Arms: randomised pick-and-place",
        "",
        f"Each cell runs the same test {data['trials']} times with the cube's position (σ = 15 mm) and yaw (σ = 0.6 rad) "
        "randomised by seed, the same seeds in every column: `robot.pick(cube)`, `robot.place(on=bin)`, "
        "`expect(cube).to_be_inside(bin)`, `expect(cube).to_be_at_rest()`. "
        "The interval is a 95% Wilson interval.",
        "",
        "| robot | " + " | ".join(backends) + " |",
        "|---|" + "---:|" * len(backends),
    ]
    arms = [n for n in robots.names("arm") if any((b, n) in by for b in backends)]
    for n in arms:
        cells = []
        for b in backends:
            r = by.get((b, n))
            if r is None:
                cells.append("–")
            elif "error" in r:
                cells.append("error")
            else:
                mark = "✅" if r["passed"] == r["n"] else ("⚠️" if r["passed"] else "❌")
                cells.append(f"{mark} {r['passed']}/{r['n']} ({100 * r['ci'][0]:.0f}–{100 * r['ci'][1]:.0f}%)")
        lines.append(f"| {robots.get(n).title} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Arms: grip strength and step cost",
        "",
        "Grip: the largest sideways shove (0.1 s, at the cube's centre) a held 30 g cube survives, found by bisection. "
        "Step: wall-clock milliseconds per 20 ms control step during pick-and-place (physics only, no tracing).",
        "",
        "| robot | " + " | ".join(f"{b} grip (N)" for b in backends) + " | " + " | ".join(f"{b} ms/step" for b in backends) + " |",
        "|---|" + "---:|" * (2 * len(backends)),
    ]
    for n in arms:
        g, s = [], []
        for b in backends:
            r = by.get((b, n))
            if r is None or "error" in r:
                g.append("–")
                s.append("–")
                continue
            gr = r.get("grip", {})
            g.append(str(gr.get("max_shove_n", "drops")) if gr.get("held") else "no grasp")
            s.append(f"{r['ms_per_step']:.2f}")
        lines.append(f"| {robots.get(n).title} | " + " | ".join(g) + " | " + " | ".join(s) + " |")
    agree = data.get("agreement", {})
    if agree:
        lines += [
            "",
            "## Engines disagree about where the cube lands",
            "",
            "Median distance between the cube's final position in each engine and in MuJoCo, same robot, same seed, same commands.",
            "",
            "| robot | " + " | ".join(b for b in backends if b != "mujoco") + " |",
            "|---|" + "---:|" * (len(backends) - 1),
        ]
        for n in arms:
            cells = [f"{agree[f'{n}/{b}']} mm" if f"{n}/{b}" in agree else "–" for b in backends if b != "mujoco"]
            lines.append(f"| {robots.get(n).title} | " + " | ".join(cells) + " |")
    legged = [n for n in robots.names("legged") if any((b, n) in by for b in backends)]
    if legged:
        lines += [
            "",
            "## Legged robots: standing and push recovery",
            "",
            "Joint servos only, no balance controller. Push: the largest sideways shove (0.1 s, at the base, "
            "as a multiple of body weight) after which the robot is still upright two seconds later.",
            "",
            "| robot | " + " | ".join(backends) + " |",
            "|---|" + "---:|" * len(backends),
        ]
        for n in legged:
            cells = []
            for b in backends:
                r = by.get((b, n))
                if r is None:
                    cells.append("–")
                elif "error" in r:
                    cells.append("error")
                elif not r["stands"]:
                    cells.append("❌ falls")
                else:
                    cells.append(f"✅ {r['max_push_bw']}× weight, {r['ms_per_step']:.2f} ms/step")
            lines.append(f"| {robots.get(n).title} | " + " | ".join(cells) + " |")
    fails = [(r["backend"], r["robot"], t) for r in res if r["kind"] == "arm" and "trials" in r for t in r["trials"] if not t["ok"]]
    if fails:
        lines += ["", "## Why trials failed", "", "First failure per cell:", ""]
        seen = set()
        for b, n, t in fails:
            if (b, n) in seen:
                continue
            seen.add((b, n))
            lines.append(f"- **{n} on {b}** (seed {t['seed']}): `{t.get('error', '?')}`")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default=None)
    ap.add_argument("--robots", default=None, help="comma-separated; default: every robot")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--markdown-only", action="store_true")
    ap.add_argument("--merge", action="store_true", help="add these backends' cells to the existing matrix.json")
    ap.add_argument("--out", default=None, help="write results here instead of bench/results/matrix.json")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = Path(a.out) if a.out else OUT / "matrix.json"
    if a.markdown_only:
        data = json.loads(path.read_text())
    else:
        from run import env

        from robowright import robots
        from robowright.backends.base import available

        backends = a.backends.split(",") if a.backends else available()
        names = a.robots.split(",") if a.robots else robots.names()
        jobs = [(b, n, robots.get(n).family, a.trials, a.seed) for b in backends for n in names]
        # Slowest engines first so the pool drains evenly.
        jobs.sort(key=lambda j: {"genesis": 0, "drake": 1, "pybullet": 2}.get(j[0], 3))
        print(f"{len(jobs)} cells on {a.jobs} processes: backends {backends}, {len(names)} robots, {a.trials} trials each", flush=True)
        t0 = time.time()
        with mp.get_context("spawn").Pool(a.jobs, maxtasksperchild=1) as pool:
            results = pool.map(_cell, jobs, chunksize=1)
        data = {
            "env": env(),
            "backends": backends,
            "trials": a.trials,
            "seconds": round(time.time() - t0, 1),
            "results": results,
        }
        if a.merge and path.exists():
            old = json.loads(path.read_text())
            ran = {(r["backend"], r["robot"]) for r in results}
            keep = [r for r in old["results"] if (r["backend"], r["robot"]) not in ran]
            data["results"] = keep + results
            data["backends"] = list(dict.fromkeys(old["backends"] + backends))
            data["env"] = old["env"]
            data.setdefault("env_extra", {})
            data["env_extra"].update({b: env() for b in backends})
            results = data["results"]
        data["agreement"] = agreement(results)
        path.write_text(json.dumps(data, indent=1))
    (ROOT / "MATRIX.md").write_text(markdown(data))
    print(f"wrote {path} and MATRIX.md")


if __name__ == "__main__":
    main()
