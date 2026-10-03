"""robowright benchmarks.

    python bench/run.py            # full run, writes bench/results/results.json and BENCHMARKS.md
    python bench/run.py --quick    # fewer repetitions

Every number in BENCHMARKS.md comes from this script on the machine named
in its header. Nothing is copied in by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import robowright as rw
from robowright import condition, expect
from robowright.backends.base import available
from robowright.policies import ScriptedPickPlace
from robowright.replay import replay
from robowright.stats import wilson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "results"


def env() -> dict:
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    import mujoco

    info = {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "mujoco": mujoco.__version__,
        "robowright": rw.__version__,
        "date": time.strftime("%Y-%m-%d"),
    }
    try:
        import pybullet  # noqa: F401

        info["pybullet"] = "installed"
    except ImportError:
        pass
    return info


def _settings(trace="off", frame_every=5):
    return rw.Settings(trace=trace, trace_dir=tempfile.mkdtemp(prefix="rwbench-"), frame_every=frame_every)


def bench_throughput(backends, seconds=10.0):
    """Control steps per wall second with the arm moving, for each tracing mode."""
    rows = []
    for b in backends:
        for mode, trace, frame_every in (
            ("no trace", "off", 5),
            ("trace, no frames", "on", 10**9),
            ("trace + frames every 5 steps", "on", 5),
        ):
            w = rw.launch(backend=b, settings=_settings(trace, frame_every))
            w.robot.reset_to()
            n = int(seconds / w.dt)
            home = w.robot._target.copy()
            w.step(5)  # warm-up: first trace frame creates the GL context
            t0 = time.perf_counter()
            for i in range(n):
                w.robot._target[0] = home[0] + 0.5 * np.sin(i * 0.02)
                w.step()
            wall = time.perf_counter() - t0
            w.close(trace_path=Path(w.settings.trace_dir) / "t.zip") if trace == "on" else w.close()
            size = (Path(w.settings.trace_dir) / "t.zip").stat().st_size if trace == "on" else 0
            rows.append(
                {
                    "backend": b,
                    "mode": mode,
                    "steps_per_s": n / wall,
                    "realtime_factor": seconds / wall,
                    "trace_kb_per_sim_s": size / 1024 / seconds,
                }
            )
    return rows


def bench_pick_place(backends, reps):
    """Wall time of a complete pick-and-place test (skills + 5 expectations)."""
    rows = []
    for b in backends:
        times, sim = [], []
        for seed in range(reps):
            w = rw.launch(backend=b, seed=seed, settings=_settings("off"))
            w.robot.reset_to()
            t0 = time.perf_counter()
            cube, bin_ = w.scene["cube"], w.scene["bin"]
            w.robot.pick(cube)
            expect(w.robot.gripper).to_be_holding(cube)
            w.robot.place(on=bin_)
            expect(cube).to_be_inside(bin_)
            expect(cube).to_be_at_rest()
            times.append(time.perf_counter() - t0)
            sim.append(w.time)
            w.close()
        rows.append(
            {
                "backend": b,
                "median_wall_s": statistics.median(times),
                "sim_s": statistics.median(sim),
                "realtime_factor": statistics.median(sim) / statistics.median(times),
            }
        )
    return rows


def bench_parallel(workers_list):
    """Wall time of a 64-test randomised pick-and-place suite vs pytest-xdist workers."""
    rows = []
    for n in workers_list:
        args = [sys.executable, "-m", "pytest", "bench/suite", "-q", "-p", "no:cacheprovider", "--rw-trace", "off"]
        if n > 1:
            args += ["-n", str(n)]
        t0 = time.perf_counter()
        r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        rows.append({"workers": n, "wall_s": wall, "ok": r.returncode == 0})
    base = rows[0]["wall_s"]
    for r in rows:
        r["speedup"] = base / r["wall_s"]
    return rows


def bench_replay(backends, reps):
    """Re-simulate traces of runs with injected faults and compare every step."""
    rows = []
    for b in backends:
        identical, worst_q, worst_o, steps = 0, 0.0, 0.0, 0
        for seed in range(reps):
            w = rw.launch(backend=b, seed=seed, name=f"replay{seed}", settings=_settings("on", 10**9))
            w.robot.reset_to()
            w.faults.jitter("cube", 0.015, yaw_std=0.4)
            w.faults.action_delay(2)
            w.robot.pick(w.scene["cube"])
            w.faults.push("cube", (0.0, 0.4, 0.0), duration=0.05)
            w.robot.place(on=w.scene["bin"])
            path = w.close()
            r = replay(path)
            steps += r.steps
            identical += r.identical
            worst_q, worst_o = max(worst_q, r.max_qpos_error), max(worst_o, r.max_object_error)
        rows.append(
            {
                "backend": b,
                "runs": reps,
                "bit_identical": identical,
                "steps_replayed": steps,
                "max_joint_error_rad": worst_q,
                "max_object_error_m": worst_o,
            }
        )
    return rows


def bench_parity(backends, reps):
    """The same randomised pick-and-place test on each engine."""
    rows, finals = [], {}
    for b in backends:
        ok, ends = 0, []
        for seed in range(reps):
            w = rw.launch(backend=b, seed=seed, settings=_settings("off"))
            w.robot.reset_to()
            w.faults.jitter("cube", 0.02, yaw_std=0.5)
            try:
                w.robot.pick(w.scene["cube"])
                w.robot.place(on=w.scene["bin"])
                expect(w.scene["cube"]).to_be_inside(w.scene["bin"])
                ok += 1
            except (AssertionError, rw.RobowrightError):
                pass
            ends.append(w.scene["cube"].position.copy())
            w.close()
        lo, hi = wilson(ok, reps)
        rows.append({"backend": b, "passed": ok, "n": reps, "ci_low": lo, "ci_high": hi})
        finals[b] = np.array(ends)
    if len(finals) == 2:
        a, c = finals.values()
        d = np.linalg.norm(a - c, axis=1)
        rows.append(
            {
                "backend": "difference",
                "median_final_cube_distance_mm": float(np.median(d) * 1000),
                "max_final_cube_distance_mm": float(d.max() * 1000),
            }
        )
    return rows


def bench_faults(reps):
    """Reference policy success rate as faults get worse (MuJoCo)."""
    rows = []
    sweeps = [
        ("action delay (steps of 20 ms)", "action_delay", [0, 10, 20, 30, 40]),
        ("joint noise (rad std)", "joint_noise", [0.0, 0.03, 0.06, 0.09, 0.12]),
        ("weak shoulder_lift (gain scale)", "weak_joint", [1.0, 0.05, 0.02, 0.01, 0.005]),
    ]
    for label, fault, levels in sweeps:
        for level in levels:
            ok, durations = 0, []
            for seed in range(reps):
                w = rw.launch(seed=seed, settings=_settings("off"))
                w.robot.reset_to()
                w.faults.jitter("cube", 0.02, yaw_std=0.5)
                if fault == "action_delay" and level:
                    w.faults.action_delay(level)
                elif fault == "joint_noise" and level:
                    w.faults.joint_noise(level)
                elif fault == "weak_joint" and level != 1.0:
                    w.faults.weak_joint("shoulder_lift", level)
                r = w.robot.run_policy(
                    ScriptedPickPlace(), until=condition(w.scene["cube"], "to_be_inside", "bin"), timeout=20, privileged=True
                )
                ok += r.success
                if r.success:
                    durations.append(r.sim_seconds)
                w.close()
            lo, hi = wilson(ok, reps)
            rows.append(
                {
                    "fault": label,
                    "level": level,
                    "passed": ok,
                    "n": reps,
                    "ci_low": lo,
                    "ci_high": hi,
                    "median_task_s": statistics.median(durations) if durations else None,
                }
            )
    return rows


def bench_micro(reps=2000):
    """Per-call costs of the core building blocks."""
    w = rw.launch(settings=_settings("off"))
    w.robot.reset_to()
    kin = w.robot.kin
    rng = np.random.default_rng(0)
    targets = [np.array([rng.uniform(0.16, 0.26), rng.uniform(-0.08, 0.08), rng.uniform(0.02, 0.07)]) for _ in range(200)]
    t0 = time.perf_counter()
    for t in targets:
        kin.ik(t, w.robot.home_q, (0, 0, -1), yaw=0.0)
    ik_ms = (time.perf_counter() - t0) / len(targets) * 1000
    t0 = time.perf_counter()
    for _ in range(reps):
        w.step()
    base = (time.perf_counter() - t0) / reps
    expect(w.scene["cube"]).always.to_be_near(w.scene["cube"].position, tol=0.05)
    expect(w.robot).always.to_have_no_collisions()
    t0 = time.perf_counter()
    for _ in range(reps):
        w.step()
    inv = (time.perf_counter() - t0) / reps
    w.close()
    return {"ik_solve_ms": ik_ms, "step_us": base * 1e6, "step_with_2_invariants_us": inv * 1e6}


def write_markdown(res: dict, path: Path):
    e = res["env"]
    L = [
        "# Benchmarks",
        "",
        f"Generated by `python bench/run.py` on {e['date']}: {e['cpu']} ({e['cores']} threads), {e['os']}, "
        f"Python {e['python']}, MuJoCo {e['mujoco']}, robowright {e['robowright']}. Raw data: "
        "[`bench/results/results.json`](bench/results/results.json).",
        "",
        "These are framework costs on the default robot (SO-101). One control step is 20 ms of simulated time "
        "(50 Hz) and four physics substeps; the robot x engine matrix is in MATRIX.md. Single-process throughput "
        "numbers moved by about 20% between two consecutive runs on this machine; treat them as rough.",
        "",
    ]
    L += [
        "## Simulation throughput",
        "",
        "The arm tracks a moving target; tracing records joints, objects and contacts every step.",
        "",
        "| backend | mode | control steps/s | x real time | trace size (KB per simulated s) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in res["throughput"]:
        size = f"{r['trace_kb_per_sim_s']:.0f}" if r["trace_kb_per_sim_s"] else "-"
        L.append(f"| {r['backend']} | {r['mode']} | {r['steps_per_s']:,.0f} | {r['realtime_factor']:.0f}x | {size} |")
    L += [
        "",
        "## One pick-and-place test",
        "",
        "`robot.pick`, `robot.place` and three expectations, tracing off. Median of the runs.",
        "",
        "| backend | wall time | simulated time | x real time |",
        "|---|---:|---:|---:|",
    ]
    for r in res["pick_place"]:
        L.append(f"| {r['backend']} | {r['median_wall_s'] * 1000:.0f} ms | {r['sim_s']:.2f} s | {r['realtime_factor']:.0f}x |")
    L += [
        "",
        "## Parallel test runs",
        "",
        "`pytest bench/suite`: 64 randomised pick-and-place tests (about 3.5 s of simulated robot time each), MuJoCo, "
        "tracing off, with pytest-xdist workers. Wall time includes interpreter and worker start-up.",
        "",
        "| workers | wall time | tests/s | speed-up |",
        "|---:|---:|---:|---:|",
    ]
    for r in res["parallel"]:
        L.append(f"| {r['workers']} | {r['wall_s']:.1f} s | {64 / r['wall_s']:.1f} | {r['speedup']:.1f}x |")
    L += [
        "",
        "## Deterministic replay",
        "",
        "Each run randomises the cube, delays commands by 2 steps and shoves the cube mid-task; the trace is then re-simulated "
        "from its recorded commands and compared at every step.",
        "",
        "| backend | runs | bit-identical | steps compared | max joint error | max object error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in res["replay"]:
        L.append(
            f"| {r['backend']} | {r['runs']} | {r['bit_identical']}/{r['runs']} | {r['steps_replayed']:,} | "
            f"{r['max_joint_error_rad']:.1e} rad | {r['max_object_error_m']:.1e} m |"
        )
    L += [
        "",
        "## Same test, every engine",
        "",
        "Randomised pick-and-place (cube position +/-2 cm, yaw +/-0.5 rad), identical seeds on every engine.",
        "",
        "| backend | passed | 95% CI |",
        "|---|---:|---:|",
    ]
    for r in res["parity"]:
        if r["backend"] == "difference":
            continue
        L.append(f"| {r['backend']} | {r['passed']}/{r['n']} | {r['ci_low']:.0%}-{r['ci_high']:.0%} |")
    diff = [r for r in res["parity"] if r["backend"] == "difference"]
    if diff:
        d = diff[0]
        L += [
            "",
            f"Where the cube ends up differs between engines by {d['median_final_cube_distance_mm']:.1f} mm (median) and "
            f"{d['max_final_cube_distance_mm']:.1f} mm (worst) for the same seed.",
        ]
    L += [
        "",
        "## Fault sensitivity of the reference policy",
        "",
        "`ScriptedPickPlace` (closed-loop, chunked) under increasing faults, MuJoCo, randomised cube. This is the kind of curve "
        "`@pytest.mark.trials` is meant to guard: a regression shows up as the curve moving left.",
        "",
        "| fault | level | passed | 95% CI | median task time |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in res["faults"]:
        t = f"{r['median_task_s']:.1f} s" if r["median_task_s"] else "-"
        L.append(f"| {r['fault']} | {r['level']} | {r['passed']}/{r['n']} | {r['ci_low']:.0%}-{r['ci_high']:.0%} | {t} |")
    m = res["micro"]
    L += [
        "",
        "## Building blocks",
        "",
        "| operation | cost |",
        "|---|---:|",
        f"| IK solve (5-DOF, position + approach + yaw) | {m['ik_solve_ms']:.2f} ms |",
        f"| control step, no tracing | {m['step_us']:.0f} us |",
        f"| control step with 2 `expect(...).always` invariants | {m['step_with_2_invariants_us']:.0f} us |",
        "",
    ]
    path.write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--markdown-only", action="store_true", help="re-render BENCHMARKS.md from results.json")
    ap.add_argument("--steps", help="comma-separated stages to (re)run, merged into the existing results.json")
    a = ap.parse_args()
    if a.markdown_only:
        write_markdown(json.loads((OUT / "results.json").read_text()), ROOT / "BENCHMARKS.md")
        return
    reps = 5 if a.quick else 20
    backends = available()
    path = OUT / "results.json"
    res = json.loads(path.read_text()) if a.steps and path.exists() else {}
    res["env"] = env()
    steps = [
        ("throughput", lambda: bench_throughput(backends, 5.0 if a.quick else 20.0)),
        ("pick_place", lambda: bench_pick_place(backends, reps)),
        ("parallel", lambda: bench_parallel([1, 2, 4])),
        ("replay", lambda: bench_replay(backends, reps)),
        ("parity", lambda: bench_parity(backends, 10 if a.quick else 50)),
        ("faults", lambda: bench_faults(5 if a.quick else 20)),
        ("micro", bench_micro),
    ]
    if a.steps:
        steps = [s for s in steps if s[0] in a.steps.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in steps:
        t0 = time.perf_counter()
        res[name] = fn()
        print(f"{name}: {time.perf_counter() - t0:.1f}s", flush=True)
        path.write_text(json.dumps(res, indent=2))  # after every stage: a late crash loses one stage, not the run
    write_markdown(res, ROOT / "BENCHMARKS.md")
    print((ROOT / "BENCHMARKS.md").read_text())


if __name__ == "__main__":
    main()
