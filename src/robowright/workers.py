"""How many test workers this machine can run: ``pytest -n auto``.

pytest-xdist's own ``auto`` starts one worker per CPU, which is wrong both ways for physics
tests. A 16 GB laptop with 8 cores cannot hold 8 Genesis workers, and a worker's memory, not
its cores, is what runs out first; on a large machine the cores are the limit. The count here
is the smaller of the two, from what each engine's workers were measured to use, and from what
this process may actually use: the CPU quota and memory limit of its cgroup (a container, a
``systemd-run`` scope) as well as the machine's free memory.

``PYTEST_XDIST_AUTO_NUM_WORKERS`` overrides it, and ``-n N`` bypasses it.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# Per worker, running every robot's tests: (CPU cores kept busy, peak resident memory in GB).
# Measured on a 32-thread Ryzen running every robot's contract tests (and, for MuJoCo and
# PyBullet, the examples) with 4 to 9 workers, rounded up: a worker that runs more robots peaks
# higher, so memory is from the 4-worker runs. Isaac Sim: 2 workers on a 16-thread Ryzen 7
# 9800X3D with an RTX 5080.
WORKER_COST = {
    "mujoco": (1.0, 2.7),
    "pybullet": (1.0, 2.2),
    "drake": (1.1, 2.3),
    "genesis": (1.3, 4.6),
    "isaac": (4.3, 5.5),
}
HEADROOM_GB = 2.0  # left for everything else on the machine
MEMORY_SHARE = 0.85  # of what remains


def auto_workers(backends: list[str]) -> int:
    """Workers for a run on ``backends``: as many as both the cores and the memory allow."""
    cpu, mem = (max(WORKER_COST.get(b, WORKER_COST["mujoco"])[i] for b in backends or ["mujoco"]) for i in (0, 1))
    by_cpu = math.floor(cpu_budget() / cpu)
    by_mem = math.floor(max(0.0, memory_budget_gb() - HEADROOM_GB) * MEMORY_SHARE / mem)
    return max(1, min(by_cpu, by_mem))


def cpu_budget() -> float:
    """Cores this process may use: its CPU affinity, capped by any cgroup CPU quota."""
    try:
        n = float(len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS, Windows
        n = float(os.cpu_count() or 1)
    for d in _cgroup_dirs():
        try:
            quota, period = (d / "cpu.max").read_text().split()[:2]
        except (OSError, ValueError):
            continue
        if quota != "max":
            n = min(n, int(quota) / int(period))
    return n


def memory_budget_gb() -> float:
    """Memory this process may still use: the machine's available memory, capped by cgroup limits."""
    avail = _meminfo_available()
    for d in _cgroup_dirs():
        try:
            limit = (d / "memory.max").read_text().strip()
            used = int((d / "memory.current").read_text())
        except (OSError, ValueError):
            continue
        if limit != "max":
            avail = min(avail, int(limit) - used)
    return avail / 2**30


def _meminfo_available() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil

        return float(psutil.virtual_memory().available)
    except ImportError:
        return 8 * 2**30  # unknown: assume a modest machine


def _cgroup_dirs() -> list[Path]:
    """This process's cgroup (v2) and its ancestors, innermost first; empty off Linux."""
    try:
        rel = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    except (OSError, StopIteration):
        return []
    root = Path("/sys/fs/cgroup")
    d = root / rel.strip().lstrip("/")
    out = []
    while True:
        out.append(d)
        if d == root:
            return out
        d = d.parent
