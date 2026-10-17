"""``robowright test --watch``: run the tests, then run them again whenever a file they could depend on changes.

What runs again: when only test files changed, those files (with the run's own options); when anything else changed
(the code under test, a robot's model file, robowright.toml), the whole run. Enter runs everything again at any time.
Each run is a fresh ``python -m pytest``, so nothing an engine or a module kept from the last run carries over.
Files are watched by polling their modification times, which needs nothing installed and works on every platform.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

WATCHED = {".py", ".toml", ".ini", ".cfg", ".yaml", ".yml", ".json", ".xml", ".urdf", ".xacro", ".sdf", ".mjcf", ".srdf"}
SKIPPED_DIRS = {"__pycache__", "node_modules", "site", "build", "dist", "robowright-traces", "__snapshots__"}


def snapshot(root: Path) -> dict[str, float]:
    """Every watched file under ``root`` and when it last changed. Hidden directories (.git, .venv) and generated
    ones are left out."""
    out: dict[str, float] = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in SKIPPED_DIRS and not x.endswith(".egg-info")]
        for f in files:
            if Path(f).suffix in WATCHED:
                p = os.path.join(d, f)
                try:
                    out[p] = os.stat(p).st_mtime
                except OSError:  # gone between listing and looking
                    pass
    return out


def changed(before: dict[str, float], after: dict[str, float]) -> list[str]:
    """Files added, removed or modified, in order."""
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


def _is_test_file(path: str) -> bool:
    name = Path(path).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _is_path(arg: str) -> bool:
    return not arg.startswith("-") and os.path.exists(arg.split("::")[0])


def plan(args: list[str], files: list[str]) -> list[str]:
    """The pytest arguments for a run after ``files`` changed: the changed test files (which still exist) with the
    run's options, when nothing but test files changed; otherwise the whole run again."""
    if files and all(_is_test_file(f) for f in files):
        alive = [os.path.relpath(f) for f in files if os.path.exists(f)]
        if alive:
            return [a for a in args if not _is_path(a)] + alive
    return list(args)


def _run_pytest(args: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", "pytest", *args])


def _enter_presses() -> threading.Event:
    """Set whenever Enter is pressed (when stdin is a terminal)."""
    pressed = threading.Event()

    def read():
        for _ in sys.stdin:
            pressed.set()

    if sys.stdin is not None and sys.stdin.isatty():
        threading.Thread(target=read, daemon=True, name="robowright-watch-enter").start()
    return pressed


def watch(
    args: list[str],
    root: Path | str = ".",
    *,
    run: Callable[[list[str]], int] = _run_pytest,
    interval: float = 0.5,
    settle: float = 0.2,
    max_runs: int | None = None,
    out=None,
) -> int:
    """Run ``pytest args``, then again on every change under ``root``, until Ctrl-C (or ``max_runs`` runs). Returns
    the last run's exit status."""
    out = out or sys.stdout
    root = Path(root)
    enter = _enter_presses()
    before = snapshot(root)
    status, runs, now = 0, 0, list(args)
    try:
        while True:
            status = run(now)
            runs += 1
            if max_runs is not None and runs >= max_runs:
                return status
            result = "passed" if status == 0 else "failed"
            print(f"\nrobowright: {result}. Watching for changes (Enter runs everything, Ctrl-C stops).", file=out, flush=True)
            while True:
                time.sleep(interval)
                if enter.is_set():
                    enter.clear()
                    now = list(args)
                    print("robowright: running everything again", file=out, flush=True)
                    break
                files = changed(before, snapshot(root))
                if files:
                    time.sleep(settle)  # an editor may write a file in more than one go: take the whole save
                    after = snapshot(root)
                    files = changed(before, after)
                    before = after
                    now = plan(args, files)
                    shown = ", ".join(os.path.relpath(f) for f in files[:5]) + (f" and {len(files) - 5} more" if len(files) > 5 else "")
                    print(f"robowright: {shown} changed", file=out, flush=True)
                    break
    except KeyboardInterrupt:
        return status
