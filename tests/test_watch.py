"""robowright test --watch: what runs again after which files change."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from robowright.watch import changed, plan, snapshot, watch


def _project(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_arm.py").write_text("def test_arm():\n    assert True\n")
    (tmp_path / "tests" / "test_base.py").write_text("def test_base():\n    assert True\n")
    (tmp_path / "controller.py").write_text("GAIN = 1\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("")
    (tmp_path / "robowright-traces").mkdir()
    (tmp_path / "robowright-traces" / "t.json").write_text("{}")
    return tmp_path


def _touch(p: Path, text: str):
    p.write_text(text)
    t = time.time() + 1  # past the last snapshot even on a coarse clock
    os.utime(p, (t, t))


def test_what_is_watched(tmp_path):
    root = _project(tmp_path)
    seen = {Path(p).relative_to(root).as_posix() for p in snapshot(root)}
    assert seen == {"tests/test_arm.py", "tests/test_base.py", "controller.py"}  # not .venv, not the traces


def test_a_change_an_addition_and_a_removal_are_each_seen(tmp_path):
    root = _project(tmp_path)
    before = snapshot(root)
    _touch(root / "controller.py", "GAIN = 2\n")
    (root / "tests" / "test_base.py").unlink()
    (root / "robot.urdf").write_text("<robot/>")
    assert [Path(p).name for p in changed(before, snapshot(root))] == ["controller.py", "robot.urdf", "test_base.py"]


def test_only_changed_tests_run_again_unless_something_else_changed(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.chdir(root)
    args = ["tests", "-x", "--rw-backend", "mujoco"]
    arm = str(root / "tests" / "test_arm.py")
    assert plan(args, [arm]) == ["-x", "--rw-backend", "mujoco", os.path.join("tests", "test_arm.py")]
    assert plan(args, [arm, str(root / "controller.py")]) == args, "the code under test changed: everything"
    assert plan(args, [str(root / "tests" / "test_gone.py")]) == args, "a deleted test file: everything"


def test_a_save_runs_the_tests_again(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.chdir(root)
    runs = []

    def run(args):
        runs.append(args)
        if len(runs) == 1:  # the user edits a test once the first run is done
            threading.Timer(0.3, _touch, (root / "tests" / "test_arm.py", "def test_arm():\n    assert False\n")).start()
        return 0

    status = watch(["tests", "-q"], root, run=run, interval=0.05, settle=0.05, max_runs=2, out=open(os.devnull, "w"))
    assert status == 0 and runs == [["tests", "-q"], ["-q", os.path.join("tests", "test_arm.py")]]


def test_the_command_runs_pytest_and_reports(tmp_path):
    root = _project(tmp_path)
    code = "import sys; from robowright.watch import watch; sys.exit(watch(['tests', '-q', '-p', 'no:cacheprovider'], '.', max_runs=1))"
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 passed" in r.stdout
