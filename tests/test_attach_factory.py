"""``--factory``: the function that builds an attached run's environment, given on the command line.

The runs are the hand-built environment of test_attach_replay (its ToyEnv, a gymnasium-shaped stand-in: gymnasium
is not in the project's venv), recorded without a spec, so only a factory can rebuild them. Generated tests run in
a fresh process; the factory module is written into their directory, so the test does not depend on this one.
"""

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
from test_attach import CONTROL_STEPS, XML, ToyEnv
from test_attach_replay import _record

import robowright as rw
from robowright.cli import main
from robowright.codegen import companion, generate
from robowright.replay import load_factory, replay

FACTORY_HEADER = """import mujoco
import numpy as np

XML = %r
CONTROL_STEPS = %r


"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _toy_factory(tmp_path: Path, name: str = "make_toy.py") -> Path:
    """A factory module that builds the same ToyEnv the recording used."""
    return _write(
        tmp_path / name, FACTORY_HEADER % (XML, CONTROL_STEPS) + inspect.getsource(ToyEnv) + "\n\ndef make():\n    return ToyEnv()\n"
    )


def _run_fresh(test_file: Path) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    src = str(Path(rw.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [src, env.get("PYTHONPATH", "")] if p)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
        cwd=test_file.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_load_factory_says_what_is_wrong_with_the_spec(tmp_path):
    with pytest.raises(ValueError, match="module:function or path/to/file.py:function"):
        load_factory("no_colon_here")
    with pytest.raises(ValueError, match="cannot import the factory's module 'no_such_module_for_robowright'"):
        load_factory("no_such_module_for_robowright:build")
    with pytest.raises(ValueError, match="does not exist"):
        load_factory(f"{tmp_path / 'missing.py'}:build")
    f = _toy_factory(tmp_path)
    with pytest.raises(ValueError, match="has no 'nothing'"):
        load_factory(f"{f}:nothing")
    _write(tmp_path / "not_callable.py", "build = 3\n")
    with pytest.raises(ValueError, match="is not callable"):
        load_factory(f"{tmp_path / 'not_callable.py'}:build")
    _write(tmp_path / "needs_args.py", "def build(seed):\n    return None\n")
    with pytest.raises(ValueError, match="must take no arguments"):
        load_factory(f"{tmp_path / 'needs_args.py'}:build")
    _write(tmp_path / "raises.py", "raise RuntimeError('boom at import')\n")
    with pytest.raises(ValueError, match="RuntimeError: boom at import"):
        load_factory(f"{tmp_path / 'raises.py'}:build")


def test_load_factory_returns_the_function_from_a_file_and_from_a_module(tmp_path, monkeypatch):
    f = _toy_factory(tmp_path)
    assert type(load_factory(f"{f}:make")()).__name__ == "ToyEnv"  # the file's own class, not this module's
    _write(tmp_path / "fac_by_module.py", "def build():\n    return 'built'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert load_factory("fac_by_module:build")() == "built"


def test_replay_with_a_factory_rebuilds_a_run_that_has_no_spec(tmp_path, capsys):
    path = _record(tmp_path, spec_id=None)
    with pytest.raises(ValueError, match="cannot rebuild this run's environment"):
        replay(path)  # without a factory a hand-built run is refused, with the reason
    code = main(["replay", str(path), "--factory", f"{_toy_factory(tmp_path)}:make"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "bit-identical" in out


def test_replay_with_a_factory_that_returns_nothing_says_so(tmp_path, capsys):
    path = _record(tmp_path, spec_id=None)
    _write(tmp_path / "returns_none.py", "def build():\n    return None\n")
    code = main(["replay", str(path), "--factory", f"{tmp_path / 'returns_none.py'}:build"])
    assert code == 2
    assert "make_env returned None" in capsys.readouterr().err


def test_codegen_with_a_factory_file_writes_a_test_that_passes_in_a_fresh_process(tmp_path, capsys):
    path = _record(tmp_path, spec_id=None)
    out = tmp_path / "test_toy_regression.py"
    code = main(["codegen", str(path), "-o", str(out), "--factory", f"{_toy_factory(tmp_path)}:make"])
    assert code == 0, capsys.readouterr().err
    text = out.read_text()
    assert "# Stub:" not in text and "make_env = _load_factory()" in text
    r = _run_fresh(out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 passed" in r.stdout


def test_codegen_with_a_module_factory_imports_it(tmp_path, monkeypatch):
    path = _record(tmp_path, spec_id=None)
    _write(
        tmp_path / "toy_factory_module.py",
        FACTORY_HEADER % (XML, CONTROL_STEPS) + inspect.getsource(ToyEnv) + "\n\ndef make():\n    return ToyEnv()\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    text = generate(path, test_name="test_toy_module_regression", factory="toy_factory_module:make")
    assert "from toy_factory_module import make as make_env" in text
    out = tmp_path / "test_toy_module_regression.py"
    out.write_text(text)
    if (extra := companion(path, "test_toy_module_regression")) is not None:  # a long run's actions, beside the test
        out.with_name(extra[0]).write_bytes(extra[1])
    r = _run_fresh(out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 passed" in r.stdout
