"""``pytest --rw-only-changed [REF]``: run only the test files a change since a git commit can affect, as
Playwright's ``--only-changed``.

A change counts against the working tree (uncommitted and untracked files included), from ``REF`` (default
``HEAD``: what you have not committed yet). Which tests it affects:

- a changed test file: that file;
- a changed Python module: every test file that imports it, directly or through other modules of the project;
- anything else a run can read (``conftest.py``, a module pytest loaded as a plugin, ``robowright.toml``,
  ``pytest.ini``, a robot's model file): every test, since there is no telling which test it touches.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

from .watch import SKIPPED_DIRS, WATCHED, _is_test_file

ALL = None  # the change could affect any test


class GitError(RuntimeError):
    pass


def changed_files(ref: str, cwd: Path) -> list[Path]:
    """Files changed between ``ref`` and the working tree, untracked ones included, as absolute paths."""

    def git(*args: str) -> str:
        try:
            r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            raise GitError("git is not installed") from None
        if r.returncode:
            raise GitError(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout

    top = Path(git("rev-parse", "--show-toplevel").strip())
    names = git("diff", "--name-only", "--no-renames", ref, "--").splitlines()
    names += git("ls-files", "--others", "--exclude-standard").splitlines()
    return sorted({(top / n).resolve() for n in names if n})


def affected(changed: list[Path], root: Path, plugins: set[Path] = frozenset()) -> set[Path] | None:
    """The test files under ``root`` that the ``changed`` files can affect, or ``ALL`` (None) when one of them
    could affect any test. Only files a run reads count: an edited README changes nothing. ``plugins`` are the
    files pytest loaded as plugins, which every test runs under."""
    root = root.resolve()
    if any(p in plugins for p in changed):
        return ALL
    tests: set[Path] = set()
    modules: list[Path] = []
    for p in changed:
        if p.suffix not in WATCHED or not _inside(p, root):
            continue
        if p.suffix != ".py" or p.name == "conftest.py":
            return ALL
        if _is_test_file(str(p)):
            tests.add(p)
        else:
            modules.append(p)
    if not modules:
        return tests
    names = _module_names(root)
    imports = {p: _imports(p) for p in names}
    # Everything that imports a changed module, then everything that imports those, until nothing new.
    reached = set(modules)
    frontier = list(modules)
    while frontier:
        m = frontier.pop()
        as_names = set(names.get(m, ()))
        for p, (imported, files) in imports.items():
            if p not in reached and (m in files or as_names & imported):
                reached.add(p)
                frontier.append(p)
    return tests | {p for p in reached if _is_test_file(str(p))}


def _module_names(root: Path) -> dict[Path, list[str]]:
    """The project's Python files, each with every name it could be imported as: ``c``, ``b.c`` and ``a.b.c``
    for ``a/b/c.py``, since which is right depends on sys.path."""
    out: dict[Path, list[str]] = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not _skipped_dir(x)]
        for f in files:
            if f.endswith(".py"):
                p = Path(d, f).resolve()
                parts = list(p.relative_to(root).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                out[p] = [".".join(parts[i:]) for i in range(len(parts))]
    return out


def _imports(path: Path) -> tuple[set[str], set[Path]]:
    """What ``path`` imports: absolute module names (for ``from a import b`` both ``a`` and ``a.b``, as b may be
    a module), and the files its relative imports can point at."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return set(), set()
    names: set[str] = set()
    files: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level:
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            if node.module:
                base = base.joinpath(*node.module.split("."))
            for target in [base, *(base / a.name for a in node.names)]:
                files.update({target.with_suffix(".py"), target / "__init__.py"})
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names, files


def _skipped_dir(name: str) -> bool:
    return name.startswith(".") or name in SKIPPED_DIRS or name.endswith(".egg-info")


def _inside(p: Path, root: Path) -> bool:
    try:
        rel = p.relative_to(root)
    except ValueError:
        return False
    return not any(_skipped_dir(x) for x in rel.parts[:-1])
