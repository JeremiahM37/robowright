"""Builds the command-line reference from the commands' own ``--help``, so it says what the installed version does."""

from __future__ import annotations

import contextlib
import io
import re
import subprocess
import sys

from mkdocs.structure.files import File

# `test` hands its arguments to pytest, so its --help is pytest's (the options are below, under pytest)
SKIP = {"test"}


def _help(argv: list[str]) -> str:
    from robowright.cli import main

    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
        main(argv)
    return out.getvalue().rstrip()


def _commands(top: str) -> list[str]:
    m = re.search(r"\{([a-z0-9,-]+)\}", top)
    return [c for c in m.group(1).split(",") if c not in SKIP] if m else []


def _pytest_options() -> str:
    """The robowright section of ``pytest --help``."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--help", "-p", "no:cacheprovider"], capture_output=True, text=True, check=True
    ).stdout
    m = re.search(r"^robowright:\n(.*?)(?=^\S)", out, flags=re.S | re.M)
    if not m:
        raise RuntimeError("pytest --help has no robowright section: is the plugin installed?")
    return m.group(1).rstrip()


def page() -> str:
    top = _help(["--help"])
    parts = [
        "# Command line",
        "",
        "Generated from the commands' own `--help` when this site was built.",
        "",
        "## pytest",
        "",
        "robowright's pytest plugin is loaded on install; `robowright test` is `pytest` with it. Its options:",
        "",
        "```text",
        _pytest_options(),
        "```",
        "",
        "## robowright",
        "",
        "```text",
        top,
        "```",
    ]
    for cmd in _commands(top):
        parts += ["", f"### robowright {cmd}", "", "```text", _help([cmd, "--help"]), "```"]
    return "\n".join(parts) + "\n"


def on_files(files, config):
    files.append(File.generated(config, "api/cli.md", content=page()))
    return files
