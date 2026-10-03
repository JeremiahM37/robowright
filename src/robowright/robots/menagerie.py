"""Fetch robot models from MuJoCo Menagerie on first use.

Only the directories a test needs are checked out (a sparse, blobless clone
pinned to one commit), so using a robot costs a few megabytes rather than the
whole 2 GB repository. Set ``ROBOWRIGHT_MENAGERIE`` to use an existing checkout.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "feadf76d42f8a2162426f7d226a3b539556b3bf5"


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "robowright" / "menagerie"


def root() -> Path | None:
    """An existing checkout to read from, if there is one."""
    env = os.environ.get("ROBOWRIGHT_MENAGERIE")
    if env:
        return Path(env)
    for p in (cache_dir(), Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie"):
        if (p / ".git").exists() or (p / "README.md").exists():
            return p
    return None


def path(relative: str) -> Path:
    """Absolute path of ``relative`` (e.g. ``franka_emika_panda/panda.xml``), fetching its directory if needed."""
    top = relative.split("/")[0]
    r = root()
    if r is not None and (r / relative).exists():
        return r / relative
    if os.environ.get("ROBOWRIGHT_MENAGERIE"):
        raise FileNotFoundError(f"{relative} not found under ROBOWRIGHT_MENAGERIE={r}")
    _fetch(top)
    return cache_dir() / relative


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _fetch(directory: str) -> None:
    d = cache_dir()
    if not (d / ".git").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--filter=blob:none", "--no-checkout", "--sparse", REPO, str(d))
        _git("sparse-checkout", "set", "--no-cone", "/assets/", cwd=d)
    listed = subprocess.run(["git", "sparse-checkout", "list"], cwd=d, capture_output=True, text=True).stdout.split()
    want = f"/{directory}/"
    if want not in listed:
        _git("sparse-checkout", "add", want, cwd=d)
    _git("checkout", "-q", COMMIT, cwd=d)
