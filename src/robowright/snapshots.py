"""Image snapshots: ``expect(world).to_match_snapshot("tidy.png")``, Playwright's ``toHaveScreenshot``.

A camera's view is compared with a baseline image kept next to the test, in
``__snapshots__/<test file>/<name>-<engine>-<robot>.png``: engines draw the same scene differently,
so each engine and robot has its own. The first run writes the baseline and fails, so that a
missing baseline is never a pass; ``pytest --rw-update-snapshots`` (or ``ROBOWRIGHT_UPDATE_SNAPSHOTS=1``)
writes them all again. On a mismatch the actual image and a diff (changed pixels in red) are written
next to the traces, and the failure names them.
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import numpy as np

UPDATE_ENV = "ROBOWRIGHT_UPDATE_SNAPSHOTS"
_PACKAGE = Path(__file__).resolve().parent


def updating() -> bool:
    return os.environ.get(UPDATE_ENV, "") not in ("", "0")


def _caller_file() -> Path:
    """The first file up the stack that is not robowright's own: the test that asked."""
    for frame in inspect.stack()[1:]:
        f = Path(frame.filename).resolve()
        if _PACKAGE not in f.parents and f.suffix == ".py":
            return f
    return Path.cwd() / "snapshots.py"


def _slug(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s).strip("_") or "x"


def baseline_path(name: str, engine: str, robot: str, test_file: Path | None = None) -> Path:
    f = test_file or _caller_file()
    stem, suffix = Path(name).stem, Path(name).suffix or ".png"
    return f.parent / "__snapshots__" / f.stem / f"{_slug(stem)}-{_slug(engine)}-{_slug(robot)}{suffix}"


def compare(actual: np.ndarray, expected: np.ndarray, threshold: float) -> tuple[int, np.ndarray | None]:
    """The number of pixels that differ by more than ``threshold`` (0..1 of full scale) in any channel,
    and a diff image (the expected image greyed, changed pixels in red); None if the sizes differ."""
    if actual.shape != expected.shape:
        return -1, None
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max(axis=-1)
    changed = delta > threshold * 255
    grey = expected[..., :3].mean(axis=-1, keepdims=True).astype(np.uint8)
    diff = np.repeat(grey // 3 + 170, 3, axis=-1)  # faded, so the red stands out
    diff[changed] = (220, 30, 30)
    return int(changed.sum()), diff


def read(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def write(path: Path, image: np.ndarray) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image[..., :3])).save(path)
    return path
