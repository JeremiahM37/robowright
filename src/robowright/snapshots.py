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


MASK_COLOR = (255, 0, 255)


def compare(actual: np.ndarray, expected: np.ndarray, threshold: float, mask: np.ndarray | None = None) -> tuple[int, np.ndarray | None]:
    """The number of pixels that differ by more than ``threshold`` (0..1 of full scale) in any channel,
    and a diff image (the expected image greyed, changed pixels in red, masked ones magenta); None if the sizes
    differ. Pixels where ``mask`` is true are not compared."""
    if actual.shape != expected.shape:
        return -1, None
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max(axis=-1)
    changed = delta > threshold * 255
    if mask is not None:
        changed &= ~mask
    grey = expected[..., :3].mean(axis=-1, keepdims=True).astype(np.uint8)
    diff = np.repeat(grey // 3 + 170, 3, axis=-1)  # faded, so the red stands out
    diff[changed] = (220, 30, 30)
    if mask is not None:
        diff[mask] = MASK_COLOR
    return int(changed.sum()), diff


def box_of(world, subject, camera: str, width: int, height: int, margin: float = 0.01, pad: int = 2):
    """Where ``subject`` (a scene object or its name) is in ``camera``'s image: its bounding box, ``margin`` metres
    wider, projected through the camera robowright placed (the same on every engine), as pixel bounds
    ``(u0, v0, u1, v1)``; None when it is behind the camera."""
    from .locators import as_subject

    cam = next(c for c in world.spec.cameras if c.name == camera)
    pos, look = np.asarray(cam.pos, float), np.asarray(cam.lookat, float)
    fwd = (look - pos) / np.linalg.norm(look - pos)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = (height / 2) / np.tan(np.radians(cam.fovy) / 2)  # fovy is vertical
    h = as_subject(world, subject)
    if not hasattr(h, "bounds"):
        raise TypeError(f"a snapshot region or mask takes scene objects or their names, not {h!r}")
    lo, hi = h.bounds()
    lo, hi = lo - margin, hi + margin
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - pos
    depth = corners @ fwd
    if np.any(depth <= 1e-6):
        return None
    u = width / 2 + f * (corners @ right) / depth
    v = height / 2 - f * (corners @ up) / depth
    return int(np.floor(u.min()) - pad), int(np.floor(v.min()) - pad), int(np.ceil(u.max()) + pad), int(np.ceil(v.max()) + pad)


def mask_of(world, subjects, camera: str, width: int, height: int, margin: float = 0.01, pad: int = 2) -> np.ndarray:
    """Where ``subjects`` (scene objects, or their names) are in ``camera``'s image, as a boolean image: each one's
    :func:`box_of`, ``margin`` metres wider for the shadow and shading it casts around itself."""
    out = np.zeros((height, width), bool)
    for s in subjects:
        box = box_of(world, s, camera, width, height, margin, pad)
        if box is None:
            continue  # behind the camera
        u0, v0, u1, v1 = max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])
        if u0 < u1 and v0 < v1:
            out[v0:v1, u0:u1] = True
    return out


def crop(image: np.ndarray, center: tuple[float, float], size: tuple[int, int]) -> np.ndarray:
    """A ``size`` (width, height) window of ``image`` centred on ``center`` (u, v); outside the image it is black,
    so the window is always that size."""
    w, h = size
    u0, v0 = int(round(center[0] - w / 2)), int(round(center[1] - h / 2))
    out = np.zeros((h, w, *image.shape[2:]), image.dtype)
    su0, sv0 = max(0, u0), max(0, v0)
    su1, sv1 = min(image.shape[1], u0 + w), min(image.shape[0], v0 + h)
    if su0 < su1 and sv0 < sv1:
        out[sv0 - v0 : sv1 - v0, su0 - u0 : su1 - u0] = image[sv0:sv1, su0:su1]
    return out


def read(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def write(path: Path, image: np.ndarray) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image[..., :3])).save(path)
    return path


# State snapshots -----------------------------------------------------------------------------------------------
# ``expect(world).to_match_state_snapshot("placed")``, Playwright's ``toMatchAriaSnapshot`` for a world: where every
# object is, the robot's joints and, for a robot that moves about, its base, kept as text a person can read and edit.
# Only the entries the baseline has are compared: delete a line to stop checking it.

STATE_SUFFIX = ".state.txt"


def state_of(world) -> dict[str, dict[str, list[float]]]:
    """The world's state as the snapshot keeps it: object positions (m), joint positions (rad, or m for a slide),
    and the base's position for a mobile or legged robot."""
    b = world.backend
    objects = {o.name: [float(x) for x in b.object_pose(o.name)[0]] for o in world.spec.objects}
    joints = {j: [float(q)] for j, q in zip(b.joint_names, b.qpos())}
    out = {"objects": objects, "joints": joints}
    if getattr(getattr(b, "robot_model", None), "family", "arm") in ("mobile", "legged"):
        out["base"] = {"position": [float(x) for x in b.base_pose()[0]]}
    return out


def format_state(state: dict, header: str) -> str:
    lines = [f"# {header}", "# Positions in m, joints in rad (m for a slide). Delete a line to stop checking it."]
    for section, entries in state.items():
        if not entries:
            continue
        lines.append(f"{section}:")
        for name, v in entries.items():
            lines.append(f"  {name}: " + (f"{v[0]:.4f}" if len(v) == 1 else "[" + ", ".join(f"{x:.4f}" for x in v) + "]"))
    return "\n".join(lines) + "\n"


def parse_state(text: str, path: Path | str = "snapshot") -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    section = None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        m = re.fullmatch(r"(\w+):", line)
        if m:
            section = out.setdefault(m.group(1), {})
            continue
        m = re.fullmatch(r"\s+(.+?):\s*\[?([^\]]*)\]?", line)
        if not m or section is None:
            raise ValueError(f"{path}:{n}: cannot read {raw.strip()!r} (want 'section:' or '  name: value')")
        try:
            section[m.group(1)] = [float(x) for x in m.group(2).split(",")]
        except ValueError:
            raise ValueError(f"{path}:{n}: {raw.strip()!r} is not a number or a list of numbers") from None
    return out


def compare_state(actual: dict, expected: dict, tol: float, joint_tol: float) -> list[str]:
    """What differs, one line per entry the baseline has: positions further apart than ``tol`` (m), joints than
    ``joint_tol``, and entries the world no longer has."""
    out = []
    for section, entries in expected.items():
        have = actual.get(section, {})
        limit = joint_tol if section == "joints" else tol
        for name, want in entries.items():
            if name not in have:
                out.append(f"{section}.{name}: in the snapshot, not in this world")
                continue
            got = have[name]
            if len(got) != len(want):
                out.append(f"{section}.{name}: {len(got)} values, the snapshot has {len(want)}")
                continue
            d = float(np.linalg.norm(np.subtract(got, want)))
            if d > limit:
                fmt = ", ".join
                out.append(
                    f"{section}.{name}: ({fmt(f'{x:.3f}' for x in got)}), snapshot ({fmt(f'{x:.3f}' for x in want)}): "
                    f"{d:.3f} off, {limit:g} allowed"
                )
    return out
