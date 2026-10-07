"""Extending robowright from outside it: engines, robots, and a project's own robots.

Three ways in, none of which needs a change to robowright itself:

**A project's robots, by name** (no code). A ``robowright.toml`` next to your tests, or a
``[tool.robowright]`` table in ``pyproject.toml``, names model files::

    [robots.my_arm]
    file = "models/my_arm.urdf"     # MJCF, URDF or xacro (with "?arg=value" arguments)
    gripper_open = 0.04             # optional: any RobotModel field, over what detection works out

after which ``pytest --rw-robot my_arm`` runs every robot test on it. ``robowright robots add
FILE --name my_arm`` checks the file and writes the entry.

**A package of robots** (a ``robowright.robots`` entry point): a callable returning
:class:`~robowright.robots.RobotModel` objects, registered under their own names::

    [project.entry-points."robowright.robots"]
    my_lab = "my_lab_robots:robots"

**An engine, or real hardware** (a ``robowright.backends`` entry point): a
:class:`~robowright.backends.base.Backend` subclass, registered under the entry point's name
and chosen with ``--rw-backend``::

    [project.entry-points."robowright.backends"]
    my_engine = "my_engine_robowright:MyEngineBackend"

``robowright check --backend my_engine --robot so101`` then runs the contract every engine
meets (:mod:`robowright.contract`). See ``docs/extending.md``.
"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from pathlib import Path

BACKENDS = "robowright.backends"
ROBOTS = "robowright.robots"
CONFIG = "robowright.toml"


def _entry_points(group: str) -> list:
    from importlib.metadata import entry_points

    return sorted(entry_points(group=group), key=lambda ep: ep.name)


# --- engines ------------------------------------------------------------------------------------
def backend_names() -> list[str]:
    """Engines installed as plugins (not imported: an engine can take seconds to import)."""
    return [ep.name for ep in _entry_points(BACKENDS)]


def load_backend(name: str) -> bool:
    """Register the plugin engine ``name``; False if no plugin provides it."""
    from .backends.base import Backend, register

    for ep in _entry_points(BACKENDS):
        if ep.name != name:
            continue
        try:
            obj = ep.load()
        except Exception as e:  # noqa: BLE001 - whatever the plugin raised, say which plugin it was
            raise ImportError(f"the robowright backend plugin {name!r} ({ep.value}) failed to load: {e}") from e
        if isinstance(obj, type) and issubclass(obj, Backend):
            register(name)(obj)
        # Anything else is a module that registered itself with @register on import.
        return True
    return False


# --- robots -------------------------------------------------------------------------------------
@functools.cache
def plugin_robots() -> tuple[str, ...]:
    """Register the robots every ``robowright.robots`` plugin provides; their names.

    A plugin that fails is reported and skipped: one broken robot package should not stop the
    tests on every other robot.
    """
    from . import robots
    from .robots.model import RobotModel

    names = []
    for ep in _entry_points(ROBOTS):
        try:
            obj = ep.load()
            out = obj() if callable(obj) and not isinstance(obj, RobotModel) else obj
            models = [out] if isinstance(out, RobotModel) else list(out or ())
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"robowright robot plugin {ep.name!r} ({ep.value}) failed to load: {e}", stacklevel=2)
            continue
        for m in models:
            if not isinstance(m, RobotModel):
                warnings.warn(f"robowright robot plugin {ep.name!r} returned {type(m).__name__}, not a RobotModel", stacklevel=2)
                continue
            robots.register(m)
            names.append(m.name)
    return tuple(names)


def find_config(start: str | Path | None = None) -> tuple[Path, dict] | None:
    """The project's robowright settings: ``$ROBOWRIGHT_CONFIG``, else the nearest
    ``robowright.toml``, or ``pyproject.toml`` with a ``[tool.robowright]`` table, from
    ``start`` (the working directory) up."""
    env = os.environ.get("ROBOWRIGHT_CONFIG")
    if env:
        path = Path(env).expanduser().resolve()
        return path, _read(path, path.name == "pyproject.toml")
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / CONFIG).is_file():
            return d / CONFIG, _read(d / CONFIG, False)
        if (d / "pyproject.toml").is_file():
            table = _read(d / "pyproject.toml", True)
            if table:
                return d / "pyproject.toml", table
    return None


def _read(path: Path, pyproject: bool) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - Python 3.10
        import tomli as tomllib
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("tool", {}).get("robowright", {}) if pyproject else data


def project_robots(start: str | Path | None = None) -> dict[str, tuple[str, dict]]:
    """``{name: (model file, field overrides)}`` from the project's settings (see
    :func:`find_config`), with files relative to the settings file made absolute."""
    found = find_config(start)
    if not found:
        return {}
    path, table = found
    out = {}
    for name, entry in (table.get("robots") or {}).items():
        entry = {"file": entry} if isinstance(entry, str) else dict(entry)
        if "file" not in entry:
            raise ValueError(f'{path}: [robots.{name}] needs a file = "path/to/model"')
        file, _, query = str(entry.pop("file")).partition("?")
        p = Path(file).expanduser()
        p = p if p.is_absolute() else (path.parent / p).resolve()
        out[name] = (str(p) + (f"?{query}" if query else ""), {k: _tuples(v) for k, v in entry.items()})
    return out


def _tuples(v):
    """TOML arrays as the tuples RobotModel's fields hold."""
    if isinstance(v, list):
        return tuple(_tuples(x) for x in v)
    if isinstance(v, dict):
        return {k: _tuples(x) for k, x in v.items()}
    return v


def load_project_robot(name: str) -> bool:
    """Register the project robot ``name`` (detected from its file on first use); False if the
    project names no such robot."""
    entry = project_robots().get(name)
    if entry is None:
        return False
    from .robots.detect import load

    file, overrides = entry
    load(file, name, **overrides)
    return True


def add_project_robot(file: str, name: str, config: Path | None = None) -> Path:
    """Append ``[robots.<name>]`` for ``file`` to ``robowright.toml`` (the one found from the
    working directory, else a new one there); the file is written relative to it."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"robot names are letters, digits, _ and -: not {name!r}")
    found = find_config()
    if config is None and found and found[0].name == "pyproject.toml":
        # A robowright.toml here would hide the robots pyproject.toml already names.
        raise ValueError(f"{found[0]} holds this project's robowright settings: add [tool.robowright.robots.{name}] there")
    target = config or (found[0] if found else Path.cwd() / CONFIG)
    existing = project_robots(target.parent) if target.exists() else {}
    if name in existing:
        raise ValueError(f"{target} already names a robot {name!r}")
    path, _, query = file.partition("?")
    p = Path(path).expanduser().resolve()
    try:
        rel = p.relative_to(target.parent.resolve()).as_posix()
    except ValueError:
        rel = p.as_posix()
    text = target.read_text() if target.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    text += f'\n[robots.{name}]\nfile = "{rel}{"?" + query if query else ""}"\n'
    target.write_text(text.lstrip("\n"))
    return target
