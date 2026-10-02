"""Trace viewer: a single self-contained HTML file per trace.

The page embeds the camera frames and telemetry, so it opens from disk,
attaches to a CI artifact or an issue, and needs no server.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .trace import Trace

_TEMPLATE = Path(__file__).with_name("viewer.html")


def build_html(trace: str | Path | Trace) -> str:
    tr = trace if isinstance(trace, Trace) else Trace(trace)
    a = tr.arrays
    frames: dict[str, dict[int, str]] = {}
    for name in tr.frame_names:
        _, cam, file = name.split("/")
        frames.setdefault(cam, {})[int(file.split(".")[0])] = "data:image/jpeg;base64," + base64.b64encode(tr.frame(name)).decode()
    # Keep contacts that involve the robot or a movable object resting on something other than the floor.
    contacts = [
        [c for c in step if not ("floor" in (c[0], c[1]) and not any(x.startswith("robot:") for x in c[:2]))] for step in tr.contacts
    ]
    data = {
        "meta": tr.meta,
        "t": np.round(a["t"], 4).tolist(),
        "qpos": np.round(a["qpos"], 4).tolist(),
        "ctrl": np.round(a["ctrl"], 4).tolist(),
        "obj_pos": np.round(a["obj_pos"], 4).tolist(),
        "contacts": contacts,
        "frames": frames,
        "source": tr.path.name,
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.read_text().replace("__TRACE_DATA__", payload).replace("__TITLE__", tr.meta["name"])


def write_html(trace: str | Path, out: str | Path | None = None) -> Path:
    trace = Path(trace)
    out = Path(out) if out else trace.with_suffix(".html")
    out.write_text(build_html(trace))
    return out
