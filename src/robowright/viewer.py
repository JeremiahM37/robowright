"""Trace viewer: a single self-contained HTML file per trace.

The page embeds the camera frames and telemetry, so it opens from disk,
attaches to a CI artifact or an issue, and needs no server. A trace without
frames captured during the run gets them drawn from its recorded state here,
so only the traces someone opens pay for rendering.
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
    jpegs: dict[str, dict[int, bytes]] = {}
    for name in tr.frame_names:
        _, cam, file = name.split("/")
        jpegs.setdefault(cam, {})[int(file.split(".")[0])] = tr.frame(name)
    if not jpegs:
        jpegs = _redraw(tr)
    for cam, by_step in jpegs.items():
        frames[cam] = {step: "data:image/jpeg;base64," + base64.b64encode(data).decode() for step, data in by_step.items()}
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


def _redraw(tr: Trace) -> dict[str, dict[int, bytes]]:
    """Frames drawn from the trace's recorded state; none if this machine cannot render."""
    from .render import jpeg_frames

    try:
        return jpeg_frames(tr, every=int(tr.meta.get("frame_every", 5)), size=(480, 360))
    except Exception as e:  # no GL: the viewer still has the telemetry
        import warnings

        warnings.warn(f"robowright: trace viewer without camera frames, rendering failed ({type(e).__name__}: {e})", stacklevel=3)
        return {}


def write_html(trace: str | Path, out: str | Path | None = None) -> Path:
    trace = Path(trace)
    out = Path(out) if out else trace.with_suffix(".html")
    out.write_text(build_html(trace))
    return out
