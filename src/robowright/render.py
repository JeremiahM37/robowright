"""Render a trace after the fact, from the state it recorded.

A trace holds every step's joint readings and object poses (and a legged
robot's base pose), which is all it takes to redraw the run: the scene is
rebuilt in MuJoCo, posed from each step and drawn, with no physics. So a
test spends nothing on rendering while it runs, a picture is made only of the
runs someone looks at, at any size and frame rate, and runs from engines that
cannot render (Isaac Sim here) are drawn the same way as the rest.

    robowright render trace.zip -o run.mp4 --size 1280x720
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .trace import Trace


class Renderer:
    """Poses the trace's scene in MuJoCo, step by step, and draws it."""

    def __init__(self, trace: str | Path | Trace):
        from .backends.base import create

        self.trace = tr = trace if isinstance(trace, Trace) else Trace(trace)
        self.scene = tr.scene()
        self.backend = create("mujoco", self.scene)
        self.steps = len(tr.arrays["t"])
        self._movable = [(i, n) for i, n in enumerate(tr.meta["object_names"]) if not self.scene.object(n).static]

    @property
    def cameras(self) -> list[str]:
        return [c.name for c in self.scene.cameras]

    def pose(self, step: int) -> None:
        a, b = self.trace.arrays, self.backend
        if "base" in a:
            base = a["base"][step]
            b.set_base_pose(base[:3], base[3:7])
        b.set_joint_positions(a["qpos"][step])
        for i, name in self._movable:
            b.set_object_pose(name, a["obj_pos"][step][i], a["obj_quat"][step][i])

    def frame(self, step: int, camera: str = "front", size=(640, 480)) -> np.ndarray:
        self.pose(step)
        return self.backend.render(camera, int(size[0]), int(size[1]))

    def frames(self, camera: str = "front", size=(640, 480), every: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        for step in range(0, self.steps, every):
            yield step, self.frame(step, camera, size)

    def close(self):
        self.backend.close()


def jpeg_frames(trace: str | Path | Trace, every: int, size=(320, 240), cameras=None) -> dict[str, dict[int, bytes]]:
    """Frames for the trace viewer: ``{camera: {step: jpeg}}``, every ``every`` steps."""
    from PIL import Image

    r = Renderer(trace)
    out: dict[str, dict[int, bytes]] = {}
    try:
        for cam in cameras or r.cameras:
            for step, img in r.frames(cam, size, every):
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="JPEG", quality=85)
                out.setdefault(cam, {})[step] = buf.getvalue()
    finally:
        r.close()
    return out


def render_video(trace: str | Path | Trace, out: str | Path, camera: str = "front", size=(1280, 720), fps: float | None = None) -> Path:
    """Every recorded step as an MP4 (needs robowright[video]) or GIF, at the run's own pace by default."""
    r = Renderer(trace)
    out = Path(out)
    try:
        if camera not in r.cameras:
            raise ValueError(f"no camera {camera!r} in this trace; it has: {', '.join(r.cameras)}")
        dt = float(r.trace.meta["control_dt"])
        fps = fps or 1.0 / dt
        every = max(1, round(1.0 / (fps * dt)))  # skip steps rather than slow the run down
        frames = (img for _, img in r.frames(camera, size, every))
        if out.suffix.lower() == ".gif":
            from PIL import Image

            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 * every * dt), loop=0)
        else:
            import imageio.v2 as imageio

            with imageio.get_writer(out, fps=1.0 / (every * dt), codec="libx264", quality=8, macro_block_size=2) as w:
                for f in frames:
                    w.append_data(f)
    finally:
        r.close()
    return out
