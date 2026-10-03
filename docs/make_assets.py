"""Regenerate the README media: docs/demo.gif, docs/gallery.png and docs/viewer.png.

    python docs/make_assets.py

The GIF is the front camera of a real test run (the example in the README);
the screenshot is the trace viewer opened on a real failing test's trace.
The screenshot needs Playwright's Chromium (`pip install playwright && playwright install chromium`).
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import robowright as rw
from robowright import CameraSpec, expect, tabletop
from robowright.viewer import write_html

DOCS = Path(__file__).resolve().parent


def demo_gif():
    cams = [CameraSpec("front", pos=(0.62, -0.46, 0.46), lookat=(0.15, 0.02, 0.09), fovy=40)]
    tmp = Path(tempfile.mkdtemp())
    w = rw.launch(
        tabletop(cameras=cams), name="demo", settings=rw.Settings(trace="on", trace_dir=str(tmp), frame_every=3, image_size=(440, 330))
    )
    w.robot.reset_to()
    cube, bin_ = w.scene["cube"], w.scene["bin"]
    w.robot.pick(cube)
    expect(w.robot.gripper).to_be_holding(cube)
    w.robot.place(on=bin_)
    expect(cube).to_be_inside(bin_)
    w.robot.arm.home()
    path = w.close()
    tr = rw.Trace(path)
    rgb = [Image.open(io.BytesIO(tr.frame(n))).convert("RGB") for n in tr.frame_names]
    # One palette for the whole clip, built from a sample of frames, keeps colours stable.
    sheet = Image.new("RGB", (rgb[0].width, rgb[0].height * 4))
    for k, f in enumerate(rgb[:: max(1, len(rgb) // 4)][:4]):
        sheet.paste(f, (0, k * f.height))
    palette = sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in rgb]
    frames[0].save(DOCS / "demo.gif", save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True)
    print(DOCS / "demo.gif", len(frames), "frames")


def gallery(backend: str = "mujoco", out: str = "gallery.png", cols: int = 5):
    """Every robot mid-task: arms holding the cube over the table, legged robots standing."""
    tiles = []
    for name in rw.robots.names():
        m = rw.robots.get(name)
        w = rw.launch(robot=name, backend=backend, settings=rw.Settings(trace="off"))
        w.robot.reset_to()
        if m.family == "arm":
            w.robot.pick(w.scene["cube"])
            w.robot.arm.move_to((0.2, 0.03, 0.09))
        else:
            w.wait(0.5)
        img = Image.fromarray(w.backend.render("front", 400, 300))
        w.close()
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=17)
        d.rectangle((0, 0, 400, 26), fill=(255, 255, 255))
        d.text((8, 4), m.title, fill=(20, 20, 20), font=font)
        tiles.append(img)
        print(" ", name)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (400 * cols, 300 * rows), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * 400, (k // cols) * 300))
    sheet.save(DOCS / out, optimize=True)
    print(DOCS / out)


def viewer_png():
    out = Path(tempfile.mkdtemp())
    w = rw.launch(name="tests/test_bin.py::test_cube_lands_in_bin[mujoco]", settings=rw.Settings(trace="on", trace_dir=str(out)))
    w.robot.reset_to()
    w.robot.pick(w.scene["cube"])
    w.robot.place(on=(0.26, -0.11, 0.0))  # the bug: wrong drop point
    try:
        expect(w.scene["cube"]).to_be_inside(w.scene["bin"], timeout=0.5)
    except rw.ExpectationError:
        pass
    html = write_html(w.close(), out / "trace.html")
    code = (
        "from playwright.sync_api import sync_playwright\n"
        "with sync_playwright() as p:\n"
        "    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1400, 'height': 860}, device_scale_factor=1)\n"
        f"    pg.goto({html.resolve().as_uri()!r}); pg.wait_for_timeout(400)\n"
        f"    pg.screenshot(path={str(DOCS / 'viewer.png')!r}); b.close()\n"
    )
    python = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    subprocess.run([python, "-c", code], check=True)
    print(DOCS / "viewer.png")


if __name__ == "__main__":
    demo_gif()
    gallery()
    viewer_png()
