"""Record docs/ui-demo.gif, the README's first picture: robowright's UI mode running a small suite with the camera
live, one test failing, its error, then its trace at the failed expectation.

    python scripts/record_ui_demo.py            # needs Playwright's Chromium and ffmpeg

The suite is written into a temporary folder: two tests that pass and one that lifts the cube 5 cm where 12 are
needed. Recorded at 1600x900 (the trace viewer lays out its three panels from 1000 px), then a 1000 px GIF.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

from robowright import ui as rw_ui

SUITE = """from robowright import expect


def test_cube_goes_in_the_bin(robot, scene):
    cube, bin = scene["cube"], scene["bin"]
    robot.pick(cube)
    robot.place(on=bin)
    expect(cube).to_be_inside(bin)


def test_arm_never_hits_the_table(robot, scene):
    expect(robot).always.to_have_no_collisions()
    robot.pick(scene["cube"])


def test_cube_clears_the_shelf_lip(robot, scene):
    robot.pick(scene["cube"], lift=0.05)
    expect(scene["cube"]).to_be_above(0.12, timeout=1.0)  # the lip is 12 cm: a 5 cm lift is too low
"""
FAIL = "tests/test_pick_place.py::test_cube_clears_the_shelf_lip[mujoco]"
OUT = Path(__file__).resolve().parent.parent / "docs" / "ui-demo.gif"
PORT = 8951


def record(project: Path, videos: Path) -> Path:
    u = rw_ui.UI(["tests"], root=project)
    u.collect()
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), rw_ui._handler(u, "demo", lambda: PORT))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            size = {"width": 1600, "height": 900}
            ctx = browser.new_context(viewport=size, record_video_dir=str(videos), record_video_size=size, color_scheme="dark")
            page = ctx.new_page()
            page.goto(f"http://127.0.0.1:{PORT}/")
            page.wait_for_selector("#tree li.test")
            time.sleep(1.5)
            page.click("#live")
            page.click('button[data-tab="live"]')
            time.sleep(0.6)
            page.click("#runAll")
            page.wait_for_selector("#runAll[disabled]")
            page.wait_for_selector("#runAll:not([disabled])", timeout=180_000)
            page.click('button[data-tab="error"]')
            page.click(f'li[data-id="{FAIL}"] .name')
            time.sleep(3.0)
            page.click('button[data-tab="trace"]')
            frame = page.frame_locator("#pane iframe")
            frame.locator("#events li.failed").first.wait_for(timeout=30_000)
            time.sleep(1.5)
            frame.locator("#events li").nth(1).click()
            time.sleep(1.2)
            frame.locator("#events li.failed").first.click()
            time.sleep(3.0)
            video = Path(page.video.path())
            ctx.close()
            browser.close()
    finally:
        httpd.shutdown()
        u.close()
    return video


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rw-ui-demo-") as tmp:
        project = Path(tmp) / "project"
        (project / "tests").mkdir(parents=True)
        (project / "tests" / "test_pick_place.py").write_text(SUITE)
        video = record(project, Path(tmp) / "video")
        graph = (
            "fps=10,scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
        )
        ffmpeg = shutil.which("ffmpeg") or sys.exit("ffmpeg is needed to make the GIF")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", "0.8", "-i", str(video), "-vf", graph, str(OUT)], check=True)
    print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
