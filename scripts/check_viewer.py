"""Drive the trace viewer in headless Chromium and assert on what it shows.

    python scripts/check_viewer.py /tmp/robowright-verify/trace.html

Needs Playwright (`pip install playwright && playwright install chromium`).
"""

import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

url = Path(sys.argv[1]).resolve().as_uri()
with sync_playwright() as p:
    browser = p.chromium.launch()
    for label, ctx_args in (("desktop", {"viewport": {"width": 1400, "height": 900}}), ("phone", dict(p.devices["Pixel 7"]))):
        page = browser.new_context(**ctx_args).new_page()
        errors = []
        page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
        page.goto(url)
        expect(page.locator("#status")).to_have_text("failed")
        expect(page.locator("#events li.failed .name")).to_contain_text("to_be_inside")
        expect(page.locator("#pane")).to_contain_text("bin spans")  # first failure is preselected
        page.locator("#events li", has_text="robot.pick").click()
        expect(page.locator("#pane")).to_contain_text("robot.pick")
        page.locator(".tabs button[data-tab=state]").click()
        expect(page.locator("#pane")).to_contain_text("shoulder_pan")
        before = page.locator("#clock").inner_text()
        page.keyboard.press("ArrowRight")
        assert page.locator("#clock").inner_text() != before, "arrow key did not move the playhead"
        assert page.evaluate("document.querySelector('#frame').src.startsWith('data:image/jpeg')"), "no camera frame"
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), f"{label}: horizontal scroll"
        assert not errors, f"{label}: page errors {errors}"
        print(f"viewer ok ({label})")
    browser.close()
