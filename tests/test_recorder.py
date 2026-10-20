"""robowright codegen with no trace: a test written by doing it in a browser page (Playwright's codegen).

The recorder's own calls are tested directly; the page is driven in a real headless browser where Playwright is
installed: pick the cube, place it in the bin, check it landed, save, and the saved test passes under pytest.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import robowright as rw
from robowright.recorder import Recorder

SRC = Path(rw.__file__).resolve().parent.parent


def run_pytest(path: Path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in [str(SRC), os.environ.get("PYTHONPATH", "")] if p)}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path)],
        cwd=path.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_what_is_done_becomes_a_test_that_passes(tmp_path):
    rec = Recorder(robot="panda", output=tmp_path / "test_put_away.py")
    try:
        assert rec.act("pick", {"object": "cube"})["ok"]
        assert rec.act("place", {"on": "bin"})["ok"]
        check = rec.act("expect", {"subject": "cube", "matcher": "to_be_inside", "args": {"container": "bin"}})
        assert check["ok"] and check["result"].startswith("PASS"), check
        code = rec.code()
        assert "def test_put_away" in code and "pick" in code and "place" in code and "to_be_inside" in code, code
        saved = Path(rec.save())
    finally:
        rec.close()
    run = run_pytest(saved)
    assert run.returncode == 0 and "1 passed" in run.stdout, run.stdout + run.stderr


def test_a_failure_is_shown_and_the_session_goes_on(tmp_path):
    rec = Recorder(robot="panda", output=tmp_path / "test_x.py")
    try:
        bad = rec.act("pick", {"object": "no_such_thing"})
        assert not bad["ok"] and "no_such_thing" in bad["result"]
        assert not rec.act("rm_rf", {})["ok"]  # only the session's own actions are taken
        failed = rec.act("expect", {"subject": "cube", "matcher": "to_be_inside", "args": {"container": "bin"}})
        assert not failed["ok"] and failed["result"].startswith("FAIL")
        assert rec.act("pick", {"object": "cube"})["ok"]  # and it goes on
        assert [e["ok"] for e in rec.state()["log"]] == [False, False, False, True]
    finally:
        rec.close()


def test_the_page_records_a_test_in_a_real_browser(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    out = tmp_path / "test_from_the_page.py"
    started, box = threading.Event(), {}

    def serve():  # the recorder is made in the server's thread: MuJoCo renders in the thread that made its context
        rec = box["rec"] = Recorder(robot="panda", output=out)
        rec.serve(port=0, ready=lambda url: (box.update(url=url), started.set()))

    threading.Thread(target=serve, daemon=True).start()
    assert started.wait(120), "the recorder did not start"
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(box["url"])
        page.wait_for_function("document.getElementById('obj').options.length > 0")
        page.select_option("#obj", "cube")
        page.click("button[data-act=pick]")
        page.wait_for_function("document.getElementById('log').textContent.includes('pick')", timeout=60000)
        page.select_option("#onto", "bin")
        page.click("button[data-act=place]")
        page.wait_for_function("document.getElementById('log').textContent.includes('place')", timeout=60000)
        page.select_option("#subject", "cube")
        page.select_option("#matcher", "to_be_inside")
        page.select_option("#arg", "bin")
        page.click("button[data-act=expect]")
        page.wait_for_function("document.getElementById('log').textContent.includes('PASS')", timeout=60000)
        assert "to_be_inside" in page.text_content("#code")
        page.wait_for_function("(f => f.complete && f.naturalWidth > 0)(document.getElementById('frame'))")  # camera image
        page.click("#save")
        page.wait_for_function("document.getElementById('saved').textContent.includes('saved')")
        browser.close()
    run = run_pytest(out)
    assert run.returncode == 0 and "1 passed" in run.stdout, run.stdout + run.stderr


def test_clicking_an_object_in_the_page_picks_its_locator(tmp_path, capfd):
    """Playwright's Pick locator: click the cube in the camera image; its locators show, and Use prints the one chosen
    (what VS Code's Pick locator reads)."""
    sync_api = pytest.importorskip("playwright.sync_api")
    started, box = threading.Event(), {}

    def serve():
        rec = box["rec"] = Recorder(robot="so101", output=tmp_path / "test_x.py")
        rec.serve(port=0, ready=lambda url: (box.update(url=url), started.set()))

    threading.Thread(target=serve, daemon=True).start()
    assert started.wait(120), "the recorder did not start"
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(box["url"])
        page.wait_for_function("(f => f.complete && f.naturalWidth > 0)(document.getElementById('frame'))")
        cam = page.eval_on_selector("#camera", "s => s.value")
        state = page.evaluate("async () => (await fetch('state')).json()")
        u0, v0, u1, v1 = next(b for n, b, _ in state["picks"][cam] if n == "cube")
        size = page.eval_on_selector("#frame", "f => [f.clientWidth, f.clientHeight]")
        page.click("#frame", position={"x": (u0 + u1) / 2 * size[0], "y": (v0 + v1) / 2 * size[1]})
        assert 'scene["cube"]' in page.text_content("#picked")
        page.click("#picked button")
        page.wait_for_function("document.querySelector('#picked button').textContent === 'sent'")
        browser.close()
    assert 'robowright picked: scene["cube"]' in capfd.readouterr().out
