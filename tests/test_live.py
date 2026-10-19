import json
import urllib.request

import pytest

import robowright as rw
from robowright.live import LiveView


def _get(url):
    return urllib.request.urlopen(url, timeout=10).read()


def test_live_view_serves_the_page_the_state_and_frames():
    w = rw.launch(settings=rw.Settings(trace="on"))
    with LiveView(w, port=0, quiet=True) as v:
        w.robot.pick(w.scene["cube"])
        page = _get(v.url + "/").decode()
        assert "robowright live" in page and "/stream" in page
        s = json.loads(_get(v.url + "/state"))
        assert s["robot"] == "so101" and s["backend"] == "mujoco" and s["fidelity"] == w.fidelity
        assert s["t"] > 1.0 and "pick" in s["action"]
        assert any("cube" in c for c in s["contacts"])
        assert _get(v.url + "/frame.jpg?camera=front")[:3] == b"\xff\xd8\xff"
    w.close()


def test_watching_costs_nothing_once_closed():
    w = rw.launch(settings=rw.Settings(trace="off"))
    v = LiveView(w, port=0, quiet=True)
    assert v._hook in w._step_hooks
    v.close()
    assert v._hook not in w._step_hooks
    w.close()


def test_rw_live_option(pytester):
    pytester.makepyfile(
        test_l="""
import json, urllib.request

def test_watched(world, rw_live):
    assert rw_live is not None and world.settings.realtime
    world.wait(0.2)
    s = json.loads(urllib.request.urlopen(rw_live.url + "/state").read())
    assert s["name"].endswith("test_watched[mujoco]") and s["t"] >= 0.2
"""
    )
    pytester.runpytest("-p", "no:cacheprovider", "--rw-live", "0").assert_outcomes(passed=1)


def _post(url, headers=None):
    headers = {"X-RW-Control": "1"} if headers is None else headers
    req = urllib.request.Request(url, data=b"", method="POST", headers=headers)
    try:
        return urllib.request.urlopen(req, timeout=10).status
    except urllib.error.HTTPError as e:
        return e.code


def _wait_paused(url, what, timeout=60.0):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s = json.loads(_get(url + "/state"))
        if s.get("paused") and s["paused"]["what"].startswith(what):
            return s["paused"]
        time.sleep(0.05)
    raise AssertionError(f"never paused before {what}: {s.get('paused')}")


def _person(steps):
    """A person at the page, in a thread: each step is (what to wait to be paused before, the button to press then).
    The world runs on the test's own thread, as in a real run (MuJoCo renders on the thread that made it)."""
    import threading

    box = {"seen": []}

    def run():
        try:
            while "url" not in box:
                threading.Event().wait(0.02)
            for what, cmd in steps:
                box["seen"].append(_wait_paused(box["url"], what))
                box.setdefault("time_at_pause", []).append(json.loads(_get(box["url"] + "/state"))["t"])
                if cmd == "csrf":  # a page on another site cannot send the header: refused
                    box["csrf"] = _post(box["url"] + "/control?cmd=resume", headers={})
                    cmd = "step"
                assert _post(box["url"] + f"/control?cmd={cmd}") == 204
        except BaseException as e:  # noqa: BLE001 - handed to the test
            box["error"] = e
            if "url" in box:
                _post(box["url"] + "/control?cmd=resume")  # never leave the test held

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def test_stepping_pauses_before_each_action_and_expectation():
    from robowright import expect

    person, box = _person([("robot.pick(obj=cube)", "csrf"), ("expect(cube).to_be_above(target=0.03", "resume")])
    w = rw.launch(settings=rw.Settings(trace="off"))
    with LiveView(w, port=0, quiet=True) as v:
        box["url"] = v.url
        v.stepping = True
        w.robot.pick(w.scene["cube"])  # held here
        expect(w.scene["cube"]).to_be_above(0.03)
        w.robot.gripper.open()  # not held: resumed
    w.close()
    person.join(30)
    assert "error" not in box, box["error"]
    first, second = box["seen"]
    assert first["at"].endswith(f'test_live.py:{_line("# held here")}: w.robot.pick(w.scene["cube"])  # held here')
    assert box["time_at_pause"][0] == 0.0 and box["time_at_pause"][1] > 1.0  # held before the pick, then after it
    assert box["csrf"] == 404  # only the page drives it


def _line(text):
    from pathlib import Path

    return next(i for i, line in enumerate(Path(__file__).read_text().splitlines(), 1) if line.endswith(text))


def test_pause_opens_the_live_view_and_waits(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    person, box = _person([("world.pause()", "resume")])
    w = rw.launch(settings=rw.Settings(trace="off"))
    real = LiveView.__init__

    def opened(self, *a, **k):  # tell the person where the page is
        real(self, *a, **k)
        box["url"] = self.url

    monkeypatch.setattr(LiveView, "__init__", opened)
    w.pause()
    w.robot.gripper.close()
    w.close()
    person.join(30)
    assert "error" not in box, box["error"]
    assert box["seen"][0]["what"] == "world.pause()"
    assert "_own_live" not in w.__dict__  # closed with the world


def test_pause_does_nothing_on_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    w = rw.launch(settings=rw.Settings(trace="on"))
    w.pause()  # returns at once
    assert "_live" not in w.__dict__
    assert any(e.name == "pause() skipped: CI is set" for e in w.trace.events)
    w.close()


def test_rw_debug_pauses_each_test_before_its_first_action(pytester):
    """In a real pytest run: the page's address and each pause reach the terminal past pytest's capture."""
    import subprocess
    import sys
    import threading
    import time

    pytester.makepyfile(
        test_d="""
from robowright import expect

def test_pick(robot, scene):
    robot.pick(scene["cube"])
    expect(scene["cube"]).to_be_above(0.03)
"""
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != "CI"}
    p = subprocess.Popen(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--rw-debug", "--rw-live", "0", "test_d.py"],
        cwd=pytester.path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    lines = []
    threading.Thread(target=lambda: lines.extend(p.stdout), daemon=True).start()
    try:
        end = time.monotonic() + 120
        url = None
        while url is None and time.monotonic() < end:
            url = next((ln.split("watch live at ")[1].strip() for ln in list(lines) if "watch live at " in ln), None)
            time.sleep(0.1)
        assert url, "".join(lines)
        _wait_paused(url, "robot.pick(obj=cube)")
        _post(url + "/control?cmd=step")
        _wait_paused(url, "expect(cube).to_be_above")
        _post(url + "/control?cmd=resume")
        assert p.wait(120) == 0, "".join(lines)
    finally:
        if p.poll() is None:
            p.kill()
    out = "".join(lines)
    assert 'paused before robot.pick(obj=cube) (test_d.py:4: robot.pick(scene["cube"]))' in out
    assert "1 passed" in out


def test_clicking_an_object_in_the_view_gives_its_locator():
    """The locator picker: click the cube in the camera image, get the code that finds it (Playwright's pick locator)."""
    import threading

    sync_api = pytest.importorskip("playwright.sync_api")
    from robowright.snapshots import box_of

    box = {}

    def person():
        try:
            while "url" not in box:
                threading.Event().wait(0.02)
            _wait_paused(box["url"], "robot.pick")
            with sync_api.sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": 1200, "height": 800})
                page.goto(box["url"])
                page.wait_for_function("document.getElementById('cam').naturalWidth > 0", timeout=20000)
                r = page.locator("#cam").bounding_box()
                u0, v0, u1, v1 = box["cube"]
                page.mouse.click(r["x"] + r["width"] * (u0 + u1) / 2 / 640, r["y"] + r["height"] * (v0 + v1) / 2 / 480)
                page.wait_for_selector("#picked:not([hidden])")
                box["picked"] = page.inner_text("#picked")
                page.mouse.click(r["x"] + 5, r["y"] + 5)  # the sky: nothing there
                box["empty"] = page.inner_text("#picked")
                browser.close()
        except BaseException as e:  # noqa: BLE001 - handed to the test
            box["error"] = e
        finally:
            if "url" in box:
                _post(box["url"] + "/control?cmd=resume")

    helper = threading.Thread(target=person, daemon=True)
    helper.start()
    w = rw.launch(settings=rw.Settings(trace="off"))
    with LiveView(w, port=0, quiet=True) as v:
        box["cube"] = box_of(w, "cube", "front", 640, 480, margin=0.0)
        box["url"] = v.url
        v.stepping = True
        w.robot.pick(w.scene["cube"])
    w.close()
    helper.join(60)
    assert "error" not in box, box["error"]
    assert 'scene["cube"]' in box["picked"] and 'scene.get(color="red")' in box["picked"]
    assert "nothing there" in box["empty"]
