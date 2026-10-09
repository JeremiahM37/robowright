import json
import urllib.request

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
