"""rw.http_stub: a stand-in HTTP service for the code under test (Playwright's route.fulfill for an API)."""

import json
import subprocess
import sys
import urllib.error
import urllib.request

import robowright as rw


def _get(url, method="GET", data=None):
    req = urllib.request.Request(url, method=method, data=None if data is None else json.dumps(data).encode())
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read()
            return r.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def test_routes_answer_and_every_call_is_recorded():
    seen = []

    def done(req):
        seen.append(req.json)
        return 204, None

    routes = {
        "GET /missions/next": {"id": 7, "pick": "cube", "place": "bin"},
        "POST /missions/7/done": done,
        "GET /map": rw.http_stub.status(503, {"error": "map server down"}),
        "GET /assets/*": {"kind": "asset"},
    }
    with rw.http_stub(routes) as fleet:
        assert _get(fleet.url + "/missions/next") == (200, {"id": 7, "pick": "cube", "place": "bin"})
        assert _get(fleet.url + "/missions/7/done", "POST", {"ok": True}) == (204, None)
        assert _get(fleet.url + "/map")[0] == 503
        assert _get(fleet.url + "/assets/arm.urdf") == (200, {"kind": "asset"})
        assert _get(fleet.url + "/nope")[0] == 404
        assert fleet.called("POST /missions/7/done") and seen == [{"ok": True}]
        assert [str(r) for r in fleet.unmatched] == ["GET /nope"]
        assert len(fleet.calls()) == 5


def test_a_slow_service_and_a_handler_that_raises():
    import time

    def broken(req):
        raise RuntimeError("database locked")

    with rw.http_stub({"GET /slow": rw.http_stub.status(200, {"ok": 1}, delay=0.3), "GET /broken": broken}) as s:
        t = time.monotonic()
        assert _get(s.url + "/slow") == (200, {"ok": 1}) and time.monotonic() - t >= 0.3
        code, body = _get(s.url + "/broken")
        assert code == 500 and "database locked" in body["error"]


def test_another_process_reaches_it():
    """A stack the test starts (a ROS 2 launch, a controller process) is pointed at the stub by its URL."""
    with rw.http_stub({"GET /missions/next": {"id": 3}}) as fleet:
        code = "import json,sys,urllib.request; print(json.load(urllib.request.urlopen(sys.argv[1] + '/missions/next'))['id'])"
        out = subprocess.run([sys.executable, "-c", code, fleet.url], capture_output=True, text=True, timeout=30)
        assert out.stdout.strip() == "3"
        assert fleet.called("GET /missions/next")


def test_a_real_service_is_recorded_and_replayed_without_it(tmp_path):
    """record() passes calls through and writes them; replay() answers from the file (recordHar / routeFromHAR)."""
    counter = {"n": 0}

    def next_mission(req):
        counter["n"] += 1
        return {"id": counter["n"]}

    real = rw.http_stub({"GET /missions/next": next_mission, "POST /missions/1/done": lambda req: (204, None)})
    har = tmp_path / "fleet.json"
    with rw.http_stub.record(real.url, har) as rec:
        assert _get(rec.url + "/missions/next") == (200, {"id": 1})
        assert _get(rec.url + "/missions/next") == (200, {"id": 2})
        assert _get(rec.url + "/missions/1/done", "POST", {"ok": True}) == (204, None)
        assert _get(rec.url + "/gone")[0] == 404  # the real service's own 404, recorded too
    real.close()  # the real service is gone from here on
    with rw.http_stub.replay(har) as fake:
        assert _get(fake.url + "/missions/next") == (200, {"id": 1})  # the recorded answers, in order
        assert _get(fake.url + "/missions/next") == (200, {"id": 2})
        assert _get(fake.url + "/missions/next") == (200, {"id": 2})  # the last one once they run out
        assert _get(fake.url + "/missions/1/done", "POST", {"ok": True}) == (204, None)
        assert _get(fake.url + "/never-recorded")[0] == 404
        assert [str(r) for r in fake.unmatched] == ["GET /never-recorded"]
