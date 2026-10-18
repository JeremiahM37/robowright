"""robowright test --ui: the suite listed, runs started from the page, results and traces, the guard on its API."""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from robowright import ui as rw_ui

WORLD_TEST = """from robowright import expect


def test_cube_stays_put(world, scene):
    expect(scene["cube"]).to_be_at_rest()
"""
PLAIN_TEST = """def test_sum():
    assert 1 + 1 == 2


def test_wrong():
    assert 1 + 1 == 3
"""


@pytest.fixture
def project(tmp_path):
    # no ini file, as in a new project: pytest then finds a rootdir from the paths it is given
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_world.py").write_text(WORLD_TEST)
    (tmp_path / "tests" / "test_plain.py").write_text(PLAIN_TEST)
    return tmp_path


@pytest.fixture
def ui(project):
    u = rw_ui.UI(["tests", "-p", "no:cacheprovider"], root=project)
    yield u
    u.close()


def _finish(u, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        if u.proc is None and u.status is not None:
            return u.status
        time.sleep(0.1)
    raise AssertionError("the run did not finish:\n" + "\n".join(u.log))


def test_xdist_options_are_left_out_of_collection():
    assert rw_ui._without_xdist(["-n", "4", "tests", "-nauto", "--dist=load", "--numprocesses=2", "--no-header", "-x"]) == [
        "tests",
        "--no-header",
        "-x",
    ]


def test_the_suite_is_listed_run_and_each_test_keeps_its_result_and_trace(ui):
    ui.collect()
    ids = list(ui.tests)
    assert ids == ["tests/test_plain.py::test_sum", "tests/test_plain.py::test_wrong", "tests/test_world.py::test_cube_stays_put[mujoco]"]
    assert ui.run([]) and not ui.run([]), "one run at a time"
    assert _finish(ui) == 1
    outcomes = {n: t["outcome"] for n, t in ui.tests.items()}
    assert outcomes == dict(zip(ids, ["passed", "failed", "passed"]))
    assert "assert (1 + 1) == 3" in ui.tests["tests/test_plain.py::test_wrong"]["message"]
    trace = ui.tests[ids[2]]["traces"][0]  # a passing test's trace too, as Playwright's UI mode keeps one
    assert ui.trace_file(trace) is not None and ui.trace_file("/etc/passwd") is None

    # one test run alone reports under the same id (a lone id must not find another rootdir)
    assert ui.run([ids[0]]) and _finish(ui) == 0
    assert list(ui.tests) == ids and ui.tests[ids[0]]["outcome"] == "passed"


def test_watch_runs_the_saved_test_file_again(ui, project):
    ui.collect()
    ui.watching = True
    threading.Thread(target=ui.watch_loop, kwargs={"interval": 0.1}, daemon=True).start()
    time.sleep(0.5)
    (project / "tests" / "test_plain.py").write_text(PLAIN_TEST.replace("1 + 1 == 3", "1 + 1 == 2"))
    end = time.time() + 60
    while ui.status is None and time.time() < end:
        time.sleep(0.1)
    assert _finish(ui) == 0, "\n".join(ui.log)
    assert "tests/test_plain.py" in ui.log[0] and "test_world" not in ui.log[0], "only the saved file runs"
    assert ui.tests["tests/test_plain.py::test_wrong"]["outcome"] == "passed"


def test_the_api_needs_the_pages_token_and_its_own_host(ui):
    token = "t0ken"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), rw_ui._handler(ui, token, lambda: httpd.server_address[1]))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path, headers=None, data=None):
        req = urllib.request.Request(base + path, headers=headers or {}, data=data)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        code, page = get("/")
        assert code == 200 and b'"t0ken"' in page
        assert get("/api/state")[0] == 403
        assert get("/api/state", {"X-RW-Token": token, "Host": "attacker.example"})[0] == 403  # DNS rebinding
        code, body = get("/api/state", {"X-RW-Token": token})
        assert code == 200 and json.loads(body)["running"] is False
        assert get("/api/run", {"X-RW-Token": "wrong"}, b'{"ids": []}')[0] == 403
        assert get("/api/run", {"X-RW-Token": token}, b'{"ids": ["--help"]}')[0] == 400, "an id is not an option"
        assert get("/trace?path=/etc/passwd", {"X-RW-Token": token})[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
