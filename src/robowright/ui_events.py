"""The pytest side of ``robowright test --ui``: each test's start and result, written as JSON lines to the file named
by ``$ROBOWRIGHT_UI_EVENTS`` for the UI's server to read. Loaded with ``-p robowright.ui_events``; does nothing
without the variable. Results are gathered with the HTML report's own ``Run``, so a test reads the same in both.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .report import Run

_RUN = Run()
_STARTED: dict[str, float] = {}
_CONFIG: dict = {}


def _emit(record: dict) -> None:
    path = os.environ.get("ROBOWRIGHT_UI_EVENTS")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def pytest_collection_finish(session):
    # where each test is (file relative to the rootdir, 0-based line), for an editor to put its result on
    locations = {i.nodeid: [i.location[0], i.location[1]] for i in session.items}
    _emit({"event": "collected", "tests": list(locations), "locations": locations, "rootdir": str(session.config.rootpath)})


def pytest_runtest_logstart(nodeid, location):
    _STARTED[nodeid] = time.time()
    _emit({"event": "start", "nodeid": nodeid})


def pytest_runtest_logreport(report):
    _RUN.add(report)


def pytest_runtest_logfinish(nodeid, location):
    t = dict(_RUN.tests.get(nodeid, {"outcome": "passed", "duration": 0.0, "message": "", "traces": []}))
    # A passing test's trace is kept too (the UI runs with --rw-trace on): the report only lists a failure's.
    from .pytest_plugin import _trace_path

    config = _CONFIG.get("config")
    if config is not None:
        p = _trace_path(config, nodeid)
        if p.exists() and str(p) not in t["traces"]:
            t["traces"] = [str(p), *t["traces"]]
    # only what this run wrote: a trace left by an earlier run is not this test's
    since = _STARTED.get(nodeid, 0.0) - 1.0
    t["traces"] = [str(Path(p).resolve()) for p in t["traces"] if Path(p).exists() and Path(p).stat().st_mtime >= since]
    _emit({"event": "result", "nodeid": nodeid, **t})


def pytest_configure(config):
    _CONFIG["config"] = config


def pytest_sessionfinish(session, exitstatus):
    _emit({"event": "finished", "status": int(exitstatus)})
