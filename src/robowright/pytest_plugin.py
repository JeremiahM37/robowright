"""pytest integration.

Fixtures: ``world``, ``robot``, ``scene`` (fresh per test), ``rw_scene``
(override in a conftest to change the default scene).

Options::

    --rw-backend mujoco,pybullet     run every robot test on each backend
    --rw-robot so101,panda | all | legged   run every robot test on each robot ("all": every arm)
    --rw-trace on|off|retain-on-failure
    --rw-trace-dir DIR
    --rw-seed N                      base seed

Markers::

    @pytest.mark.scene(spec_or_factory)
    @pytest.mark.seed(7)
    @pytest.mark.backends("mujoco")              # restrict
    @pytest.mark.robots("panda", "ur5e")         # restrict
    @pytest.mark.trials(20, min_success=0.9)     # run across 20 seeds, judge the rate
"""

from __future__ import annotations

from pathlib import Path

import pytest

import robowright as rw

from . import robots as _robots
from .errors import RobowrightError
from .scene import SceneSpec, default_camera, default_scene
from .stats import TrialReport
from .world import Settings, World, _safe

_TRACES = pytest.StashKey[list]()
_REPORTS = pytest.StashKey[dict]()


def pytest_addoption(parser):
    g = parser.getgroup("robowright")
    g.addoption("--rw-backend", default="mujoco", help="comma-separated backends to run robot tests on")
    g.addoption("--rw-robot", default="so101", help="comma-separated robots to run robot tests on, or 'all'")
    g.addoption("--rw-trace", default="retain-on-failure", choices=["on", "off", "retain-on-failure"])
    g.addoption("--rw-trace-dir", default="robowright-traces")
    g.addoption("--rw-seed", type=int, default=0, help="base seed added to each test's seed")


def pytest_configure(config):
    config.addinivalue_line("markers", "scene(spec): scene spec or zero-arg factory for this test")
    config.addinivalue_line("markers", "seed(n): seed for this test")
    config.addinivalue_line("markers", "backends(*names): only run on these backends")
    config.addinivalue_line("markers", "robots(*names): only run on these robots")
    config.addinivalue_line("markers", "trials(n, min_success=1.0, lower_bound=False): run across n seeds and judge the success rate")
    config.stash[_TRACES] = []
    config.stash[_REPORTS] = {}


def _robot_names(config) -> list[str]:
    raw = [r.strip() for r in config.getoption("--rw-robot").split(",") if r.strip()]
    out = []
    for r in raw:
        if r in ("all", "arms", "legged"):
            out.extend(_robots.names({"all": "arm", "arms": "arm", "legged": "legged"}[r]))
        else:
            out.append(_robots.get(r).name)
    return list(dict.fromkeys(out))


def pytest_generate_tests(metafunc):
    if "rw_backend" in metafunc.fixturenames:
        names = [b.strip() for b in metafunc.config.getoption("--rw-backend").split(",") if b.strip()]
        m = metafunc.definition.get_closest_marker("backends")
        if m:
            names = [n for n in names if n in m.args] or list(m.args[:1])
        metafunc.parametrize("rw_backend", names, ids=names, scope="function")
    if "rw_robot" in metafunc.fixturenames:
        names = _robot_names(metafunc.config)
        m = metafunc.definition.get_closest_marker("robots")
        if m:
            names = [n for n in names if n in m.args] or list(m.args[:1])
        if len(names) > 1 or m:
            metafunc.parametrize("rw_robot", names, ids=names, scope="function")


@pytest.fixture
def rw_robot(request) -> str:  # parametrized by pytest_generate_tests when several robots are selected
    return _robot_names(request.config)[0]


@pytest.fixture
def rw_scene(rw_robot) -> SceneSpec:
    return default_scene(rw_robot)


def _scene_for(item, request) -> SceneSpec:
    m = item.get_closest_marker("scene")
    if m:
        s = m.args[0]
        spec = s() if callable(s) else s
    else:
        spec = request.getfixturevalue("rw_scene")
    robot = request.getfixturevalue("rw_robot")
    if spec.robot != robot:
        import dataclasses

        cams = [default_camera(robot)] if spec.cameras == [default_camera(spec.robot)] else spec.cameras
        spec = dataclasses.replace(spec, robot=robot, cameras=cams)
    return spec


def _settings(config) -> Settings:
    return Settings(trace=config.getoption("--rw-trace"), trace_dir=config.getoption("--rw-trace-dir"))


def _trace_path(config, nodeid: str) -> Path:
    return Path(config.getoption("--rw-trace-dir")) / f"{_safe(nodeid.replace('::', '__').replace('/', '_'))}.zip"


def _make_world(item, request, backend: str, seed: int, suffix: str = "") -> World:
    w = World(_scene_for(item, request), backend=backend, seed=seed, name=item.nodeid + suffix, settings=_settings(item.config))
    rw._current.set(w)
    return w


def _seed(item) -> int:
    m = item.get_closest_marker("seed")
    return item.config.getoption("--rw-seed") + (m.args[0] if m else 0)


@pytest.fixture
def rw_backend():  # parametrized by pytest_generate_tests
    return "mujoco"


@pytest.fixture
def world(request, rw_backend, rw_robot):
    item = request.node
    if item.get_closest_marker("trials"):
        yield None  # pytest_pyfunc_call builds one world per trial
        return
    w = _make_world(item, request, rw_backend, _seed(item))
    w.robot.reset_to()
    yield w
    rep = getattr(item, "_rw_call_report", None)
    failed = rep is None or rep.failed
    path = w.close(failed=failed, trace_path=_trace_path(item.config, item.nodeid))
    if path and failed:
        item.config.stash[_TRACES].append((item.nodeid, str(path)))


@pytest.fixture
def seed(request) -> int:
    """This test's seed (each trial gets its own under ``@pytest.mark.trials``)."""
    return _seed(request.node)


@pytest.fixture
def robot(world):
    return world.robot if world is not None else None


@pytest.fixture
def scene(world):
    return world.scene if world is not None else None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        item._rw_call_report = rep
        if rep.failed and "world" in getattr(item, "fixturenames", ()) and not item.get_closest_marker("trials"):
            rep.user_properties.append(("robowright_trace", str(_trace_path(item.config, item.nodeid))))
        if hasattr(item, "_rw_trials"):
            rep.user_properties.append(("robowright_trials", item._rw_trials.summary()))


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    m = pyfuncitem.get_closest_marker("trials")
    if m is None:
        return None
    n = m.args[0] if m.args else m.kwargs.get("n", 10)
    report = TrialReport(pyfuncitem.nodeid, n, 0, m.kwargs.get("min_success", 1.0), m.kwargs.get("lower_bound", False))
    request = pyfuncitem._request
    backend = pyfuncitem.funcargs.get("rw_backend", "mujoco")
    argnames = pyfuncitem._fixtureinfo.argnames
    base = _seed(pyfuncitem)
    for i in range(n):
        w = _make_world(pyfuncitem, request, backend, base + i, f"[trial {i}]")
        w.robot.reset_to()
        args = {k: pyfuncitem.funcargs[k] for k in argnames}
        args.update({k: v for k, v in (("world", w), ("robot", w.robot), ("scene", w.scene)) if k in args})
        if "seed" in args:
            args["seed"] = base + i
        failed, msg = False, ""
        try:
            pyfuncitem.obj(**args)
        except (AssertionError, RobowrightError) as e:
            failed, msg = True, f"{type(e).__name__}: {e}"
        path = None
        try:
            path = w.close(failed=failed, trace_path=_trace_path(pyfuncitem.config, f"{pyfuncitem.nodeid}[trial {i}]"))
        except AssertionError as e:  # soft expectations
            failed, msg = True, str(e)
        if failed:
            report.failures.append((base + i, msg.splitlines()[0] if msg else "", str(path) if path else None))
            if path:
                pyfuncitem.config.stash[_TRACES].append((f"{pyfuncitem.nodeid}[seed {base + i}]", str(path)))
        else:
            report.passed += 1
    pyfuncitem._rw_trials = report
    pyfuncitem.config.stash[_REPORTS][pyfuncitem.nodeid] = report
    if not report.ok:
        __tracebackhide__ = True
        lines = [f"trials: {report.summary()}"] + [
            f"  seed {s}: {msg}" + (f"\n    trace: {p}" if p else "") for s, msg, p in report.failures[:10]
        ]
        raise AssertionError("\n".join(lines))
    return True


def pytest_runtest_logreport(report):
    # Runs on the xdist controller too, so traces from workers are listed.
    for k, v in report.user_properties:
        if k == "robowright_trace" and report.failed:
            _collected.setdefault("traces", []).append((report.nodeid, v))
        if k == "robowright_trials":
            _collected.setdefault("trials", []).append((report.nodeid, v, report.outcome))


_collected: dict = {}


def pytest_terminal_summary(terminalreporter):
    trials = _collected.get("trials", [])
    traces = list(dict.fromkeys(_collected.get("traces", [])))
    if trials:
        terminalreporter.section("robowright trials")
        for nodeid, summary, outcome in trials:
            terminalreporter.line(f"{'PASS' if outcome == 'passed' else 'FAIL'} {nodeid}: {summary}")
    if traces:
        terminalreporter.section("robowright traces")
        for nodeid, path in traces:
            terminalreporter.line(f"{nodeid}\n    robowright show-trace {path}")
    _collected.clear()
