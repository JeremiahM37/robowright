"""pytest integration.

Fixtures: ``world``, ``robot``, ``scene`` (fresh per test), ``rw_scene``
(override in a conftest to change the default scene).

Options::

    --rw-backend mujoco,pybullet     run every robot test on each backend
    --rw-robot so101,panda | all | legged   run every robot test on each robot ("all": every arm)
    --rw-robot path/to/robot.xml             ...or on any robot, from its model file (MJCF or URDF)
    --rw-trace on|off|retain-on-failure
    --rw-trace-dir DIR
    --rw-seed N                      base seed
    --rw-all-trials                  run every trial of a trials test, not just until the verdict is settled
    -n auto                          (pytest-xdist) as many workers as the cores and memory allow

Markers::

    @pytest.mark.scene(spec_or_factory)
    @pytest.mark.seed(7)
    @pytest.mark.backends("mujoco")              # restrict
    @pytest.mark.robots("panda", "ur5e")         # restrict
    @pytest.mark.trials(20, min_success=0.9)     # run across up to 20 seeds, judge the rate
"""

from __future__ import annotations

import os
import re
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
    g.addoption(
        "--rw-robot", default="so101", help="robots to run robot tests on (comma-separated): names, 'all', 'legged', or MJCF/URDF files"
    )
    g.addoption("--rw-trace", default="retain-on-failure", choices=["on", "off", "retain-on-failure"])
    g.addoption("--rw-trace-dir", default="robowright-traces")
    g.addoption("--rw-seed", type=int, default=0, help="base seed added to each test's seed")
    g.addoption(
        "--rw-all-trials",
        action="store_true",
        help="run all n trials of a trials test; by default it stops once the rest cannot change the verdict",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "scene(spec): scene spec or zero-arg factory for this test")
    config.addinivalue_line("markers", "seed(n): seed for this test")
    config.addinivalue_line("markers", "backends(*names): only run on these backends")
    config.addinivalue_line("markers", "robots(*names): only run on these robots")
    config.addinivalue_line("markers", "trials(n, min_success=1.0, lower_bound=False): run across n seeds and judge the success rate")
    config.addinivalue_line("markers", "xdist_group(name): pytest-xdist's grouping, set per robot and engine")
    config.stash[_TRACES] = []
    config.stash[_REPORTS] = {}
    config.pluginmanager.register(_Summary(), "robowright-summary")
    if any(_robots.is_file(r.strip()) for r in config.getoption("--rw-robot").split(",")):
        _robot_names(config)  # a robot model file robowright cannot drive is a usage error, said once
    if getattr(config.option, "dist", None) == "loadgroup" and "--loadscope-reorder" not in config.invocation_params.args:
        # xdist hands out the largest groups first, which puts a lone trials test, the longest
        # kind, at the end of the run; robowright's own order (pytest_collection_modifyitems)
        # starts those first.
        config.option.loadscopereorder = False


@pytest.hookimpl(optionalhook=True, tryfirst=True)
def pytest_xdist_auto_num_workers(config):
    """``-n auto``: workers by the cores and memory the selected engines need (see :mod:`robowright.workers`)."""
    if os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return None  # xdist's own hook reads it
    from .workers import auto_workers

    return auto_workers([b.strip() for b in config.getoption("--rw-backend").split(",") if b.strip()])


def _robot_names(config) -> list[str]:
    raw = [r.strip() for r in config.getoption("--rw-robot").split(",") if r.strip()]
    out = []
    for r in raw:
        if r in ("all", "arms", "legged"):
            out.extend(_robots.names({"all": "arm", "arms": "arm", "legged": "legged"}[r]))
        elif _robots.is_file(r):  # a model file: tests get its absolute path, so traces replay from anywhere
            from pathlib import Path

            file, _, query = r.partition("?")  # arm.urdf.xacro?ur_type=ur5e: xacro arguments
            out.append(str(Path(file).expanduser().resolve()) + (f"?{query}" if query else ""))
            from .robots.detect import DetectionError

            try:
                _robots.get(out[-1])
            except (DetectionError, FileNotFoundError) as e:
                raise pytest.UsageError(f"--rw-robot {r}: robowright cannot drive this robot: {e}") from None
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
            metafunc.parametrize("rw_robot", names, ids=[_robots.get(n).name for n in names], scope="function")


@pytest.hookimpl(tryfirst=True)  # before xdist reads the groups into node ids
def pytest_collection_modifyitems(config, items):
    """Run each module's tests robot by robot, and group them for ``-n N --dist loadgroup``.

    A closed world's scene is kept and restored for the next world that needs it (see
    ``backends.create``), which runs bit-for-bit like a new build, so the order cannot change a
    result. Ordered and grouped, the tests that share a scene run one after another in one
    process and reuse it, instead of every robot's scene being built for every test.
    """
    module = {}

    def key(item):
        p = getattr(item, "callspec", None)
        params = p.params if p is not None else {}
        # Trials tests first: they are the longest, and handed out last they were the tail of
        # every parallel run.
        first = item.get_closest_marker("trials") is None
        return (first, module.setdefault(item.path, len(module)), params.get("rw_backend", ""), params.get("rw_robot", ""))

    items[:] = sorted(items, key=key)
    for item in items:
        p = getattr(item, "callspec", None)
        if p is not None and "rw_backend" in p.params:
            robot = p.params.get("rw_robot", "")
            group = f"{p.params['rw_backend']}-{_robots.get(robot).name if robot else ''}-{item.path.stem}"
            if item.get_closest_marker("trials"):
                # A trials test builds a world per trial, so it shares little with its robot's
                # other tests, and it is the longest: on its own it can start while they run.
                group += f"-{item.originalname}"
            item.add_marker(pytest.mark.xdist_group(group))


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


def _plain(nodeid: str) -> str:
    """``nodeid`` without the ``@group`` that ``--dist loadgroup`` appends to it."""
    return re.sub(r"@[\w.-]+$", "", nodeid)


def _trace_path(config, nodeid: str) -> Path:
    return Path(config.getoption("--rw-trace-dir")) / f"{_safe(_plain(nodeid).replace('::', '__').replace('/', '_'))}.zip"


def _make_world(item, request, backend: str, seed: int, suffix: str = "") -> World:
    w = World(_scene_for(item, request), backend=backend, seed=seed, name=_plain(item.nodeid) + suffix, settings=_settings(item.config))
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
        item.config.stash[_TRACES].append((_plain(item.nodeid), str(path)))


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
    report = TrialReport(_plain(pyfuncitem.nodeid), n, 0, m.kwargs.get("min_success", 1.0), m.kwargs.get("lower_bound", False))
    request = pyfuncitem._request
    backend = pyfuncitem.funcargs.get("rw_backend", "mujoco")
    argnames = pyfuncitem._fixtureinfo.argnames
    base = _seed(pyfuncitem)
    every = pyfuncitem.config.getoption("--rw-all-trials")
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
            path = w.close(failed=failed, trace_path=_trace_path(pyfuncitem.config, f"{_plain(pyfuncitem.nodeid)}[trial {i}]"))
        except AssertionError as e:  # soft expectations
            failed, msg = True, str(e)
        if failed:
            report.failures.append((base + i, msg.splitlines()[0] if msg else "", str(path) if path else None))
            if path:
                pyfuncitem.config.stash[_TRACES].append((f"{_plain(pyfuncitem.nodeid)}[seed {base + i}]", str(path)))
        else:
            report.passed += 1
        report.ran = i + 1
        if report.settled and not every:
            break  # same seeds, same order: the verdict all n trials would give
    pyfuncitem._rw_trials = report
    pyfuncitem.config.stash[_REPORTS][pyfuncitem.nodeid] = report
    if not report.ok:
        __tracebackhide__ = True
        lines = [f"trials: {report.summary()}"] + [
            f"  seed {s}: {msg}" + (f"\n    trace: {p}" if p else "") for s, msg, p in report.failures[:10]
        ]
        raise AssertionError("\n".join(lines))
    return True


class _Summary:
    """The trials and traces of one session, listed at its end.

    One per session, not module state: a session run inside another (pytester, or a test
    worker running the plugin's own tests) printed the outer run's results and wiped them.
    """

    def __init__(self):
        self.trials: list = []
        self.traces: list = []

    def pytest_runtest_logreport(self, report):
        # Runs on the xdist controller too, so traces from workers are listed.
        for k, v in report.user_properties:
            if k == "robowright_trace" and report.failed:
                self.traces.append((_plain(report.nodeid), v))
            if k == "robowright_trials":
                self.trials.append((_plain(report.nodeid), v, report.outcome))

    def pytest_terminal_summary(self, terminalreporter):
        if self.trials:
            terminalreporter.section("robowright trials")
            for nodeid, summary, outcome in self.trials:
                terminalreporter.line(f"{'PASS' if outcome == 'passed' else 'FAIL'} {nodeid}: {summary}")
        if self.traces:
            terminalreporter.section("robowright traces")
            for nodeid, path in dict.fromkeys(self.traces):
                terminalreporter.line(f"{nodeid}\n    robowright show-trace {path}")
