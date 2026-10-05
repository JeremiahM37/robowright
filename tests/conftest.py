import pytest

import robowright as rw

pytest_plugins = ["pytester"]


@pytest.fixture
def quiet_world():
    """A world with tracing off, for unit tests that drive the API directly."""
    worlds = []

    def make(**kw):
        kw.setdefault("settings", rw.Settings(trace="off"))
        w = rw.launch(**kw)
        w.robot.reset_to()
        worlds.append(w)
        return w

    yield make
    for w in worlds:
        if w.status == "running":
            w.backend.close()


@pytest.hookimpl(tryfirst=True)  # before xdist reads the groups
def pytest_collection_modifyitems(config, items):
    """Keep each heavy engine's tests in this repo's own engine sweeps on few workers.

    ``test_backends`` and ``test_reuse`` run Genesis and Drake whatever ``--rw-backend`` says, and
    ``-n auto`` sizes workers by the engines selected there. Grouped per engine, robot and module, a
    MuJoCo run with ``--dist loadgroup`` loads Genesis (~5 GB a worker) on a few workers, not all.
    """
    for item in items:
        p = getattr(item, "callspec", None)
        if p is not None and p.params.get("backend") in ("genesis", "drake", "isaac"):
            item.add_marker(pytest.mark.xdist_group(f"{p.params['backend']}-{p.params.get('robot', '')}-{item.path.stem}"))
