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
