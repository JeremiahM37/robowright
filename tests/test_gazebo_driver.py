"""robowright.gazebo's bookkeeping of what Gazebo publishes, without Gazebo: messages are fed to
the driver in the order (and with the timing) Gazebo's transport can deliver them."""

import threading
import time
from types import SimpleNamespace

from robowright.gazebo import Gazebo


class _Msg(SimpleNamespace):
    """Stands in for a gz.msgs message the driver builds a request from."""

    MODEL = 2

    def __init__(self, **fields):
        super().__init__(**fields)
        self.pose = SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace())


def _driver() -> Gazebo:
    g = Gazebo.__new__(Gazebo)  # the state Gazebo.__init__ sets up, with no simulator behind it
    g._Entity = SimpleNamespace(Entity=_Msg)
    g._Factory = SimpleNamespace(EntityFactory=_Msg)
    g._lock = threading.Lock()
    g._fresh = threading.Condition(g._lock)
    g._poses, g._history, g._stamp, g._removing = {}, {}, 0.0, set()
    g._stats = {"paused": False}
    return g


def _poses(t: float, *names: str):
    """A ``pose/info`` message: every model in the world, stamped ``t`` seconds."""
    point = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    quat = SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0)
    stamp = SimpleNamespace(sec=int(t), nsec=int(round((t - int(t)) * 1e9)))
    return SimpleNamespace(
        pose=[SimpleNamespace(name=n, position=point, orientation=quat) for n in names],
        header=SimpleNamespace(stamp=stamp),
    )


def test_a_removed_model_stays_removed_when_older_poses_arrive_late():
    g = _driver()
    g._call = lambda *args: None  # Gazebo removed it
    g._on_poses(_poses(1.0, "cube", "ball"))

    def gazebo():
        time.sleep(0.05)
        g._on_poses(_poses(1.1, "cube", "ball"))  # published before the removal, delivered after it
        time.sleep(0.05)
        g._on_poses(_poses(1.2, "cube"))  # the world without it

    later = threading.Thread(target=gazebo)
    later.start()
    g.remove("ball")
    later.join()

    assert g.models() == ["cube"]
    assert not g.has("ball")


def test_a_model_spawned_again_after_its_removal_is_seen():
    g = _driver()
    g._call = lambda *args: None
    g._on_poses(_poses(1.0, "ball"))
    threading.Timer(0.05, g._on_poses, args=(_poses(1.1),)).start()
    g.remove("ball")

    g.spawn("ball", "<sdf/>", (0.1, 0.2, 0.3))
    g._on_poses(_poses(1.2, "ball"))

    assert g.has("ball")
