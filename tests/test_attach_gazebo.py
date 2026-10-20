"""robowright attached to Gazebo's own shipped worlds, with no ROS: the worlds run in their own ``gz sim`` process.

Two of Gazebo's example worlds, unchanged, with the plugins they load and no ROS anywhere:

* ``lift_drag.sdf``: a revolute rod (``rod_1_joint``) that stays at rest until a force is commanded on its
  ``cmd_force`` topic, which the world's ApplyJointForce plugin applies, and whose JointStatePublisher publishes
  its joints. The test is the commanding side: it publishes the force, as ``gz topic -p`` would.
* ``diff_drive.sdf``: loads no JointStatePublisher, so a joint read has to say so.

The shipped worlds are the ones in the Gazebo install of this environment. Needs Gazebo Harmonic's Python
bindings (``gz.transport13``, ``gz.msgs10``) and its ``gz`` command (which runs under the env's ruby). Skipped without them.
Each simulation gets its own ``GZ_PARTITION`` and its own process group, which the test stops by the PID it recorded.
"""

import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

try:
    from gz.msgs10 import double_pb2
    from gz.transport13 import Node
except ImportError:  # pragma: no cover - depends on the install
    pytest.skip("needs Gazebo Harmonic's Python bindings (gz.transport13, gz.msgs10)", allow_module_level=True)

ENV_BIN = Path(sys.executable).parent
GZ = ENV_BIN / "gz"
WORLDS = Path(sys.prefix) / "share/gz/gz-sim8/worlds"
if not GZ.exists() or not (WORLDS / "lift_drag.sdf").exists():
    pytest.skip("needs Gazebo's gz command and shipped worlds in this Python's environment", allow_module_level=True)

os.environ.setdefault("GZ_PARTITION", f"rw-gzattach-{os.getpid()}")  # before any transport node exists

import robowright as rw  # noqa: E402
from robowright import expect  # noqa: E402
from robowright.errors import CapabilityError, InvariantViolation  # noqa: E402
from robowright.gazebo import worlds  # noqa: E402
from robowright.stack import popen_reaped  # noqa: E402

FORCE_TOPIC = "/model/lift_drag_demo_model/joint/rod_1_joint/cmd_force"
COMMANDED_FORCE = 0.7  # N m, the shipped world's own example value
DT = 0.001  # the shipped world's max_step_size


def _start(world_file: Path, *, running: bool, log: Path) -> subprocess.Popen:
    """``gz sim -s [-r] --headless-rendering`` in its own process group; the caller stops it with _stop."""
    args = [str(GZ), "sim", "-s"] + (["-r"] if running else []) + ["--headless-rendering", str(world_file)]
    env = dict(os.environ, PATH=f"{ENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}")
    with open(log, "w") as out:
        return popen_reaped(args, stdout=out, stderr=subprocess.STDOUT, env=env)  # gone with this process however it ends


def _wait_for_world(name: str, proc: subprocess.Popen, log: Path, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while name not in worlds(timeout=1.0):
        if proc.poll() is not None or time.monotonic() > deadline:
            _stop(proc)
            raise RuntimeError(f"gz sim did not offer world {name!r}; log: {log}")


def _stop(proc: subprocess.Popen) -> None:
    """Interrupt the simulation's process group by the PID it was started with, then kill that group if it does not stop."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


@pytest.fixture
def gz_world(tmp_path):
    """Start a shipped world (``running`` decides -r), wait until Gazebo offers it, and stop its process group afterwards."""
    started = []

    def start(world_file: str, name: str, *, running: bool) -> None:
        log = tmp_path / f"gz-{name}.log"
        proc = _start(WORLDS / world_file, running=running, log=log)
        started.append(proc)
        _wait_for_world(name, proc, log)

    yield start
    for proc in started:
        _stop(proc)


def _command_force(value: float) -> None:
    """Publish the force on the world's cmd_force topic: the commanding side of the example, as ``gz topic -p`` does it."""
    node = Node()
    pub = node.advertise(FORCE_TOPIC, double_pb2.Double)
    deadline = time.monotonic() + 15
    while not pub.has_connections():  # the world's plugin subscribes when it loads; the message must reach it
        if time.monotonic() > deadline:
            raise RuntimeError(f"nothing subscribes to {FORCE_TOPIC} in this Gazebo world")
        time.sleep(0.05)
    assert pub.publish(double_pb2.Double(data=value)), "Gazebo refused the force message"
    return node, pub


def test_running_world_moves_on_its_own_and_robowright_observes_its_clock(gz_world):
    gz_world("lift_drag.sdf", "lift_drag", running=True)
    _command = _command_force(COMMANDED_FORCE)  # kept referenced: the publisher's connections live with it

    w = rw.attach(engine="gazebo", settings=rw.Settings(trace="off"))
    try:
        assert w.backend.trace_meta()["mode"] == "observed"
        assert "rod_1" in w.bodies() and "body" in w.bodies()

        # The world's own clock runs without any step of robowright's.
        t_a = w.backend.time
        time.sleep(0.2)
        moved = w.backend.time - t_a
        assert moved > 0.1, f"the running world's clock moved {moved:.3f} s in 0.2 s of wall time"

        # The commanded torque turns the rod: the joint passes 1 rad, and the rod's link spins about -z (the joint's axis).
        # Read it there: the velocity is a finite difference over 0.02 s, which aliases once the rod turns past pi in that
        # span (about 157 rad/s; by 1 rad the rod turns at about -15 rad/s).
        expect(w.joint("rod_1_joint")).to_be_between(1.0, 1e6, timeout=20)
        # Read until it holds, not once: in a world running on its own, two pose samples can land close together under
        # load and one reading come out low (seen once in a full gate: -0.31 rad/s with the rod well past 1 rad).
        expect(w.body("rod_1")).to_satisfy(lambda b: b.velocity[5] < -1.0, "turning about the joint's axis (-z)", timeout=2)

        # world.step() in observed mode takes the world's own physics steps, so it runs in real time.
        # The clock is read before the first wait, so the world may run free for up to a step before it: the
        # sim time covers the 1000 steps and that sliver, and the wall time can never be shorter than the sim time.
        n0, t0, wall0 = w.step_count, w.backend.time, time.monotonic()
        w.step(1000)
        sim, wall = w.backend.time - t0, time.monotonic() - wall0
        assert w.step_count - n0 == 1000
        assert 1000 * DT - 1e-9 <= sim < 1000 * DT + 0.01, f"1000 steps covered {sim:.6f} s of sim time"
        assert wall >= 0.95 * sim, f"1000 steps took {wall:.3f} s of wall time for {sim:.3f} s of sim time"
        print(f"observed: 1000 steps = {sim:.6f} s sim in {wall:.3f} s wall (ratio {wall / sim:.3f})")
    finally:
        w.close(failed=False)


def test_paused_world_is_stepped_by_robowright_and_its_invariant_fires(gz_world, tmp_path):
    gz_world("lift_drag.sdf", "lift_drag", running=False)  # started paused: robowright will step it
    _command = _command_force(COMMANDED_FORCE)

    w = rw.attach(engine="gazebo", settings=rw.Settings(trace="on", trace_dir=str(tmp_path)))
    try:
        assert w.backend.trace_meta()["mode"] == "stepped"
        t0 = w.backend.time
        w.step(100)
        t1 = w.backend.time
        assert t1 - t0 == pytest.approx(100 * DT, abs=1e-9)
        theta100 = w.joint("rod_1_joint").value
        w.step(100)
        t2 = w.backend.time
        assert t2 - t1 == pytest.approx(100 * DT, abs=1e-9)
        theta200 = w.joint("rod_1_joint").value

        # Constant torque from rest: the angle grows as t squared (the ratio is 4; damping pulls it slightly under).
        assert 0 < theta100 < theta200
        assert theta200 / theta100 == pytest.approx(4.0, rel=0.02)
        # The pose and the joint come from the same step: the link's yaw is minus the joint angle, the joint's axis being -z.
        assert w.body("rod_1").yaw == pytest.approx(-theta200, abs=1e-6)
        print(f"stepped: t={t2:.3f} s theta100={theta100:.6f} theta200={theta200:.6f} ratio={theta200 / theta100:.4f}")

        # The anchor the rod hangs from does not move (the shipped model's body link, at (0, 0, 0.5)): an invariant on
        # it holds through the run, and the joint's invariant below is the one that fires.
        expect(w.body("body")).always.to_be_near((0.0, 0.0, 0.5), tol=1e-4)
        # The angle passes 3 rad a little after step 200: an invariant that bounds it must fire then, and not before.
        expect(w.joint("rod_1_joint")).always.to_be_between(-10.0, 3.0)
        with pytest.raises(InvariantViolation):
            w.step(200)
        assert w.step_count > 200
        assert w.backend.joint_value("rod_1_joint") > 3.0
        print(f"invariant fired at step {w.step_count}, theta={w.backend.joint_value('rod_1_joint'):.6f}")
    finally:
        saved = w.close(failed=False)
    # The trace keeps the bodies' poses per step, so the run can be checked after the simulation is gone.
    assert saved is not None and saved.exists()
    with zipfile.ZipFile(saved) as z:
        assert "bodies.npz" in z.namelist()


def test_world_without_joint_state_publisher_says_so(gz_world):
    gz_world("diff_drive.sdf", "diff_drive", running=False)
    w = rw.attach(engine="gazebo", settings=rw.Settings(trace="off"))
    try:
        with pytest.raises(CapabilityError, match="JointStatePublisher"):
            w.backend.named_joints()
        with pytest.raises(CapabilityError, match="JointStatePublisher"):
            w.joint("vehicle_blue::left_wheel_joint")
    finally:
        w.close(failed=False)
