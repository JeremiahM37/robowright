"""robowright attached to one of Webots' own shipped worlds, while that world's own controllers move the robot.

The world is Webots R2025a's ``universal_robots/worlds/ure.wbt``: a UR3e with a Robotiq gripper picking
cans from a conveyor, driven by the world's ``ure_can_grasper`` controller. This test copies that world
into a scratch project, names the gripper and one can with DEF names, and declares one robot with
``controller "<extern>"`` and ``supervisor TRUE``. This process connects to that robot as a Supervisor
and steps the simulation; the world's own controllers run inside Webots and are not touched.

Needs an installed Webots R2025a (``WEBOTS_HOME``), an X display (``DISPLAY``; Xvfb works), and the
world's EXTERNPROTO files reachable (Webots fetches them from GitHub). A host whose OpenGL lacks
``libGLU.so.1`` needs it on ``LD_LIBRARY_PATH``. Skipped without ``WEBOTS_HOME``.
"""

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import robowright as rw
from robowright import expect
from robowright.stack import popen_reaped

WEBOTS_HOME = os.environ.get("WEBOTS_HOME")
PORT = 1234  # the port Webots' extern controller listens on
BODIES = {"gripper": "ROBOWRIGHT_GRIPPER", "can": "ROBOWRIGHT_CAN"}
# Where the gripper is when the world's controller has it at the can, measured in the world's
# own run (about 12 s of simulated time after the start): a point the world's controller reaches,
# not one this test chose to pass.
GRASP_POINT = (0.948, 0.219, 1.070)
LIFT_HEIGHT = 1.0  # the can is lifted above this height at some point in the world's cycle

pytestmark = pytest.mark.skipif(
    not WEBOTS_HOME or not Path(WEBOTS_HOME, "webots").exists(),
    reason="needs an installed Webots R2025a: set WEBOTS_HOME to its directory",
)


def _probe_world(home: Path) -> str:
    """The shipped world with two DEF names and a declared Supervisor robot, and nothing else changed."""
    src = (home / "projects/robots/universal_robots/worlds/ure.wbt").read_text()
    # The first gripper is the UR3e's, the arm whose controller the test observes.
    gripper = "  toolSlot [\n    Robotiq3fGripper {\n    }\n"
    at = src.index(gripper)
    assert src.rfind("\nUR3e {", 0, at) > src.rfind("\nUR5e {", 0, at), "the shipped world changed"
    src = src[:at] + gripper.replace("Robotiq3fGripper", f"DEF {BODIES['gripper']} Robotiq3fGripper") + src[at + len(gripper) :]
    src = src.replace("\nCan {\n", f"\nDEF {BODIES['can']} Can {{\n", 1)
    return src + ('\nRobot {\n  name "robowright_probe"\n  controller "<extern>"\n  supervisor TRUE\n  synchronization TRUE\n}\n')


def _start_world(project: Path) -> subprocess.Popen:
    """The probe world in ``project``, started headless with its extern port open. The caller stops it with _stop."""
    home = Path(WEBOTS_HOME)
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"port {PORT} is already in use: Webots' extern controller needs it")
    (project / "worlds").mkdir(exist_ok=True)
    if not (project / "controllers").exists():
        (project / "controllers").symlink_to(home / "projects/robots/universal_robots/controllers")
    world = project / "worlds" / "ure_probe.wbt"
    world.write_text(_probe_world(home))
    log_path = project / "webots.log"
    with open(log_path, "w") as log:
        proc = popen_reaped(  # its own process group, so _stop reaches the controllers too, and gone with this process
            [str(home / "webots"), "--no-rendering", "--batch", "--mode=fast", "--stdout", "--stderr", "--minimize", str(world)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
        )
    deadline = time.monotonic() + 180
    while "Waiting for local or remote connection" not in log_path.read_text(errors="replace"):
        if proc.poll() is not None or time.monotonic() > deadline:
            _stop(proc)
            raise RuntimeError(f"Webots did not open its extern port; log: {log_path}")
        time.sleep(0.5)
    return proc


def _stop(proc: subprocess.Popen) -> None:
    """Stops the whole Webots session (its controllers are children of webots-bin and hold the extern port
    with it). Terminating only webots-bin left them orphaned, and the next world then found port 1234 taken."""
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)  # controllers can outlive webots-bin: sweep the group


def _connect_supervisor():
    """This process as the world's extern Supervisor. The controller module reads its URL from the environment, once."""
    sys.path.insert(0, str(Path(WEBOTS_HOME) / "lib/controller/python"))
    os.environ["WEBOTS_CONTROLLER_URL"] = f"tcp://127.0.0.1:{PORT}/robowright_probe"
    os.environ["WEBOTS_ROBOT_NAME"] = "robowright_probe"
    from controller import Supervisor

    return Supervisor()


@pytest.fixture(scope="module")
def webots_project(tmp_path_factory):
    """Start Webots headless on the probe world, wait for the extern port, stop it by PID afterwards."""
    if not os.environ.get("DISPLAY"):
        pytest.skip("needs an X display: set DISPLAY (for example to an Xvfb started with -nolisten tcp)")
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            pytest.skip(f"port {PORT} is already in use: Webots' extern controller needs it")
    proj = tmp_path_factory.mktemp("webots-project")
    proc = _start_world(proj)
    try:
        yield proj
    finally:
        _stop(proc)


@pytest.fixture
def supervisor(webots_project):
    sup = _connect_supervisor()
    yield sup
    sys.path.remove(str(Path(WEBOTS_HOME) / "lib/controller/python"))


def test_expectations_hold_while_the_worlds_own_controller_moves_the_arm(supervisor, tmp_path):
    from robowright.backends.attached_webots import WebotsEnv

    env = WebotsEnv(supervisor, BODIES)
    # The steps go through the wrapped step: advance=env.step, bound before attach wraps it, would take steps the trace never sees.
    w = rw.attach(env, engine="webots", advance=lambda: env.step(), settings=rw.Settings(trace_dir=str(tmp_path)))
    try:
        # Their controller reaches for the can and lifts it; these checks only observe that.
        expect(w.body("gripper")).to_be_near(GRASP_POINT, tol=0.05, timeout=25)
        expect(w.body("can")).to_be_above(LIFT_HEIGHT, timeout=25)
        assert w.backend.time > 10.0
    finally:
        w.close(failed=False)
