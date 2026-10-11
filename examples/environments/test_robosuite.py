"""robosuite's Lift, with robowright attached through ``env.sim``. robosuite ships demo playback and device
scripts but no policy, so this is an attach check: random actions through robosuite's own OSC controller.

Nothing here asserts the task's success. robosuite's success is its own private check, and random actions
never meet it; the checks are what robowright reads of the cube and gripper, an invariant, a fault after
reset, and a trace redrawn to a PNG.

Needs mujoco 3.3.x in the test venv. With mujoco 3.15, robosuite 1.5.2 fails in its own robot setup
(``get_joint_qpos_addr`` asserts ``joint_type in (mjJNT_HINGE, mjJNT_SLIDE)``, and a numpy int is not
equal to the enum member), so the suite cannot build a robot there at all (measured).
"""

import os
from pathlib import Path

import numpy as np
import pytest

import robowright as rw
from robowright import expect

suite = pytest.importorskip("robosuite")

STEPS = 60


def _trace_dir(tmp_path_factory) -> Path:
    d = os.environ.get("RS_TRACE_DIR")
    return Path(d) if d else tmp_path_factory.mktemp("rs_traces")


def test_lift_attach_checks_fault_and_redraw(tmp_path_factory):
    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=STEPS + 10,
    )
    world = rw.attach(env, name="robosuite_lift", settings=rw.Settings(trace_dir=str(_trace_dir(tmp_path_factory))))
    try:
        env.reset()
        cube = world.body("cube_main")
        table = float(world.body("table").position[2])  # the table body sits at its surface
        expect(cube).always.to_satisfy(lambda s: s.position[2] > table, "cube above the table surface")
        z0 = float(cube.position[2])  # on the table, so the table surface is below it
        world.move_object("cube_main", cube.position + np.array([0.0, 0.05, 0.0]))  # the fault: the cube moved after reset
        y_fault = float(cube.position[1])
        low, high = env.action_spec
        rng = np.random.default_rng(0)
        for _ in range(STEPS):
            env.step(rng.uniform(low, high))
        contacts = cube.contacts()
        print(
            f"\nlift cube z0={z0:.3f} y_after_fault={y_fault:.3f} y_end={float(cube.position[1]):.3f} "
            f"z_end={float(cube.position[2]):.3f} contacts={contacts}"
        )
        assert world.step_count == STEPS
        path = world.close(failed=False)
        from PIL import Image

        from robowright.attached import AttachedRenderer
        from robowright.trace import Trace

        r = AttachedRenderer(Trace(path))
        try:
            img = r.frame(r.steps - 1, size=(320, 240))
        finally:
            r.close()
        assert img.any()
        Image.fromarray(img).save(_trace_dir(tmp_path_factory) / "lift-final.png")
    except BaseException:
        world.close(failed=True)
        raise
