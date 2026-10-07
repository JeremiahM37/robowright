# Extending robowright

robowright has three layers, and each can be extended from outside it, without a fork:

| Layer | What it is | Extend it with |
|---|---|---|
| **Robots** | a `RobotModel`: the model file, which joints are the arm, the hand, the fingers, the gripper | a model file and a name in `robowright.toml`; or a package with a `robowright.robots` entry point |
| **Engines** | a `Backend`: physics (or hardware) for one scene, nothing else | a package with a `robowright.backends` entry point |
| **The core** | IK, planning, waiting, `expect`, faults, traces, replay, codegen | written once, shared by every robot and engine |

The split matters: a test means the same thing on every engine because everything a test
*says* lives in the core, and a backend only reports joints, steps time and, if it can,
contacts and object poses. A new engine or robot gets the whole API at once.

Whatever you add is held to the same **contract** the built-in robots and engines pass,
which ships with robowright:

```bash
robowright check --robot my_arm --backend mujoco          # a robot
robowright check --robot all,legged --backend my_engine   # an engine, on every robot
```

`check` runs `pytest --pyargs robowright.contract` with those options; any other argument
goes to pytest (`-k`, `-x`, `-n auto`).

## A robot

### From its model file, with no code

Any MJCF, URDF or xacro file works directly: `pytest --rw-robot path/to/arm.urdf`.
robowright works out the arm, hand, fingers and gripper from the model itself, and
`robowright robots --inspect path/to/arm.urdf` shows what it decided and why.

To give it a name your team can use, add it to the project:

```bash
robowright robots add models/my_arm.urdf --name my_arm
```

That checks robowright can drive it, then writes `robowright.toml`:

```toml
[robots.my_arm]
file = "models/my_arm.urdf"          # relative to this file; xacro arguments after a "?"
```

Now `pytest --rw-robot my_arm` and `robowright check --robot my_arm` work from anywhere in
the project. The same table can live in `pyproject.toml` as `[tool.robowright.robots.my_arm]`.
(`$ROBOWRIGHT_CONFIG` points at another settings file. A `robowright.toml` takes precedence
over `pyproject.toml` in the same directory.)

If detection gets something wrong, or the model leaves something out, set the
[`RobotModel`](../src/robowright/robots/model.py) field in the same table. What you set is
used as given, and what depends on it is worked out from it:

```toml
[robots.my_arm]
file = "models/my_arm.urdf"
gripper_open = 0.04                  # the model's gripper range is a placeholder
gripper_closed = 0.0
home = [0.2, 0.0, 0.15]              # TCP position of the home pose
grip_force = 70.0                    # N per jaw, from the gripper's datasheet...
grip_force_source = "https://example.com/datasheet.pdf"   # ...and where it says so
```

### As a package

A lab with several robots, or a vendor, can ship them as a Python package. Its entry point
is a callable returning `RobotModel`s:

```toml
# the package's pyproject.toml
[project.entry-points."robowright.robots"]
acme = "acme_robots.robowright:robots"
```

```python
# acme_robots/robowright.py
from pathlib import Path

from robowright.robots.detect import load

HERE = Path(__file__).parent


def robots():
    return [
        load(HERE / "models/acme_r1.urdf", "acme_r1"),
        load(HERE / "models/acme_r2.urdf", "acme_r2", gripper_open=0.05),
    ]
```

Once installed, `--rw-robot acme_r1` works and the robots are listed by `robowright robots`.
A robot can also be written out field by field as a `RobotModel`, as the built-in ones are
in `src/robowright/robots/__init__.py`. A plugin that fails to load is reported as a warning
and skipped, so it never stops the tests on other robots.

## An engine, or hardware

A backend subclasses [`Backend`](../src/robowright/backends/base.py). Six methods are
required:

| Method | Does |
|---|---|
| `qpos()`, `qvel()` | measured arm joint positions and velocities, then the gripper opening (0 closed .. 1 open) |
| `set_ctrl(target)`, `ctrl()` | position targets in the same layout |
| `step()` | advance one control period (`self.control_dt`) |
| `time` | seconds since the start |

Everything else is optional, and a backend says what it offers in `capabilities`:

| Capability | Methods | Without it |
|---|---|---|
| `GROUND_TRUTH` | `object_pose`, `object_velocity`, `set_object_pose` | object assertions fail with a clear message |
| `CONTACTS` | `contacts()` | `to_be_holding` and contact checks are unavailable |
| `RENDER` | `render(camera, w, h)` | traces carry no images |
| `STATE` | `get_state`, `set_state` | no mid-test restore; no scene reuse between tests |
| `DETERMINISTIC` | (same inputs, same outputs) | the bit-identical replay tests are skipped |
| `FORCES` | `apply_force` | fault injection (shoves) is unavailable |

The contract skips what a backend does not claim, and says so, rather than failing.
That is what makes a **hardware backend** a backend like any other. A real arm reports
joints, takes targets and steps in real time, and claims none of the simulator
capabilities. The same test then runs on the simulated and the real robot, with object
assertions reading perception instead of ground truth: a camera or motion capture registered
as `world.perception["cube"] = lambda: (pos, quat)`.

Register it with an entry point named for how users select it:

```toml
[project.entry-points."robowright.backends"]
my_engine = "my_engine_robowright:MyEngineBackend"
```

```python
from robowright.backends.base import CONTACTS, GROUND_TRUTH, Backend


class MyEngineBackend(Backend):
    capabilities = frozenset({GROUND_TRUTH, CONTACTS})

    def __init__(self, spec, seed=0):
        super().__init__(spec, seed)
        # spec.robot_model.mjcf() is the robot's MJCF. For an engine that reads URDF,
        # robowright.robots.urdf.load(spec.robot_model) returns it as one, with the metadata
        # robowright adds (finger couplings, telescopes, servo gains), as the built-ins use it.
        ...
```

`pytest --rw-backend my_engine` and `robowright check --backend my_engine` then select it,
and `robowright info` lists it. The built-in backends are worked examples. MuJoCo's
(`mujoco_backend.py`, 330 lines) is the shortest; PyBullet's builds the robot from the URDF
export, as most engines will; Isaac Sim's shows an engine that owns a process-wide app
(`exclusive = True`).

Rules a backend keeps:

- **Physics only.** No IK, no waiting, no assertions. If a robot needs special handling
  on your engine, the robot model is usually what is wrong; fix it there, so every
  engine benefits.
- **Ramp servo targets** across physics substeps (`TargetRamp` in `base.py`). Stepped
  targets shake light grips loose.
- **When the engine genuinely disagrees** with the others for a robot, record the
  measurement as a known divergence (see `conftest.py`) rather than tuning around it.

## Where it grows

The extension points above are where growth is expected to go, each without changes to the
core:

- **Hardware**: a backend per robot family or middleware (LeRobot for SO-101-class arms,
  ROS 2 for anything with a `ros2_control` driver). It claims no simulator capabilities and
  gets a perception source for object poses.
- **More engines**: any simulator with joint position control fits the six required
  methods.
- **More robots**: model files in a project's `robowright.toml`, or robot packs from labs
  and vendors.

Changes that touch the core (new assertions, new skills like `pick`, new fault types)
belong in robowright itself; see [CONTRIBUTING.md](../CONTRIBUTING.md).
