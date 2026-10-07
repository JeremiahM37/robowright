# Extending robowright

robowright has three layers, and what runs on them (robots, engines, policies) can be added
from outside it, without a fork:

| Layer | What it is | Extend it with |
|---|---|---|
| **Robots** | a `RobotModel`: the model file, which joints are the arm, the hand, the fingers, the gripper | a model file and a name in `robowright.toml`; or a package with a `robowright.robots` entry point |
| **Engines** | a `Backend`: physics (or hardware) for one scene, nothing else | a package with a `robowright.backends` entry point; robots behind ROS 2 need none |
| **Learned policies** | a trained model, run by `LearnedPolicy` | a model file or object; a package with a `robowright.policies` entry point for a framework's checkpoints |
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
as `world.perception["cube"] = lambda: (pos, quat)`, or sources the backend has itself,
returned by its `perception()` method (the ROS 2 backend's TF frames).

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

## A robot behind ROS 2

A robot with a ROS 2 driver needs no backend of its own: the built-in `ros2` backend reads
`/joint_states` and drives the standard `ros2_control` controllers, so describing the robot
(its model file, as above) and its controllers is enough:

```toml
[ros2]
namespace = ""                                   # prefixes the topics below
arm = { controller = "forward_position_controller", interface = "position" }
gripper = { interface = "with_arm" }              # or { controller = "...", interface = "action" | "position" | "trajectory" | "none" }
joints = { shoulder_pan = "joint1" }              # robowright name -> the driver's, where they differ
frame = "base_link"                               # the robot's base frame, for TF
objects = { cube = "cube" }                       # object -> TF frame (its pose, relative to `frame`)
cameras = { front = "/camera/image_raw" }         # camera -> sensor_msgs/Image topic
use_sim_time = false                              # true: follow /clock (Gazebo, Isaac Sim)
```

| `arm` / `gripper` interface | Controller | What is sent each control step |
|---|---|---|
| `position` | `position_controllers/JointGroupPositionController` | `Float64MultiArray` on `/<controller>/commands` |
| `trajectory` | `joint_trajectory_controller/JointTrajectoryController` | a one-point `JointTrajectory` reaching the targets one period later |
| `action` (gripper) | `GripperActionController` (`gripper_controllers`, `parallel_gripper_controller`) | a `GripperCommand` goal when the target changes |
| `with_arm` (gripper) | the arm's forward controller, the gripper joint last | in the arm's message |

Then `pytest --rw-backend ros2` and `robowright check --backend ros2` run on the robot. The
settings can also be passed to `rw.launch(..., backend="ros2", namespace="/arm1")`. What the
backend cannot do on a real robot it does not pretend to: there is no ground truth, so object
assertions read the TF frames; there are no contact sensors, so `to_be_holding` is judged from
the jaws (told to close, stopped short on something, the object within the hand) and says so;
`robot.reset_to` drives to the pose (at `reset_speed`, 0.5 rad/s) instead of teleporting.

Run tests from a ROS 2 environment (`source /opt/ros/jazzy/setup.bash`, then `pip install
robowright`). ROS 2's `launch_testing` pytest plugins do not load under pytest 9; `robowright
check` leaves them out, and a plain `pytest` run needs `-p no:launch_testing -p no:launch_ros`.
`scripts/ros2_env.sh` builds a ROS 2 Jazzy environment without root (RoboStack) for robowright's
own ROS 2 tests, which run the backend against `ros2_control`'s controllers on mock hardware
and a simulated arm behind the same topics.

## A learned policy

[`LearnedPolicy`](../src/robowright/learned.py) runs any trained model in `run_policy`. Its
settings say how the model was trained, and it translates both ways: the observation
(`state` features, `images` and their layout, the task) into the model's batch, and the
model's action (one, or a chunk) back into joint targets, through the model's joint order
(`joints`), `units`, `gripper` range and `normalize` statistics.

A model is a Python object, an ONNX, `torch.export` or TorchScript file, `"module:name"`, or
`"loader:reference"` for a framework's own checkpoint format. A loader is a plugin:

```toml
[project.entry-points."robowright.policies"]
my_framework = "my_framework_robowright:load"
```

```python
def load(reference: str, device: str | None = None):
    """The model for ``reference`` (a hub id, a run directory: whatever the framework uses)."""
    return MyFrameworkPolicy.from_pretrained(reference).to(device or "cpu").eval()
```

after which `LearnedPolicy("my_framework:org/pick-v2", images=["front"])` loads it. Models
with a `select_action(batch)` method are called through it, and their `reset()` is called
before each rollout. A project names its policies, with their settings, next to its robots:

```toml
[policies.pick_v2]
model = "checkpoints/pick_v2.onnx"         # relative to this file
state = ["qpos", "objects.cube"]
images = ["front"]
units = "deg"
gripper = [0, 100]
normalize = "checkpoints/pick_v2_stats.json"
```

`LearnedPolicy("pick_v2")` then loads it with those settings (anything passed as a keyword
overrides them), and a regression test generated from a trace recreates it by that name.

## Where it grows

The extension points above are where growth is expected to go, each without changes to the
core:

- **Hardware**: robots with a ROS 2 driver run through the `ros2` backend today. A robot
  without one gets a backend of its own, which claims no simulator capabilities and reports
  object poses through `perception()`, as the ROS 2 backend does from TF.
- **Learned policies**: a framework plugs in its checkpoint format with a policy loader;
  robowright stays independent of all of them.
- **More engines**: any simulator with joint position control fits the six required
  methods.
- **More robots**: model files in a project's `robowright.toml`, or robot packs from labs
  and vendors.

Changes that touch the core (new assertions, new skills like `pick`, new fault types)
belong in robowright itself; see [CONTRIBUTING.md](../CONTRIBUTING.md).
