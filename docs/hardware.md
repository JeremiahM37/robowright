# Testing on a real arm

robowright reaches a real robot the same way it reaches Gazebo: over ROS 2, through the
standard `ros2_control` interfaces. Nothing in a test changes between simulation and the arm;
only `--rw-backend` does. This page is the order to bring an arm up in, so that the first thing
that moves is something you chose to move.

None of this has run on a physical robot yet. The same backend passes against `ros2_control`
on mock hardware, against robowright's own simulation served over ROS 2, against an SO-101
in Gazebo, and observe-only against Universal Robots' own Gazebo simulation driven by MoveIt
(`tests/test_ros2.py`, `tests/test_ros2_bridge.py`, `tests/test_gazebo.py`,
`tests/test_moveit_gazebo.py`). Expect the first real run to find things those could not:
timing, a driver's quirks, sensor noise.

## 1. Describe the robot

robowright needs the robot's model file (MJCF, URDF or xacro: usually its ROS 2 description
package) to know its joints, gripper and kinematics. Check what it makes of it first:

```bash
robowright robots --inspect path/to/my_arm.urdf.xacro
robowright robots add path/to/my_arm.urdf.xacro --name my_arm   # names it in robowright.toml
```

Give it the description the driver uses, as it is. An arm with no gripper in its description
(a UR on its own, say) is loaded with `robots.load(path, gripper=False)`; without that, robowright
attaches a Robotiq 2F-85 to it, as it does for its own simulations, and then waits for a gripper
joint the driver never publishes.

## 2. Point robowright at the driver

In the project's `robowright.toml`:

```toml
[ros2]
namespace = ""                                    # prefixes every topic below
arm = { controller = "joint_trajectory_controller", interface = "trajectory" }   # or a forward position controller
gripper = { controller = "gripper_controller", interface = "action" }            # GripperCommand; "none" to leave it alone
joints = { shoulder_pan = "joint1" }              # robowright's name -> the driver's, where they differ
frame = "base_link"                               # the robot's base frame in TF
objects = { cube = "cube" }                       # objects your perception publishes as TF frames
command = false                                   # start observe-only: robowright never sends a command
```

## 3. Watch before anything moves

With `command = false` robowright only reads `/joint_states` and TF. Every action that would move
the robot (`pick`, `move_to`, `run_policy`) fails at once with a `CapabilityError` instead, and a
reset takes the robot as it is rather than driving it anywhere. Start here, with the arm under its own control or held by its brakes:

```python
from robowright import expect


def test_the_arm_is_where_its_driver_says(robot):
    expect(robot).to_have_joint("joint1", 0.0, tol=0.05)  # read, not commanded


def test_perception_sees_the_cube(scene):
    expect(scene["cube"]).to_be_near((0.30, 0.0, 0.02), tol=0.02)
```

```bash
pytest --rw-backend ros2 --rw-robot my_arm --rw-live   # and watch the readings in a browser
```

## 4. Your stack drives, robowright watches

Still observe-only: your own software moves the arm and the test asserts on the outcome. This
is the test you will keep:

```python
import subprocess


def test_my_stack_puts_the_cube_in_the_bin(scene):
    stack = subprocess.Popen(["ros2", "launch", "my_robot", "pick.launch.py"])
    try:
        expect(scene["cube"]).to_be_inside(scene["bin"], timeout=60)
    finally:
        stack.terminate()
```

Run the same file against robowright's simulation first (`ros2` fixture, or `robowright sim
--ros2`), then Gazebo, then the arm. Where the arm disagrees with the simulation, that is a
finding: record it the way engine differences are recorded, in the project's registry of known
divergences, with what the trace showed.

## 5. Only then let robowright command it

Set `command = true` to let robowright's reference controller and `run_policy` drive the arm.
Keep the speeds low at first: `Settings(max_joint_speed=0.3, max_tcp_speed=0.05)`, and
`[ros2] reset_speed = 0.2`. Keep an emergency stop in reach; robowright is not a safety system.
