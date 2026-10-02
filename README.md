# robowright

**Playwright-style testing for robots.** Write a robot test once, with actions that wait until they're actually done and assertions that retry until the physical world catches up. Run it on MuJoCo or PyBullet. Every failure comes with a trace you can scrub through, a replay that reproduces it bit-for-bit, and a generated regression test.

<p align="center"><img src="docs/demo.gif" width="480" alt="SO-101 arm picking up a red cube and placing it in a blue bin"></p>

```python
from robowright import expect

def test_pick_and_place(robot, scene):
    cube, bin = scene["cube"], scene["bin"]

    robot.pick(cube)
    expect(robot.gripper).to_be_holding(cube)

    robot.place(on=bin)
    expect(cube).to_be_inside(bin)
    expect(cube).to_be_at_rest()
```

```console
$ pytest --rw-backend mujoco,pybullet
```

> **Status: pre-alpha prototype.** One robot (SO-101 arm), two simulators, top-down
> grasping. See [Limitations](#limitations) before relying on it.

## Why

Most robot software is tested by running it and watching. The tools that exist each cover
one slice of the problem:

- `launch_testing` checks that processes start and exit.
- Benchmark suites such as LIBERO and ManiSkill score policies on fixed tasks.
- Rerun and Foxglove record and visualise runs, but have no notion of pass or fail.

None of them gives you what web developers have had for years: a test that says what
should happen, waits for it, fails with a clear explanation, and leaves behind enough
evidence to debug it. robowright brings that workflow to robots:

| Playwright | robowright |
|---|---|
| auto-waiting actions | `robot.arm.move_to(...)` returns only once the arm has settled; `gripper.close()` returns once the jaws have stalled |
| web-first assertions | `expect(cube).to_be_inside(bin)` re-checks every control step until it holds or times out, in *simulated* time |
| locators | `scene["cube"]`, `scene.get(color="red")`, `scene.nearest(to=robot.tcp)` are live handles, not snapshots |
| trace viewer | every failure leaves a `.zip` trace with camera frames, joints, contacts and the action timeline, viewable as HTML |
| codegen | `robowright codegen trace.zip` rebuilds the exact failing situation as a pytest test |
| projects (browsers) | `--rw-backend mujoco,pybullet` runs every test on each physics engine |

On top of that, it adds things robots need and web pages don't:

- **Statistical tests:** a policy that works 92% of the time is normal. `@pytest.mark.trials(20, min_success=0.9)` judges a rate, with confidence intervals.
- **Fault injection:** sensor noise, command latency, weak servos, shoves and camera dropout, all seeded.
- **Invariants:** `expect(robot).always.to_have_no_collisions()` is checked after every step.
- **Deterministic replay:** re-simulates a trace from its recorded motor commands and reports the first step where anything diverges.

## Install

```bash
git clone https://github.com/JeremiahM37/robowright && cd robowright
pip install -e ".[dev]"      # MuJoCo is required; PyBullet, xdist and ruff come with [dev]
robowright info              # versions, backends, and whether offscreen rendering works
pytest examples
```

On a headless Linux machine robowright renders through EGL. Without a working GL, tests
still run and traces are still recorded, just without camera frames.

## A tour

### Actions wait

```python
robot.arm.move_to((0.22, 0.05, 0.04))              # IK, joint-space trajectory, then wait until settled
robot.arm.move_to(cube, linear=True, speed=0.05)   # straight-line Cartesian approach
robot.gripper.close()                              # returns when the jaws stop: on an object, or shut
robot.pick(cube); robot.place(on=bin)              # skills built from the above
```

When an action can't finish, it says why:

```
ActionTimeoutError: arm did not settle within 1.0s; worst joint shoulder_lift is 0.408 rad off target
UnreachableError: no joint configuration reaches [0.5, 0.0, 0.3] (closest 264.6 mm)
```

### Assertions wait, then explain

```python
expect(cube).to_be_inside(bin)                     # retries for up to 2 s of simulated time
expect(cube).to_be_at_rest(hold=0.3)               # ...and must stay true for 0.3 s
expect(robot.gripper).to_be_holding(cube)          # both jaws in contact
expect(cube).not_.to_be_touching("floor")
expect(robot).always.to_have_no_collisions()       # invariant for the rest of the test
expect.soft(cube).to_be_upright()                  # record and keep going; fail at the end
```

```
ExpectationError: expect(cube).to_be_inside failed after 2.00s (timeout 2.0s)
  cube at (0.256, -0.102, 0.012); bin spans (0.150, 0.070, 0.000)..(0.250, 0.170, 0.040)
```

Time spent waiting is *simulated* time, so a 2-second timeout costs a few milliseconds of
wall time, and a condition that is already true costs nothing.

Matchers: `to_be_near`, `to_be_inside`, `to_be_above`, `to_have_position`, `to_be_at_rest`,
`to_be_upright`, `to_be_touching`, `to_be_holding`, `to_be_open`, `to_be_closed`,
`to_have_joint`, `to_have_no_collisions`, `to_satisfy(fn)`.

### Policies are first-class

Any callable `obs -> joint targets` is a policy, including action chunks (`(n, 6)` arrays),
which is how ACT- and diffusion-style policies emit actions:

```python
from robowright import condition

done = condition(scene["cube"], "to_be_inside", scene["bin"])
rollout = robot.run_policy(my_policy, until=done, timeout=15, cameras=("front",))
assert rollout.success
```

### Statistics instead of flakes

```python
@pytest.mark.trials(20, min_success=0.9)
def test_policy_with_randomized_cube(world, robot, scene):
    world.faults.jitter("cube", xy_std=0.02, yaw_std=0.5)
    ...
```

```
================================ robowright trials =================================
PASS examples/test_pick_and_place.py::test_policy_with_randomized_cube[mujoco]: 20/20 passed (100%, 95% CI 84%-100%); required rate >= 90%
```

Each trial gets its own seed. A failing trial keeps its own trace, and its seed is printed,
so you can rerun exactly that one.

### Faults

```python
world.faults.joint_noise(std=0.02)            # encoder noise, seen by robot and policy, not by physics
world.faults.action_delay(steps=3)            # 60 ms command latency
world.faults.weak_joint("shoulder_lift", 0.3) # a tired servo
world.faults.push("cube", force=(0, 1.5, 0), duration=0.1)
world.faults.camera_dropout(p=0.1)
world.faults.jitter("cube", xy_std=0.02)      # domain randomisation, seeded
```

### Traces, replay, codegen

With the default `--rw-trace retain-on-failure`, every failing test leaves a trace:

```
================================ robowright traces =================================
tests/test_bin.py::test_cube_lands_in_bin[mujoco]
    robowright show-trace robowright-traces/tests_test_bin.py__test_cube_lands_in_bin[mujoco].zip
```

<p align="center"><img src="docs/viewer.png" alt="robowright trace viewer showing the action list, camera frame, timeline and joint plots of a failing test"></p>

The viewer is one self-contained HTML file, so you can attach it to a CI run or an issue.
It has an action list, a scrubbable camera view, a timeline, joint plots (measured against
commanded), and per-step joints, objects and contacts.

```console
$ robowright replay trace.zip
replayed 223 steps: bit-identical

$ robowright codegen trace.zip -o tests/test_regression_bin.py
```

`replay` rebuilds the scene, restores the saved simulator state and feeds back the
recorded motor commands and forces. No test code runs. If anything diverges, it reports
the first step where it does. `codegen` writes the run back out as plain robowright code.
That includes the exact randomised start poses and injected faults, so a one-in-fifty
failure becomes a test you can run every time.

### Two physics engines

```console
$ pytest --rw-backend mujoco,pybullet
```

The robot, kinematics, actions and assertions are shared; only physics differs. That's
useful because a test that passes on one engine and fails on the other usually means the
behaviour depends on contact details neither engine models faithfully. For example,
running the examples on PyBullet showed the cube tumbling 90 degrees as it drops into the
bin, which didn't happen in any of the MuJoCo runs.

## Benchmarks

Measured with `python bench/run.py`. Full tables and machine details are in
[BENCHMARKS.md](BENCHMARKS.md).

On an AMD Ryzen AI Max+ 395 (32 threads):

| | MuJoCo | PyBullet |
|---|---:|---:|
| one pick-and-place test (2 actions, 3 assertions) | 90 ms wall for 3.6 s simulated | 146 ms for 3.3 s |
| control steps/s, no tracing / tracing / tracing + camera frames | 25k / 18k / 3.1k | 5.2k / 5.0k / 0.46k |
| replays of faulted runs that were bit-identical | 20/20 | 20/20 |
| randomised pick-and-place passing (same 50 seeds) | 50/50 | 50/50 |

- **Parallel runs:** 64 randomised tests take 8.8 s on one worker and 2.3 s with 8 pytest-xdist workers. Speed-up flattens around 4x because a suite that small is dominated by worker start-up.
- **Invariants:** two `expect(...).always` invariants add 15 µs to each 31 µs control step.
- **Fault curves:** the reference policy still passes 20/20 with 0.06 rad of encoder noise, 12/20 at 0.09 rad and 0/20 at 0.12 rad. With the shoulder servo's gain cut to 1% it passes 15/20; at 0.5% it passes 1/20. Those are the curves `@pytest.mark.trials` exists to guard.

## How it works

```
 test code ──► Robot / expect / locators / faults          (backend-independent core)
                   │  IK from the URDF, trajectories, waiting, invariants
                   ▼
               World.step()  ── one 20 ms control period ──► Recorder ──► trace.zip
                   │                                             │
                   ▼                                             ├─► viewer (HTML)
               Backend: MuJoCo | PyBullet | (hardware)           ├─► replay
                 physics only: qpos, set_ctrl, contacts,         └─► codegen
                 render, state save/restore
```

- Kinematics come from the SO-101 URDF and are shared by all backends. The URDF and MuJoCo
  models agree to within 2.5 µm (`tests/test_kinematics.py`).
- Backends advertise capabilities (`ground_truth`, `contacts`, `render`, `state`, ...).
  A matcher that needs one fails with a clear message on a backend that lacks it, instead
  of passing silently. On hardware, object poses come from a perception hook:
  `world.perception["cube"] = lambda: (pos, quat)`.
- Anything that changes the world mid-test (teleports, faults) is written to the trace,
  which is what makes replay and codegen exact.

## Limitations

- **One robot:** the SO-101 arm, with top-down grasps. Its reachable top-down workspace is
  roughly 13–31 cm out and 1–9 cm up.
- **Simulation only:** no hardware backend yet. The backend interface is written with one
  in mind (capabilities, perception hooks, `Settings(realtime=True)`), but nothing has run
  on a real arm.
- **Privileged policies:** the bundled `ScriptedPickPlace` reads ground-truth object poses.
  Camera-based learned policies plug into the same `run_policy`, but no LeRobot adapter
  ships yet.
- **`to_be_upright`** is too strict for symmetric objects: a cube lying on another face
  is still a cube.

## Roadmap

1. LeRobot policy adapter (`lerobot/smolvla_*`, ACT) and LeRobot hardware backend for a
   real SO-101.
2. ROS 2 backend (topics/actions in, the same `expect` out), so existing robots can be
   tested without a simulator.
3. More engines (Isaac Sim / Genesis) and more robots via URDF + MJCF pairs.
4. An agent-facing MCP server: "pick up the red cube" becomes a recorded, assertable run.

## License

Apache-2.0. The SO-101 model (MJCF, URDF and meshes) is from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), both Apache-2.0.
See [NOTICE](NOTICE).
