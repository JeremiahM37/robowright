# Changelog

## 0.2.0 (not yet released)

The first release on PyPI (0.1.0 was tagged but never published). 121 changes since 0.1.0, most of them the parts
of Playwright Test that a robot test needs; [Coming from Playwright](docs/playwright.md) lists each, with what is
partial.

### JavaScript, TypeScript, Java and .NET

- `clients/js`: drive robowright from Node. It starts `robowright mcp` and makes every tool a typed method
  (`rw.robot.pick`, `rw.ros2.inspect`, ...), with `expect(subject).toBeInside("bin")` and `.not`, throwing on a
  failure as Playwright does. No dependencies; the tool table and types are generated from the server's tool
  list, and a test fails when they fall behind.
- `clients/java`: the same for Java 17 or later, in Playwright for Java's style (required arguments as
  parameters, the rest in an options class; `rw.expect("cube").not().toBeInside("bin")` throws an `AssertionError`
  subclass, so JUnit reports it). No dependencies; generated from the tool list the same way.
- `clients/dotnet`: the same for .NET 8, in Playwright for .NET's style (`await rw.Robot.PickAsync("cube")`,
  options objects, `rw.Expect("cube").Not.ToBeInsideAsync("bin")`). No dependencies; generated the same way.
- `robowright/test`: robot tests in Node written as Playwright Test writes them, on Node's own runner. Each test gets
  a fresh world from one server per file, `expect("cube")` needs no connection passed, `test.use({ robot, backend,
  seed, retries, ... })` sets a file's options, `ROBOWRIGHT_ROBOT` and `ROBOWRIGHT_BACKEND` run every test on each
  robot and engine, a failure saves its trace and names the `robowright show-trace` command that opens it, and a test
  that passes on a retry is reported flaky. `trials: 20, minSuccess: 0.9` judges a success rate over randomized
  seeds as `@pytest.mark.trials` does, and `ROBOWRIGHT_RESULTS=dir` writes results that `robowright merge-reports
  dir/*.json` makes into the HTML report. TypeScript types in `test.d.ts`.
- MCP `robot_launch(randomize=true)` moves the free objects to places drawn from the seed, as each trial does.
- Snapshot matchers through the MCP server and every client: `expect("world").toMatchStateSnapshot("tidy")` and
  `toMatchSnapshot("tidy.png")`, the baseline kept beside the test file the client names (JavaScript names its own;
  Java and .NET pass `test_file`).
- `@RobotTest` for JUnit 5 (`robowright.junit` in the Java client): a fresh world per test from one server per class,
  the `Robowright` it drives as a parameter, a run per robot and engine (`ROBOWRIGHT_ROBOT`, `ROBOWRIGHT_BACKEND`),
  retries reported as flaky, and a failure that saves its trace and names it. JUnit's API is a provided dependency.
- `RobotTest.RunAsync` and `RobotMatrix` in the .NET client, for xUnit, NUnit or MSTest: a fresh world per target,
  retries on a fresh world, and a failure that saves its trace and names it in `RobotTestException`. No NuGet
  dependency.
- `robowright init --js`: a Node project (package.json, an example test with a trials test, settings, a CI workflow
  that writes the HTML report, the MCP server). The JavaScript client ships inside the wheel, at its server's
  version; `robowright js-client` says where, and the project installs it from there.
- `ROBOWRIGHT_PROJECT=nightly` runs robowright/test in `robowright.toml`'s `[projects.nightly]`; `robowright projects
  [--json]` prints the projects, for reading and for runners in other languages.

### Running tests

- `--rw-retries`, `@pytest.mark.rw_retries(n)` and the flaky outcome; `--rw-fail-on-flaky`; `--rw-repeat-each N`.
- `--rw-timeout` and `@pytest.mark.rw_timeout` (fixtures included, a hung test says where it hung, a test stuck in a
  C call ends the process instead of the CI job); `@pytest.mark.rw_slow`; `--rw-global-timeout`, which stops a test
  still running when the time is up.
- `@pytest.mark.only` with `--rw-forbid-only` (on by itself on CI); `@pytest.mark.fixme`; serial mode
  (`pytestmark = pytest.mark.serial`).
- `pytest.mark.use(...)` (Playwright's `test.use`): waits, speeds, trace, video, time limit and retries for a module,
  class or test.
- Projects: `[projects.NAME]` in `robowright.toml`, `--rw-project a,b` or `all`, `depends` (setup projects first,
  dependents skipped when they fail), `match` and `ignore`.
- `--rw-shard K/N`, `--rw-blob` and `robowright merge-reports`; `--rw-only-changed [REF]`.
- `rw_attempt`, `rw_repeat` and `rw_project` fixtures.
- `world.checkpoint()` and `world.restore()` (Playwright's `storageState`): set up once and start each case from
  that moment, in this world or a fresh one of the same scene; traces replay through it and codegen writes it.

### Assertions and locators

- `expect(obj).to_be_visible(camera=...)` and `to_be_hidden()` (Playwright's `toBeVisible`, `toBeHidden`): whether it
  shows in a camera's image, in the frame and not behind anything (MuJoCo).
- `expect.poll`, `expect(block).to_pass()` (both with `intervals=`), `expect.extend` (custom matchers),
  `expect.configure`.
- `expect(world).to_match_snapshot` (whole view or one object's window, `mask=`), `to_match_state_snapshot`
  (positions and joints as editable text), `--rw-update-snapshots [all|changed|missing|none]`.
- Locators by where things are now (`inside=`, `on=`, `touching=`, `near=`, `held=`, `where=`), by what the robot can
  do with them (`graspable=True`), and lazy ones: `scene.filter(...)` with `expect(...).to_have_count(n)`.

### Tools

- MCP: `robot_wait_until` (a condition, not a duration), `robot_process_output` (a running stack's output),
  `ros2_list` (the graph's topics, services and actions), `robot_list_backends`.
- MCP: `ros2_traffic` (what went over the graph: Playwright MCP's network requests) and `ros2_logs` (the nodes'
  /rosout lines by level and node: its console).
- MCP: worlds open side by side, as tabs (`robot_launch keep_open=true`, `robot_worlds`, `robot_switch`).
- MCP actions say what touched what on the way ("during this call: cube stopped touching floor at 1.36 s"),
  which the state they end in does not show.
- `robowright test --ui` (UI mode) and `robowright test --watch`.
- `pytest --rw-debug` and `world.pause()`: a debugger in the live view, with a locator picker.
- `robowright codegen` with no trace: write a test by doing it in a browser page, with a locator picker.
- A VS Code extension (`editors/vscode`): Test Explorer, run, debug under the Python Debugger, live view, record a new
  test, pick locator, failures on their line, traces in a tab.
- Traces: the Source tab, `rw.step`, ROS 2 topics and node logs, `show-trace --perfetto`; `--rw-video`;
  `pytest --rw-log` (Playwright's `DEBUG=pw:api`).
- `robowright init` (sharded CI with a merged report, `--ros2` for your own stack) and `robowright init-agents`
  (planner, generator and healer agents on the MCP server).

### Reports

- Each failure in the HTML report shows what the camera saw at the step it failed, redrawn from its trace.
- The HTML report keeps its own copies of traces, videos and attachments; tags and annotations
  (`@pytest.mark.annotate`, `rw_annotate`); `--rw-json`; GitHub annotations on the failing line; JUnit properties
  for each test's robot, engine, seed, project and trace.

### ROS 2 and other simulations

- `rw.ros2_stack` / `[ros2] stack`: a stack started once and reset between tests; `world.ros.publish`, `call`,
  `send_goal`, `serve`, `serve_action`; routes (`ros2.run(..., route=...)`) and rosbag record/replay; ROS 2 Humble.
- `rw.http_stub` with record and replay.
- `rw.attach` on gymnasium, dm_control, Meta-World, robosuite, ManiSkill 3, Isaac Lab, Isaac Sim, Stable-Baselines3
  vector envs and async vector envs.
- Wheeled and omni bases on PyBullet, Drake and Genesis.
- Python 3.14: tested on CI; every extra installs (PyBullet builds from source there, as on 3.12 and 3.13).
- `[ros2.limits] workspace = {min = [x, y, z], max = [x, y, z]}`: a box the tool centre point must stay in on a real
  arm (ROS 2 or LeRobot), checked at every point of a command from the arm's kinematics, robowright's own moves
  included; a command that would leave it is refused whole, saying where the tool would have gone. A misspelt key
  under `[ros2.limits]` is refused instead of ignored.

### Fixes

- `import robowright` no longer fails on a machine without EGL.
- The CI workflow `robowright init` writes installs robowright the way it is installed here.
- Closing a ROS 2 world never hangs, and nothing a test started outlives a killed run.
- Genesis 1.4.5 works (it renamed the arrays a saved state carries, and every Genesis world failed to start).
- On pytest 9.2, `--dist loadgroup` keeps a robot's tests on one worker again. pytest-xdist 3.8 adds the group to a
  test's id in a field pytest 9.2 no longer reads, so every group was silently spread over all workers (and serial
  modules ran in parallel); robowright now writes it where pytest reads it.
- A worker that crashes under `--dist loadgroup` no longer hangs the run. pytest-xdist 3.8 puts a crashed worker's
  finished tests back on the queue, and a worker handed only finished tests waits for good (xdist #1323 and #1327,
  fixed upstream but not yet released); robowright applies xdist's fix to the versions that still have the bug.
- `place()` fails with a `GraspError` when the object slipped out of the hand on the way, instead of opening the jaws
  over the target and passing wherever the object happened to fall. The SO-101 on PyBullet, whose published squeeze
  pushes the cube out of its jaws within a second, passed its plain pick-and-place that way on one PyBullet build and
  failed it on another; it is now listed with that robot's other known PyBullet divergences.
- `robowright.GraspError` is importable (it was listed in `robowright.__all__`, so `from robowright import *` failed).
- On MuJoCo 3.17, a closed world no longer breaks the next one's drawing (3.17 frees a renderer in whichever GL context
  is current).
- A Dockerfile (MuJoCo, PyBullet and video, rendering on the CPU).
- An expectation with no timeout (a snapshot's) says it failed, not that it "failed after 0.00s (timeout 0.0s)".
- `robot_move_object` with `[x, y]` failed with NumPy's broadcasting error, reported as an engine refusing to move the
  object; `[x, y]` now keeps the object's height, and any other length says what it takes.
- A short move of robowright's own could break `[ros2.limits] joint_speed` and be refused: its peak speed was estimated
  from as few samples as the move had steps, and two samples of a smooth move average its peak away (an SO-101 wrist
  at 1.16 rad/s under a 1 rad/s limit). Moves are now sampled finely to find it.
- First run, from a walkthrough by someone new to it: `robowright info` gave 0.1.0 (the version is now read from the
  installed package); a `--rw-backend` engine that is not installed is one line before anything runs, saying what to
  install, not a traceback from every test; Drake's hint no longer says it needs Python 3.12 on 3.12; `robowright
  init` says the MCP server it registers needs `robowright[mcp]`; a failed `run_policy` says why its `until` never
  held (`<Rollout success=False ...: cube at (0.220, -0.060, 0.012); bin spans (0.150, 0.070, 0.000)..(0.250, 0.170, 0.040)>`); `--rw-debug` says the test waits for you,
  and refuses to run on CI; the live view's address comes with the `ssh -L` that reaches it over SSH; `robowright
  test --help` lists robowright's options, not pytest's 450 lines; `robowright show-trace test_pick` finds the newest
  trace of that test (no quoting of `[...]`), and `robowright show-trace` alone opens the newest.

## 0.1.0 (2026-10-15, tagged, not published)

The first tagged version.
