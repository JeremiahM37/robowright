# Changelog

## 0.2.0 (not yet released)

The first release on PyPI (0.1.0 was tagged but never published). 121 changes since 0.1.0, most of them the parts
of Playwright Test that a robot test needs; [Coming from Playwright](docs/playwright.md) lists each, with what is
partial.

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

### Assertions and locators

- `expect.poll`, `expect(block).to_pass()`, `expect.extend` (custom matchers), `expect.configure`.
- `expect(world).to_match_snapshot` (whole view or one object's window, `mask=`), `to_match_state_snapshot`
  (positions and joints as editable text), `--rw-update-snapshots [all|changed|missing|none]`.
- Locators by where things are now (`inside=`, `on=`, `touching=`, `near=`, `held=`, `where=`), by what the robot can
  do with them (`graspable=True`), and lazy ones: `scene.filter(...)` with `expect(...).to_have_count(n)`.

### Tools

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

### Fixes

- `import robowright` no longer fails on a machine without EGL.
- The CI workflow `robowright init` writes installs robowright the way it is installed here.
- Closing a ROS 2 world never hangs, and nothing a test started outlives a killed run.
- A Dockerfile (MuJoCo, PyBullet and video, rendering on the CPU).

## 0.1.0 (2026-10-15, tagged, not published)

The first tagged version.
