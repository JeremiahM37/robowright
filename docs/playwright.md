# Coming from Playwright

Playwright Test's features, by the sections of its own documentation, and what robowright has in its place. A
robot is not a web page, so some features become something else (a browser becomes a physics engine, the network
becomes ROS 2 topics) and a few have no counterpart. The gaps are listed as plainly as the matches. An independent
audit against playwright.dev (by an agent that did not see this page) put robowright at about 70 % of the Playwright
Test features that apply to robots, counting partial ones as half; its gaps are the partial and missing rows below.

**Status**: **same** works the way Playwright's does; **robot** is the robot counterpart of a web feature;
**partial** covers part of it; **missing** is not there yet; **n/a** has no meaning for robots.

## Writing tests

| Playwright | robowright | Status |
|---|---|---|
| `test()`, fixtures, `test.describe`, hooks | pytest tests and fixtures; `world`, `robot`, `scene`, `seed`, `ros2`, `rw_attach`, `rw_live` fixtures | same |
| auto-waiting actions | `robot.pick(cube)`, `arm.move_to(...)`, `gripper.close()` return once the motion has settled, or raise `ActionTimeoutError` | same |
| web-first assertions (auto-retry) | `expect(cube).to_be_inside(bin)` re-checks every control step, in simulated time | same |
| locators, `filter({has})` | `scene["cube"]`, `scene.get(color="red")`, `scene.nearest(to=robot.tcp)`, and by where things are now: `scene.get(inside="bin")`, `on="tray"`, `touching="floor"`, `near=(p, r)`, `held=True`, `where=fn`; `scene.count(...)` | robot |
| `expect.soft` | `expect.soft(cube)...`: recorded, the test carries on and fails at the end | same |
| `expect.poll`, `toPass` | `expect.poll(fn).to_equal(...)`, `expect(block).to_pass()` | same |
| `expect.extend` (custom matchers) | `expect.extend(to_be_level=fn)`, retried, with `.not_`, `.always`, soft, poll | same |
| `.not` | `.not_` | same |
| `expect.configure` | `expect.configure(timeout=10, soft=True)` | same |
| `test.use` (options per file) | `pytestmark = [pytest.mark.robots(...), pytest.mark.backends(...), pytest.mark.scene(...)]` per module or class; no per-file settings beyond what markers carry | partial |
| invariants | `expect(robot).always.to_have_no_collisions()`: checked every step for the rest of the test | robot (Playwright has none) |
| `test.step` | `with rw.step("put the cube in the bin"):` in the trace, the viewer and the report | same |
| `testInfo.attach` | `rw_attach("stack.log", path=...)`, `rw_attach.camera(world)` | same |
| annotations: `skip`, `fail`, `fixme`, `slow`, tags | pytest markers: `skip`, `xfail(strict=True)`, custom markers with `-m` | same (pytest's) |
| `test.only` | `@pytest.mark.only` runs only the marked tests | same |
| `test.fixme` | `@pytest.mark.fixme("why")`: skipped, saying why | same |
| `test.slow` | `@pytest.mark.rw_slow` (or `rw_slow(5)`): the test's time limit three (five) times over | same |
| `testInfo.retry` | the `rw_attempt` fixture: 0, then 1 on the first retry | same |
| serial mode (`describe.configure({mode: 'serial'})`) | `pytestmark = pytest.mark.serial` on a module or class: in order, on one worker (`--dist loadgroup`), the rest skipped once one fails | same |
| parameterize | `pytest.mark.parametrize`; `--rw-robot`, `--rw-backend` run every test on each | same |
| page object model | plain Python classes and fixtures | same |
| statistical tests | `@pytest.mark.trials(20, min_success=0.9)`: randomized scenes, a success rate with confidence intervals | robot (Playwright has none) |

## Running tests

| Playwright | robowright | Status |
|---|---|---|
| projects (browsers) | `--rw-backend mujoco,pybullet,genesis,drake,ros2`, `--rw-robot panda,ur5e,all` | robot |
| devices / emulation | 19 robots built in, any MJCF or URDF file with `--rw-robot path` | robot |
| `webServer` | `[ros2] stack = "ros2 launch ..."`: the stack started once per worker, ready-checked, reset between tests | robot |
| workers, `fullyParallel` | `-n auto` (pytest-xdist), grouped so tests sharing a scene reuse it | same |
| `--shard 2/4` | `--rw-shard 2/4` | same |
| retries, flaky | `--rw-retries 2`; a test that passes on retry is reported flaky | same |
| `--fail-on-flaky-tests` | `--rw-fail-on-flaky` | same |
| `--repeat-each` | `--rw-repeat-each 10` (same seed each time) | same |
| `--only-changed` | `--rw-only-changed [REF]`, followed through imports | same |
| `--last-failed` | pytest's `--lf` | same (pytest's) |
| `forbidOnly` | `--rw-forbid-only` (on by itself when `CI` is set): a focus left in fails the run | same |
| `globalTimeout` | `--rw-global-timeout 3600`: the run stops and fails once it has taken that long | same |
| `--grep`, `--list` | pytest's `-k`, `--collect-only` | same (pytest's) |
| test timeout | `--rw-timeout 120`, `@pytest.mark.rw_timeout(300)`; a hung test says where it hung | same |
| `globalSetup` | session fixtures in `conftest.py`; the `[ros2]` stack | same (pytest's) |
| `npm init playwright` | `robowright init`: a test, settings, a CI workflow, the MCP server | same |
| clock control | everything runs in simulated time; nothing waits on the wall clock | robot |

## Debugging and tools

| Playwright | robowright | Status |
|---|---|---|
| trace viewer | `robowright show-trace`: actions, camera drawn from the recorded state, timeline, joint plots, contacts | robot |
| trace: source tab | the Source tab, at the line of your code each action came from | same |
| trace: network, console | the Topics and Log tabs of a ROS 2 run | robot |
| `trace: 'on-first-retry'` | `--rw-trace on-first-retry` | same |
| `--debug`, `page.pause()`, Inspector | `--rw-debug`, `world.pause()`: Step, Resume, Pause in the live view | same |
| `--headed` | `--rw-live` (a browser page), `--rw-headed` (MuJoCo's window) | same |
| pick locator | click an object in the live view's camera image: its locators, most robust first (`scene["cube"]`, `scene.get(color="red")`, `scene.get(inside="bin")`), each with a copy button | same |
| UI mode | `robowright test --ui` | same |
| watch mode | `robowright test --watch` | same |
| codegen | `robowright codegen` (in a browser, or from a trace: the failure as a regression test) | same |
| VS Code extension | `editors/vscode`: Test Explorer, run, failures on their line, open the trace | partial: built and tested, not on the Marketplace |
| MCP server, test agents (`init-agents`) | `robowright mcp`; `robowright init-agents` writes a planner, a generator and a healer for Claude Code that use it | same |
| `--repeat-each` to find flakes | see above; plus `robowright crosscheck` (the same calls on another engine) | robot |
| bit-for-bit replay | `robowright replay trace.zip` | robot (Playwright has none) |

## Reporting

| Playwright | robowright | Status |
|---|---|---|
| HTML reporter, `show-report` | `--rw-report report.html`, `robowright show-report` | same |
| blob reporter, `merge-reports` | `--rw-blob`, `robowright merge-reports` | same |
| `github` reporter | annotations at the failing line on GitHub Actions, by itself | same |
| JUnit, JSON reporters | pytest's `--junitxml`; `--rw-json results.json` (every test's outcome, message, traces, trials and attachments) | same |
| line, dot, list reporters | pytest's `-q`, `-v` | same (pytest's) |
| custom reporters | pytest plugins and hooks | same (pytest's) |
| `video` | `--rw-video retain-on-failure`, played in the HTML report | same |
| screenshots | `rw_attach.camera(world)`, `robowright render` | same |

## Assertions on what is seen

| Playwright | robowright | Status |
|---|---|---|
| `toHaveScreenshot`, element screenshots | `expect(world).to_match_snapshot("tidy.png")` (the whole view) or `expect(cube).to_match_snapshot("cube.png")` (the window around one object, following it), per engine and robot, with a diff | same |
| `toMatchAriaSnapshot` | `expect(world).to_match_state_snapshot("tidy")`: positions and joints as editable text | robot |
| `--update-snapshots` | `--rw-update-snapshots` | same |
| `mask` in screenshots | `to_match_snapshot(..., mask=["cube"])`: where the object is, now and in the baseline, left out | same |

## Network and services

| Playwright | robowright | Status |
|---|---|---|
| `page.route` | `ros2.run(cmd, route={"/scan": rw.route.cut(...)})` for a ROS 2 stack's topics; `robot.run_policy(policy, route={"observation": ..., "action": ...})` for an in-process controller: drop, delay, cut, throttle, abort, edit. Not a controller's own serial or HTTP link to hardware | robot |
| `route.fulfill` (mock an API) | `world.ros.serve("/detect", ..., answer)`: a stub ROS service or action; `rw.http_stub({"GET /missions/next": {...}})`: a stand-in HTTP service (fleet manager, map server, cloud API) any process can call, with every request recorded | robot |
| `request` fixture (API testing) | `world.ros.call`, `send_goal`, `publish` | robot |
| `recordHar`, `routeFromHAR` for HTTP | `rw.http_stub.record("https://fleet.local", "fleet.json")` passes calls through and records them; `rw.http_stub.replay("fleet.json")` answers from the file | same |
| `recordHar`, `routeFromHAR` | `rw.route.record(bag)`, `rw.route.from_bag(bag)` (rosbags, including ones from the real robot) | robot |
| fault injection | `world.faults`: pushes, delays, sensor noise, weak joints | robot (Playwright has none) |

## Where it runs

| Playwright | robowright | Status |
|---|---|---|
| the real browser engines | MuJoCo, PyBullet, Genesis, Drake built in; Gazebo, Isaac Sim, Webots and others attached; any robot behind ROS 2 | robot |
| your app under test | your controller, learned policy or ROS 2 stack; `rw.attach(env)` tests someone else's simulation as they built it | robot |
| real devices | a real arm behind ROS 2 (`--rw-backend ros2`) | partial: never yet run on a physical robot |
| installing with one command | `pip install robowright` | missing: not yet on PyPI (it installs from GitHub) |
| JavaScript, Python, Java, .NET | Python only | partial |
| years of use by many teams | new, with no outside users yet | missing |

## Not applicable to robots

Browser contexts and storage state (a test gets a fresh world instead), cookies and authentication, dialogs,
downloads, frames, file choosers, accessibility checks, component testing of a UI framework, and mobile web
emulation have no robot meaning. Each test's isolation, which browser contexts give a page, comes from a fresh world
per test, restored bit for bit from a kept scene when another test already built it.
