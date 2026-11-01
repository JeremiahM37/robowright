# robowright for JavaScript and TypeScript

The same simulated robots, matchers and traces as the Python library, from Node. The client starts the
robowright MCP server (`robowright mcp`) and talks to it over stdio, so every tool an AI agent gets is a typed
method here, and a world you drive from JavaScript is the same deterministic world a pytest test sees. It has no
dependencies.

It needs robowright itself, with its MCP extra, where the server will run:

```sh
pip install "robowright[mcp] @ git+https://github.com/JeremiahM37/robowright"
```

With `robowright/test`, tests read as Playwright Test's do: each one gets a fresh world, `expect` needs no
connection passed, and a failure saves its trace:

The examples are ES modules: give your `package.json` `"type": "module"` (or name the files `.mjs`).
`robowright init --js` sets up such a project: `package.json`, an example test, settings, CI, and the MCP server for
AI agents, with the client installed from robowright itself (`robowright js-client` says where it is).

```js
// tests/pick.test.js, run with: node --test
import { expect, test } from "robowright/test";

test.use({ robot: "so101" });

test("the arm puts the cube in the bin", async ({ rw }) => {
  await rw.robot.pick({ object: "cube" });
  await expect("gripper").toBeHolding("cube");
  await rw.robot.place({ on: "bin" });
  await expect("cube").toBeInside("bin", { timeout: 2 });
});
```

- **One server per test file**, started on the first test; each test launches a fresh world in it with the
  file's options (`test.use({ robot, backend, seed, objects, fidelity, ros2, robotOptions, retries, timeout })`)
  and its own (`test(name, { retries: 2 }, fn)`).
- **Every robot and engine**: `ROBOWRIGHT_ROBOT=panda,ur5e ROBOWRIGHT_BACKEND=mujoco,drake node --test` runs each
  test on every pair, named `name [mujoco-panda]`, as `--rw-robot` and `--rw-backend` do in pytest.
- **A failed test** saves its trace to `robowright-traces/<test>.zip` and its error ends with the
  `robowright show-trace` command that opens it.
- **Retries**: `ROBOWRIGHT_RETRIES=2` (or `retries`) runs a failing test again on a fresh world; one that passes on
  a retry is reported flaky (`# flaky: failed 1 time, then passed on retry 1`).
- **Trials**: `test(name, { trials: 20, minSuccess: 0.9 }, fn)` runs it on up to 20 seeds, each with the scene's
  free objects moved to places drawn from the seed, and judges the success rate with its 95% interval, as
  `@pytest.mark.trials` does (`lowerBound: true` judges the interval's lower bound; it stops once the rest cannot
  change the verdict, unless `ROBOWRIGHT_ALL_TRIALS=1`).
- **The HTML report**: `ROBOWRIGHT_RESULTS=results node --test`, then `robowright merge-reports results/*.json -o
  report.html`: robowright's own report, each failure with its trace.
- **Projects**: `ROBOWRIGHT_PROJECT=nightly` (or `nightly,smoke`, or `all`) runs each test in `robowright.toml`'s
  `[projects.nightly]` (its robot, backend, retries, seed, fidelity and timeout), named `name [nightly]`, as
  `--rw-project` does; `robowright projects` lists them.
- **Snapshots**: `expect("world").toMatchStateSnapshot("tidy")` (positions and joints as text) and
  `toMatchSnapshot("tidy.png", { mask: ["cube"] })` keep one baseline per engine and robot beside the test file, as in
  Python: `tests/__snapshots__/pick.test/tidy-mujoco-so101.state.txt` for `tests/pick.test.js` (`[snapshots] path` in
  robowright.toml moves them). The first run writes it and fails; `ROBOWRIGHT_UPDATE_SNAPSHOTS=all` writes them again.
- **Shards**: Node's `--test-shard=1/3`.
- `test.only`, `test.skip`, `test.fixme`, `test.describe`, and Node's own reporters (`--test-reporter=junit`,
  `spec`, `tap`) and `--test-name-pattern`.

The client underneath it can be used directly, with any test runner:

```js
import { test } from "node:test";
import { launch } from "robowright";

test("the arm puts the cube in the bin", async () => {
  const rw = await launch();
  try {
    await rw.robot.launch({ robot: "so101" });
    await rw.robot.pick({ object: "cube" });
    await rw.expect("gripper").toBeHolding("cube");
    await rw.robot.place({ on: "bin" });
    await rw.expect("cube").toBeInside("bin", { timeout: 2 });
    await rw.expect("gripper").not.toBeHolding("cube");
  } finally {
    await rw.robot.saveTrace({ path: "trace.zip" });
    await rw.close();
  }
});
```

- **Tools** are grouped as `rw.robot.*`, `rw.ros2.*` and `rw.sim.*`, named after the MCP tools in camelCase
  (`robot_move_to` is `rw.robot.moveTo`). Arguments are one object; camelCase names are sent in snake_case. Each
  returns the tool's text: a snapshot of the world, a result or a report.
- **`expect(subject)`** has the Python matchers in camelCase (`toBeInside`, `toBeHolding`, `toBeNear`,
  `toHaveJoint`, `toBeVisible`, ...), `.not`, and a `timeout` in seconds of simulated time, retried as in
  Python. A matcher's first argument can be given bare (`toBeInside("bin")`) or as an object
  (`toBeNear({ target: "bin", tol: 0.02 })`). A failing one throws `ExpectationError` with the reason.
- **Errors** from a tool throw `RobowrightError`, with `.tool` naming it.
- **`launch({ command })`** picks the server command; default `robowright mcp`, or `ROBOWRIGHT_COMMAND`.
- `rw.robot.generateTest()` writes what you did as a pytest test that reproduces it, and `rw.robot.saveTrace()`
  keeps the trace for `robowright show-trace`.

`src/tools.js` and `index.d.ts` are generated from the server's own tool list by
`python scripts/gen_js_client.py`; the Python test suite fails when they fall behind. `npm test` runs the client's
tests, against a stand-in server and, when robowright is installed, the real one.
