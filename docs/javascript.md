# JavaScript and TypeScript

Playwright has one API in four languages. robowright's test runner is pytest, but its robots can be driven from
Node as well: `clients/js` is a client, with TypeScript types, for the robowright MCP server. Every tool an AI agent
gets is a method there, and `expect` has the same matchers.

```js
import { launch } from "robowright";

const rw = await launch(); // starts `robowright mcp`
await rw.robot.launch({ robot: "so101" });
await rw.robot.pick({ object: "cube" });
await rw.expect("gripper").toBeHolding("cube");
await rw.robot.place({ on: "bin" });
await rw.expect("cube").toBeInside("bin", { timeout: 2 });
await rw.close();
```

It is not on npm yet; install it from the repository (`npm install ./clients/js` from a checkout). Its README in
[clients/js](https://github.com/JeremiahM37/robowright/tree/main/clients/js) has the details.

`robowright/test` runs them as Playwright Test does, on Node's own runner (`node --test`): a fresh world per test,
`expect` with no connection passed, every test on each robot and engine (`ROBOWRIGHT_ROBOT=panda,ur5e
ROBOWRIGHT_BACKEND=mujoco,drake`), a failure's trace saved with the `robowright show-trace` command that opens it,
and retries that report a test flaky when it passes on one:

```js
import { expect, test } from "robowright/test";

test.use({ robot: "so101", retries: 1 });

test("puts the cube in the bin", async ({ rw }) => {
  await rw.robot.pick({ object: "cube" });
  await rw.robot.place({ on: "bin" });
  await expect("cube").toBeInside("bin", { timeout: 2 });
});
```

`trials: 20, minSuccess: 0.9` judges a test as a success rate over seeds with the scene randomized, as
`@pytest.mark.trials` does; `ROBOWRIGHT_RESULTS=results` writes the results that `robowright merge-reports
results/*.json` makes into robowright's HTML report; `ROBOWRIGHT_PROJECT=nightly` runs in `robowright.toml`'s
`[projects.nightly]`; Node's `--test-shard=1/3` shards. What it does not have yet, of the pytest plugin's runner:
image and state snapshots, project dependencies, and the live view and debugger.
`rw.robot.generateTest()` writes a session as a pytest test when you want it in that suite.
