// Run by runner.test.js against the fake server, with ROBOWRIGHT_ROBOT=panda,ur5e.
import assert from "node:assert/strict";

import { expect, test } from "../../src/test.js";

test.use({ seed: 7 });

test("passes", async ({ rw }) => {
  await expect("cube").toBeInside("bin");
  const launched = JSON.parse(await rw.call("robot_last_launch"));
  assert.equal(launched.seed, 7);
  assert.ok((process.env.ROBOWRIGHT_ROBOT || "so101").split(",").includes(launched.robot), launched.robot);
  assert.equal(launched.backend, "mujoco");
});

test("fails with its trace", async ({ expect }) => {
  await expect("cube").toBeInside("box");
});

test("flaky", { retries: 1, robot: "so101" }, async ({ rw }) => {
  await rw.call("robot_flaky");
});
