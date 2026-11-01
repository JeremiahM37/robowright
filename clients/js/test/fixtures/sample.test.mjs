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

test("trials", { trials: 4, minSuccess: 0.5, robot: "so101" }, async ({ rw }) => {
  const launched = JSON.parse(await rw.call("robot_last_launch"));
  assert.equal(launched.randomize, true); // each trial's scene drawn from its seed
  assert.equal(launched.seed % 2, 1, `seed ${launched.seed}`); // odd seeds pass: 7 and 9 of 7..10
});
