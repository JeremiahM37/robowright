// Run by runner.test.js with ROBOWRIGHT_PROJECT and the stand-in projects above.
import assert from "node:assert/strict";

import { test } from "../../src/test.js";

test("launches", async ({ rw }) => {
  const launched = JSON.parse(await rw.call("robot_last_launch"));
  assert.equal(launched.seed, launched.robot === "so101" ? 0 : 5); // the project's seed, where it has one
});
