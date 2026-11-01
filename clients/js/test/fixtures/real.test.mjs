// Run by runner.test.js against the real robowright server.
import { expect, test } from "../../src/test.js";

test("puts the cube in the bin", async ({ rw }) => {
  await rw.robot.pick({ object: "cube" });
  await expect("gripper").toBeHolding("cube");
  await rw.robot.place({ on: "bin" });
  await expect("cube").toBeInside("bin", { timeout: 2 });
});

test("leaves the cube where it was", async () => {
  await expect("cube").toBeInside("bin", { timeout: 0.2 });
});
