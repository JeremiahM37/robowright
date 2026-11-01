// robowright/test: robot tests on Node's own test runner, in Playwright Test's style.

import type { Assertions, RobotTools, Robowright } from "./index.js";

export interface TestOptions {
  /** A robot name or model file, or several (each is a separate test). ROBOWRIGHT_ROBOT overrides it. Default so101. */
  robot?: string | string[];
  /** mujoco (default), pybullet, drake, genesis, isaac or ros2, or several. ROBOWRIGHT_BACKEND overrides it. */
  backend?: string | string[];
  seed?: number;
  objects?: Record<string, unknown>[];
  fidelity?: "published" | "adjusted";
  ros2?: Record<string, unknown>;
  robotOptions?: Record<string, unknown>;
  /** Run a failing test again this many times (ROBOWRIGHT_RETRIES); passing on a retry is reported flaky. */
  retries?: number;
  /** Node's per-test time limit, in milliseconds. */
  timeout?: number;
  /** Where a failed test's trace is saved (ROBOWRIGHT_TRACE_DIR). Default robowright-traces. */
  traceDir?: string;
  skip?: boolean | string;
  todo?: boolean | string;
  only?: boolean;
}

export interface Fixtures {
  /** This test's server connection, its world freshly launched. */
  rw: Robowright;
  robot: RobotTools;
  expect(subject: string, options?: { timeout?: number }): Assertions;
}

type Body = (fixtures: Fixtures) => Promise<void> | void;

export interface TestFunction {
  (name: string, body: Body): void;
  (name: string, options: TestOptions, body: Body): void;
  /** Options for every test after it in this file (Playwright's test.use). */
  use(options: TestOptions): void;
  only: TestFunction;
  skip: TestFunction;
  fixme: TestFunction;
  describe: typeof import("node:test").describe;
}

export const test: TestFunction;

/** expect(subject) for the world of the test it is called in. */
export function expect(subject: string, options?: { timeout?: number }): Assertions;
