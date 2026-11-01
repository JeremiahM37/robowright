// robowright/test: robot tests in Node, written as Playwright Test writes web tests, on Node's own test runner.
//
//   import { test, expect } from "robowright/test";
//   test.use({ robot: "panda" });
//   test("puts the cube in the bin", async ({ rw }) => {
//     await rw.robot.pick({ object: "cube" });
//     await rw.robot.place({ on: "bin" });
//     await expect("cube").toBeInside("bin");
//   });
//
// Each test gets a fresh world from one robowright server per test file. ROBOWRIGHT_ROBOT and ROBOWRIGHT_BACKEND
// (comma lists, as --rw-robot and --rw-backend) run every test on each, as Playwright runs every project. A failed
// test saves its trace and says how to open it; ROBOWRIGHT_RETRIES (or `retries`) runs it again, and a test that
// passes on a retry is reported flaky. Run with `node --test`.

import { AsyncLocalStorage } from "node:async_hooks";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { after, describe, test as nodeTest } from "node:test";

import { launch } from "./index.js";

const current = new AsyncLocalStorage();
const fileOptions = {};
let server = null;

const list = (v) =>
  (Array.isArray(v) ? v : String(v ?? "").split(","))
    .map((s) => String(s).trim())
    .filter(Boolean);

function server_() {
  server ??= launch({ stderr: process.env.ROBOWRIGHT_SERVER_LOG ? "inherit" : "ignore" });
  return server;
}

after(async () => {
  if (server) await (await server).close();
});

const slug = (s) => s.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");

function settings(own) {
  const o = { ...fileOptions, ...own };
  const robots = list(process.env.ROBOWRIGHT_ROBOT || o.robot || "so101");
  const backends = list(process.env.ROBOWRIGHT_BACKEND || o.backend || "mujoco");
  const retries = Number(o.retries ?? process.env.ROBOWRIGHT_RETRIES ?? 0);
  const traceDir = o.traceDir ?? process.env.ROBOWRIGHT_TRACE_DIR ?? "robowright-traces";
  return { o, robots, backends, retries, traceDir };
}

async function attempt(rw, launchArgs, fn) {
  await rw.robot.launch(launchArgs);
  const fixtures = { rw, robot: rw.robot, expect: (subject, opts) => rw.expect(subject, opts) };
  await current.run(rw, () => fn(fixtures));
}

function register(name, own, fn) {
  const { o, robots, backends, retries, traceDir } = settings(own);
  const many = robots.length * backends.length > 1;
  for (const backend of backends) {
    for (const robot of robots) {
      const full = many ? `${name} [${backend}-${robot}]` : name;
      const nodeOpts = {};
      for (const k of ["timeout", "skip", "todo", "only"]) if (o[k] !== undefined) nodeOpts[k] = o[k];
      nodeTest(full, nodeOpts, async (t) => {
        const rw = await server_();
        const launchArgs = { robot, backend, seed: o.seed ?? 0 };
        for (const k of ["objects", "fidelity", "ros2", "robotOptions"]) if (o[k] !== undefined) launchArgs[k] = o[k];
        let lastError = null;
        for (let i = 0; i <= retries; i++) {
          try {
            await attempt(rw, launchArgs, fn);
            if (i > 0) t.diagnostic(`flaky: failed ${i} time${i > 1 ? "s" : ""}, then passed on retry ${i}`);
            return;
          } catch (e) {
            lastError = e;
            const path = join(traceDir, `${slug(full)}${i ? `-retry${i}` : ""}.zip`);
            try {
              mkdirSync(traceDir, { recursive: true });
              const saved = (await rw.robot.saveTrace({ path })).trim().split("\n").pop();
              const at = /\S+\.zip/.exec(saved)?.[0] ?? path;
              e.message += `\n\ntrace: ${at}\n  robowright show-trace ${at}    (or --text for a plain summary)`;
              t.diagnostic(`trace: ${at}`);
            } catch {
              // no world to save (the launch itself failed): the error says why
            }
          }
        }
        throw lastError;
      });
    }
  }
}

export function test(name, optionsOrFn, maybeFn) {
  const [own, fn] = typeof optionsOrFn === "function" ? [{}, optionsOrFn] : [optionsOrFn ?? {}, maybeFn];
  register(name, own, fn);
}

// Options for every test in this file (Playwright's test.use): robot, backend, seed, objects, fidelity, ros2,
// robotOptions, retries, timeout, traceDir. Call it before the tests it applies to.
test.use = (options) => Object.assign(fileOptions, options);
test.only = (name, optionsOrFn, maybeFn) =>
  typeof optionsOrFn === "function" ? test(name, { only: true }, optionsOrFn) : test(name, { ...optionsOrFn, only: true }, maybeFn);
test.skip = (name, optionsOrFn, maybeFn) =>
  typeof optionsOrFn === "function" ? test(name, { skip: true }, optionsOrFn) : test(name, { ...optionsOrFn, skip: true }, maybeFn);
test.fixme = test.skip;
test.describe = describe;

// expect(subject) inside a test: the current test's world, as Playwright's expect needs no page passed.
export function expect(subject, options) {
  const rw = current.getStore();
  if (!rw) throw new Error("expect() from robowright/test is for use inside a test(); outside one, use rw.expect()");
  return rw.expect(subject, options);
}
