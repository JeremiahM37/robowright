# Java

Playwright has one API in four languages. robowright's test runner is pytest, but its robots can be driven from
Java as well: `clients/java` is a client for the robowright MCP server, for Java 17 or later, with no dependencies.
Every tool an AI agent gets is a method there, and `expect` has the same matchers.

```java
try (Robowright rw = Robowright.launch()) { // starts `robowright mcp`
  rw.robot().launch(new RobotTools.LaunchOptions().setRobot("so101"));
  rw.robot().pick("cube");
  rw.expect("gripper").toBeHolding("cube");
  rw.robot().place("bin");
  rw.expect("cube", 2).toBeInside("bin");
}
```

A failing expectation throws `ExpectationException`, an `AssertionError`, so it fails a JUnit or TestNG test the
way a Playwright assertion does. It is not on Maven Central yet; its README in
[clients/java](https://github.com/JeremiahM37/robowright/tree/main/clients/java) shows how to build the jar.

Snapshot matchers keep their baseline beside the file named as `test_file`, as Python keeps it beside the test:
`rw.expect("world").toMatchStateSnapshot("tidy", Map.of("test_file", "src/test/java/TidyTest.java"))`.

With JUnit 5, `@RobotTest` (in `robowright.junit`) writes robot tests as Playwright Test writes web tests: one
robowright server per test class, a fresh world per test, the `Robowright` it drives as a parameter, a run per robot
and engine, retries that report a test flaky when it passes on one, and a failure that saves its trace and names it:

```java
import robowright.Robowright;
import robowright.junit.RobotTest;

class PickTest {
  @RobotTest(robot = {"so101", "panda"}, retries = 1)
  void putsTheCubeInTheBin(Robowright rw) {
    rw.robot().pick("cube");
    rw.robot().place("bin");
    rw.expect("cube", 2).toBeInside("bin");
  }
}
```

`ROBOWRIGHT_ROBOT` and `ROBOWRIGHT_BACKEND` (comma lists) override the robots and engines, as `--rw-robot` and
`--rw-backend` do; `ROBOWRIGHT_RETRIES` the retries. JUnit's API is a provided dependency of the jar: a project with
JUnit 5 has it. What it does not have yet, of the pytest plugin's runner: trials, projects and robowright's HTML
report (JUnit's own reports work), and `rw.robot().generateTest()` writes a session as a pytest test when you want
those.
