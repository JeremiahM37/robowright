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

What it does not have yet: the pytest plugin's runner features (retries, projects, shards, the HTML report) are
Python's. From Java you run the world and assert on it in your own test framework, and `rw.robot().generateTest()`
writes the session as a pytest test when you want it in the suite.
