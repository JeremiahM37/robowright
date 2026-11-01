package robowright.junit;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import robowright.Robowright;

/** Run by tests/test_java_client.py against the stand-in server: what @RobotTest does, each outcome on purpose. */
class SampleTest {
  @RobotTest(robot = {"panda", "ur5e"}, seed = 7)
  void passes(Robowright rw) {
    rw.expect("cube").toBeInside("bin");
    String launched = rw.call("robot_last_launch", Map.of()); // the stand-in's echo of the launch, as JSON
    assertTrue(launched.matches(".*\"robot\":\"(panda|ur5e)\".*") && launched.contains("\"seed\":7"), launched);
  }

  @RobotTest
  void failsWithItsTrace(Robowright rw) {
    rw.expect("cube").toBeInside("box");
  }

  @RobotTest(retries = 1)
  void flaky(Robowright rw) {
    rw.call("robot_flaky", Map.of());
  }
}
