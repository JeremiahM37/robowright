package robowright.junit;

import robowright.Robowright;

/** Run by tests/test_java_client.py against the real robowright server: a pick and place, and a failure on purpose. */
class RealServerTest {
  @RobotTest
  void putsTheCubeInTheBin(Robowright rw) {
    rw.robot().pick("cube");
    rw.expect("gripper").toBeHolding("cube");
    rw.robot().place("bin");
    rw.expect("cube", 2).toBeInside("bin");
  }

  @RobotTest
  void leavesTheCubeWhereItWas(Robowright rw) {
    rw.expect("cube", 0.2).toBeInside("bin");
  }
}
