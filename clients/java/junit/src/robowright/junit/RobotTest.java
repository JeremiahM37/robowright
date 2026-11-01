package robowright.junit;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import org.junit.jupiter.api.TestTemplate;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * A robot test, as Playwright Test writes a web test: JUnit runs it once per robot and engine, each run with a fresh
 * world, and a failure saves its trace and names it. Take a {@code Robowright} parameter to drive the world:
 *
 * <pre>{@code
 * @RobotTest(robot = "so101")
 * void putsTheCubeInTheBin(Robowright rw) {
 *   rw.robot().pick("cube");
 *   rw.robot().place("bin");
 *   rw.expect("cube", 2).toBeInside("bin");
 * }
 * }</pre>
 *
 * <p>ROBOWRIGHT_ROBOT and ROBOWRIGHT_BACKEND (comma lists, as --rw-robot and --rw-backend) override {@link #robot}
 * and {@link #backend}; ROBOWRIGHT_RETRIES overrides {@link #retries}; ROBOWRIGHT_TRACE_DIR is where traces go.
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@TestTemplate
@ExtendWith(RobowrightExtension.class)
public @interface RobotTest {
  /** The robots it runs on, each a separate run (default so101). */
  String[] robot() default {};

  /** The engines it runs on: mujoco (default), pybullet, drake, genesis, isaac or ros2. */
  String[] backend() default {};

  int seed() default 0;

  /** Run a failed test again this many times, on a fresh world; one that then passes is reported flaky. */
  int retries() default 0;
}
