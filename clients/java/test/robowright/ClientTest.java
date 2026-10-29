package robowright;

import java.io.File;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * The client's tests, with no test framework so they run on a bare JDK: java -cp out robowright.ClientTest.
 * Prints one line per test and "# pass N" / "# fail N"; exits 1 on any failure.
 */
public final class ClientTest {
  interface Body {
    void run() throws Exception;
  }

  private static int passed;
  private static final List<String> failed = new ArrayList<>();

  private static void test(String name, Body body) {
    try {
      body.run();
      passed++;
      System.out.println("ok - " + name);
    } catch (Throwable t) {
      failed.add(name);
      System.out.println("not ok - " + name + ": " + t);
    }
  }

  private static void check(boolean ok, String what) {
    if (!ok) throw new AssertionError(what);
  }

  private static <T extends Throwable> T raises(Class<T> type, Body body) throws Exception {
    try {
      body.run();
    } catch (Throwable t) {
      if (type.isInstance(t)) return type.cast(t);
      throw new AssertionError("expected " + type.getSimpleName() + ", got " + t, t);
    }
    throw new AssertionError("expected " + type.getSimpleName() + ", nothing was thrown");
  }

  private static Robowright fake() {
    String java = System.getProperty("java.home") + File.separator + "bin" + File.separator + "java";
    return Robowright.launch(new Robowright.LaunchOptions()
        .setCommand(List.of(java, "-cp", System.getProperty("java.class.path"), "robowright.FakeServer")));
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Object> echoed(String out) {
    return (Map<String, Object>) Json.parse(out);
  }

  public static void main(String[] argv) throws Exception {
    test("every tool is a public method of its namespace", () -> {
      try (Robowright rw = fake()) {
        check(rw.server().equals(Map.of("name", "fake", "version", "9")), "server " + rw.server());
        int n = 0;
        for (Class<?> c : List.of(RobotTools.class, Ros2Tools.class, SimTools.class)) {
          for (Method m : c.getDeclaredMethods()) if (Modifier.isPublic(m.getModifiers())) n++;
        }
        check(n > 50, "only " + n + " tool methods");
        Map<String, Object> out = echoed(rw.robot().moveTo(List.of(0.2, 0, 0.1)));
        check(out.get("name").equals("robot_move_to"), "name " + out);
        check(out.get("args").equals(Map.of("target", List.of(0.2, 0L, 0.1))), "args " + out);
      }
    });

    test("options reach the server under their own names, and unset ones are left out", () -> {
      try (Robowright rw = fake()) {
        Map<String, Object> out = echoed(rw.robot().launch(new RobotTools.LaunchOptions().setRobot("panda").setKeepOpen(true)));
        check(out.get("args").equals(Map.of("robot", "panda", "keep_open", true)), "args " + out);
        check(echoed(rw.robot().waitSeconds(1.5)).get("args").equals(Map.of("seconds", 1.5)), "waitSeconds");
        check(echoed(rw.robot().switchTo("b")).get("name").equals("robot_switch"), "switchTo");
      }
    });

    test("a tool error throws RobowrightException naming the tool", () -> {
      try (Robowright rw = fake()) {
        RobowrightException e = raises(RobowrightException.class, () -> rw.robot().pick("nope"));
        check(e.tool().equals("robot_pick") && e.getMessage().contains("no object named"), e.toString());
      }
    });

    test("expect passes, fails with the reason, negates with not(), and takes the matcher's first argument bare", () -> {
      try (Robowright rw = fake()) {
        check(rw.expect("cube").toBeInside("bin").startsWith("PASS"), "pass");
        raises(ExpectationException.class, () -> rw.expect("cube").toBeInside("box"));
        check(rw.expect("cube").not().toBeInside("box").startsWith("PASS"), "not");
        String out = rw.expect("cube", 2).toBeInside(Map.of("container", "bin"), Map.of("min_fraction", 0.5));
        check(echoed(out.substring(6)).equals(Map.of("subject", "cube", "matcher", "to_be_inside",
            "args", Map.of("container", "bin", "min_fraction", 0.5), "negate", false, "timeout", 2.0)), out);
        check(echoed(rw.expect("cube").toBeInside("bin").substring(6)).get("timeout") == null, "no timeout sent");
      }
    });

    test("a matcher with no first argument refuses a bare value", () -> {
      try (Robowright rw = fake()) {
        raises(IllegalArgumentException.class, () -> rw.expect("cube").toBeAtRest(3));
      }
    });

    test("calls in flight fail, not hang, when the server dies", () -> {
      Robowright rw = fake();
      RobowrightException e = raises(RobowrightException.class, () -> rw.call("robot_crash"));
      check(e.getMessage().contains("exited (code 3)"), e.getMessage());
      raises(RobowrightException.class, () -> rw.robot().snapshot());
      rw.close();
    });

    test("a command that does not exist fails launch", () -> {
      RobowrightException e = raises(RobowrightException.class,
          () -> Robowright.launch(new Robowright.LaunchOptions().setCommand(List.of("robowright-no-such-command"))));
      check(e.getMessage().contains("could not start"), e.getMessage());
    });

    String real = System.getenv("ROBOWRIGHT_COMMAND");
    if (real != null && !real.isBlank()) {
      test("against the real server: pick the cube, hold it, and fail to find it in the bin", () -> {
        try (Robowright rw = Robowright.launch(new Robowright.LaunchOptions().setInheritStderr(false))) {
          check("robowright".equals(rw.server().get("name")), "server " + rw.server());
          check(rw.robot().launch().contains("cube: red box"), "launch");
          rw.robot().pick("cube");
          check(rw.expect("gripper").toBeHolding("cube").startsWith("PASS"), "holding");
          ExpectationException e = raises(ExpectationException.class, () -> rw.expect("cube", 0.2).toBeInside("bin"));
          check(Pattern.compile("bin spans").matcher(e.getMessage()).find(), e.getMessage());
          RobowrightException err = raises(RobowrightException.class, () -> rw.robot().pick("nope"));
          check(err.getMessage().contains("scene has"), err.getMessage());
        }
      });
    } else {
      System.out.println("ok - against the real server # SKIP set ROBOWRIGHT_COMMAND to run it");
    }

    System.out.println("# pass " + passed);
    System.out.println("# fail " + failed.size());
    System.exit(failed.isEmpty() ? 0 : 1);
  }
}
