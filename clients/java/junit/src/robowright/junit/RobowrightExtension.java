package robowright.junit;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.Extension;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.InvocationInterceptor;
import org.junit.jupiter.api.extension.ParameterContext;
import org.junit.jupiter.api.extension.ParameterResolver;
import org.junit.jupiter.api.extension.ReflectiveInvocationContext;
import org.junit.jupiter.api.extension.TestTemplateInvocationContext;
import org.junit.jupiter.api.extension.TestTemplateInvocationContextProvider;
import robowright.RobotTools;
import robowright.Robowright;

/**
 * What runs a {@link RobotTest}: one robowright server per test class, a fresh world per run, one run per robot and
 * engine, retries, and a failure's trace saved under robowright-traces/ and named in the failure.
 */
public final class RobowrightExtension implements TestTemplateInvocationContextProvider {
  private static final ExtensionContext.Namespace NS = ExtensionContext.Namespace.create(RobowrightExtension.class);
  private static final Pattern ZIP = Pattern.compile("[^\\s\"']+\\.zip");

  @Override
  public boolean supportsTestTemplate(ExtensionContext context) {
    return context.getTestMethod().map(m -> m.isAnnotationPresent(RobotTest.class)).orElse(false);
  }

  @Override
  public Stream<TestTemplateInvocationContext> provideTestTemplateInvocationContexts(ExtensionContext context) {
    RobotTest spec = context.getRequiredTestMethod().getAnnotation(RobotTest.class);
    List<String> robots = list("ROBOWRIGHT_ROBOT", spec.robot(), "so101");
    List<String> backends = list("ROBOWRIGHT_BACKEND", spec.backend(), "mujoco");
    boolean many = robots.size() * backends.size() > 1;
    List<TestTemplateInvocationContext> runs = new ArrayList<>();
    for (String backend : backends) {
      for (String robot : robots) {
        runs.add(new Run(robot, backend, spec, many));
      }
    }
    return runs.stream();
  }

  private static List<String> list(String env, String[] given, String fallback) {
    String v = System.getenv(env);
    List<String> out = v != null && !v.isBlank()
        ? Arrays.stream(v.split(",")).map(String::trim).filter(s -> !s.isEmpty()).toList()
        : Arrays.asList(given);
    return out.isEmpty() ? List.of(fallback) : out;
  }

  /** The class's server, started on its first test and stopped when the class is done. */
  static Robowright server(ExtensionContext context) {
    ExtensionContext owner = context.getParent().flatMap(ExtensionContext::getParent).orElse(context.getRoot());
    Server s = owner.getStore(NS).getOrComputeIfAbsent(Server.class, k -> new Server(), Server.class);
    return s.rw;
  }

  static final class Server implements ExtensionContext.Store.CloseableResource {
    final Robowright rw = Robowright.launch(new Robowright.LaunchOptions().setInheritStderr(System.getenv("ROBOWRIGHT_SERVER_LOG") != null));

    @Override
    public void close() {
      rw.close();
    }
  }

  static final class Run implements TestTemplateInvocationContext {
    final String robot;
    final String backend;
    final RobotTest spec;
    final boolean many;

    Run(String robot, String backend, RobotTest spec, boolean many) {
      this.robot = robot;
      this.backend = backend;
      this.spec = spec;
      this.many = many;
    }

    @Override
    public String getDisplayName(int index) {
      return many ? "[" + backend + "-" + robot + "]" : "[" + robot + "]";
    }

    void launch(Robowright rw) {
      rw.robot().launch(new RobotTools.LaunchOptions().setRobot(robot).setBackend(backend).setSeed(spec.seed()));
    }

    @Override
    public List<Extension> getAdditionalExtensions() {
      Run run = this;
      ParameterResolver params = new ParameterResolver() {
        @Override
        public boolean supportsParameter(ParameterContext p, ExtensionContext c) {
          Class<?> t = p.getParameter().getType();
          return t == Robowright.class || t == RobotTools.class;
        }

        @Override
        public Object resolveParameter(ParameterContext p, ExtensionContext c) {
          Robowright rw = server(c);
          return p.getParameter().getType() == RobotTools.class ? rw.robot() : rw;
        }
      };
      BeforeEachCallback launch = c -> run.launch(server(c));
      InvocationInterceptor retried = new InvocationInterceptor() {
        @Override
        public void interceptTestTemplateMethod(
            Invocation<Void> invocation, ReflectiveInvocationContext<Method> call, ExtensionContext c) throws Throwable {
          Robowright rw = server(c);
          String env = System.getenv("ROBOWRIGHT_RETRIES");
          int retries = env != null && !env.isBlank() ? Integer.parseInt(env.trim()) : spec.retries();
          Throwable last = null;
          for (int i = 0; i <= retries; i++) {
            try {
              if (i == 0) {
                invocation.proceed();
              } else {
                run.launch(rw); // a fresh world for the retry
                Method m = call.getExecutable();
                m.setAccessible(true);
                try {
                  m.invoke(call.getTarget().orElse(null), call.getArguments().toArray());
                } catch (InvocationTargetException e) {
                  throw e.getCause();
                }
              }
              if (i > 0) c.publishReportEntry("flaky", "failed " + i + " time" + (i > 1 ? "s" : "") + ", then passed on retry " + i);
              return;
            } catch (Throwable t) {
              last = withTrace(t, rw, c, i);
            }
          }
          throw last;
        }
      };
      return List.of(params, launch, retried);
    }

    /** The failure, its message ending with where its trace was saved. */
    static Throwable withTrace(Throwable t, Robowright rw, ExtensionContext c, int attempt) {
      String dir = Optional.ofNullable(System.getenv("ROBOWRIGHT_TRACE_DIR")).orElse("robowright-traces");
      String name = (c.getRequiredTestClass().getSimpleName() + "." + c.getRequiredTestMethod().getName() + c.getDisplayName())
          .replaceAll("[^A-Za-z0-9._-]+", "_").replaceAll("^_+|_+$", "") + (attempt > 0 ? "-retry" + attempt : "");
      String path = dir + "/" + name + ".zip";
      try {
        Files.createDirectories(Path.of(dir));
        String saved = rw.robot().saveTrace(new RobotTools.SaveTraceOptions().setPath(path));
        Matcher m = ZIP.matcher(saved);
        String at = m.find() ? m.group() : path;
        c.publishReportEntry("trace", at);
        AssertionError out = new AssertionError(
            t.getMessage() + "\n\ntrace: " + at + "\n  robowright show-trace " + at + "    (or --text for a plain summary)", t);
        out.setStackTrace(t.getStackTrace());
        return out;
      } catch (Exception e) {
        return t; // no world to save (the launch itself failed): the failure says why
      }
    }
  }
}
