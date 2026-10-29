package robowright;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * robowright for Java: the same simulated robots, matchers and traces as the Python library, driven through the
 * robowright MCP server ({@code robowright mcp}) over stdio. No dependencies.
 *
 * <pre>{@code
 * try (Robowright rw = Robowright.launch()) {
 *   rw.robot().launch(new RobotTools.LaunchOptions().setRobot("panda"));
 *   rw.robot().pick("cube");
 *   rw.robot().place("bin");
 *   rw.expect("cube").toBeInside("bin");
 * }
 * }</pre>
 */
public final class Robowright implements AutoCloseable {
  static final String PROTOCOL = "2025-06-18";

  // The argument a matcher takes first, so expect("cube").toBeInside("bin") reads as it does in Python.
  private static final Map<String, String> PRIMARY = Map.of(
      "to_be_inside", "container",
      "to_be_holding", "obj",
      "to_be_near", "target",
      "to_be_above", "target",
      "to_be_touching", "other",
      "to_have_joint", "joint",
      "to_be_visible", "camera",
      "to_be_hidden", "camera");

  /** How to start the server. */
  public static final class LaunchOptions {
    List<String> command;
    Map<String, String> env;
    java.io.File cwd;
    boolean inheritStderr = true;

    /** The command that starts the server; default robowright mcp (or ROBOWRIGHT_COMMAND, split on spaces). */
    public LaunchOptions setCommand(List<String> command) {
      this.command = command;
      return this;
    }

    /** Extra environment variables for the server process. */
    public LaunchOptions setEnv(Map<String, String> env) {
      this.env = env;
      return this;
    }

    public LaunchOptions setCwd(java.io.File cwd) {
      this.cwd = cwd;
      return this;
    }

    /** Whether the server's stderr goes to this process's (default true) or is discarded. */
    public LaunchOptions setInheritStderr(boolean inherit) {
      this.inheritStderr = inherit;
      return this;
    }
  }

  private final Process proc;
  private final OutputStream stdin;
  private final Map<Long, CompletableFuture<Map<String, Object>>> pending = new ConcurrentHashMap<>();
  private final AtomicLong next = new AtomicLong(1);
  private volatile RuntimeException closed;
  private final RobotTools robot = new RobotTools(this);
  private final Ros2Tools ros2 = new Ros2Tools(this);
  private final SimTools sim = new SimTools(this);
  private Map<String, Object> server = Map.of();

  private Robowright(Process proc) {
    this.proc = proc;
    this.stdin = proc.getOutputStream();
    Thread reader = new Thread(this::read, "robowright-reader");
    reader.setDaemon(true);
    reader.start();
  }

  /** Start {@code robowright mcp} (or ROBOWRIGHT_COMMAND) and connect to it. */
  public static Robowright launch() {
    return launch(new LaunchOptions());
  }

  /** Start the server and connect to it. */
  public static Robowright launch(LaunchOptions options) {
    List<String> command = options.command;
    if (command == null) {
      String env = System.getenv("ROBOWRIGHT_COMMAND");
      command = env != null && !env.isBlank()
          ? Arrays.stream(env.trim().split(" +")).toList()
          : List.of("robowright", "mcp");
    }
    ProcessBuilder pb = new ProcessBuilder(command);
    if (options.cwd != null) pb.directory(options.cwd);
    if (options.env != null) pb.environment().putAll(options.env);
    pb.redirectError(options.inheritStderr ? ProcessBuilder.Redirect.INHERIT : ProcessBuilder.Redirect.DISCARD);
    Process proc;
    try {
      proc = pb.start();
    } catch (IOException e) {
      throw new RobowrightException(null, "could not start " + String.join(" ", command) + ": " + e.getMessage(), e);
    }
    Robowright rw = new Robowright(proc);
    try {
      Map<String, Object> init = rw.request("initialize", Map.of(
          "protocolVersion", PROTOCOL,
          "capabilities", Map.of(),
          "clientInfo", Map.of("name", "robowright-java", "version", "0.1.0")));
      rw.send(Map.of("jsonrpc", "2.0", "method", "notifications/initialized"));
      Object info = init.get("serverInfo");
      if (info instanceof Map<?, ?> m) {
        @SuppressWarnings("unchecked")
        Map<String, Object> cast = (Map<String, Object>) m;
        rw.server = cast;
      }
      return rw;
    } catch (RuntimeException e) {
      proc.destroy();
      throw e;
    }
  }

  public RobotTools robot() {
    return robot;
  }

  public Ros2Tools ros2() {
    return ros2;
  }

  public SimTools sim() {
    return sim;
  }

  /** The server's name and version, from its answer to initialize. */
  public Map<String, Object> server() {
    return server;
  }

  /** Playwright-style assertions on an object name, or robot, gripper, tcp, base. */
  public Assertions expect(String subject) {
    return new Assertions(this, subject, null, false);
  }

  /** The same, retrying for up to timeout seconds of simulated time. */
  public Assertions expect(String subject, double timeout) {
    return new Assertions(this, subject, timeout, false);
  }

  /** Call any tool by its MCP name, e.g. call("robot_pick", Map.of("object", "cube")); argument names as the server has them. */
  public String call(String tool, Map<String, ?> args) {
    Map<String, Object> params = new LinkedHashMap<>();
    if (args != null) args.forEach((k, v) -> {
      if (v != null) params.put(k, v);
    });
    Map<String, Object> result = request("tools/call", Map.of("name", tool, "arguments", params));
    String out = text(result);
    if (Boolean.TRUE.equals(result.get("isError"))) throw new RobowrightException(tool, out);
    return out;
  }

  public String call(String tool) {
    return call(tool, null);
  }

  String expectCall(String subject, String matcher, Object arg, Map<String, ?> rest, Double timeout, boolean negate) {
    Map<String, Object> args = new LinkedHashMap<>();
    if (arg instanceof Map<?, ?> m) {
      m.forEach((k, v) -> args.put(String.valueOf(k), v));
    } else if (arg != null) {
      String key = PRIMARY.get(matcher);
      if (key == null) throw new IllegalArgumentException(matcher + " takes a map of arguments, not " + arg);
      args.put(key, arg);
    }
    if (rest != null) args.putAll(rest);
    Map<String, Object> call = new LinkedHashMap<>();
    call.put("subject", subject);
    call.put("matcher", matcher);
    call.put("args", args);
    call.put("negate", negate);
    call.put("timeout", timeout);
    String out = call("robot_expect", call);
    if (!out.startsWith("PASS")) throw new ExpectationException(out);
    return out;
  }

  /** Close the worlds and stop the server. */
  @Override
  public void close() {
    if (!proc.isAlive()) return;
    try {
      call("robot_close");
    } catch (RuntimeException e) {
      // nothing open, or the server is already going
    }
    try {
      stdin.close();
    } catch (IOException e) {
      // already closed
    }
    try {
      if (!proc.waitFor(5, TimeUnit.SECONDS)) {
        proc.destroy();
        proc.waitFor(5, TimeUnit.SECONDS);
      }
    } catch (InterruptedException e) {
      proc.destroy();
      Thread.currentThread().interrupt();
    }
  }

  private static String text(Map<String, Object> result) {
    List<String> parts = new ArrayList<>();
    if (result.get("content") instanceof List<?> content) {
      for (Object c : content) {
        if (c instanceof Map<?, ?> m && "text".equals(m.get("type"))) parts.add(String.valueOf(m.get("text")));
      }
    }
    return String.join("\n", parts);
  }

  private synchronized void send(Map<String, Object> msg) {
    if (closed != null) throw closed;
    try {
      stdin.write((Json.write(msg) + "\n").getBytes(StandardCharsets.UTF_8));
      stdin.flush();
    } catch (IOException e) {
      throw new RobowrightException(null, "the robowright server is not reading: " + e.getMessage(), e);
    }
  }

  private Map<String, Object> request(String method, Map<String, Object> params) {
    long id = next.getAndIncrement();
    CompletableFuture<Map<String, Object>> f = new CompletableFuture<>();
    pending.put(id, f);
    if (closed != null) f.completeExceptionally(closed);
    else {
      try {
        send(Map.of("jsonrpc", "2.0", "id", id, "method", method, "params", params));
      } catch (RuntimeException e) {
        pending.remove(id);
        throw e;
      }
    }
    try {
      return f.get();
    } catch (ExecutionException e) {
      if (e.getCause() instanceof RuntimeException r) throw r;
      throw new RobowrightException(null, e.getCause().toString(), e.getCause());
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RobowrightException(null, "interrupted waiting for " + method, e);
    }
  }

  @SuppressWarnings("unchecked")
  private void read() {
    try (BufferedReader r = new BufferedReader(new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8))) {
      String line;
      while ((line = r.readLine()) != null) {
        line = line.trim();
        if (line.isEmpty()) continue;
        Object msg;
        try {
          msg = Json.parse(line);
        } catch (IllegalArgumentException e) {
          continue; // not a protocol message
        }
        if (!(msg instanceof Map<?, ?> m) || !(m.get("id") instanceof Number id)) continue; // a notification
        CompletableFuture<Map<String, Object>> f = pending.remove(id.longValue());
        if (f == null) continue; // a request from the server we do not serve
        if (m.get("error") instanceof Map<?, ?> err) f.completeExceptionally(new RobowrightException(null, String.valueOf(err.get("message"))));
        else f.complete(m.get("result") instanceof Map<?, ?> res ? (Map<String, Object>) res : Map.of());
      }
    } catch (IOException e) {
      // the server went away; reported below
    }
    String how;
    try {
      how = "code " + proc.waitFor();
    } catch (InterruptedException e) {
      how = "unknown status";
    }
    closed = new RobowrightException(null, "the robowright server exited (" + how + ")");
    for (CompletableFuture<Map<String, Object>> f : pending.values()) f.completeExceptionally(closed);
    pending.clear();
  }
}
