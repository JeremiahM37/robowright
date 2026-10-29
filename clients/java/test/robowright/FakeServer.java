package robowright;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A stand-in for robowright mcp: answers the MCP handshake and tools/call from a script, so the client's own code is
 * tested without Python. Each call is echoed back, so tests can see what was sent.
 */
public final class FakeServer {
  private static void send(Object m) {
    System.out.println(Json.write(m));
    System.out.flush();
  }

  private static Map<String, Object> text(Object id, String text, boolean error) {
    Map<String, Object> result = new LinkedHashMap<>();
    if (error) result.put("isError", true);
    result.put("content", List.of(Map.of("type", "text", "text", text)));
    return Map.of("jsonrpc", "2.0", "id", id, "result", result);
  }

  @SuppressWarnings("unchecked")
  public static void main(String[] argv) throws Exception {
    System.out.println("not a protocol line");
    BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
    String line;
    while ((line = in.readLine()) != null) {
      Map<String, Object> m = (Map<String, Object>) Json.parse(line);
      Object id = m.get("id");
      Map<String, Object> params = (Map<String, Object>) m.get("params");
      if ("initialize".equals(m.get("method"))) {
        send(Map.of("jsonrpc", "2.0", "method", "notifications/message", "params", Map.of("level", "info", "data", "hello")));
        send(Map.of("jsonrpc", "2.0", "id", id, "result", Map.of(
            "protocolVersion", params.get("protocolVersion"), "capabilities", Map.of(),
            "serverInfo", Map.of("name", "fake", "version", "9"))));
      } else if ("tools/call".equals(m.get("method"))) {
        String name = (String) params.get("name");
        Map<String, Object> args = (Map<String, Object>) params.get("arguments");
        if (name.equals("robot_crash")) System.exit(3);
        if (name.equals("robot_pick") && "nope".equals(args.get("object"))) {
          send(text(id, "KeyError: no object named 'nope'", true));
        } else if (name.equals("robot_expect")) {
          boolean ok = "bin".equals(((Map<String, Object>) args.get("args")).get("container")) != (Boolean) args.get("negate");
          send(text(id, (ok ? "PASS" : "FAIL") + ": " + Json.write(args), false));
        } else {
          send(text(id, Json.write(Map.of("name", name, "args", args)), false));
        }
      }
    }
  }
}
