package robowright;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** The little JSON the MCP protocol needs: objects become LinkedHashMap, arrays ArrayList, numbers Long or Double. */
final class Json {
  private Json() {}

  static String write(Object v) {
    StringBuilder sb = new StringBuilder();
    write(sb, v);
    return sb.toString();
  }

  private static void write(StringBuilder sb, Object v) {
    if (v == null) {
      sb.append("null");
    } else if (v instanceof String s) {
      string(sb, s);
    } else if (v instanceof Double d && (d.isNaN() || d.isInfinite()) || v instanceof Float f && (f.isNaN() || f.isInfinite())) {
      throw new IllegalArgumentException("JSON has no " + v);
    } else if (v instanceof Number || v instanceof Boolean) {
      sb.append(v);
    } else if (v instanceof Map<?, ?> m) {
      sb.append('{');
      boolean first = true;
      for (Map.Entry<?, ?> e : m.entrySet()) {
        if (!first) sb.append(',');
        first = false;
        string(sb, String.valueOf(e.getKey()));
        sb.append(':');
        write(sb, e.getValue());
      }
      sb.append('}');
    } else if (v instanceof Iterable<?> it) {
      sb.append('[');
      boolean first = true;
      for (Object o : it) {
        if (!first) sb.append(',');
        first = false;
        write(sb, o);
      }
      sb.append(']');
    } else if (v.getClass().isArray()) {
      sb.append('[');
      for (int i = 0; i < Array.getLength(v); i++) {
        if (i > 0) sb.append(',');
        write(sb, Array.get(v, i));
      }
      sb.append(']');
    } else {
      string(sb, v.toString());
    }
  }

  private static void string(StringBuilder sb, String s) {
    sb.append('"');
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '"' -> sb.append("\\\"");
        case '\\' -> sb.append("\\\\");
        case '\n' -> sb.append("\\n");
        case '\r' -> sb.append("\\r");
        case '\t' -> sb.append("\\t");
        default -> {
          if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
          else sb.append(c);
        }
      }
    }
    sb.append('"');
  }

  static Object parse(String s) {
    Parser p = new Parser(s);
    Object v = p.value();
    p.ws();
    if (p.i != s.length()) throw p.error("trailing text");
    return v;
  }

  private static final class Parser {
    final String s;
    int i;

    Parser(String s) {
      this.s = s;
    }

    IllegalArgumentException error(String what) {
      return new IllegalArgumentException(what + " at " + i + " in JSON");
    }

    void ws() {
      while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
    }

    Object value() {
      ws();
      if (i >= s.length()) throw error("end");
      char c = s.charAt(i);
      if (c == '{') return object();
      if (c == '[') return array();
      if (c == '"') return string();
      if (s.startsWith("true", i)) {
        i += 4;
        return Boolean.TRUE;
      }
      if (s.startsWith("false", i)) {
        i += 5;
        return Boolean.FALSE;
      }
      if (s.startsWith("null", i)) {
        i += 4;
        return null;
      }
      return number();
    }

    Map<String, Object> object() {
      Map<String, Object> m = new LinkedHashMap<>();
      i++;
      ws();
      if (i < s.length() && s.charAt(i) == '}') {
        i++;
        return m;
      }
      while (true) {
        ws();
        if (i >= s.length() || s.charAt(i) != '"') throw error("expected a key");
        String k = string();
        ws();
        if (i >= s.length() || s.charAt(i) != ':') throw error("expected ':'");
        i++;
        m.put(k, value());
        ws();
        if (i >= s.length()) throw error("end");
        char c = s.charAt(i++);
        if (c == '}') return m;
        if (c != ',') throw error("expected ',' or '}'");
      }
    }

    List<Object> array() {
      List<Object> a = new ArrayList<>();
      i++;
      ws();
      if (i < s.length() && s.charAt(i) == ']') {
        i++;
        return a;
      }
      while (true) {
        a.add(value());
        ws();
        if (i >= s.length()) throw error("end");
        char c = s.charAt(i++);
        if (c == ']') return a;
        if (c != ',') throw error("expected ',' or ']'");
      }
    }

    String string() {
      StringBuilder sb = new StringBuilder();
      i++;
      while (true) {
        if (i >= s.length()) throw error("unterminated string");
        char c = s.charAt(i++);
        if (c == '"') return sb.toString();
        if (c != '\\') {
          sb.append(c);
          continue;
        }
        if (i >= s.length()) throw error("end");
        char e = s.charAt(i++);
        switch (e) {
          case 'n' -> sb.append('\n');
          case 't' -> sb.append('\t');
          case 'r' -> sb.append('\r');
          case 'b' -> sb.append('\b');
          case 'f' -> sb.append('\f');
          case 'u' -> {
            if (i + 4 > s.length()) throw error("bad \\u escape");
            sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
            i += 4;
          }
          default -> sb.append(e);
        }
      }
    }

    Number number() {
      int start = i;
      while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
      String n = s.substring(start, i);
      if (n.isEmpty()) throw error("unexpected character");
      if (n.contains(".") || n.contains("e") || n.contains("E")) return Double.parseDouble(n);
      try {
        return Long.parseLong(n);
      } catch (NumberFormatException e) {
        return Double.parseDouble(n);
      }
    }
  }
}
