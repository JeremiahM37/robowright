package robowright;

/** A tool failed (the server answered isError), or the server could not be reached. */
public class RobowrightException extends RuntimeException {
  private static final long serialVersionUID = 1L;
  private final String tool;

  public RobowrightException(String tool, String message) {
    super(message);
    this.tool = tool;
  }

  public RobowrightException(String tool, String message, Throwable cause) {
    super(message, cause);
    this.tool = tool;
  }

  /** The tool that failed, e.g. "robot_pick"; null when the server itself failed. */
  public String tool() {
    return tool;
  }
}
