package robowright;

/** An expectation that did not hold: robot_expect answered FAIL, and the message says why. */
public class ExpectationException extends AssertionError {
  private static final long serialVersionUID = 1L;

  public ExpectationException(String message) {
    super(message);
  }
}
