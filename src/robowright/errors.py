class RobowrightError(Exception):
    """Base class for robowright errors."""


class ExpectationError(AssertionError, RobowrightError):
    """An ``expect(...)`` assertion did not hold within its timeout."""


class ActionTimeoutError(RobowrightError, TimeoutError):
    """A robot action did not complete within its timeout."""


class ActionError(RobowrightError):
    """The robot's own controller refused a move, or gave it up part-way (a trajectory controller
    aborting on its path tolerance, say): the message is the controller's."""


class InvariantViolation(ExpectationError):
    """A condition registered with ``expect(...).always`` stopped holding."""


class CapabilityError(RobowrightError):
    """The backend cannot provide what an action or matcher needs."""


class UnreachableError(RobowrightError):
    """Inverse kinematics found no joint configuration for a target."""


class GraspError(RobowrightError):
    """``pick`` lifted without the object in both jaws."""


class TooWideError(GraspError):
    """The object is wider than the gripper opens: the robot cannot do what the test asks. Run
    under pytest, the test is skipped on that robot, saying so (Lite6's narrow gripper opens
    12 mm; the default cube is 25 mm)."""


class TimeLimitError(Exception):
    """A test ran past its time limit (``--rw-timeout``, ``@pytest.mark.rw_timeout``), wall clock.

    Not a RobowrightError on purpose: a trials test counts those as one failed trial and goes on,
    and a test out of time must stop.
    """


class SafetyLimitError(ActionError):
    """A command was refused because it broke the robot's ``[ros2] limits`` (speed or position):
    nothing of it was sent."""
