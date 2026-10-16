"""expect.poll (a value the test computes, waited for in simulated time) and expect(block).to_pass() (a block of
expectations tried every control step until it passes): Playwright's expect.poll and toPass."""

import pytest

import robowright as rw
from robowright import expect
from robowright.codegen import generate
from robowright.errors import ExpectationError, InvariantViolation
from robowright.trace import Trace


@pytest.fixture
def world(tmp_path):
    with rw.launch(robot="panda", backend="mujoco", settings=rw.Settings(trace="on", trace_dir=str(tmp_path))) as w:
        yield w


def test_poll_waits_in_simulated_time_until_the_value_holds(world):
    start = world.time
    expect.poll(lambda: world.time - start, name="elapsed").to_be_greater_than(0.5, timeout=2)
    assert 0.5 < world.time - start < 0.6  # it stepped only as long as it had to


def test_poll_failure_says_the_last_value_it_read(world):
    with pytest.raises(ExpectationError, match=r"answer = 7 \(want 42\)"):
        expect.poll(lambda: 7, name="answer").to_equal(42, timeout=0.2)
    with pytest.raises(ExpectationError, match=r"value = 3 \(want < 2\)"):
        expect.poll(lambda: 3).to_be_less_than(2, timeout=0.1)
    with pytest.raises(ExpectationError, match=r"off by 0.5"):
        expect.poll(lambda: [1.0, 2.5]).to_be_close_to([1.0, 2.0], tol=0.1, timeout=0.1)


def test_poll_matchers_negate_and_name_a_function_by_its_name(world):
    def gripper_state():
        return "open"

    expect.poll(gripper_state).not_.to_equal("closed", timeout=0)
    expect.poll(gripper_state).to_be_truthy(timeout=0)
    with pytest.raises(ExpectationError, match=r"gripper_state\(\) = 'open'"):
        expect.poll(gripper_state).to_equal("closed", timeout=0.05)


def test_poll_as_an_invariant_fails_at_the_step_it_breaks(world):
    start = world.time
    expect.poll(lambda: world.time - start, name="elapsed").always.to_be_less_than(0.3)
    with pytest.raises(InvariantViolation, match="elapsed"):
        world.step(1000)
    assert world.time - start == pytest.approx(0.3, abs=0.01)


def test_to_pass_tries_the_block_until_every_expectation_in_it_holds(world):
    start = world.time
    cube = world.scene["cube"]
    tries = []

    def block():
        tries.append(world.time)
        expect(cube).to_be_above(0.0)
        expect.poll(lambda: world.time).to_be_greater_than(start + 0.4)

    expect(block).to_pass(timeout=2)
    assert 0.4 < world.time - start < 0.5
    assert len(tries) > 10  # once per control step, each expectation checked once there


def test_to_pass_failure_says_which_expectation_failed_last(world):
    cube = world.scene["cube"]

    def block():
        expect(cube).to_be_above(0.0)
        expect(cube).to_be_near((9, 9, 9), tol=0.01)

    with pytest.raises(ExpectationError, match=r"to_pass failed after 0\.\d+s[\s\S]*expect\(cube\)\.to_be_near: cube at"):
        expect(block).to_pass(timeout=0.3)


def test_a_plain_assert_in_the_block_is_retried_too(world):
    start = world.time

    def block():
        assert world.time - start >= 0.2, "not yet"

    expect(block).to_pass(timeout=1)
    with pytest.raises(ExpectationError, match="never"):
        expect(lambda: (_ for _ in ()).throw(AssertionError("never"))).to_pass(timeout=0.05)


def test_to_pass_as_an_invariant_and_what_it_refuses(world):
    cube = world.scene["cube"]
    expect(lambda: expect(cube).to_be_above(-1.0)).always.to_pass()
    world.step(20)  # holds at every step
    with pytest.raises(TypeError, match="registered once"):
        expect(lambda: expect(cube).always.to_be_above(0.0)).to_pass(timeout=0.05)
    with pytest.raises(TypeError, match="not_.to_pass"):
        expect(lambda: None).not_.to_pass()


def test_codegen_says_it_cannot_write_the_tests_own_code_again(world):
    expect.poll(lambda: 1, name="one").to_equal(1, timeout=0)
    expect(lambda: None).to_pass(timeout=0)
    expect(world.scene["cube"]).to_be_above(0.0, timeout=0)
    path = world.close()
    code = generate(Trace(path), test_name="test_again")
    assert code.count("# not written again (it ran the original test's own code)") == 2, code
    assert "expect(scene['cube']).to_be_above(" in code or 'expect(scene["cube"]).to_be_above(' in code, code
    compile(code, "test_again.py", "exec")
