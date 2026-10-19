"""expect.extend: your own matchers, as Playwright's expect.extend, retried and traced like the built-in ones."""

import numpy as np
import pytest

import robowright as rw
from robowright import condition, expect
from robowright.errors import ExpectationError


def to_be_left_of(obj, other, by=0.0):
    """``obj`` at least ``by`` metres further along -y than ``other`` (the robot's left, seen from the front)."""
    gap = float(other.position[1] - obj.position[1])
    return gap >= by, f"{gap * 100:.1f} cm left of {other.name}"


def to_be_slower_than(value, limit):
    return value < limit, f"{value:.2f}"


@pytest.fixture(autouse=True)
def matchers():
    expect.extend(to_be_left_of=to_be_left_of, to_be_slower_than=to_be_slower_than)


@pytest.fixture
def w():
    w = rw.launch(settings=rw.Settings(trace="on"))
    yield w
    w.close()


def test_a_custom_matcher_passes_and_is_in_the_trace(w):
    cube, bin_ = w.scene["cube"], w.scene["bin"]
    expect(cube).to_be_left_of(bin_, by=0.1)
    e = w.trace.events[-1]
    assert e.name == "expect(cube).to_be_left_of" and e.status == "ok"
    assert e.args["kwargs"] == {"other": {"$ref": "bin"}, "by": 0.1}
    assert e.args.get("code")  # codegen leaves it as a comment: the generated test would not have the matcher


def test_a_failing_custom_matcher_says_what_it_saw_after_its_timeout(w):
    with pytest.raises(ExpectationError, match=r"to_be_left_of failed after 0.20s.*\n.*18.0 cm left of bin"):
        expect(w.scene["cube"]).to_be_left_of(w.scene["bin"], by=0.5, timeout=0.2)
    assert w.time >= 0.2  # it waited, in simulated time


def test_it_waits_until_the_world_catches_up(w):
    moved = []

    def later(world):  # 0.3 s in, something moves the cube half a metre to the left
        if world.time >= 0.3 and not moved:
            moved.append(world.time)
            world.move_object("cube", (0.22, -0.6, 0.0125))

    w._step_hooks.append(later)
    expect(w.scene["cube"]).to_be_left_of(w.scene["bin"], by=0.5, timeout=1.0)
    assert moved and w.time >= moved[0]  # it was still retrying when the cube moved


def test_not_soft_poll_always_and_condition(w):
    cube, bin_ = w.scene["cube"], w.scene["bin"]
    expect(cube).not_.to_be_left_of(bin_, by=0.5)
    expect.soft(cube).to_be_left_of(bin_, by=0.5, timeout=0.05)  # recorded, and the test goes on
    assert len(w._soft_failures) == 1
    w._soft_failures.clear()
    speeds = iter(np.linspace(1.0, 0.0, 200))
    expect.poll(lambda: next(speeds), name="speed").to_be_slower_than(0.5)
    expect(cube).always.to_be_left_of(bin_, by=0.1)
    w.wait(0.1)  # the invariant holds while the world runs
    assert condition(cube, "to_be_left_of", bin_)()
    assert not condition(cube, "to_be_left_of", bin_, by=0.5)()


def test_a_custom_matcher_cannot_replace_a_built_in_one_or_be_misnamed():
    with pytest.raises(ValueError, match="built-in"):
        expect.extend(to_be_inside=lambda obj, other: True)
    with pytest.raises(ValueError, match="starts with to_"):
        expect.extend(left_of=to_be_left_of)
    expect.extend(to_be_left_of=to_be_left_of)  # registering one of yours again is fine (a re-imported conftest)
