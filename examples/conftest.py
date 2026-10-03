import pytest


@pytest.fixture(autouse=True)
def _arms_only(rw_robot):
    """These examples pick and place; legged robots have their own (tests/test_legged.py)."""
    from robowright import robots

    if robots.get(rw_robot).family != "arm":
        pytest.skip("pick-and-place examples need an arm")
