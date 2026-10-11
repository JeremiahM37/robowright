import pytest


@pytest.fixture(autouse=True)
def _arms_only():
    """Overrides examples/conftest.py: these are other people's simulations, not robowright's robots."""
