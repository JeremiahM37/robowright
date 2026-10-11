import pytest


@pytest.fixture(autouse=True)
def _arms_only():
    """Overrides examples/conftest.py: this simulation is not one of robowright's robots."""
