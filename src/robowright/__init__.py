"""robowright: Playwright-style testing for robots.

import robowright as rw
from robowright import expect

with rw.launch() as world:
    world.robot.pick(world.scene["cube"])
    world.robot.place(on=world.scene["bin"])
    expect(world.scene["cube"]).to_be_inside(world.scene["bin"])
"""

import contextvars
import os
import sys

# Headless Linux (CI, servers) has no display for GLFW; render offscreen via EGL.
# Must happen before anything imports mujoco.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

__version__ = "0.1.0.dev0"

_current: contextvars.ContextVar = contextvars.ContextVar("robowright_world", default=None)

from .errors import (  # noqa: E402
    ActionTimeoutError,
    CapabilityError,
    ExpectationError,
    InvariantViolation,
    RobowrightError,
    UnreachableError,
)
from .expect import condition, expect  # noqa: E402
from .locators import ObjectHandle, Point  # noqa: E402
from .scene import CameraSpec, ObjectSpec, SceneSpec, default_scene, open_floor, tabletop  # noqa: E402
from .trace import Trace  # noqa: E402
from .world import Settings, World  # noqa: E402
from .world import launch as _launch  # noqa: E402


def launch(*args, **kwargs) -> World:
    world = _launch(*args, **kwargs)
    _current.set(world)
    return world


launch.__doc__ = _launch.__doc__

__all__ = [
    "ActionTimeoutError",
    "CameraSpec",
    "CapabilityError",
    "ExpectationError",
    "InvariantViolation",
    "ObjectHandle",
    "ObjectSpec",
    "Point",
    "RobowrightError",
    "SceneSpec",
    "Settings",
    "Trace",
    "UnreachableError",
    "World",
    "condition",
    "expect",
    "launch",
    "tabletop",
    "open_floor",
    "default_scene",
]
