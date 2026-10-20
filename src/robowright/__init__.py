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


def _has_glx(display: str) -> bool:
    """Whether the X server at ``display`` offers GLX (a virtual one, such as Xvfb, may not)."""
    import ctypes
    import ctypes.util

    try:
        x11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
    except OSError:
        return False
    x11.XOpenDisplay.argtypes, x11.XOpenDisplay.restype = [ctypes.c_char_p], ctypes.c_void_p
    x11.XQueryExtension.argtypes = [ctypes.c_void_p, ctypes.c_char_p, *[ctypes.POINTER(ctypes.c_int)] * 3]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    handle = x11.XOpenDisplay(display.encode())
    if not handle:
        return False
    codes = [ctypes.c_int() for _ in range(3)]
    try:
        return bool(x11.XQueryExtension(handle, b"GLX", *map(ctypes.byref, codes)))
    finally:
        x11.XCloseDisplay(handle)


def _offscreen_gl(find=None) -> str | None:
    """The offscreen OpenGL that loads here: EGL, else OSMesa (CPU only), else None. MuJoCo loads the one named in
    MUJOCO_GL when it is imported, so naming one whose library is missing (EGL on a bare CI runner) made
    ``import robowright`` itself fail; with neither, nothing is named, and only rendering is unavailable."""
    import ctypes.util

    find = find or ctypes.util.find_library
    if find("EGL"):
        return "egl"
    if find("OSMesa"):
        return "osmesa"
    return None


# Render offscreen on headless Linux (CI, servers), and where the display cannot do OpenGL through GLX: MuJoCo picks
# GLFW whenever DISPLAY is set, which then fails with "GLX extension not found". Must happen before anything imports
# mujoco.
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    _display = os.environ.get("DISPLAY")
    if not _display or not _has_glx(_display):
        _gl = _offscreen_gl()
        if _gl:
            os.environ["MUJOCO_GL"] = _gl
            if _gl == "osmesa":
                os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

__version__ = "0.1.0"

_current: contextvars.ContextVar = contextvars.ContextVar("robowright_world", default=None)

from .attached import attach  # noqa: E402
from .attached_remote import attachable  # noqa: E402
from .errors import (  # noqa: E402
    ActionError,
    ActionTimeoutError,
    CapabilityError,
    ExpectationError,
    InvariantViolation,
    RobowrightError,
    SafetyLimitError,
    TimeLimitError,
    UnreachableError,
)
from .expect import condition, expect  # noqa: E402
from .http_stub import http_stub  # noqa: E402
from .locators import ObjectHandle, Point  # noqa: E402
from .routes import route  # noqa: E402
from .scene import CameraSpec, ObjectSpec, SceneSpec, default_scene, open_floor, tabletop  # noqa: E402
from .stack import Ros2Stack, ros2_stack  # noqa: E402
from .steps import step  # noqa: E402
from .trace import Trace  # noqa: E402
from .world import (  # noqa: E402
    Settings,
    World,
    ros2_robot,  # noqa: E402
)
from .world import launch as _launch  # noqa: E402


def launch(*args, **kwargs) -> World:
    world = _launch(*args, **kwargs)
    _current.set(world)
    return world


launch.__doc__ = _launch.__doc__

__all__ = [
    "ActionError",
    "ActionTimeoutError",
    "CameraSpec",
    "CapabilityError",
    "ExpectationError",
    "GraspError",
    "InvariantViolation",
    "ObjectHandle",
    "ObjectSpec",
    "Point",
    "RobowrightError",
    "SafetyLimitError",
    "SceneSpec",
    "TimeLimitError",
    "Settings",
    "Trace",
    "UnreachableError",
    "World",
    "attach",
    "attachable",
    "condition",
    "expect",
    "route",
    "http_stub",
    "ros2_stack",
    "Ros2Stack",
    "step",
    "launch",
    "ros2_robot",
    "tabletop",
    "open_floor",
    "default_scene",
]
