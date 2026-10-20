"""Which OpenGL robowright renders MuJoCo with, chosen when it is imported."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the choice is Linux's")


def _mujoco_gl(**env) -> str:
    base = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "MUJOCO_GL")}
    code = "import os, robowright; print(os.environ.get('MUJOCO_GL'))"
    out = subprocess.run([sys.executable, "-c", code], env={**base, **env}, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_no_display_renders_with_egl():
    assert _mujoco_gl() == "egl"


def test_a_display_without_glx_renders_with_egl():
    # MuJoCo would pick GLFW for any DISPLAY and fail with "GLX extension not found"
    assert _mujoco_gl(DISPLAY=":4711") == "egl"


def test_an_explicit_choice_is_kept():
    assert _mujoco_gl(DISPLAY=":4711", MUJOCO_GL="osmesa") == "osmesa"


def test_offscreen_gl_is_one_whose_library_loads():
    """EGL named on a machine without libEGL made `import robowright` itself fail (a bare CI runner)."""
    from robowright import _offscreen_gl

    libs = {"EGL": "libEGL.so.1", "OpenGL": "libOpenGL.so.0", "OSMesa": "libOSMesa.so.8"}
    assert _offscreen_gl(lambda n: libs.get(n)) == "egl"
    assert _offscreen_gl(lambda n: libs.get(n) if n == "EGL" else None) is None  # EGL alone: PyOpenGL cannot use it
    assert _offscreen_gl(lambda n: libs.get(n) if n == "OSMesa" else None) == "osmesa"
    assert _offscreen_gl(lambda n: None) is None
