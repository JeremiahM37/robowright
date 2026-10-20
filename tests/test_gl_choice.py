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


def test_offscreen_gl_is_one_that_loads():
    """EGL named where PyOpenGL cannot load it made `import robowright` itself fail (GitHub's runner, which has libEGL
    and libGL): a library is named only if it is installed and loads."""
    from robowright import _offscreen_gl

    libs = {"EGL": "libEGL.so.1", "OSMesa": "libOSMesa.so.8"}
    assert _offscreen_gl(lambda n: libs.get(n), lambda p: True) == "egl"
    assert _offscreen_gl(lambda n: libs.get(n), lambda p: p == "osmesa") == "osmesa"  # EGL installed but broken
    assert _offscreen_gl(lambda n: libs.get(n), lambda p: False) is None
    assert _offscreen_gl(lambda n: None, lambda p: True) is None  # nothing installed: nothing tried


def test_the_probe_matches_this_machine():
    """Here EGL renders (the test suite renders through it), so the probe must say it loads."""
    from robowright import _loads

    assert _loads("egl")
