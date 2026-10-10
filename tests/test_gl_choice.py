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
