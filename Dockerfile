# robowright, ready to run robot tests headless: MuJoCo and PyBullet, rendering on the CPU through Mesa's EGL (no GPU,
# no display), and video export. Mount your project at /work:
#
#   docker build -t robowright .
#   docker run --rm -v "$PWD":/work robowright                       # pytest
#   docker run --rm -v "$PWD":/work robowright robowright info
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libosmesa6 git \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md LICENSE NOTICE /opt/robowright/
COPY src /opt/robowright/src
RUN pip install --no-cache-dir "/opt/robowright[pybullet,video]" pytest-xdist

WORKDIR /work
CMD ["pytest"]
