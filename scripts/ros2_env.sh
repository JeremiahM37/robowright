#!/usr/bin/env bash
# A ROS 2 Jazzy environment for robowright's ROS 2 tests, without root or a ROS install:
# RoboStack's conda packages (rclpy, ros2_control, ros2_controllers, tf2_ros, Gazebo with
# gz_ros2_control, MoveIt, and Universal Robots' Gazebo simulation and MoveIt configuration),
# with robowright installed into it in development mode, and pymoveit2 (a MoveIt client
# library the UR tests run the examples of) at a pinned commit with one fix applied
# (scripts/patches: it could not read a robot description that move_group gets on topics),
# and ros2_control's controller manager built with one fix (scripts/patches: in simulation it
# timed controllers by the /clock messages it had received, not by the simulator's step, so a
# late /clock stopped a trajectory and then aborted it). The UR rig preloads that build.
#
#   scripts/ros2_env.sh create                  # once (about 2 GB)
#   scripts/ros2_env.sh pytest tests/test_ros2.py
#
# The environment lives in $ROBOWRIGHT_ROS2_ENV (default ~/.cache/robowright/ros2-jazzy).
# With ROS 2 already installed, source its setup.bash and pip install -e . instead.
set -euo pipefail
ENV="${ROBOWRIGHT_ROS2_ENV:-$HOME/.cache/robowright/ros2-jazzy}"
TOOLS="$(dirname "$ENV")/micromamba"
MM="$TOOLS/bin/micromamba"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if [ "${1:-}" = "create" ]; then
    if [ ! -x "$MM" ]; then
        mkdir -p "$TOOLS"
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$TOOLS" bin/micromamba
    fi
    MAMBA_ROOT_PREFIX="$TOOLS/root" "$MM" create -y -p "$ENV" -c conda-forge -c robostack-jazzy \
        python=3.12 ros-jazzy-ros-base ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
        ros-jazzy-controller-manager ros-jazzy-tf2-ros-py ros-jazzy-control-msgs \
        ros-jazzy-robot-state-publisher ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge ros-jazzy-gz-ros2-control \
        ros-jazzy-moveit ros-jazzy-ur-simulation-gz ros-jazzy-ur-moveit-config ros-jazzy-ur-description ros-jazzy-ur-robot-driver uv
    "$ENV/bin/uv" pip install --python "$ENV/bin/python" -e "$REPO[dev,onnx]"
    "$0" pymoveit2
    "$0" ros2_control
    exit 0
fi
if [ "${1:-}" = "pymoveit2" ]; then
    PYMOVEIT2_COMMIT=4c83fb30f1004aa26bb75483637fbc7e917e508b
    rm -rf "$ENV/src/pymoveit2"
    git clone -q https://github.com/AndrejOrsula/pymoveit2.git "$ENV/src/pymoveit2"
    git -C "$ENV/src/pymoveit2" checkout -q "$PYMOVEIT2_COMMIT"
    git -C "$ENV/src/pymoveit2" apply "$REPO"/scripts/patches/pymoveit2-*.patch
    exit 0
fi
if [ "${1:-}" = "ros2_control" ]; then
    # The installed version, patched, as $ENV/overlay/lib/libcontroller_manager.so
    version="$(ls "$ENV/conda-meta" | sed -n 's/^ros-jazzy-controller-manager-\([0-9.]*\)-.*/\1/p' | head -1)"
    [ -n "$version" ] || { echo "ros-jazzy-controller-manager is not installed in $ENV" >&2; exit 1; }
    MAMBA_ROOT_PREFIX="$TOOLS/root" "$MM" install -y -q -p "$ENV" -c conda-forge -c robostack-jazzy --freeze-installed gxx_linux-64 make
    rm -rf "$ENV/src/ros2_control" "$ENV/overlay/build"
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$version" https://github.com/ros-controls/ros2_control.git "$ENV/src/ros2_control"
    git -C "$ENV/src/ros2_control" apply "$REPO"/scripts/patches/ros2_control-*.patch
    mkdir -p "$ENV/overlay/build" "$ENV/overlay/lib"
    "$MM" run -p "$ENV" cmake -S "$ENV/src/ros2_control/controller_manager" -B "$ENV/overlay/build" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF -DCMAKE_PREFIX_PATH="$ENV" > "$ENV/overlay/build.log" 2>&1
    "$MM" run -p "$ENV" make -C "$ENV/overlay/build" -j"$(nproc)" controller_manager >> "$ENV/overlay/build.log" 2>&1 \
        || { tail -20 "$ENV/overlay/build.log" >&2; exit 1; }
    cp "$ENV/overlay/build/libcontroller_manager.so" "$ENV/overlay/lib/"
    rm -rf "$ENV/overlay/build"
    exit 0
fi
[ -x "$ENV/bin/python" ] || { echo "no ROS 2 environment at $ENV: run $0 create" >&2; exit 1; }
cmd="${1:-python}"; shift || true
# The environment's activation scripts set AMENT_PREFIX_PATH, which ros2_control's plugin
# loader needs to find controllers and mock hardware.
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
if [ "$cmd" = "pytest" ]; then
    # ROS 2's launch_testing pytest plugins do not load under pytest 9.
    exec "$MM" run -p "$ENV" python -m pytest -p no:launch_testing -p no:launch_ros "$@"
fi
exec "$MM" run -p "$ENV" "$cmd" "$@"
