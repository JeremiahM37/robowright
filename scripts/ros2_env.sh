#!/usr/bin/env bash
# A ROS 2 Jazzy environment for robowright's ROS 2 tests, without root or a ROS install:
# RoboStack's conda packages (rclpy, ros2_control, ros2_controllers, tf2_ros, Gazebo with
# gz_ros2_control, MoveIt, and Universal Robots' Gazebo simulation and MoveIt configuration),
# with robowright installed into it in development mode.
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
