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
#   scripts/ros2_env.sh kortex                  # Kinova's ros2_kortex and Robotiq's descriptions, for the Kinova Gazebo rig
#   scripts/ros2_env.sh nav2                    # a second environment with Nav2 and its TurtleBot simulations
#   scripts/ros2_env.sh franka                  # Franka's franka_ros2 and franka_description, built, for the Franka Gazebo tests
#   scripts/ros2_env.sh pytest tests/test_ros2.py
#
# The environment lives in $ROBOWRIGHT_ROS2_ENV (default ~/.cache/robowright/ros2-jazzy).
# With ROS 2 already installed, source its setup.bash and pip install -e . instead.
set -euo pipefail
ENV="${ROBOWRIGHT_ROS2_ENV:-$HOME/.cache/robowright/ros2-jazzy}"
TOOLS="$(dirname "$ENV")/micromamba"
MM="${ROBOWRIGHT_MICROMAMBA:-$TOOLS/bin/micromamba}"
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
        ros-jazzy-moveit ros-jazzy-ur-simulation-gz ros-jazzy-ur-moveit-config ros-jazzy-ur-description ros-jazzy-ur-robot-driver \
        ros-jazzy-moveit-resources-panda-moveit-config ros-jazzy-moveit-resources-panda-description \
        ros-jazzy-kortex-description ros-jazzy-robotiq-description ros-jazzy-gripper-controllers \
        ros-jazzy-picknik-twist-controller ros-jazzy-picknik-reset-fault-controller uv
    "$ENV/bin/uv" pip install --python "$ENV/bin/python" -e "$REPO[dev,onnx]"
    "$0" pymoveit2
    "$0" kortex
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
if [ "${1:-}" = "nav2" ]; then
    # Nav2 and its TurtleBot simulations, in an environment of their own ($ENV-nav2): RoboStack's
    # Nav2 builds do not install beside the MoveIt and UR packages above. Run its tests with
    #   ROBOWRIGHT_ROS2_ENV="$ENV-nav2" scripts/ros2_env.sh pytest tests/test_nav2_examples.py tests/test_nav2_turtlebot.py
    [ -x "$MM" ] || { echo "run $0 create first (it fetches micromamba)" >&2; exit 1; }
    MAMBA_ROOT_PREFIX="$TOOLS/root" "$MM" create -y -p "$ENV-nav2" -c conda-forge -c robostack-jazzy \
        ros-jazzy-ros-base ros-jazzy-nav2-bringup ros-jazzy-navigation2 ros-jazzy-nav2-simple-commander \
        ros-jazzy-turtlebot3-gazebo ros-jazzy-turtlebot3-navigation2 ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
        ros-jazzy-tf2-ros-py ros-jazzy-robot-state-publisher uv
    "$ENV-nav2/bin/uv" pip install --python "$ENV-nav2/bin/python" -e "$REPO[dev]"
    exit 0
fi
if [ "${1:-}" = "xarm" ]; then
    # UFACTORY's xarm_ros2 (jazzy) at a pinned commit, for the xArm Gazebo rig (tests/xarm_gz_rig.py):
    # its launch files, descriptions and controller configs are used as shipped; the rig makes them
    # visible to ROS 2 through an ament index of its own. The one library built here is xarm_gazebo's
    # mimic-joint plugin, which the gripper's joints need, into $XARM_WS/lib (not the env).
    XARM_WS="${ROBOWRIGHT_XARM_WS:-/mnt/bulk/rw-ext/xarm-ws}"
    rm -rf "$ENV/src/xarm_ros2"
    git init -q "$ENV/src/xarm_ros2"
    git -C "$ENV/src/xarm_ros2" fetch -q --depth 1 https://github.com/xArm-Developer/xarm_ros2.git 3dc2b5e8294758d96b54b15fa5920d581b7cbb3d
    git -C "$ENV/src/xarm_ros2" checkout -q FETCH_HEAD
    V="$ENV/src/xarm_ros2"
    mkdir -p "$XARM_WS/lib"
    "$MM" run -p "$ENV" bash -c "x86_64-conda-linux-gnu-g++ -shared -fPIC -std=c++17 -O2 -I$V/xarm_gazebo/include \$(pkg-config --cflags gz-sim8 sdformat14) $V/xarm_gazebo/src/mimic_joint_plugin.cpp \$(pkg-config --libs gz-sim8 sdformat14) -o $XARM_WS/lib/libgazebo_mimic_joint_plugin.so -w" \
        || { echo "mimic plugin build failed" >&2; exit 1; }
    exit 0
fi
if [ "${1:-}" = "kortex" ]; then
    # Kinova's ros2_kortex, for kortex_bringup's simulation launch (tests/kinova_gz_rig.py), and
    # the Robotiq 2F-85 description its main branch expects: RoboStack's robotiq_description is
    # older and has no mock_sensor_commands argument, so the description will not expand with
    # it. Only the launch files and the descriptions are used here (nothing is built); the rig
    # makes them visible to ROS 2 through an ament index of its own.
    fetch() {  # fetch URL COMMIT DIR
        rm -rf "$3"
        git init -q "$3"
        git -C "$3" fetch -q --depth 1 "$1" "$2"
        git -C "$3" checkout -q FETCH_HEAD
    }
    fetch https://github.com/Kinovarobotics/ros2_kortex.git b2aa46b29ae8fadd5b293c57e3500925c61eb8e8 "$ENV/src/ros2_kortex"
    fetch https://github.com/picknikrobotics/ros2_robotiq_gripper.git a74d007d8f2f06dc6a503ad21038ba869d4999a6 "$ENV/src/ros2_robotiq_gripper"
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
if [ "${1:-}" = "franka" ]; then
    # Franka Robotics' own stack for tests/test_franka_gazebo.py: franka_ros2 (jazzy, 4acc05c) and
    # franka_description (2.9.0, 7aeeddc), built with colcon into $FRANKA_WS, in an environment of their
    # own ($FRANKA_ENV, default $ENV-franka). libfranka is built from its 0.20.5 source, which is Franka's
    # pin, against Pinocchio 4.1.0 and Eigen 5.0.1 from conda-forge: those two are conda-forge's current
    # versions, not Franka's pins (libfranka 0.20.5 declares no version for them here). The 0.20.5 package
    # on conda-forge conflicts with the Gazebo vendor's Poco, so it is built instead. Nothing is installed
    # into $ENV. On this machine: ROBOWRIGHT_FRANKA_ENV=/mnt/bulk/ros2-env/jazzy-franka
    # ROBOWRIGHT_FRANKA_WS=/mnt/bulk/rw-ext/franka-ws, and the tests run with ROBOWRIGHT_ROS2_ENV set to the same env.
    FRANKA_ENV="${ROBOWRIGHT_FRANKA_ENV:-$ENV-franka}"
    FRANKA_WS="${ROBOWRIGHT_FRANKA_WS:-$FRANKA_ENV-ws}"  # tests/franka_gz_rig.py finds it at the same place
    [ -x "$MM" ] || { echo "run $0 create first (it fetches micromamba)" >&2; exit 1; }
    if [ ! -x "$FRANKA_ENV/bin/python" ]; then
        MAMBA_ROOT_PREFIX="$TOOLS/root" "$MM" create -y -p "$FRANKA_ENV" -c conda-forge -c robostack-jazzy \
            python=3.12 ros-jazzy-ros-base ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
            ros-jazzy-controller-manager ros-jazzy-gz-ros2-control ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
            ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher-gui ros-jazzy-sdformat-urdf \
            ros-jazzy-realsense2-description ros-jazzy-ros2controlcli ros-jazzy-xacro ros-jazzy-realtime-tools \
            ros-jazzy-hardware-interface ros-jazzy-ament-cmake-gmock ros-jazzy-rclcpp-action \
            ros-jazzy-rclcpp-components ros-jazzy-pluginlib ros-jazzy-tf2-ros-py ros-jazzy-control-msgs \
            ros-jazzy-trajectory-msgs ros-jazzy-test-msgs ros-jazzy-moveit-core \
            ros-jazzy-ament-cmake-clang-format ros-jazzy-ament-cmake-copyright ros-jazzy-ament-cmake-cppcheck \
            ros-jazzy-ament-cmake-flake8 ros-jazzy-ament-cmake-lint-cmake ros-jazzy-ament-cmake-pep257 \
            ros-jazzy-ament-cmake-xmllint ros-jazzy-ament-lint-auto ros-jazzy-ament-cmake-clang-tidy \
            pinocchio gxx_linux-64 make cmake pkg-config uv
    fi
    "$FRANKA_ENV/bin/uv" pip install --python "$FRANKA_ENV/bin/python" -q colcon-core colcon-common-extensions
    "$FRANKA_ENV/bin/uv" pip install --python "$FRANKA_ENV/bin/python" -q -e "$REPO[dev]"
    mkdir -p "$FRANKA_WS/src"
    fetch() {  # fetch URL COMMIT DIR (as the kortex block)
        rm -rf "$3"
        git init -q "$3"
        git -C "$3" fetch -q --depth 1 "$1" "$2"
        git -C "$3" checkout -q FETCH_HEAD
    }
    if [ ! -f "$FRANKA_WS/libfranka-install/lib/cmake/Franka/FrankaConfig.cmake" ] && [ ! -f "$FRANKA_WS/libfranka-install/lib/cmake/Franka/FrankaTargets.cmake" ]; then
        rm -rf "$FRANKA_WS/libfranka-src" "$FRANKA_WS/libfranka-build"
        git -c advice.detachedHead=false clone -q --depth 1 --branch 0.20.5 --recurse-submodules --shallow-submodules \
            https://github.com/frankarobotics/libfranka.git "$FRANKA_WS/libfranka-src"
        "$MM" run -p "$FRANKA_ENV" cmake -S "$FRANKA_WS/libfranka-src" -B "$FRANKA_WS/libfranka-build" \
            -DEigen3_DIR="$FRANKA_ENV/share/eigen3/cmake" -DEIGEN3_INCLUDE_DIRS="$FRANKA_ENV/include/eigen3" \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$FRANKA_WS/libfranka-install" \
            -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCUMENTATION=OFF \
            -DCMAKE_C_COMPILER="$FRANKA_ENV/bin/x86_64-conda-linux-gnu-cc" \
            -DCMAKE_CXX_COMPILER="$FRANKA_ENV/bin/x86_64-conda-linux-gnu-c++" > "$FRANKA_WS/libfranka-cmake.log" 2>&1
        "$MM" run -p "$FRANKA_ENV" cmake --build "$FRANKA_WS/libfranka-build" -j"$(nproc)" >> "$FRANKA_WS/libfranka-cmake.log" 2>&1
        "$MM" run -p "$FRANKA_ENV" cmake --install "$FRANKA_WS/libfranka-build" >> "$FRANKA_WS/libfranka-cmake.log" 2>&1
    fi
    fetch https://github.com/frankarobotics/franka_ros2.git 4acc05c01ebfcdf916ee6b3df966df0ad7b2f9ec "$FRANKA_WS/src/franka_ros2"
    fetch https://github.com/frankarobotics/franka_description.git 7aeeddc449edf8d62b594f9e36a81da53e7796f9 "$FRANKA_WS/src/franka_description"
    # conda-forge's pinocchio exports sdformat::requested in its link interface, but no target by that name
    # exists in the env's sdformat configs, so every consumer fails with "-lsdformat::requested" at link
    # time. Only the name is dropped: pinocchio's own libraries are unchanged. This edits a file in
    # $FRANKA_ENV, which is ours; it is not Franka's code, and the original is kept beside it as .orig.
    P="$FRANKA_ENV/lib/cmake/pinocchio/pinocchioTargets.cmake"
    if grep -q ";sdformat::requested" "$P"; then
        cp -n "$P" "$P.orig"
        sed -i 's/;sdformat::requested//' "$P"
    fi
    cd "$FRANKA_WS" && CMAKE_PREFIX_PATH="$FRANKA_WS/libfranka-install" "$MM" run -p "$FRANKA_ENV" \
        colcon build --base-paths src --symlink-install --packages-up-to franka_gazebo_bringup \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        "-DCMAKE_PREFIX_PATH=$FRANKA_WS/libfranka-install;$FRANKA_ENV" > "$FRANKA_WS/colcon.log" 2>&1 \
        || { grep -n "Failed\|Aborted" "$FRANKA_WS/colcon.log" >&2; exit 1; }
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
