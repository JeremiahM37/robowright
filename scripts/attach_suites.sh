#!/bin/sh
# The attach tests and examples against other people's environments, each suite in the environment it is
# installed in (they pin different MuJoCo and gymnasium versions, so no one environment holds them all).
# Usage: scripts/attach_suites.sh [DIR]   (DIR holds the environments; default /mnt/bulk/rw-ext)
# A suite whose environment is missing is reported and counted as a failure: nothing is skipped silently.
E="${1:-/mnt/bulk/rw-ext}"
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/src"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$E/vk/share/vulkan/icd.d/lvp_icd.x86_64.json}"  # ManiSkill needs a Vulkan driver
O="-q -p no:cacheprovider"
bad=0
run() {  # run ENV TESTS...
    py="$E/$1/bin/python"; shift
    echo "== $py: $*"
    if [ ! -x "$py" ]; then echo "missing environment"; bad=1; return; fi
    # robowright's pytest plugin is loaded by name where the environment has no robowright installed of its own
    plugin="-p robowright.pytest_plugin"
    "$py" -c "import importlib.metadata as m; m.distribution('robowright')" 2>/dev/null && plugin=""
    "$py" -m pytest $O $plugin "$@" || bad=1
}
run envs-venv tests/test_attach.py tests/test_attach_gymnasium_mujoco.py tests/test_attach_metaworld_all.py tests/test_attach_robosuite_all.py tests/test_attach_envs.py tests/test_attach_vector.py tests/test_attach_sb3.py tests/test_attach_replay.py tests/test_attach_factory.py \
    examples/environments/test_gymnasium_robotics.py examples/environments/test_metaworld.py examples/environments/test_robosuite.py
run envs-dm-venv examples/environments/test_dm_control.py tests/test_attach_dm_control_suite.py
run venv-pybullet/.venv tests/test_attach_replay_engines.py examples/environments/test_pybullet_envs.py
run venv-drones/.venv examples/environments/test_pybullet_drones.py
run venv-maniskill tests/test_attach_replay_engines.py tests/test_attach_sapien.py tests/test_attach_maniskill_all.py examples/environments/test_maniskill.py
run agent-venv tests/test_mcp_attach.py
run lerobot-venv tests/test_lerobot_backend.py  # LeRobot's own SOFollower (lerobot[feetech]), its serial bus stood in for
exit $bad
