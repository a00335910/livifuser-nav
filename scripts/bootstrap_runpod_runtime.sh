#!/usr/bin/env bash
# Bootstrap the prospectively frozen RunPod RTX 3090 development runtime.
set -euo pipefail

# Two honest answers are accepted, and which one was given is recorded.
#   YES         - a human confirmed volume encryption in the RunPod console.
#   UNCONFIRMED - the console exposes no encryption setting for this volume
#                 type, so the property could not be verified. RunPod pod
#                 volumes have no such control; only network volumes may.
# There is deliberately no way to claim YES without meaning it: recording an
# unverified property as verified is the failure mode this gate exists to stop.
case "${LIVIFUSER_ENCRYPTED_VOLUME_ACK:-}" in
  YES)
    VOLUME_ENCRYPTION_STATUS="confirmed_by_operator"
    ;;
  UNCONFIRMED)
    VOLUME_ENCRYPTION_STATUS="not_independently_confirmed"
    echo "NOTE: volume encryption is NOT confirmed; proceeding under a recorded deviation." >&2
    ;;
  *)
    echo "Set LIVIFUSER_ENCRYPTED_VOLUME_ACK to YES (confirmed in the RunPod" >&2
    echo "console) or UNCONFIRMED (no such setting is exposed for this volume)." >&2
    exit 2
    ;;
esac
export VOLUME_ENCRYPTION_STATUS
if [[ ! -d /workspace ]]; then
  echo "/workspace is missing; attach the persistent RunPod volume first." >&2
  exit 2
fi
if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run this bootstrap as root inside the RunPod container." >&2
  exit 2
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "Frozen runtime requires Ubuntu 22.04 (Jammy); observed ${ID:-unknown} ${VERSION_ID:-unknown}." >&2
  echo "This is not a version preference. ROS 2 Humble has no 24.04 binaries, and" >&2
  echo "Gazebo Fortress is the Humble pairing. Substituting Jazzy and Gazebo" >&2
  echo "Harmonic would change the simulator that produced the confirmatory" >&2
  echo "dataset, which is a scientific input, not an infrastructure detail." >&2
  echo "Select a RunPod image whose tag contains ubuntu2204." >&2
  exit 2
fi

# ROS 2 Humble's rclpy and rosbag2_py binaries are built against the Jammy
# system interpreter. A container that redirects python3 elsewhere imports
# nothing from ROS, and the failure looks like a missing package rather than a
# version mismatch.
SYSTEM_PYTHON_VERSION="$(/usr/bin/python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$SYSTEM_PYTHON_VERSION" != "3.10" ]]; then
  echo "ROS 2 Humble requires the Jammy system interpreter 3.10 at /usr/bin/python3;" >&2
  echo "observed $SYSTEM_PYTHON_VERSION." >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg lsb-release locales python3-venv software-properties-common
locale-gen en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /etc/apt/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
  > /etc/apt/sources.list.d/ros2.list
apt-get update
apt-get install -y --no-install-recommends \
  ros-humble-ros-base \
  ros-humble-ros-gz \
  ros-humble-rosbag2-storage-mcap \
  ros-humble-rosbag2-compression-zstd \
  ros-humble-tf2-ros \
  python3-colcon-common-extensions \
  python3-rosdep

# Gazebo Fortress renders the camera through the ogre1 path. In a headless
# container that means software rasterization, which needs Mesa's DRI drivers
# and the GLX/EGL loaders. Those are Recommends of the GL packages, so the
# --no-install-recommends above silently drops them and --headless-rendering
# then fails at the first camera frame. C0 and C1 are camera conditions, so
# this is fatal rather than cosmetic.
apt-get install -y --no-install-recommends \
  libgl1-mesa-dri \
  libgl1-mesa-glx \
  libglapi-mesa \
  libegl1-mesa \
  libglu1-mesa \
  libxrandr2 \
  libxi6 \
  libxcursor1 \
  libxinerama1 \
  xvfb \
  gcc \
  libc6-dev

python3 - <<'RENDERCHECK'
import ctypes.util
import sys

# Fail here, cheaply, rather than forty minutes into a rollout batch.
missing = [name for name in ("GL", "EGL") if ctypes.util.find_library(name) is None]
if missing:
    sys.exit(f"missing OpenGL loaders after install: {missing}")
print("OpenGL loaders present: GL, EGL")
RENDERCHECK

install -d -m 0755 /workspace/livifuser/{input,repo,evidence,runtime,tmp}

# Build the CPU-report shim and apply it to everything that follows.
# /sys reports the host's 256 CPUs while the cgroup allows 27, and every
# program that auto-sizes from that number misbehaves: OGRE builds a 256
# thread render pool and fails to load, colcon spawns 256 compilers and is
# OOM-killed. Correcting the reported value once covers all of them.
SHIM_SOURCE=/workspace/livifuser/runtime/limit_cpu_report.c
SHIM_LIBRARY=/workspace/livifuser/runtime/limit_cpu_report.so
cat > "$SHIM_SOURCE" <<'SHIMSOURCE'
/* Report a modest CPU count to callers of sysconf().
 *
 * OGRE sizes its render work queue from the reported processor count. In this
 * container /sys reports the host's 256 CPUs while the cgroup allows 27, so
 * OGRE builds a thread pool for a machine that is not there, fails to load the
 * render engine, and Gazebo retries until the process dies. Everything else,
 * including the policy and the physics step, is unaffected by this value.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <unistd.h>

#ifndef LIVIFUSER_REPORTED_CPUS
#define LIVIFUSER_REPORTED_CPUS 8
#endif

static long (*real_sysconf)(int) = 0;

long sysconf(int name) {
  if (!real_sysconf) {
    real_sysconf = dlsym(RTLD_NEXT, "sysconf");
  }
  if (name == _SC_NPROCESSORS_ONLN || name == _SC_NPROCESSORS_CONF) {
    return LIVIFUSER_REPORTED_CPUS;
  }
  return real_sysconf(name);
}
SHIMSOURCE
gcc -shared -fPIC -o "$SHIM_LIBRARY" "$SHIM_SOURCE" -ldl
reported=$(LD_PRELOAD="$SHIM_LIBRARY" getconf _NPROCESSORS_ONLN)
if [[ "$reported" != "8" ]]; then
  echo "CPU-report shim is not effective: getconf reports $reported" >&2
  exit 2
fi
echo "CPU-report shim built; processes see $reported CPUs instead of $(getconf _NPROCESSORS_ONLN)"
export LD_PRELOAD="$SHIM_LIBRARY${LD_PRELOAD:+:${LD_PRELOAD}}"

python3 -m venv --system-site-packages /workspace/livifuser/runtime/venv
source /workspace/livifuser/runtime/venv/bin/activate
python -m pip install --upgrade "pip==25.2"

# Ubuntu 22.04 ships packaging 21.3, but the setuptools present here (80.9.0
# system-wide, 78.1.0 in the venv after the pinned torch install) calls
# canonicalize_version(strip_trailing_zero=...), which 21.3 rejects. Without
# this, every ament_python and ament_cmake_python package fails to build with a
# TypeError that names setuptools and never mentions packaging. The venv is
# created with --system-site-packages, so the old copy shadows a newer one
# unless each interpreter gets its own.
python -m pip install "packaging==24.2"
/usr/bin/python3 -m pip install "packaging==24.2"
python -m pip install \
  "huggingface-hub==0.34.4" \
  "numpy==2.0.2" \
  "pillow==11.3.0" \
  "safetensors==0.6.2" \
  "scipy==1.15.3" \
  "transformers==4.56.0"

# Some Ubuntu 22.04 images ship no torch, or ship a version other than the one
# the runtime contract pins. Install the exact pinned build from the CUDA 12.8
# index rather than accepting whatever is present; the parity gate compares CPU
# against CUDA inside one build, so the build must be deterministic.
if ! python -c 'import torch' 2>/dev/null || \
   [[ "$(python -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null)" != "2.10.0" ]]; then
  echo "Installing the pinned torch 2.10.0+cu128 build."
  python -m pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.10.0"
fi

python - <<'PY'
import sys
import torch
expected = (2, 10, 0)
observed = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:3])
if observed != expected:
    raise SystemExit(
        f"PyTorch version drift: expected 2.10.0, observed {torch.__version__}. "
        "The bootstrap installs the pinned build; a drift here means the install "
        "was overridden, not that the image was wrong."
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
if "RTX 3090" not in name:
    raise SystemExit(f"GPU identity drift: expected RTX 3090, observed {name}")
if torch.cuda.get_device_properties(0).total_memory < 23 * 1024**3:
    raise SystemExit("GPU VRAM is below the frozen 24-GiB class")
print(sys.version)
print(torch.__version__, torch.version.cuda, name)
PY

# The verified input handoff must already be unpacked into the repository root
# below before the workspace can be built. Building is a prerequisite for every
# readiness gate and launches nothing.
REPO=/workspace/livifuser/repo
if [[ -d "$REPO/ros2_ws/src" ]]; then
  echo "Building the ROS workspace from the verified handoff."
  # ROS's setup.bash dereferences AMENT_TRACE_SETUP_FILES and friends without
  # defaulting them, so `set -u` aborts on the first source. Relax nounset only
  # around the sourcing, then restore it: the rest of this script depends on it.
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  # The CPU-report shim above already caps what colcon and make see. This
  # explicit bound is kept as a second line of defence: an OOM-killed link
  # leaves a half-built workspace that fails much later and less obviously.
  BUILD_JOBS=8
  (cd "$REPO/ros2_ws" && MAKEFLAGS="-j${BUILD_JOBS}" \
     colcon build --symlink-install \
       --parallel-workers "$BUILD_JOBS" \
       --cmake-args "-DCMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}")

  # colcon runs under the system interpreter, so setuptools stamps
  # #!/usr/bin/python3 onto every generated console script. That interpreter has
  # rclpy but no torch, and the policy runner dies at import. The venv has torch
  # and resolves rclpy, rosbag2_py, and the generated message packages through
  # the PYTHONPATH that setup.bash exports. Repoint the interpreter line only.
  VENV_PYTHON=/workspace/livifuser/runtime/venv/bin/python
  python3 - "$REPO/ros2_ws/install" "$VENV_PYTHON" <<'SHEBANG'
import pathlib
import sys

install_root, venv_python = pathlib.Path(sys.argv[1]), sys.argv[2]
rewritten = []
for path in sorted(install_root.glob("*/lib/*/*")):
    if not path.is_file():
        continue
    resolved = path.resolve()
    try:
        first, _, rest = resolved.read_text(encoding="utf-8").partition(chr(10))
    except (UnicodeDecodeError, OSError):
        continue
    if first.startswith("#!") and "python" in first and venv_python not in first:
        resolved.write_text("#!" + venv_python + chr(10) + rest, encoding="utf-8")
        rewritten.append(path.name)
print("repointed to the runtime venv:", ", ".join(rewritten) or "(none)")
SHEBANG

  # Fail loudly rather than discovering this again inside a rollout.
  for executable in live_policy_runner simulation_supervisor constant_arm_runner; do
    found="$REPO/ros2_ws/install/livifuser_sim_eval/lib/livifuser_sim_eval/$executable"
    head -1 "$found" | grep -q "^#!$VENV_PYTHON$" || {
      echo "shebang repoint failed for $executable: $(head -1 "$found")" >&2
      exit 2
    }
  done
  "$VENV_PYTHON" -c "import torch" || {
    echo "the runtime venv cannot import torch" >&2
    exit 2
  }

  set +u
  source "$REPO/ros2_ws/install/setup.bash"
  set -u
  echo "--- registered executables ---"
  ros2 pkg executables livifuser_sim_eval
  ros2 pkg executables livifuser_sim
else
  echo "NOTE: $REPO/ros2_ws/src is absent, so the workspace was not built."
  echo "Unpack the verified input handoff first:"
  echo "  python scripts/unpack_runpod_input_handoff.py <bundle.zip> $REPO"
  echo "then re-run this bootstrap to build the workspace."
fi

echo "Volume encryption status recorded as: $VOLUME_ENCRYPTION_STATUS"
echo "Bootstrap complete. No benchmark, Gazebo rollout, or confirmatory inference was launched."
