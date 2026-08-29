#!/usr/bin/env bash
# Run one excluded-development episode. Confirmatory identities are rejected.
# Nounset is enabled only after both ROS setups are sourced, matching the
# frozen scripts/run_confirmatory_sim_episode.sh. ROS's setup.bash reads
# AMENT_TRACE_SETUP_FILES without defaulting it, so enabling -u earlier
# aborts the run before the simulator starts.
set -eo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: $0 ATTEMPT_DIR WORLD_SDF WORLD_JSON LIDAR_CONDITION OBS_SEED VARIANT POLICY_SEED" >&2
  exit 2
fi
ATTEMPT_DIR="$1"
WORLD_SDF="$2"
WORLD_JSON="$3"
LIDAR_CONDITION="$4"
OBS_SEED="$5"
VARIANT="$6"
POLICY_SEED="$7"

# Scope. Development is the default and needs no authorization. Confirmatory
# worlds require an explicit acknowledgement, because section 9 makes an
# accepted scientific outcome permanent: a rollout started by accident cannot be
# taken back.
case "$WORLD_JSON" in
  */development_worlds/*)
    EPISODE_SCOPE="development"
    ;;
  */confirmatory_v3/worlds/*|*/confirmatory_worlds/*)
    if [[ "${LIVIFUSER_CONFIRMATORY_AUTHORIZED:-}" != "YES" ]]; then
      echo "confirmatory world requires LIVIFUSER_CONFIRMATORY_AUTHORIZED=YES" >&2
      echo "  refusing: $WORLD_JSON" >&2
      exit 2
    fi
    EPISODE_SCOPE="confirmatory"
    ;;
  *)
    echo "Only packaged development or confirmatory worlds are allowed." >&2
    exit 2
    ;;
esac
export EPISODE_SCOPE
case "$VARIANT" in
  full|lidar_only|concat|rgb_only|constant_training_mean|nav2) ;;
  *) echo "unknown arm: $VARIANT" >&2; exit 2 ;;
esac
# The reference arms carry no checkpoint and therefore no training seed; both
# use the reserved identity seed 0, which no learned identity may take.
case "$VARIANT" in
  constant_training_mean|nav2)
    [[ "$POLICY_SEED" == "0" ]] || {
      echo "$VARIANT must use the reserved seed 0" >&2; exit 2; }
    ;;
  *)
    case "$POLICY_SEED" in
      20260805|20260806|20260807) ;;
      *) echo "unknown policy seed: $POLICY_SEED" >&2; exit 2 ;;
    esac
    ;;
esac
if [[ -e "$ATTEMPT_DIR" ]]; then
  echo "Refusing to overwrite attempt: $ATTEMPT_DIR" >&2
  exit 2
fi
# Leftover ign/ROS nodes on domain 97 share /clock and Fast DDS SHM locks.
# They interleave proposals from prior smokes and look like stamp regression.
python3 - <<'PY'
import os, signal, glob
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    pid_i = int(pid)
    if pid_i in (os.getpid(), os.getppid()):
        continue
    try:
        cmd = open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    needles = (
        "ign gazebo",
        "ros2 launch livifuser",
        "live_policy_runner",
        "simulation_supervisor",
        "constant_arm_runner",
        "wait_sim_terminal",
        "ros2 bag record",
    )
    if any(n in cmd for n in needles) and "run_live_sim_development_episode" not in cmd:
        try:
            os.kill(pid_i, signal.SIGKILL)
        except OSError:
            pass
for path in glob.glob("/dev/shm/fastrtps*") + glob.glob("/dev/shm/sem.fastrtps*"):
    try:
        os.remove(path)
    except OSError:
        pass
PY
if pgrep -x ign >/dev/null 2>&1; then
  echo "Refusing to launch while another ign/Gazebo process is running." >&2
  echo "Shared /clock from leftover sims causes proposal stamp regression." >&2
  exit 2
fi
mkdir -p "$ATTEMPT_DIR"
python3 scripts/check_runpod_storage.py --mode development >/dev/null
export ROS_LOCALHOST_ONLY=1
# Domain 97 is reserved for simulation and the launch file refuses anything
# else: keeping every simulated run on one known domain is what guarantees a
# simulated command can never reach the physical robot's domain. A per-episode
# domain was tried to isolate leaked nodes and is wrong -- it trades a safety
# invariant for a cleanup problem that the process-group teardown below already
# solves.
export ROS_DOMAIN_ID=97

# Resolve the interpreter explicitly. Bare `python` depends on whether the
# caller activated the virtualenv, and the wrong one has no rclpy: the failure
# surfaces as a missing C extension long after launch rather than here.
if [[ -n "${LIVIFUSER_PYTHON:-}" ]]; then
  PYTHON_BIN="$LIVIFUSER_PYTHON"
elif [[ -x /workspace/livifuser/runtime/venv/bin/python ]]; then
  PYTHON_BIN=/workspace/livifuser/runtime/venv/bin/python
else
  PYTHON_BIN="$(command -v python3)"
fi
export PYTHON_BIN
printf '{"episode_scope":"%s"}\n' "$EPISODE_SCOPE" > "$ATTEMPT_DIR/scope.json"
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
# The runner, supervisor, and constant-arm nodes import livifuser_nav, which
# lives in src/ and is not installed into the ROS Python path. Prepend, never
# assign: ROS's setup.bash populates PYTHONPATH and clobbering it silently
# breaks rosbag2_py and message deserialization.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Job control: bash then puts each background job in its own process group
# whose id equals the pid reported by $!, so the whole node graph can be
# signalled with kill -- -PID. Without this, ros2 launch children survive.
set -m
set -u
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# ROS is sourced now, so this is the first point where a wrong interpreter can
# be detected. Failing here costs a second; failing later costs a launch.
if ! "$PYTHON_BIN" -c "import rclpy" 2>/dev/null; then
  echo "interpreter cannot import rclpy: $PYTHON_BIN" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
LAUNCH_START_EPOCH=$(date +%s)
DEADLINE="${SCIENTIFIC_DEADLINE_SEC:-120.0}"

# OGRE 1.9's RenderSystem_GL initializes through GLX and aborts the simulator
# when DISPLAY is unset; Gazebo's --headless-rendering only covers the ogre2/EGL
# path. ogre1 is retained deliberately: every confirmatory episode was collected
# under it, and the renderer determines the pixels this study measures, so
# switching engines would change a scientific input rather than an
# infrastructure detail. Claim a private virtual display per episode so
# concurrent rollouts never share one.
XVFB_DISPLAY=""
for candidate in $(seq 90 199); do
  lock="/tmp/.X${candidate}-lock"
  if [[ ! -e "$lock" ]]; then
    XVFB_DISPLAY=":${candidate}"
    break
  fi
  # Xvfb is killed by the process-group teardown, so it never gets to remove
  # its own lock. Those files accumulate, and once all 110 slots hold debris
  # every later episode dies at the guard below -- which is exactly what
  # stopped the batch after 110 episodes. A lock naming a pid that no longer
  # exists is debris: reclaim it. Episodes run one at a time, so there is no
  # live claimant to race against.
  lock_pid="$(tr -dc '0-9' < "$lock" 2>/dev/null)"
  if [[ -z "$lock_pid" ]] || ! kill -0 "$lock_pid" 2>/dev/null; then
    rm -f "$lock" "/tmp/.X11-unix/X${candidate}" 2>/dev/null
    XVFB_DISPLAY=":${candidate}"
    break
  fi
done
if [[ -z "$XVFB_DISPLAY" ]]; then
  echo "no free X display number for the ogre1 render path" >&2
  exit 2
fi
# Depth 24. Xvfb refuses depth 32 outright ("Couldn't add screen 0"); it accepts
# 8, 15, 16, and 24 only. Depth was briefly and wrongly believed to be the cause
# of the render-engine failure -- see the CPU-report shim below for the actual
# cause.
Xvfb "$XVFB_DISPLAY" -screen 0 1024x768x24 -nolisten tcp \
  >"$ATTEMPT_DIR/xvfb.log" 2>&1 &
XVFB_PID=$!
export DISPLAY="$XVFB_DISPLAY"

# Do not launch into a display that never came up: the simulator would abort
# with the same GLX error this exists to prevent.
for _ in $(seq 1 50); do
  if [[ -e "/tmp/.X${XVFB_DISPLAY#:}-lock" ]]; then break; fi
  sleep 0.2
done
if [[ ! -e "/tmp/.X${XVFB_DISPLAY#:}-lock" ]]; then
  echo "Xvfb did not start on $XVFB_DISPLAY" >&2
  kill "$XVFB_PID" 2>/dev/null
  exit 2
fi
if ! kill -0 "$XVFB_PID" 2>/dev/null; then
  echo "Xvfb exited immediately on $XVFB_DISPLAY" >&2
  exit 2
fi
echo "ogre1 render path using virtual display $XVFB_DISPLAY (pid $XVFB_PID)"

# OGRE sizes its render work queue from the reported CPU count, and this
# container reports the host's 256 rather than the 27 the cgroup allows. Left
# alone it spawns thousands of render workers, fails to load the render engine,
# and Gazebo retries until it segfaults on the unloaded engine. taskset does not
# help: glibc reads /sys, not the affinity mask. The shim changes what is
# reported. Measured: 256 reported -> engine load fails; 8 reported -> 8
# workers, loads first attempt, no crash.
CPU_SHIM=/workspace/livifuser/runtime/limit_cpu_report.so
if [[ ! -f "$CPU_SHIM" ]]; then
  echo "missing CPU-report shim: $CPU_SHIM (re-run the bootstrap)" >&2
  exit 2
fi
export LD_PRELOAD="$CPU_SHIM${LD_PRELOAD:+:${LD_PRELOAD}}"

LAUNCH_PID=""
BAG_PID=""
# ros2 launch does not reap its nodes when the launch process is killed, and
# killing only the launch PID leaves the entire node graph running. Both
# `set -m` above puts each background job in its own process group whose id
# equals the pid in $!, and the group is signalled as a whole. An earlier
# attempt used setsid and failed: setsid forks, so $! was the short-lived
# parent and nothing was killed. Measured then: survivors after each episode
# were 21, 27, 27, 35, with two runners publishing to the same proposal topic.
stop_group() {
  local pid="$1" grace="$2"
  [[ -z "$pid" ]] && return 0
  # With `set -m` the job's process group id equals its pid.
  kill -INT -- "-$pid" 2>/dev/null
  for _ in $(seq 1 "$grace"); do
    kill -0 -- "-$pid" 2>/dev/null || return 0
    sleep 1
  done
  kill -TERM -- "-$pid" 2>/dev/null
  sleep 2
  kill -KILL -- "-$pid" 2>/dev/null
  return 0
}

# Backstop for anything that escaped its process group. Names are matched at the
# 15-character truncation the kernel applies to comm; using full names silently
# matches nothing. Runs after the group kill so it cannot hide a failure of the
# primary mechanism.
sweep_stragglers() {
  local name
  for name in parameter_bridg static_transfor odom_waypoint_g relative_goal_p \
              analytic_lidar live_policy_run simulation_supe constant_arm_ru \
              contract_node world_pose nav2_probe ruby ign; do
    pkill -9 -x "$name" 2>/dev/null
  done
  return 0
}

cleanup() {
  set +e
  stop_group "$BAG_PID" 10
  stop_group "$LAUNCH_PID" 15
  sweep_stragglers
  if [[ -n "${XVFB_PID:-}" ]]; then kill -KILL "$XVFB_PID" 2>/dev/null; fi
  # Killed Xvfb cannot clean up after itself; do it here or the display
  # range is consumed one slot per episode.
  # ros2 launch writes a parameter file per node per episode and never
  # removes them; 6,116 had accumulated in /tmp by episode 542. Clean the
  # ones this episode's launch created.
  if [[ -n "${LAUNCH_START_EPOCH:-}" ]]; then
    find /tmp -maxdepth 1 -name 'launch_params_*' -newermt "@${LAUNCH_START_EPOCH}" -delete 2>/dev/null
  fi
  if [[ -n "${XVFB_DISPLAY:-}" ]]; then
    rm -f "/tmp/.X${XVFB_DISPLAY#:}-lock" "/tmp/.X11-unix/X${XVFB_DISPLAY#:}" 2>/dev/null
  fi
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# Compress inside the mcap file, not around it. File-mode compression writes
# bag_0.mcap.zstd, which Foxglove and mcap-native readers cannot open without
# a manual decompression step; chunk compression keeps a plain .mcap that
# tools read directly. Level Fast, because this runs during recording rather
# than at file close and must not disturb the 10 Hz control loop.
cat > "$ATTEMPT_DIR/mcap_storage.yaml" <<YAML
compression: "Zstd"
compressionLevel: "Fast"
chunkSize: 4194304
YAML
ros2 bag record --storage mcap \
  --storage-config-file "$ATTEMPT_DIR/mcap_storage.yaml" \
  --output "$ATTEMPT_DIR/bag" \
  /clock /camera/image_raw /camera/camera_info /scan /odom \
  /livifuser/goal_relative /livifuser/sim/ground_truth/odom \
  /livifuser/eval/policy_proposal /livifuser/eval/supervisor_status \
  /livifuser/sim_cmd_vel /livifuser/cmd_vel_stamped /tf /tf_static \
  >"$ATTEMPT_DIR/rosbag.log" 2>&1 &
BAG_PID=$!

ros2 launch livifuser_sim live_policy_eval.launch.py \
  world_sdf_path:="$WORLD_SDF" geometry_path:="$WORLD_JSON" \
  lidar_condition:="$LIDAR_CONDITION" observation_seed:="$OBS_SEED" \
  variant:="$VARIANT" seed:="$POLICY_SEED" device:=cuda:0 \
  scientific_deadline_sec:="$DEADLINE" \
  real_time_factor:="${SIM_REAL_TIME_FACTOR:-0.4}" \
  backbone_bundle:="artifacts/livifuser_dinov3_vits16plus_backbone_c93d816_bundle.zip" \
  policy_bundle:="artifacts/livifuser_sim_closed_loop_policy_payload_v1_bundle.zip" \
  backbone_extract_root:="/workspace/livifuser/runtime/backbone" \
  sensor_contract:="config/simulation_live_sensor_contract_v1.json" \
  >"$ATTEMPT_DIR/launch.log" 2>&1 &
LAUNCH_PID=$!

# Slow the simulator so each 100 ms control tick has wall time to complete.
# See the header comment: at real-time factor 1.0 the loop measured 163-187 ms
# per tick and the supervisor terminated the episode. Simulated time drives the
# control clock, so nothing scientific changes. Applied through the runtime
# service because the world SDFs are frozen.
WORLD_NAME="$(basename "$WORLD_SDF" .sdf)"
SIM_RTF="${SIM_REAL_TIME_FACTOR:-0.4}"
for _ in $(seq 1 40); do
  if ign service -s "/world/${WORLD_NAME}/set_physics" \
       --reqtype ignition.msgs.Physics --reptype ignition.msgs.Boolean \
       --timeout 2000 --req "real_time_factor: ${SIM_RTF}" >/dev/null 2>&1; then
    echo "simulator real-time factor set to ${SIM_RTF} for ${WORLD_NAME}"
    break
  fi
  sleep 1
done

set +e
# The wall budget must cover the scientific deadline divided by the real-time
# factor, plus simulator startup and model load.
WALL_MONITOR_SEC="${WALL_MONITOR_SEC:-900}"
timeout --signal=INT "${WALL_MONITOR_SEC}s" "$PYTHON_BIN" scripts/wait_sim_terminal.py \
  --output "$ATTEMPT_DIR/terminal.json" >"$ATTEMPT_DIR/monitor.log" 2>&1
MONITOR_STATUS=$?
set -e
cleanup
trap - EXIT INT TERM
LAUNCH_PID=""
BAG_PID=""
# The supervisor's terminal record is the authority on the scientific outcome.
# The monitor may be signalled during teardown after writing a correct result,
# so its exit code alone must not condemn an otherwise valid episode; it is
# recorded either way for audit.
TERMINAL_VALID=0
if [[ -f "$ATTEMPT_DIR/terminal.json" ]]; then
  if python3 -c "
import json, sys
record = json.load(open('$ATTEMPT_DIR/terminal.json'))
sys.exit(0 if record.get('terminal') and record.get('terminal_reason') else 1)
" 2>/dev/null; then
    TERMINAL_VALID=1
  fi
fi

printf '{"monitor_exit_code":%d,"terminal_record_valid":%d}\n' \
  "$MONITOR_STATUS" "$TERMINAL_VALID" >"$ATTEMPT_DIR/monitor_status.json"

if [[ "$TERMINAL_VALID" -ne 1 ]]; then
  printf '{"status":"operational_failure","monitor_exit_code":%d}\n' "$MONITOR_STATUS" \
    >"$ATTEMPT_DIR/operational_failure.json"
fi
# A leak silently corrupts every later episode, so record whether one happened.
survivors=$(ps -eo comm | grep -cE "parameter_brid|static_transf|analytic_lida|live_policy_run|simulation_supe|contract_node|world_pose" || true)
echo "post-teardown node survivors on this host: $survivors"
if [[ "$survivors" -ne 0 ]]; then
  # A survivor publishes into the next episode on the shared domain and
  # silently corrupts it. Record it rather than letting it propagate.
  printf '{"status":"teardown_leak","surviving_nodes":%d}\n' "$survivors" \
    >"$ATTEMPT_DIR/teardown_leak.json"
fi

"$PYTHON_BIN" scripts/seal_runtime_attempt.py "$ATTEMPT_DIR"
test "$TERMINAL_VALID" -eq 1
