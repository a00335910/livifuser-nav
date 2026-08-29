#!/usr/bin/env bash
# Launcher for the live policy overlay demonstration. Run this from WSL.
#
#   ./scripts/demo_live_overlay.sh            preflight, start overlay, then drive
#   ./scripts/demo_live_overlay.sh check      preflight only, changes nothing
#   ./scripts/demo_live_overlay.sh overlay    overlay only, no teleop, no motion
#   ./scripts/demo_live_overlay.sh drive      teleop only, against a running overlay
#   ./scripts/demo_live_overlay.sh watchdog   start the robot-local watchdog over SSH
#   ./scripts/demo_live_overlay.sh stop       stop the overlay, and the watchdog
#
# This script never publishes a command topic itself. Motion can only come from
# the `drive` step, which runs the existing keyboard teleop, whose own graph
# check refuses to arm unless the watchdog is the sole /cmd_vel authority.

set -uo pipefail

REPO="${REPO:-/mnt/d/LiViFuser}"
PORT="${PORT:-8080}"
ROBOT="${ROBOT:-a00335910@192.168.0.33}"
CHECKPOINT="${CHECKPOINT:-artifacts/experiments/pilot5_leave_one_episode_out_kaggle_t4x2_v1/held_out_center_002b/lidar_only/seed_20260805/checkpoint.pt}"
MANIFEST="${MANIFEST:-artifacts/export/protocol_clean_30/train_lab_s1_center_002b_policy_git_3f47712/manifest.json}"
LOG="${LOG:-$REPO/artifacts/live_overlay.log}"

WATCHDOG_NODE="livifuser_command_watchdog"
WATCHDOG_UNIT="livifuser-watchdog-trial"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mstopped:\033[0m %s\n' "$*" >&2; exit 1; }

source_ros() {
  # ROS's setup scripts read unbound variables, so -u has to come off around
  # them. It goes straight back on afterwards.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash || die "could not source ROS 2 Humble"
  # shellcheck disable=SC1091
  source "$REPO/ros2_ws/install/setup.bash" || die "could not source the workspace"
  set -u
  cd "$REPO" || die "could not enter $REPO"
}

# ---------------------------------------------------------------- preflight --

check_files() {
  [ -f "$REPO/$CHECKPOINT" ] || die "checkpoint not found: $CHECKPOINT"
  [ -f "$REPO/$MANIFEST" ]   || die "calibration manifest not found: $MANIFEST"
  ok "checkpoint and calibration manifest present"
}

check_scan() {
  if timeout 12 ros2 topic echo /scan --once >/dev/null 2>&1; then
    ok "/scan is publishing"
  else
    bad "/scan is not publishing"
    die "start base bringup on the robot first (see docs/LIVE_OVERLAY_DEMONSTRATION.md)"
  fi
}

check_discovery() {
  # `ros2 topic info` counts endpoints, but the teleop graph check needs node
  # NAMES. Robot-local shells have been seen to count publishers while listing
  # no nodes at all, which is exactly the failure this catches early.
  local nodes
  nodes="$(timeout 12 ros2 node list 2>/dev/null)"
  if [ -z "$nodes" ]; then
    bad "node discovery returned nothing"
    die "run this from WSL, not from a shell on the robot"
  fi
  ok "node discovery works ($(printf '%s\n' "$nodes" | wc -l) nodes visible)"
}

check_watchdog() {
  local publishers
  publishers="$(timeout 12 ros2 topic info /cmd_vel 2>/dev/null | awk '/Publisher count/{print $3}')"
  publishers="${publishers:-0}"
  if [ "$publishers" -eq 0 ]; then
    warn "no /cmd_vel publisher — the watchdog is not running"
    return 1
  fi
  if ! timeout 12 ros2 node list 2>/dev/null | grep -qx "/$WATCHDOG_NODE"; then
    bad "/cmd_vel has $publishers publisher(s), but the watchdog is not among them"
    die "refusing to continue: the watchdog must be the sole command authority"
  fi
  if [ "$publishers" -gt 1 ]; then
    bad "/cmd_vel has $publishers publishers; exactly one is required"
    die "refusing to continue: something else is publishing commands"
  fi
  ok "watchdog is the sole /cmd_vel publisher"
  return 0
}

preflight() {
  bold "Preflight"
  check_files
  check_scan
  check_discovery
}

# ------------------------------------------------------------------ overlay --

overlay_running() {
  curl -s --max-time 2 "http://127.0.0.1:$PORT/state" >/dev/null 2>&1
}

start_overlay() {
  if overlay_running; then
    warn "something is already serving port $PORT — reusing it"
    return 0
  fi
  mkdir -p "$(dirname "$LOG")"
  bold "Starting overlay"
  python3 scripts/run_live_lidar_demo.py --source ros \
    --checkpoint "$CHECKPOINT" \
    --calibration-manifest "$MANIFEST" \
    --port "$PORT" > "$LOG" 2>&1 &
  OVERLAY_PID=$!

  local waited=0
  while [ "$waited" -lt 40 ]; do
    if ! kill -0 "$OVERLAY_PID" 2>/dev/null; then
      sed -n '1,25p' "$LOG" >&2
      die "the overlay exited during startup — see $LOG"
    fi
    overlay_running && break
    sleep 1
    waited=$((waited + 1))
  done
  overlay_running || die "the overlay did not start serving within 40s — see $LOG"

  # The safety line is the one that matters; show the whole banner.
  grep -E '^(dashboard|policy|source|publishers)' "$LOG" | sed 's/^/  /'
  if grep -q 'publishers:.*no command topic' "$LOG"; then
    ok "overlay creates no command publisher"
  else
    bad "the overlay banner did not confirm it holds no command topic"
    die "refusing to continue"
  fi
  bold "Dashboard: http://localhost:$PORT/"
}

stop_overlay() {
  if [ -n "${OVERLAY_PID:-}" ] && kill -0 "$OVERLAY_PID" 2>/dev/null; then
    kill "$OVERLAY_PID" 2>/dev/null
    sleep 1
    kill -9 "$OVERLAY_PID" 2>/dev/null
  fi
}

# ----------------------------------------------------------------- watchdog --

start_watchdog() {
  bold "Starting the robot-local watchdog"
  # Built as a string because it runs in the robot's login shell, not this one.
  # Its own `bash -lc` is not under nounset, so sourcing ROS there is safe.
  local remote_shell="source /opt/ros/humble/setup.bash"
  remote_shell="$remote_shell && source ~/ros2_ws/install/setup.bash"
  remote_shell="$remote_shell && exec ros2 run livifuser_command_watchdog command_watchdog"
  ssh -o ConnectTimeout=10 "$ROBOT" \
    "systemd-run --user --quiet --unit=$WATCHDOG_UNIT --collect -p Restart=no \
     -p StandardOutput=journal -p StandardError=journal \
     /bin/bash -lc '$remote_shell'" \
    || die "could not start the watchdog on $ROBOT"
  sleep 8
  ok "watchdog unit started; it holds /cmd_vel at zero until intent arrives"
}

stop_watchdog() {
  ssh -o ConnectTimeout=10 "$ROBOT" \
    "systemctl --user stop $WATCHDOG_UNIT.service 2>/dev/null; true" \
    && ok "watchdog stopped" || warn "could not reach $ROBOT to stop the watchdog"
}

# -------------------------------------------------------------------- drive --

drive() {
  bold "Handing over to keyboard teleop"
  cat <<'KEYS'
  i forward    u forward+left   o forward+right
  j turn left  l turn right     , reverse
  k STOP       Ctrl+C to quit

  Commands are LATCHED — the robot keeps going until you press another key.
  Check the floor is clear and keep a hand near the power switch.
KEYS
  printf '\n  Press Enter when the floor is clear, or Ctrl+C to abort: '
  read -r _
  ros2 run livifuser_command_watchdog keyboard_teleop
}

# --------------------------------------------------------------------- main --

MODE="${1:-all}"
source_ros

case "$MODE" in
  check)
    preflight
    check_watchdog || warn "start it with: $0 watchdog"
    bold "Preflight complete."
    ;;
  overlay)
    preflight
    trap stop_overlay EXIT
    start_overlay
    bold "Overlay running. Ctrl+C to stop."
    wait "${OVERLAY_PID:-}" 2>/dev/null
    ;;
  watchdog)
    start_watchdog
    check_watchdog
    ;;
  drive)
    preflight
    check_watchdog || die "start the watchdog first: $0 watchdog"
    overlay_running || warn "no overlay on port $PORT — driving without the dashboard"
    drive
    ;;
  all)
    preflight
    if ! check_watchdog; then
      start_watchdog
      check_watchdog || die "the watchdog did not come up"
    fi
    trap stop_overlay EXIT
    start_overlay
    printf '\n'
    drive
    ;;
  stop)
    stop_overlay
    pkill -f run_live_lidar_demo 2>/dev/null && ok "overlay stopped" || ok "no overlay was running"
    stop_watchdog
    ;;
  *)
    die "unknown mode '$MODE' — use check, overlay, watchdog, drive, all, or stop"
    ;;
esac
