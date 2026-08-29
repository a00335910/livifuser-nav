#!/usr/bin/env bash
# Run one development-only learned LiDAR policy episode under C0/C3.
set -eo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 WORLD_JSON WORLD_SDF OBSERVATION_SEED CONDITION MODEL_ONNX OUTPUT_JSON TIMEOUT_SEC" >&2
  exit 2
fi

world_json=$1
world_sdf=$2
observation_seed=$3
condition=$4
model_onnx=$5
output_json=$6
timeout_sec=$7
workspace=/mnt/d/LiViFuser
launch_log="${output_json%.json}.launch.log"
policy_log="${output_json%.json}.policy.log"

if [[ "$condition" != "C0" && "$condition" != "C3a" && "$condition" != "C3b" ]]; then
  echo "condition must be C0, C3a, or C3b" >&2
  exit 2
fi
if pgrep -f "ros2 launch livifuser_sim fortress_burger.launch.py|ign gazebo.*worlds_dev" >/dev/null; then
  echo "refusing to start while another LiViFuser simulation is running" >&2
  exit 4
fi
for path in "$output_json" "$launch_log" "$policy_log"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite development evidence: $path" >&2
    exit 3
  fi
done
mkdir -p "$(dirname "$output_json")"

source /opt/ros/humble/setup.bash
source "$workspace/ros2_ws/install/setup.bash"
set -u
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=97

read -r goal_forward goal_left < <(
  python3 -c 'import json,math,sys; p=json.load(open(sys.argv[1])); x,y,a=p["start_pose_xy_yaw"]; gx,gy=p["goal_xy_m"]; dx=gx-x; dy=gy-y; print(math.cos(a)*dx+math.sin(a)*dy, -math.sin(a)*dx+math.cos(a)*dy)' "$world_json"
)

launch_group=""
policy_group=""
stop_group() {
  local signal=$1
  local group=$2
  if [[ -n "$group" ]] && kill -0 -- "-$group" 2>/dev/null; then
    kill "-$signal" -- "-$group" 2>/dev/null || true
  fi
}
wait_group() {
  local group=$1
  local attempts=$2
  for ((index=0; index<attempts; index++)); do
    if [[ -z "$group" ]] || ! kill -0 -- "-$group" 2>/dev/null; then
      return
    fi
    sleep 1
  done
}
cleanup() {
  stop_group TERM "$policy_group"
  wait_group "$policy_group" 3
  stop_group KILL "$policy_group"
  stop_group TERM "$launch_group"
  wait_group "$launch_group" 5
  stop_group KILL "$launch_group"
}
trap cleanup EXIT

setsid ros2 launch livifuser_sim fortress_burger.launch.py \
  start_expert:=false \
  world_sdf_path:="$world_sdf" \
  geometry_path:="$world_json" \
  observation_seed:="$observation_seed" \
  lidar_condition:="$condition" \
  goal_forward_m:="$goal_forward" \
  goal_left_m:="$goal_left" >"$launch_log" 2>&1 &
launch_group=$!

ready=false
for _ in {1..40}; do
  topic_list=$(ros2 topic list 2>/dev/null || true)
  ready=true
  for topic in /scan /odom /livifuser/goal_relative /clock; do
    if ! grep -Fxq "$topic" <<<"$topic_list"; then
      ready=false
      break
    fi
  done
  if [[ "$ready" == true ]]; then
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "simulation graph did not become ready; inspect $launch_log" >&2
  exit 5
fi

setsid ros2 run livifuser_sim lidar_policy --ros-args \
  -p use_sim_time:=true \
  -p model_path:="$model_onnx" >"$policy_log" 2>&1 &
policy_group=$!
sleep 3
if ! kill -0 "$policy_group" 2>/dev/null; then
  echo "LiDAR policy exited early; inspect $policy_log" >&2
  exit 6
fi

set +e
python3 "$workspace/scripts/verify_sim_contract.py" \
  --duration-sec "$timeout_sec" \
  --expect-actions \
  --max-linear-mps 0.10 \
  --max-angular-radps 0.50 \
  --world-json "$world_json" \
  --until-goal \
  --output "$output_json"
verifier_status=$?
set -e
cleanup
trap - EXIT
exit "$verifier_status"
