#!/usr/bin/env bash
# Record and export one generated development episode in the isolated ROS domain.
set -eo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 WORLD_JSON WORLD_SDF OBSERVATION_SEED BAG_DIR EXPORT_DIR VERIFY_JSON TIMEOUT_SEC" >&2
  exit 2
fi

world_json=$1
world_sdf=$2
observation_seed=$3
bag_dir=$4
export_dir=$5
verify_json=$6
timeout_sec=$7
workspace=/mnt/d/LiViFuser
launch_log="${verify_json%.json}.launch.log"
record_log="${verify_json%.json}.record.log"
expert_log="${verify_json%.json}.expert.log"

if pgrep -f "ros2 launch livifuser_sim fortress_burger.launch.py|ign gazebo.*worlds_dev" >/dev/null; then
  echo "refusing to start while another LiViFuser simulation is running" >&2
  exit 4
fi
for path in "$bag_dir" "$export_dir" "$verify_json" "$launch_log" "$record_log" "$expert_log"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite development evidence: $path" >&2
    exit 3
  fi
done
mkdir -p "$(dirname "$bag_dir")" "$(dirname "$export_dir")" "$(dirname "$verify_json")"

source /opt/ros/humble/setup.bash
source "$workspace/ros2_ws/install/setup.bash"
set -u
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=97

read -r goal_forward goal_left < <(
  python3 -c 'import json,math,sys; p=json.load(open(sys.argv[1])); x,y,a=p["start_pose_xy_yaw"]; gx,gy=p["goal_xy_m"]; dx=gx-x; dy=gy-y; print(math.cos(a)*dx+math.sin(a)*dy, -math.sin(a)*dx+math.cos(a)*dy)' "$world_json"
)
environment_id=$(basename "${world_json%.json}")

launch_group=""
record_group=""
expert_group=""

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
  stop_group TERM "$expert_group"
  wait_group "$expert_group" 3
  stop_group KILL "$expert_group"
  # SIGINT lets rosbag2 flush indexes and metadata before its process exits.
  stop_group INT "$record_group"
  wait_group "$record_group" 10
  stop_group TERM "$record_group"
  wait_group "$record_group" 3
  stop_group KILL "$record_group"
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
  goal_forward_m:="$goal_forward" \
  goal_left_m:="$goal_left" >"$launch_log" 2>&1 &
launch_group=$!

required_topics=(/camera/image_raw /camera/camera_info /scan /odom /livifuser/sim/ground_truth/odom /livifuser/goal_relative /tf_static /clock)
ready=false
for _ in {1..40}; do
  topic_list=$(ros2 topic list 2>/dev/null || true)
  ready=true
  for topic in "${required_topics[@]}"; do
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

setsid ros2 bag record -s mcap --use-sim-time -o "$bag_dir" \
  /camera/image_raw \
  /camera/camera_info \
  /scan \
  /odom \
  /livifuser/sim/ground_truth/odom \
  /livifuser/goal_relative \
  /livifuser/cmd_vel_stamped \
  /tf \
  /tf_static \
  /clock >"$record_log" 2>&1 &
record_group=$!
sleep 2
if ! kill -0 "$record_group" 2>/dev/null; then
  echo "rosbag recorder exited early; inspect $record_log" >&2
  exit 6
fi

setsid ros2 run livifuser_sim privileged_expert --ros-args \
  -p use_sim_time:=true \
  -p geometry_path:="$world_json" >"$expert_log" 2>&1 &
expert_group=$!

set +e
python3 "$workspace/scripts/verify_sim_contract.py" \
  --duration-sec "$timeout_sec" \
  --expect-actions \
  --world-json "$world_json" \
  --until-goal \
  --output "$verify_json"
verifier_status=$?
set -e

stop_group TERM "$expert_group"
wait_group "$expert_group" 3
stop_group KILL "$expert_group"
wait_group "$expert_group" 2
expert_group=""
stop_group INT "$record_group"
wait_group "$record_group" 15
if kill -0 -- "-$record_group" 2>/dev/null; then
  echo "rosbag recorder did not finalize after SIGINT" >&2
  exit 7
fi
record_group=""
stop_group TERM "$launch_group"
wait_group "$launch_group" 5
stop_group KILL "$launch_group"
wait_group "$launch_group" 2
launch_group=""

if [[ $verifier_status -ne 0 ]]; then
  echo "simulation contract failed; bag retained but export skipped" >&2
  exit "$verifier_status"
fi

python3 "$workspace/scripts/export_pilot_dataset.py" "$bag_dir" \
  --output "$export_dir" \
  --environment-id "$environment_id" \
  --run-id "$(basename "$bag_dir")" \
  --domain simulation \
  --view policy \
  --lidar-causal

trap - EXIT
echo "recorded: $bag_dir"
echo "exported: $export_dir"
