#!/usr/bin/env bash
# Run one immutable confirmatory expert episode in isolated Gazebo Fortress.
set -eo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 WORLD_JSON WORLD_SDF LIDAR_CONDITION OBSERVATION_SEED EPISODE_ID ATTEMPT_DIR WALL_WATCHDOG_SEC SIM_DEADLINE_SEC" >&2
  exit 2
fi

world_json=$1
world_sdf=$2
lidar_condition=$3
observation_seed=$4
episode_id=$5
attempt_dir=$6
wall_watchdog_sec=$7
sim_deadline_sec=$8
workspace=/mnt/d/LiViFuser

if [[ "$lidar_condition" != "C0" && "$lidar_condition" != "C3b" ]]; then
  echo "confirmatory LiDAR condition must be C0 or amendment-bound C3b" >&2
  exit 2
fi
if [[ ! -f "$world_json" || ! -f "$world_sdf" ]]; then
  echo "world JSON or SDF is missing" >&2
  exit 2
fi
if [[ ! -d "$attempt_dir" || -n "$(find "$attempt_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "attempt directory must exist and be empty: $attempt_dir" >&2
  exit 3
fi
if pgrep -f "ros2 launch livifuser_sim fortress_burger.launch.py|ign gazebo.*confirmatory_v[0-9]+|ros2 run livifuser_sim privileged_expert" >/dev/null; then
  echo "refusing to start while another LiViFuser simulation is running" >&2
  exit 4
fi

bag_dir="$attempt_dir/bag"
export_dir="$attempt_dir/export"
verify_json="$attempt_dir/verify.json"
launch_log="$attempt_dir/launch.log"
record_log="$attempt_dir/record.log"
expert_log="$attempt_dir/expert.log"
runtime_json="$attempt_dir/runtime.json"

source /opt/ros/humble/setup.bash
source "$workspace/ros2_ws/install/setup.bash"
set -u
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=97

read -r goal_forward goal_left environment_id < <(
  python3 -c 'import json,math,sys; p=json.load(open(sys.argv[1])); x,y,a=p["start_pose_xy_yaw"]; gx,gy=p["goal_xy_m"]; dx=gx-x; dy=gy-y; print(math.cos(a)*dx+math.sin(a)*dy, -math.sin(a)*dx+math.cos(a)*dy, p["name"])' "$world_json"
)

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
  stop_group INT "$record_group"
  wait_group "$record_group" 15
  stop_group TERM "$record_group"
  wait_group "$record_group" 3
  stop_group KILL "$record_group"
  stop_group TERM "$launch_group"
  wait_group "$launch_group" 5
  stop_group KILL "$launch_group"
}
trap cleanup EXIT

started_ns=$(date +%s%N)
setsid ros2 launch livifuser_sim fortress_burger.launch.py \
  start_expert:=false \
  world_sdf_path:="$world_sdf" \
  geometry_path:="$world_json" \
  observation_seed:="$observation_seed" \
  lidar_condition:="$lidar_condition" \
  goal_forward_m:="$goal_forward" \
  goal_left_m:="$goal_left" >"$launch_log" 2>&1 &
launch_group=$!

required_topics=(/camera/image_raw /camera/camera_info /scan /odom /livifuser/sim/ground_truth/odom /livifuser/goal_relative /tf_static /clock)
ready=false
for _ in {1..60}; do
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
  --duration-sec "$wall_watchdog_sec" \
  --expect-actions \
  --world-json "$world_json" \
  --until-goal \
  --output "$verify_json"
verifier_status=$?
set -e

stop_group TERM "$expert_group"
wait_group "$expert_group" 3
stop_group KILL "$expert_group"
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
launch_group=""

if [[ $verifier_status -ne 0 ]]; then
  echo "simulation contract failed; bag retained but export skipped" >&2
  exit "$verifier_status"
fi

python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); span=float(p["simulated_span_sec"]["action"]); deadline=float(sys.argv[2]); assert span <= deadline + 1.0, f"simulated span {span:.3f}s exceeds {deadline:.3f}s deadline"' \
  "$verify_json" "$sim_deadline_sec"

python3 "$workspace/scripts/export_pilot_dataset.py" "$bag_dir" \
  --output "$export_dir" \
  --environment-id "$environment_id" \
  --run-id "$episode_id" \
  --domain simulation \
  --view policy \
  --lidar-causal

finished_ns=$(date +%s%N)
python3 -c 'import json,sys; start=int(sys.argv[1]); end=int(sys.argv[2]); payload={"schema_version":"1.0.0","episode_id":sys.argv[3],"lidar_condition":sys.argv[4],"observation_seed":int(sys.argv[5]),"wall_elapsed_including_launch_and_teardown_sec":(end-start)/1e9,"verifier_exit_code":0}; open(sys.argv[6],"w",encoding="utf-8").write(json.dumps(payload,indent=2)+"\n")' \
  "$started_ns" "$finished_ns" "$episode_id" "$lidar_condition" "$observation_seed" "$runtime_json"

trap - EXIT
echo "recorded: $bag_dir"
echo "exported: $export_dir"
