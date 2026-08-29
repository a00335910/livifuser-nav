#!/usr/bin/env bash
# Run one C0/C4 development-only Nav2 structural-probe episode.
set -eo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 WORLD_JSON WORLD_SDF OBSERVATION_SEED CONDITION PARAMS_YAML OUTPUT_JSON TIMEOUT_SEC" >&2
  exit 2
fi

world_json=$1
world_sdf=$2
observation_seed=$3
condition=$4
params_yaml=$5
output_json=$6
timeout_sec=$7
workspace=/mnt/d/LiViFuser
stem=${output_json%.json}
map_yaml="${stem}.map.yaml"
map_pgm="${stem}.map.pgm"
map_manifest="${stem}.map.manifest.json"
status_json="${stem}.nav2_status.json"
launch_log="${stem}.launch.log"

if [[ "$condition" != "C0" && "$condition" != "C4" ]]; then
  echo "condition must be C0 or C4" >&2
  exit 2
fi
if [[ "$condition" == "C4" && "$world_json" != *.C4.json ]]; then
  echo "C4 requires a derived .C4.json world" >&2
  exit 2
fi
if [[ "$condition" == "C0" && "$world_json" == *.C4.json ]]; then
  echo "C0 may not use a derived .C4.json world" >&2
  exit 2
fi
if pgrep -f "ros2 launch livifuser_sim|ign gazebo.*worlds_dev" >/dev/null; then
  echo "refusing to start while another LiViFuser simulation is running" >&2
  exit 4
fi
for path in "$output_json" "$map_yaml" "$map_pgm" "$map_manifest" "$status_json" "$launch_log"; do
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

python3 "$workspace/scripts/materialize_nav2_map.py" "$world_json" "$map_yaml"
read -r start_x start_y start_yaw goal_forward goal_left < <(
  python3 -c 'import json,math,sys; p=json.load(open(sys.argv[1])); x,y,a=p["start_pose_xy_yaw"]; gx,gy=p["goal_xy_m"]; dx=gx-x; dy=gy-y; print(x,y,a,math.cos(a)*dx+math.sin(a)*dy,-math.sin(a)*dx+math.cos(a)*dy)' "$world_json"
)

launch_group=""
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
  stop_group TERM "$launch_group"
  wait_group "$launch_group" 8
  stop_group KILL "$launch_group"
}
trap cleanup EXIT

setsid ros2 launch livifuser_sim nav2_probe.launch.py \
  world_sdf_path:="$world_sdf" \
  geometry_path:="$world_json" \
  map_yaml_path:="$map_yaml" \
  params_file:="$params_yaml" \
  status_path:="$status_json" \
  condition:="$condition" \
  observation_seed:="$observation_seed" \
  goal_forward_m:="$goal_forward" \
  goal_left_m:="$goal_left" \
  map_to_odom_x:="$start_x" \
  map_to_odom_y:="$start_y" \
  map_to_odom_yaw:="$start_yaw" >"$launch_log" 2>&1 &
launch_group=$!

ready=false
for _ in {1..60}; do
  topic_list=$(ros2 topic list 2>/dev/null || true)
  ready=true
  for topic in /scan /odom /map /livifuser/goal_relative /clock; do
    if ! grep -Fxq "$topic" <<<"$topic_list"; then
      ready=false
      break
    fi
  done
  if [[ "$ready" == true && -s "$status_json" ]]; then
    break
  fi
  sleep 1
done
if [[ "$ready" != true || ! -s "$status_json" ]]; then
  echo "Nav2 graph did not become ready; inspect $launch_log" >&2
  exit 5
fi

set +e
python3 "$workspace/scripts/verify_sim_contract.py" \
  --duration-sec "$timeout_sec" \
  --expect-actions \
  --max-linear-mps 0.08 \
  --max-angular-radps 0.40 \
  --world-json "$world_json" \
  --until-goal \
  --output "$output_json"
verifier_status=$?
set -e
sleep 2
cleanup
trap - EXIT
exit "$verifier_status"
