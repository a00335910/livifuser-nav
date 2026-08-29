#!/usr/bin/env bash
# Run one generated development episode in the reserved isolated ROS domain.
set -eo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 WORLD_JSON WORLD_SDF OBSERVATION_SEED OUTPUT_JSON TIMEOUT_SEC" >&2
  exit 2
fi

world_json=$1
world_sdf=$2
observation_seed=$3
output_json=$4
timeout_sec=$5
workspace=/mnt/d/LiViFuser
log_path="${output_json%.json}.launch.log"
runtime_path="${output_json%.json}.runtime.json"

if pgrep -f "ros2 launch livifuser_sim fortress_burger.launch.py|ign gazebo.*worlds_dev" >/dev/null; then
  echo "refusing to start while another LiViFuser simulation is running" >&2
  exit 4
fi
if [[ -e "$output_json" || -e "$runtime_path" || -e "$log_path" ]]; then
  echo "refusing to overwrite a development-gate artifact" >&2
  exit 3
fi
mkdir -p "$(dirname "$output_json")"

source /opt/ros/humble/setup.bash
source "$workspace/ros2_ws/install/setup.bash"
set -u
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=97

read -r goal_forward goal_left < <(
  python3 -c 'import json,math,sys; p=json.load(open(sys.argv[1])); x,y,a=p["start_pose_xy_yaw"]; gx,gy=p["goal_xy_m"]; dx=gx-x; dy=gy-y; print(math.cos(a)*dx+math.sin(a)*dy, -math.sin(a)*dx+math.cos(a)*dy)' "$world_json"
)

started_ns=$(date +%s%N)
setsid ros2 launch livifuser_sim fortress_burger.launch.py \
  start_expert:=true \
  world_sdf_path:="$world_sdf" \
  geometry_path:="$world_json" \
  observation_seed:="$observation_seed" \
  goal_forward_m:="$goal_forward" \
  goal_left_m:="$goal_left" >"$log_path" 2>&1 &
launch_group=$!

cleanup() {
  kill -TERM -- "-$launch_group" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 -- "-$launch_group" 2>/dev/null; then
      return
    fi
    sleep 1
  done
  kill -KILL -- "-$launch_group" 2>/dev/null || true
}
trap cleanup EXIT

sleep 8
set +e
python3 "$workspace/scripts/verify_sim_contract.py" \
  --duration-sec "$timeout_sec" \
  --expect-actions \
  --world-json "$world_json" \
  --until-goal \
  --output "$output_json"
verifier_status=$?
set -e
cleanup
trap - EXIT
finished_ns=$(date +%s%N)

python3 -c 'import json,sys; start=int(sys.argv[1]); end=int(sys.argv[2]); payload={"schema_version":"1.0.0","wall_elapsed_including_launch_and_teardown_sec":(end-start)/1e9,"verifier_exit_code":int(sys.argv[3])}; open(sys.argv[4],"w",encoding="utf-8").write(json.dumps(payload,indent=2)+"\n")' \
  "$started_ns" "$finished_ns" "$verifier_status" "$runtime_path"

exit "$verifier_status"
