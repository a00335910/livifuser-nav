#!/usr/bin/env bash
# Record one excluded, bounded AWS Small House visual compatibility probe.
set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

workspace=/mnt/d/LiViFuser
probe_root="$workspace/artifacts/simulation/vendor_probe/aws-small-house-fortress-v5"
world="$probe_root/aws_small_house_fortress_probe.sdf"
resource_path="$probe_root/models"
output=$1
runtime_root=""

if [[ -e "$output" ]]; then
  echo "refusing to overwrite visual-probe output: $output" >&2
  exit 3
fi
if [[ ! -f "$world" || ! -d "$resource_path" ]]; then
  echo "prepared AWS Small House probe is missing" >&2
  exit 4
fi
if pgrep -f "ign gazebo.*aws_small_house_fortress_probe|visual_asset_probe.launch.py" >/dev/null; then
  echo "refusing to start while another visual probe is running" >&2
  exit 5
fi

mkdir -p "$output"
source /opt/ros/humble/setup.bash
source "$workspace/ros2_ws/install/setup.bash"
set -u
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=97

runtime_root=$(mktemp -d /tmp/livifuser-small-house.XXXXXX)
cp -a "$world" "$runtime_root/world.sdf"
cp -a "$resource_path" "$runtime_root/models"
world="$runtime_root/world.sdf"
resource_path="$runtime_root/models"
export IGN_GAZEBO_RESOURCE_PATH="$resource_path"

launch_group=""
record_group=""
cleanup() {
  ros2 topic pub --once /livifuser/sim_cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.0}, angular: {z: 0.0}}" >/dev/null 2>&1 || true
  if [[ -n "$record_group" ]]; then
    kill -INT -- "-$record_group" 2>/dev/null || true
    sleep 2
    kill -TERM -- "-$record_group" 2>/dev/null || true
  fi
  if [[ -n "$launch_group" ]]; then
    kill -TERM -- "-$launch_group" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      if ! kill -0 -- "-$launch_group" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 -- "-$launch_group" 2>/dev/null; then
      kill -KILL -- "-$launch_group" 2>/dev/null || true
    fi
  fi
  if [[ "$runtime_root" == /tmp/livifuser-small-house.* && -d "$runtime_root" ]]; then
    rm -rf -- "$runtime_root"
  fi
}
trap cleanup EXIT INT TERM

setsid ros2 launch livifuser_sim visual_asset_probe.launch.py \
  world_sdf_path:="$world" \
  >"$output/launch.log" 2>&1 &
launch_group=$!

camera_ready=false
for _ in $(seq 1 20); do
  if timeout 6s ros2 topic echo --once /camera/image_raw >/dev/null 2>&1; then
    camera_ready=true
    break
  fi
  sleep 1
done
if [[ "$camera_ready" != true ]]; then
  echo "camera did not become ready" >&2
  exit 6
fi

setsid ros2 bag record --storage mcap --output "$output/bag" \
  /camera/image_raw /camera/camera_info /odom \
  /livifuser/sim/ground_truth/odom /livifuser/sim_cmd_vel \
  /tf /tf_static /clock >"$output/record.log" 2>&1 &
record_group=$!

sleep 3
set +e
timeout 8s ros2 topic pub -r 10 /livifuser/sim_cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.05}, angular: {z: 0.0}}" >"$output/drive.log" 2>&1
drive_status=$?
set -e
if [[ $drive_status -ne 0 && $drive_status -ne 124 ]]; then
  echo "bounded simulated drive failed with status $drive_status" >&2
  exit 7
fi
ros2 topic pub --once /livifuser/sim_cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.0}}" >>"$output/drive.log" 2>&1
sleep 4

kill -INT -- "-$record_group" 2>/dev/null || true
wait "$record_group" || true
record_group=""
cleanup
launch_group=""
trap - EXIT INT TERM

ros2 bag info "$output/bag" | tee "$output/bag_info.txt"
