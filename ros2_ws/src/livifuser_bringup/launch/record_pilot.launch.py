"""Record every field required by the LiViFuser-Nav v1.1 Stage 1 contract."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

TOPICS = (
    "/camera/image_raw",
    "/camera/camera_info",
    "/scan",
    "/cmd_vel",
    "/livifuser/cmd_vel_stamped",
    "/livifuser/teleop_intent_stamped",
    "/livifuser/operator_intent_stamped",
    "/livifuser/command_watchdog_status",
    "/livifuser/command_watchdog_timing",
    "/livifuser/episode_state",
    "/odom",
    "/livifuser/goal_relative",
    "/tf",
    "/tf_static",
)


def generate_launch_description() -> LaunchDescription:
    output = LaunchConfiguration("output")
    storage_id = LaunchConfiguration("storage_id")
    qos_overrides_path = str(
        Path(get_package_share_directory("livifuser_command_watchdog"))
        / "config"
        / "rosbag_qos_overrides_v1.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "output",
                default_value="bags/pilot",
                description="Output rosbag path; it must not already exist.",
            ),
            DeclareLaunchArgument(
                "storage_id",
                default_value="mcap",
                description="Installed rosbag2 storage plugin identifier.",
            ),
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "record",
                    "-s",
                    storage_id,
                    "-o",
                    output,
                    "--qos-profile-overrides-path",
                    qos_overrides_path,
                    *TOPICS,
                ],
                output="screen",
            ),
        ]
    )
