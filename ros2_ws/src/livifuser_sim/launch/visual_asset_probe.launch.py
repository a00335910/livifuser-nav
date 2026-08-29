"""Launch the excluded AWS Small House visual compatibility probe."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORLD_NAME = "aws_small_house_fortress_probe"


def _require_isolation(_context):
    if os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError("simulation probe requires ROS_LOCALHOST_ONLY=1")
    if os.environ.get("ROS_DOMAIN_ID") != "97":
        raise RuntimeError("simulation probe requires ROS_DOMAIN_ID=97")
    return []


def _static_transform(
    name: str,
    parent: str,
    child: str,
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> Node:
    x, y, z = translation
    qx, qy, qz, qw = quaternion
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        arguments=[
            "--x", str(x), "--y", str(y), "--z", str(z),
            "--qx", str(qx), "--qy", str(qy), "--qz", str(qz), "--qw", str(qw),
            "--frame-id", parent, "--child-frame-id", child,
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    ros_gz_share = get_package_share_directory("ros_gz_sim")
    world = LaunchConfiguration("world_sdf_path")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(f"{ros_gz_share}/launch/gz_sim.launch.py"),
        launch_arguments={
            "gz_args": ["-r -s --headless-rendering --render-engine ogre ", world],
            "gz_version": "6",
            "on_exit_shutdown": "true",
        }.items(),
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="livifuser_visual_probe_bridge",
        arguments=[
            "/camera@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/model/livifuser_burger/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            "/model/livifuser_burger/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            (
                f"/world/{WORLD_NAME}/dynamic_pose/info@tf2_msgs/msg/TFMessage"
                "[ignition.msgs.Pose_V"
            ),
        ],
        remappings=[
            ("/camera", "/livifuser/sim/raw/image"),
            ("/model/livifuser_burger/odometry", "/livifuser/sim/raw/odom"),
            ("/model/livifuser_burger/cmd_vel", "/livifuser/sim_cmd_vel"),
            (
                f"/world/{WORLD_NAME}/dynamic_pose/info",
                "/livifuser/sim/raw/world_pose",
            ),
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("world_sdf_path"),
            OpaqueFunction(function=_require_isolation),
            gazebo,
            bridge,
            Node(
                package="livifuser_sim",
                executable="contract_node",
                parameters=[{"use_sim_time": True}],
                output="screen",
            ),
            Node(
                package="livifuser_sim",
                executable="world_pose",
                parameters=[{"use_sim_time": True}],
                output="screen",
            ),
            _static_transform(
                "livifuser_visual_probe_base_to_scan_tf",
                "base_link", "base_scan", (0.0, 0.0, 0.172), (0.0, 0.0, 0.0, 1.0),
            ),
            _static_transform(
                "livifuser_visual_probe_lidar_to_camera_tf",
                "base_scan", "camera", (0.0723955522, 0.0048472604, -0.0838973150),
                (-0.4806489642, 0.5212435451, -0.4930249275, 0.5041905996),
            ),
            _static_transform(
                "livifuser_visual_probe_camera_alias_tf",
                "camera", "camera_optical_frame", (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ]
    )
