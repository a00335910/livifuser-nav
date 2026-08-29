"""Launch the isolated LiViFuser Burger simulation on Gazebo Fortress."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _require_isolation(_context):
    if os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError("simulation requires ROS_LOCALHOST_ONLY=1")
    if os.environ.get("ROS_DOMAIN_ID") != "97":
        raise RuntimeError("simulation requires the reserved ROS_DOMAIN_ID=97")
    return []


def _world_pose_bridge(context):
    geometry_path = Path(LaunchConfiguration("geometry_path").perform(context))
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    world_name = str(payload.get("name", ""))
    if not re.fullmatch(r"[A-Za-z0-9_]+", world_name):
        raise RuntimeError(f"world name is not ROS-topic safe: {world_name!r}")
    gazebo_topic = f"/world/{world_name}/dynamic_pose/info"
    return [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="livifuser_sim_world_pose_bridge",
            arguments=[
                f"{gazebo_topic}@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V"
            ],
            remappings=[(gazebo_topic, "/livifuser/sim/raw/world_pose")],
            parameters=[{"use_sim_time": True}],
            output="screen",
        )
    ]


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
            "--x",
            str(x),
            "--y",
            str(y),
            "--z",
            str(z),
            "--qx",
            str(qx),
            "--qy",
            str(qy),
            "--qz",
            str(qz),
            "--qw",
            str(qw),
            "--frame-id",
            parent,
            "--child-frame-id",
            child,
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("livifuser_sim"))
    ros_gz_share = Path(get_package_share_directory("ros_gz_sim"))
    default_world = package_share / "worlds" / "livifuser_lab.sdf"
    default_geometry = package_share / "config" / "livifuser_lab_world_v2.json"
    model_path = str(package_share / "models")
    existing_resource_path = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    resource_path = (
        model_path
        if not existing_resource_path
        else f"{model_path}{os.pathsep}{existing_resource_path}"
    )
    world = LaunchConfiguration("world_sdf_path")
    geometry = LaunchConfiguration("geometry_path")
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(ros_gz_share / "launch" / "gz_sim.launch.py")),
        launch_arguments={
            "gz_args": ["-r -s --headless-rendering --render-engine ogre ", world],
            "gz_version": "6",
            "on_exit_shutdown": "true",
        }.items(),
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="livifuser_sim_bridge",
        arguments=[
            "/camera@sensor_msgs/msg/Image[ignition.msgs.Image",
            (
                "/model/livifuser_burger/odometry@nav_msgs/msg/Odometry"
                "[ignition.msgs.Odometry"
            ),
            (
                "/model/livifuser_burger/cmd_vel@geometry_msgs/msg/Twist"
                "]ignition.msgs.Twist"
            ),
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        ],
        remappings=[
            ("/camera", "/livifuser/sim/raw/image"),
            ("/model/livifuser_burger/odometry", "/livifuser/sim/raw/odom"),
            ("/model/livifuser_burger/cmd_vel", "/livifuser/sim_cmd_vel"),
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    contract = Node(
        package="livifuser_sim",
        executable="contract_node",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    world_pose = Node(
        package="livifuser_sim",
        executable="world_pose",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    analytic_lidar = Node(
        package="livifuser_sim",
        executable="analytic_lidar",
        parameters=[
            {
                "use_sim_time": True,
                "geometry_path": geometry,
                "observation_seed": ParameterValue(
                    LaunchConfiguration("observation_seed"), value_type=int
                ),
                "condition": LaunchConfiguration("lidar_condition"),
            }
        ],
        output="screen",
    )
    goal = Node(
        package="livifuser_goal_publisher",
        executable="odom_waypoint_goal_publisher",
        parameters=[
            {
                "use_sim_time": True,
                "forward_m": ParameterValue(
                    LaunchConfiguration("goal_forward_m"), value_type=float
                ),
                "left_m": ParameterValue(
                    LaunchConfiguration("goal_left_m"), value_type=float
                ),
                "rate_hz": 10.0,
                "odom_timeout_ms": 250.0,
                "odom_frame_id": "world",
            }
        ],
        remappings=[("/odom", "/livifuser/sim/ground_truth/odom")],
        output="screen",
    )
    expert = Node(
        package="livifuser_sim",
        executable="privileged_expert",
        parameters=[{"use_sim_time": True, "geometry_path": geometry}],
        condition=IfCondition(LaunchConfiguration("start_expert")),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world_sdf_path",
                default_value=str(default_world),
                description="Materialized Gazebo SDF world.",
            ),
            DeclareLaunchArgument(
                "geometry_path",
                default_value=str(default_geometry),
                description="Authoritative schema-2 layered world JSON.",
            ),
            DeclareLaunchArgument(
                "observation_seed",
                default_value="20260821",
                description="Measured LDS-03 observation RNG seed.",
            ),
            DeclareLaunchArgument(
                "lidar_condition",
                default_value="C0",
                description="Frozen analytic LiDAR observation condition: C0/C3a/C3b.",
            ),
            DeclareLaunchArgument("goal_forward_m", default_value="4.0"),
            DeclareLaunchArgument("goal_left_m", default_value="0.0"),
            DeclareLaunchArgument(
                "start_expert",
                default_value="false",
                description="Start the deterministic simulation-only expert.",
            ),
            OpaqueFunction(function=_require_isolation),
            SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", resource_path),
            gazebo,
            bridge,
            OpaqueFunction(function=_world_pose_bridge),
            contract,
            world_pose,
            analytic_lidar,
            goal,
            expert,
            _static_transform(
                "livifuser_sim_base_to_scan_tf",
                "base_link",
                "base_scan",
                (0.0, 0.0, 0.172),
                (0.0, 0.0, 0.0, 1.0),
            ),
            _static_transform(
                "livifuser_sim_lidar_to_camera_tf",
                "base_scan",
                "camera",
                (0.0723955522, 0.0048472604, -0.0838973150),
                (-0.4806489642, 0.5212435451, -0.4930249275, 0.5041905996),
            ),
            _static_transform(
                "livifuser_sim_camera_optical_alias_tf",
                "camera",
                "camera_optical_frame",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ]
    )
