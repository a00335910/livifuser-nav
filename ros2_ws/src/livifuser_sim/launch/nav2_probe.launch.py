"""Launch the isolated, competence-gated Nav2 structural probe."""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _require_isolation(_context):
    if os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError("Nav2 probe requires ROS_LOCALHOST_ONLY=1")
    if os.environ.get("ROS_DOMAIN_ID") != "97":
        raise RuntimeError("Nav2 probe requires the reserved ROS_DOMAIN_ID=97")
    return []


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("livifuser_sim"))
    sim_launch = package_share / "launch" / "fortress_burger.launch.py"
    params = LaunchConfiguration("params_file")
    map_yaml = LaunchConfiguration("map_yaml_path")
    geometry = LaunchConfiguration("geometry_path")
    common_parameters = [params]

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(sim_launch)),
        launch_arguments={
            "start_expert": "false",
            "world_sdf_path": LaunchConfiguration("world_sdf_path"),
            "geometry_path": geometry,
            "observation_seed": LaunchConfiguration("observation_seed"),
            "lidar_condition": "C0",
            "goal_forward_m": LaunchConfiguration("goal_forward_m"),
            "goal_left_m": LaunchConfiguration("goal_left_m"),
        }.items(),
    )
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        parameters=[params, {"yaml_filename": map_yaml, "use_sim_time": True}],
        output="screen",
    )
    controller = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        parameters=common_parameters,
        remappings=[("cmd_vel", "/livifuser/nav2_cmd_vel")],
        output="screen",
    )
    planner = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        parameters=common_parameters,
        output="screen",
    )
    behaviors = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        parameters=common_parameters,
        remappings=[("cmd_vel", "/livifuser/nav2_cmd_vel")],
        output="screen",
    )
    navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        parameters=common_parameters,
        output="screen",
    )
    lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_nav2_probe",
        parameters=[
            {
                "use_sim_time": True,
                "autostart": True,
                "bond_timeout": 10.0,
                "node_names": [
                    "map_server",
                    "controller_server",
                    "planner_server",
                    "behavior_server",
                    "bt_navigator",
                ],
            }
        ],
        output="screen",
    )
    map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="livifuser_nav2_map_to_odom",
        arguments=[
            "--x",
            LaunchConfiguration("map_to_odom_x"),
            "--y",
            LaunchConfiguration("map_to_odom_y"),
            "--z",
            "0.0",
            "--yaw",
            LaunchConfiguration("map_to_odom_yaw"),
            "--pitch",
            "0.0",
            "--roll",
            "0.0",
            "--frame-id",
            "map",
            "--child-frame-id",
            "odom",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    probe = Node(
        package="livifuser_sim",
        executable="nav2_probe",
        parameters=[
            {
                "use_sim_time": True,
                "geometry_path": geometry,
                "status_path": LaunchConfiguration("status_path"),
                "condition": LaunchConfiguration("condition"),
                "max_linear_mps": 0.08,
                "max_angular_radps": 0.40,
            }
        ],
        output="screen",
    )

    arguments = [
        DeclareLaunchArgument("world_sdf_path"),
        DeclareLaunchArgument("geometry_path"),
        DeclareLaunchArgument("map_yaml_path"),
        DeclareLaunchArgument("params_file"),
        DeclareLaunchArgument("status_path"),
        DeclareLaunchArgument("condition", default_value="C0"),
        DeclareLaunchArgument("observation_seed"),
        DeclareLaunchArgument("goal_forward_m"),
        DeclareLaunchArgument("goal_left_m"),
        DeclareLaunchArgument("map_to_odom_x"),
        DeclareLaunchArgument("map_to_odom_y"),
        DeclareLaunchArgument("map_to_odom_yaw"),
    ]
    return LaunchDescription(
        arguments
        + [
            OpaqueFunction(function=_require_isolation),
            simulator,
            map_to_odom,
            map_server,
            controller,
            planner,
            behaviors,
            navigator,
            lifecycle,
            probe,
        ]
    )
