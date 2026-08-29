"""Launch one verified policy identity behind the simulation-only supervisor."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from livifuser_nav.live_runtime import CONSTANT_ARM_NAME


def _runtime_actions(context):
    package_share = Path(get_package_share_directory("livifuser_sim"))
    geometry_path = Path(LaunchConfiguration("geometry_path").perform(context))
    world_sdf_path = Path(LaunchConfiguration("world_sdf_path").perform(context))
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    start_x, start_y, start_yaw = (float(value) for value in payload["start_pose_xy_yaw"])
    goal_x, goal_y = (float(value) for value in payload["goal_xy_m"])
    delta_x, delta_y = goal_x - start_x, goal_y - start_y
    goal_forward = math.cos(start_yaw) * delta_x + math.sin(start_yaw) * delta_y
    goal_left = -math.sin(start_yaw) * delta_x + math.cos(start_yaw) * delta_y
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(package_share / "launch/fortress_burger.launch.py")),
        launch_arguments={
            "world_sdf_path": str(world_sdf_path),
            "geometry_path": str(geometry_path),
            "observation_seed": LaunchConfiguration("observation_seed"),
            "lidar_condition": LaunchConfiguration("lidar_condition"),
            "goal_forward_m": str(goal_forward),
            "goal_left_m": str(goal_left),
            "start_expert": "false",
        }.items(),
    )
    # The constant reference arm of closed-loop execution amendment section 1.1
    # runs a different executable, not the learned runner with sensing disabled.
    # It takes no bundle, no device, and no seed, so there is no code path by
    # which a backbone, a checkpoint, or a sensor subscription could reach it.
    variant = LaunchConfiguration("variant").perform(context)
    # 100 ms control period / real-time factor, times a 20x allowance. The
    # watchdog is a liveness check on the policy process, not a timing gate, and
    # Gazebo's /clock stalls briefly under rendering load: with a 4x allowance a
    # healthy episode ended in watchdog_timeout while publishing at exactly the
    # expected rate. At factor 1.0 this is 2 s; at 0.4 it is 5 s.
    real_time_factor = float(LaunchConfiguration("real_time_factor").perform(context))
    stale_timeout_ms = max(2000.0, (100.0 / max(real_time_factor, 0.05)) * 20.0)
    if variant == CONSTANT_ARM_NAME:
        runner = Node(
            package="livifuser_sim_eval",
            executable="constant_arm_runner",
            name="livifuser_constant_arm_runner",
            parameters=[{"use_sim_time": True}],
            output="screen",
        )
    else:
        runner = Node(
            package="livifuser_sim_eval",
            executable="live_policy_runner",
            name="livifuser_live_policy_runner",
            parameters=[
                {
                    "use_sim_time": True,
                    "variant": LaunchConfiguration("variant"),
                    "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
                    "device": LaunchConfiguration("device"),
                    "backbone_bundle": LaunchConfiguration("backbone_bundle"),
                    "policy_bundle": LaunchConfiguration("policy_bundle"),
                    "backbone_extract_root": LaunchConfiguration("backbone_extract_root"),
                    "sensor_contract": LaunchConfiguration("sensor_contract"),
                }
            ],
            output="screen",
        )
    supervisor = Node(
        package="livifuser_sim_eval",
        executable="simulation_supervisor",
        name="livifuser_simulation_supervisor",
        parameters=[
            {
                "use_sim_time": True,
                "world_json": str(geometry_path),
                "scientific_deadline_sec": ParameterValue(
                    LaunchConfiguration("scientific_deadline_sec"), value_type=float
                ),
                "expected_variant": LaunchConfiguration("variant"),
                "expected_seed": ParameterValue(
                    LaunchConfiguration("seed"), value_type=int
                ),
                # The hung-policy watchdog is wall time, but the control loop
                # ticks in simulated time. Running the simulator below real time
                # stretches the wall gap between ticks by 1/factor, so the
                # threshold is scaled to match with headroom for a slow tick.
                "stale_proposal_wall_timeout_ms": stale_timeout_ms,
            }
        ],
        output="screen",
    )
    return [simulation, runner, supervisor]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("world_sdf_path"),
            DeclareLaunchArgument("geometry_path"),
            DeclareLaunchArgument("observation_seed"),
            DeclareLaunchArgument("lidar_condition"),
            DeclareLaunchArgument("variant"),
            DeclareLaunchArgument("seed", default_value="0"),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("backbone_bundle", default_value=""),
            DeclareLaunchArgument("policy_bundle", default_value=""),
            DeclareLaunchArgument("backbone_extract_root", default_value=""),
            DeclareLaunchArgument("sensor_contract", default_value=""),
            DeclareLaunchArgument("scientific_deadline_sec", default_value="120.0"),
            DeclareLaunchArgument("real_time_factor", default_value="1.0"),
            OpaqueFunction(function=_runtime_actions),
        ]
    )
