"""Publish renderer-independent analytic scans for the controlled sim world."""

from __future__ import annotations

import math
import random
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from .analytic_lidar import Pose2D
from .observation_model import load_observation_model, simulate_observation
from .world_layers import LAYER_LIDAR, geometry_for_layer, load_world


def _yaw_from_odometry(message: Odometry) -> float:
    quaternion = message.pose.pose.orientation
    sin_yaw = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cos_yaw = 1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2)
    return math.atan2(sin_yaw, cos_yaw)


class AnalyticLidarNode(Node):
    def __init__(self) -> None:
        super().__init__("livifuser_sim_analytic_lidar")
        default_path = (
            Path(get_package_share_directory("livifuser_sim"))
            / "config"
            / "livifuser_lab_world_v2.json"
        )
        default_observation_path = (
            Path(get_package_share_directory("livifuser_sim"))
            / "config"
            / "lds03_observation_model_v1.json"
        )
        self.declare_parameter("geometry_path", str(default_path))
        self.declare_parameter("observation_model_path", str(default_observation_path))
        self.declare_parameter("observation_seed", 20260821)
        self.declare_parameter("condition", "C0")
        geometry_path = Path(
            self.get_parameter("geometry_path").get_parameter_value().string_value
        )
        observation_path = Path(
            self.get_parameter("observation_model_path")
            .get_parameter_value()
            .string_value
        )
        observation_seed = (
            self.get_parameter("observation_seed").get_parameter_value().integer_value
        )
        self._observation_condition = str(self.get_parameter("condition").value)
        if self._observation_condition not in {"C0", "C3a", "C3b"}:
            raise ValueError(
                "unsupported LiDAR observation condition: "
                f"{self._observation_condition}"
            )
        # Cast against the LiDAR layer alone. Using the full obstacle set would
        # put LiDAR-invisible obstacles back into the scan and destroy C4.
        self._world = load_world(geometry_path)
        self._geometry = geometry_for_layer(self._world, LAYER_LIDAR)
        self._observation_model = load_observation_model(observation_path)
        self._generator = random.Random(observation_seed)
        self._pose: Pose2D | None = None
        self._publisher = self.create_publisher(
            LaserScan, "/scan", qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry,
            "/livifuser/sim/ground_truth/odom",
            self._on_world_pose,
            qos_profile_sensor_data,
        )
        self.create_timer(self._observation_model.scan_interval_sec, self._publish)
        hidden = [obstacle.name for obstacle in self._world.lidar_invisible]
        self.get_logger().info(
            "publishing analytic simulation LiDAR with measured LDS-03 "
            f"observation model {self._observation_model.name}; seed={observation_seed}; "
            f"observation_condition={self._observation_condition}; "
            f"world_condition={self._world.condition}; "
            f"world={self._world.name}; "
            f"scan sees {len(self._geometry.obstacles)}/{len(self._world.obstacles)} "
            f"obstacles; LiDAR-invisible={hidden or 'none'}; "
            "this is not a Gazebo GPU sensor"
        )

    def _on_world_pose(self, message: Odometry) -> None:
        if message.header.frame_id != "world":
            self._pose = None
            return
        position = message.pose.pose.position
        self._pose = Pose2D(position.x, position.y, _yaw_from_odometry(message))

    def _publish(self) -> None:
        if self._pose is None:
            return
        scan = simulate_observation(
            self._geometry,
            self._pose,
            self._observation_model,
            self._generator,
            self._observation_condition,
        )
        specification = scan.specification
        message = LaserScan()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = specification.frame_id
        message.angle_min = specification.angle_min_rad
        message.angle_max = specification.angle_max_rad
        message.angle_increment = specification.angle_increment_rad
        message.time_increment = specification.scan_time_sec / specification.beam_count
        message.scan_time = specification.scan_time_sec
        message.range_min = specification.range_min_m
        message.range_max = specification.range_max_m
        message.ranges = list(scan.ranges)
        self._publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = AnalyticLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
