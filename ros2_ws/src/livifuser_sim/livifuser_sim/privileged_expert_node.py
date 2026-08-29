"""ROS wrapper for the geometry-only privileged simulation expert.

This node deliberately subscribes only to Gazebo's physics-authoritative world
pose.  In particular it never subscribes to wheel odometry, ``/scan``, or camera
topics, so sensor corruptions and odometry drift cannot alter expert labels.
It publishes commands exclusively on the isolated simulation command topic and
the stamped training-label topic.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .privileged_expert import (
    build_clearance_field,
    follow_path,
    plan_path,
)
from .world_layers import load_world


def _yaw(message: Odometry) -> float:
    quaternion = message.pose.pose.orientation
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2),
    )


class PrivilegedExpertNode(Node):
    def __init__(self) -> None:
        super().__init__("livifuser_sim_privileged_expert")
        default_world = (
            Path(get_package_share_directory("livifuser_sim"))
            / "config"
            / "livifuser_lab_world_v2.json"
        )
        self.declare_parameter("geometry_path", str(default_world))
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("odom_timeout_ms", 250.0)
        geometry_path = Path(str(self.get_parameter("geometry_path").value))
        rate_hz = float(self.get_parameter("rate_hz").value)
        self._pose_timeout_s = (
            float(self.get_parameter("odom_timeout_ms").value) / 1000.0
        )
        if not 5.0 <= rate_hz <= 20.0:
            raise ValueError("rate_hz must be in [5, 20]")
        if self._pose_timeout_s <= 0.0:
            raise ValueError("odom_timeout_ms must be positive")

        self._world = load_world(geometry_path)
        if self._world.start_pose_xy_yaw is None or self._world.goal_xy_m is None:
            raise ValueError("privileged expert requires generated start and goal")
        self._goal = self._world.goal_xy_m
        self._field = build_clearance_field(self._world)
        self._path = plan_path(
            self._field,
            self._world.start_pose_xy_yaw[:2],
            self._goal,
        )
        if self._path is None:
            raise ValueError("privileged expert cannot plan a path in the world")

        self._pose: tuple[float, float, float] | None = None
        self._pose_arrival_s: float | None = None
        self._last_reason: str | None = None
        self._command_publisher = self.create_publisher(
            Twist, "/livifuser/sim_cmd_vel", 10
        )
        self._label_publisher = self.create_publisher(
            TwistStamped, "/livifuser/cmd_vel_stamped", 10
        )
        self.create_subscription(
            Odometry,
            "/livifuser/sim/ground_truth/odom",
            self._on_world_pose,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / rate_hz, self._publish)
        self.get_logger().info(
            f"geometry-only privileged expert: world={self._world.name}; "
            f"goal=({self._goal[0]:.3f}, {self._goal[1]:.3f}); "
            "sensor subscriptions=none"
        )

    def _on_world_pose(self, message: Odometry) -> None:
        if message.header.frame_id != "world":
            self._pose = None
            self._pose_arrival_s = None
            return
        position = message.pose.pose.position
        self._pose = (float(position.x), float(position.y), _yaw(message))
        self._pose_arrival_s = time.monotonic()

    def _publish(self) -> None:
        now = time.monotonic()
        command = Twist()
        reason = "world_pose_unavailable"
        if (
            self._pose is not None
            and self._pose_arrival_s is not None
            and 0.0 <= now - self._pose_arrival_s <= self._pose_timeout_s
        ):
            x_m, y_m, yaw_rad = self._pose
            clearance = self._field.clearance_at(x_m, y_m)
            decision = follow_path(
                self._path,
                x_m,
                y_m,
                yaw_rad,
                self._goal,
                clearance,
            )
            command.linear.x = decision.linear_mps
            command.angular.z = decision.angular_radps
            reason = decision.reason
        self._command_publisher.publish(command)
        label = TwistStamped()
        label.header.stamp = self.get_clock().now().to_msg()
        label.header.frame_id = "base_link"
        label.twist = command
        self._label_publisher.publish(label)
        if reason != self._last_reason:
            self.get_logger().info(f"expert state: {reason}")
            self._last_reason = reason

    def destroy_node(self) -> bool:
        self._command_publisher.publish(Twist())
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PrivilegedExpertNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
