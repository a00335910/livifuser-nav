"""Publish a dynamic robot-frame goal fixed from the first valid odometry pose."""

from __future__ import annotations

import math
import time

import rclpy
from livifuser_interfaces.msg import RelativeGoal
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .goal_policy import LockedRelativeWaypoint, RobotPose2D, yaw_from_quaternion


class OdomWaypointGoalPublisher(Node):
    def __init__(self) -> None:
        super().__init__("odom_waypoint_goal_publisher")
        self.declare_parameter("forward_m", 3.0)
        self.declare_parameter("left_m", 0.0)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("odom_timeout_ms", 250.0)
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("goal_frame_id", "base_link")

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.odom_timeout_s = float(self.get_parameter("odom_timeout_ms").value) / 1000.0
        self.odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self.goal_frame_id = str(self.get_parameter("goal_frame_id").value)
        if not math.isfinite(self.rate_hz) or not 5.0 <= self.rate_hz <= 20.0:
            raise ValueError("rate_hz must be finite and in [5, 20]")
        if not math.isfinite(self.odom_timeout_s) or self.odom_timeout_s <= 0.0:
            raise ValueError("odom_timeout_ms must be finite and positive")
        if not self.odom_frame_id or not self.goal_frame_id:
            raise ValueError("odometry and goal frame IDs must not be empty")

        self._waypoint = LockedRelativeWaypoint(
            forward_m=float(self.get_parameter("forward_m").value),
            left_m=float(self.get_parameter("left_m").value),
        )
        self._latest_pose: RobotPose2D | None = None
        self._odom_arrival_s: float | None = None
        self._target_logged = False
        self._publisher = self.create_publisher(
            RelativeGoal,
            "/livifuser/goal_relative",
            10,
        )
        self.create_subscription(
            Odometry,
            "/odom",
            self._on_odom,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / self.rate_hz, self._publish_goal)

    def _on_odom(self, message: Odometry) -> None:
        if message.header.frame_id != self.odom_frame_id:
            self._latest_pose = None
            self._odom_arrival_s = None
            return
        orientation = message.pose.pose.orientation
        try:
            yaw_rad = yaw_from_quaternion(
                x=float(orientation.x),
                y=float(orientation.y),
                z=float(orientation.z),
                w=float(orientation.w),
            )
            position = message.pose.pose.position
            self._latest_pose = RobotPose2D(
                float(position.x),
                float(position.y),
                yaw_rad,
            )
        except ValueError:
            self._latest_pose = None
            self._odom_arrival_s = None
            return
        self._odom_arrival_s = time.monotonic()

    def _publish_goal(self) -> None:
        now = time.monotonic()
        if (
            self._latest_pose is None
            or self._odom_arrival_s is None
            or now - self._odom_arrival_s < 0.0
            or now - self._odom_arrival_s > self.odom_timeout_s
        ):
            return
        goal = self._waypoint.update(self._latest_pose)
        if not self._target_logged and self._waypoint.target_xy_m is not None:
            self.get_logger().info(
                "Locked odom waypoint at "
                f"x={self._waypoint.target_xy_m[0]:.3f}, "
                f"y={self._waypoint.target_xy_m[1]:.3f} m"
            )
            self._target_logged = True
        message = RelativeGoal()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.goal_frame_id
        message.rho_m = goal.rho_m
        message.sin_alpha = goal.sin_alpha
        message.cos_alpha = goal.cos_alpha
        self._publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OdomWaypointGoalPublisher()
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
