"""Publish deterministic labels and commands on simulation-only topics.

**DISQUALIFIED for sensor-failure episode generation** — see the module
docstring of ``expert_policy``. This node subscribes to ``/scan`` and labels
from it, so any LiDAR corruption would corrupt the labels too. The
sensor-failure study labels from ``privileged_expert`` instead.
"""

from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from livifuser_interfaces.msg import RelativeGoal
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from .expert_policy import reactive_command


class ReactiveExpertNode(Node):
    def __init__(self) -> None:
        super().__init__("livifuser_sim_reactive_expert")
        self._scan: LaserScan | None = None
        self._scan_arrival_s: float | None = None
        self._goal: RelativeGoal | None = None
        self._goal_arrival_s: float | None = None
        self._command_publisher = self.create_publisher(
            Twist, "/livifuser/sim_cmd_vel", 10
        )
        self._label_publisher = self.create_publisher(
            TwistStamped, "/livifuser/cmd_vel_stamped", 10
        )
        self.create_subscription(
            LaserScan, "/scan", self._on_scan, qos_profile_sensor_data
        )
        self.create_subscription(
            RelativeGoal, "/livifuser/goal_relative", self._on_goal, 10
        )
        self.create_timer(0.1, self._publish)

    def _on_scan(self, message: LaserScan) -> None:
        self._scan = message
        self._scan_arrival_s = time.monotonic()

    def _on_goal(self, message: RelativeGoal) -> None:
        self._goal = message
        self._goal_arrival_s = time.monotonic()

    def _publish(self) -> None:
        now = time.monotonic()
        command = Twist()
        if (
            self._scan is not None
            and self._goal is not None
            and self._scan_arrival_s is not None
            and self._goal_arrival_s is not None
            and 0.0 <= now - self._scan_arrival_s <= 0.25
            and 0.0 <= now - self._goal_arrival_s <= 0.25
        ):
            scan = self._scan
            angles = [
                scan.angle_min + index * scan.angle_increment
                for index in range(len(scan.ranges))
            ]
            alpha = math.atan2(self._goal.sin_alpha, self._goal.cos_alpha)
            decision = reactive_command(
                scan.ranges,
                angles,
                goal_rho_m=float(self._goal.rho_m),
                goal_alpha_rad=alpha,
            )
            command.linear.x = decision.linear_mps
            command.angular.z = decision.angular_radps
        self._command_publisher.publish(command)
        label = TwistStamped()
        label.header.stamp = self.get_clock().now().to_msg()
        label.header.frame_id = "base_link"
        label.twist = command
        self._label_publisher.publish(label)

    def destroy_node(self) -> bool:
        zero = Twist()
        self._command_publisher.publish(zero)
        label = TwistStamped()
        label.header.stamp = self.get_clock().now().to_msg()
        label.header.frame_id = "base_link"
        self._label_publisher.publish(label)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ReactiveExpertNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
