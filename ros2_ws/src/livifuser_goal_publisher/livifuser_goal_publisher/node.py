"""Publish a validated robot-frame waypoint at the locked 10 Hz sampling rate."""

from __future__ import annotations

import math

import rclpy
from livifuser_interfaces.msg import RelativeGoal
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter


class RelativeGoalPublisher(Node):
    def __init__(self) -> None:
        super().__init__("relative_goal_publisher")
        self.declare_parameter("rho_m", 1.0)
        self.declare_parameter("alpha_rad", 0.0)
        self.declare_parameter("frame_id", "base_link")

        self._publisher = self.create_publisher(
            RelativeGoal,
            "/livifuser/goal_relative",
            10,
        )
        self.add_on_set_parameters_callback(self._validate_parameters)
        self.create_timer(0.1, self._publish_goal)

        self.get_logger().info(
            "Publishing /livifuser/goal_relative at 10 Hz; "
            "set rho_m and alpha_rad with ros2 param set."
        )

    def _validate_parameters(self, parameters: list[Parameter]) -> SetParametersResult:
        for parameter in parameters:
            if parameter.name == "rho_m":
                if not isinstance(parameter.value, float | int):
                    return SetParametersResult(successful=False, reason="rho_m must be numeric")
                if not math.isfinite(parameter.value) or parameter.value < 0:
                    return SetParametersResult(
                        successful=False,
                        reason="rho_m must be finite and non-negative",
                    )
            elif parameter.name == "alpha_rad":
                if not isinstance(parameter.value, float | int):
                    return SetParametersResult(successful=False, reason="alpha_rad must be numeric")
                if not math.isfinite(parameter.value):
                    return SetParametersResult(
                        successful=False,
                        reason="alpha_rad must be finite",
                    )
            elif parameter.name == "frame_id":
                if not isinstance(parameter.value, str) or not parameter.value.strip():
                    return SetParametersResult(
                        successful=False,
                        reason="frame_id must be a non-empty string",
                    )

        return SetParametersResult(successful=True)

    def _publish_goal(self) -> None:
        rho_m = float(self.get_parameter("rho_m").value)
        alpha_rad = float(self.get_parameter("alpha_rad").value)

        message = RelativeGoal()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = str(self.get_parameter("frame_id").value)
        message.rho_m = rho_m
        message.sin_alpha = math.sin(alpha_rad)
        message.cos_alpha = math.cos(alpha_rad)
        self._publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RelativeGoalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

