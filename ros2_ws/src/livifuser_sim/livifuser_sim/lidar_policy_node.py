"""ROS deployment wrapper for the development-only ONNX LiDAR policy."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from livifuser_interfaces.msg import RelativeGoal
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from .lidar_policy_features import lidar_only_features

CONTEXT_K = 8
LIDAR_SECTORS = 80
EXPECTED_INPUTS = {
    "lidar_features": [1, CONTEXT_K, LIDAR_SECTORS, 4],
    "goal": [1, CONTEXT_K, 3],
    "robot_state": [1, CONTEXT_K, 2],
}


class LidarPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("livifuser_sim_lidar_policy")
        self.declare_parameter("model_path", "")
        self.declare_parameter("input_timeout_ms", 250.0)
        model_path = Path(str(self.get_parameter("model_path").value))
        if not model_path.is_file():
            raise ValueError(f"LiDAR policy ONNX does not exist: {model_path}")
        self._timeout_s = float(self.get_parameter("input_timeout_ms").value) / 1000.0
        if self._timeout_s <= 0.0:
            raise ValueError("input_timeout_ms must be positive")

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        observed = {item.name: item.shape for item in self._session.get_inputs()}
        if observed != EXPECTED_INPUTS:
            raise ValueError(f"unexpected LiDAR policy input contract: {observed}")

        self._latest_goal: tuple[np.ndarray, float] | None = None
        self._latest_state: tuple[np.ndarray, float] | None = None
        self._history: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=CONTEXT_K
        )
        self._command_publisher = self.create_publisher(
            Twist, "/livifuser/sim_cmd_vel", 10
        )
        self._stamped_publisher = self.create_publisher(
            TwistStamped, "/livifuser/cmd_vel_stamped", 10
        )
        self.create_subscription(
            RelativeGoal, "/livifuser/goal_relative", self._on_goal, 10
        )
        self.create_subscription(
            Odometry, "/odom", self._on_odometry, qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan, "/scan", self._on_scan, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"development-only LiDAR policy loaded: {model_path}; "
            "RGB subscriptions=none; command envelope is embedded in ONNX"
        )

    def _on_goal(self, message: RelativeGoal) -> None:
        self._latest_goal = (
            np.asarray(
                [message.rho_m, message.sin_alpha, message.cos_alpha],
                dtype=np.float32,
            ),
            time.monotonic(),
        )

    def _on_odometry(self, message: Odometry) -> None:
        twist = message.twist.twist
        self._latest_state = (
            np.asarray([twist.linear.x, twist.angular.z], dtype=np.float32),
            time.monotonic(),
        )

    def _publish(self, linear: float, angular: float, stamp) -> None:
        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)
        self._command_publisher.publish(command)
        stamped = TwistStamped()
        stamped.header.stamp = stamp
        stamped.header.frame_id = "base_link"
        stamped.twist = command
        self._stamped_publisher.publish(stamped)

    def _on_scan(self, message: LaserScan) -> None:
        now = time.monotonic()
        if (
            self._latest_goal is None
            or self._latest_state is None
            or now - self._latest_goal[1] > self._timeout_s
            or now - self._latest_state[1] > self._timeout_s
        ):
            self._history.clear()
            self._publish(0.0, 0.0, message.header.stamp)
            return
        features = lidar_only_features(
            message.ranges,
            angle_min_rad=float(message.angle_min),
            angle_increment_rad=float(message.angle_increment),
            range_min_m=float(message.range_min),
            range_max_m=float(message.range_max),
            sectors=LIDAR_SECTORS,
            range_clip_m=10.0,
        )
        self._history.append(
            (features, self._latest_goal[0].copy(), self._latest_state[0].copy())
        )
        if len(self._history) < CONTEXT_K:
            self._publish(0.0, 0.0, message.header.stamp)
            return
        inputs = {
            "lidar_features": np.stack([item[0] for item in self._history])[None],
            "goal": np.stack([item[1] for item in self._history])[None],
            "robot_state": np.stack([item[2] for item in self._history])[None],
        }
        mean, _log_variance = self._session.run(None, inputs)
        linear, angular = mean[0, 0]
        self._publish(float(linear), float(angular), message.header.stamp)

    def destroy_node(self) -> bool:
        self._publish(0.0, 0.0, self.get_clock().now().to_msg())
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LidarPolicyNode()
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
