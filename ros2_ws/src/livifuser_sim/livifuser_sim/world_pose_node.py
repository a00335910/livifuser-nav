"""Publish Gazebo's physics-authoritative Burger pose as ROS odometry.

The DiffDrive system publishes wheel-integrated odometry even when the physical
model is obstructed or unstable.  That signal remains useful as an observation,
but it cannot certify motion or generate privileged labels.  This node extracts
the model transform from Gazebo's world pose vector and publishes it on a
separate, explicit ground-truth topic.
"""

from __future__ import annotations

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_msgs.msg import TFMessage

MODEL_FRAME_ID = "livifuser_burger"
GROUND_TRUTH_TOPIC = "/livifuser/sim/ground_truth/odom"
RAW_WORLD_POSE_TOPIC = "/livifuser/sim/raw/world_pose"


def _finite_transform(transform) -> bool:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    values = (
        translation.x,
        translation.y,
        translation.z,
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return False
    norm = math.sqrt(
        rotation.x**2 + rotation.y**2 + rotation.z**2 + rotation.w**2
    )
    return norm > 1e-9


def burger_transform(message: TFMessage):
    """Return the unique world transform for the Burger model, if valid."""

    matches = [
        transform
        for transform in message.transforms
        if transform.child_frame_id == MODEL_FRAME_ID
    ]
    if len(matches) != 1 or not _finite_transform(matches[0]):
        return None
    return matches[0]


class WorldPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("livifuser_sim_world_pose")
        self._publisher = self.create_publisher(
            Odometry, GROUND_TRUTH_TOPIC, qos_profile_sensor_data
        )
        self.create_subscription(
            TFMessage,
            RAW_WORLD_POSE_TOPIC,
            self._on_pose_vector,
            qos_profile_sensor_data,
        )
        self._warned_invalid = False

    def _on_pose_vector(self, message: TFMessage) -> None:
        transform = burger_transform(message)
        if transform is None:
            if not self._warned_invalid:
                self.get_logger().error(
                    "Gazebo world pose vector lacks one valid livifuser_burger transform"
                )
                self._warned_invalid = True
            return
        self._warned_invalid = False
        output = Odometry()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "world"
        output.child_frame_id = "base_link"
        output.pose.pose.position.x = transform.transform.translation.x
        output.pose.pose.position.y = transform.transform.translation.y
        output.pose.pose.position.z = transform.transform.translation.z
        output.pose.pose.orientation = transform.transform.rotation
        self._publisher.publish(output)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = WorldPoseNode()
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
