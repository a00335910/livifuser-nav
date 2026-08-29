"""Normalize Gazebo messages to the recorded real-robot topic contracts."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

CAMERA_K = [316.21156, 0.0, 223.13834, 0.0, 315.6497, 107.39364, 0.0, 0.0, 1.0]
CAMERA_D = [0.012344, 0.038138, -0.016819, 0.004823, 0.0]
CAMERA_R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
CAMERA_P = [
    321.69766,
    0.0,
    222.50471,
    0.0,
    0.0,
    318.87292,
    102.86607,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
]


def calibrated_camera_info(image: Image) -> CameraInfo:
    message = CameraInfo()
    message.header = image.header
    message.header.frame_id = "camera"
    message.height = 240
    message.width = 320
    message.distortion_model = "plumb_bob"
    message.d = CAMERA_D
    message.k = CAMERA_K
    message.r = CAMERA_R
    message.p = CAMERA_P
    return message


class SimulationContractNode(Node):
    def __init__(self) -> None:
        super().__init__("livifuser_sim_contract")
        self._image_publisher = self.create_publisher(
            Image, "/camera/image_raw", qos_profile_sensor_data
        )
        self._camera_info_publisher = self.create_publisher(
            CameraInfo, "/camera/camera_info", qos_profile_sensor_data
        )
        self._odom_publisher = self.create_publisher(
            Odometry, "/odom", qos_profile_sensor_data
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            Image, "/livifuser/sim/raw/image", self._on_image, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, "/livifuser/sim/raw/odom", self._on_odom, qos_profile_sensor_data
        )

    def _on_image(self, message: Image) -> None:
        message.header.frame_id = "camera"
        self._image_publisher.publish(message)
        self._camera_info_publisher.publish(calibrated_camera_info(message))

    def _on_odom(self, message: Odometry) -> None:
        message.header.frame_id = "odom"
        message.child_frame_id = "base_link"
        self._odom_publisher.publish(message)
        transform = TransformStamped()
        transform.header = message.header
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = message.pose.pose.position.x
        transform.transform.translation.y = message.pose.pose.position.y
        transform.transform.translation.z = message.pose.pose.position.z
        transform.transform.rotation = message.pose.pose.orientation
        self._tf_broadcaster.sendTransform(transform)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SimulationContractNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
