"""Publish the accepted camera-LiDAR extrinsic and optical-frame alias."""

from launch import LaunchDescription
from launch_ros.actions import Node

TRANSLATION_M = (0.0723955522, 0.0048472604, -0.0838973150)
QUATERNION_XYZW = (-0.4806489642, 0.5212435451, -0.4930249275, 0.5041905996)


def static_transform_node(
    name: str,
    parent: str,
    child: str,
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> Node:
    x, y, z = translation
    qx, qy, qz, qw = quaternion
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        arguments=[
            "--x",
            str(x),
            "--y",
            str(y),
            "--z",
            str(z),
            "--qx",
            str(qx),
            "--qy",
            str(qy),
            "--qz",
            str(qz),
            "--qw",
            str(qw),
            "--frame-id",
            parent,
            "--child-frame-id",
            child,
        ],
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            static_transform_node(
                "livifuser_lidar_to_camera_tf",
                "base_scan",
                "camera",
                TRANSLATION_M,
                QUATERNION_XYZW,
            ),
            static_transform_node(
                "livifuser_camera_optical_alias_tf",
                "camera",
                "camera_optical_frame",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ]
    )
