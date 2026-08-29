"""Explicit transforms between Gazebo-relative odometry and world geometry."""

from __future__ import annotations

import math


def compose_world_pose(
    start_pose_xy_yaw: tuple[float, float, float] | None,
    odom_pose_xy_yaw: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Place Gazebo's start-relative odometry into generated-world coordinates."""

    if start_pose_xy_yaw is None:
        return odom_pose_xy_yaw
    start_x, start_y, start_yaw = start_pose_xy_yaw
    odom_x, odom_y, odom_yaw = odom_pose_xy_yaw
    cosine = math.cos(start_yaw)
    sine = math.sin(start_yaw)
    return (
        start_x + cosine * odom_x - sine * odom_y,
        start_y + sine * odom_x + cosine * odom_y,
        math.atan2(
            math.sin(start_yaw + odom_yaw),
            math.cos(start_yaw + odom_yaw),
        ),
    )
