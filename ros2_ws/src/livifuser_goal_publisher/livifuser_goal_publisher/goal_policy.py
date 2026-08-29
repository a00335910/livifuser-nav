"""Pure geometry for a waypoint fixed from the robot's first odometry pose."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RobotPose2D:
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x_m, self.y_m, self.yaw_rad)):
            raise ValueError("robot pose must be finite")


@dataclass(frozen=True, slots=True)
class RelativeGoalFields:
    rho_m: float
    sin_alpha: float
    cos_alpha: float


def yaw_from_quaternion(*, x: float, y: float, z: float, w: float) -> float:
    values = (x, y, z, w)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-9:
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = (value / norm for value in values)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def resolve_body_offset_waypoint(
    start: RobotPose2D,
    *,
    forward_m: float,
    left_m: float,
) -> tuple[float, float]:
    if not all(math.isfinite(value) for value in (forward_m, left_m)):
        raise ValueError("waypoint offsets must be finite")
    if math.hypot(forward_m, left_m) <= 0.25:
        raise ValueError("waypoint must begin more than 0.25 m from the robot")
    cosine = math.cos(start.yaw_rad)
    sine = math.sin(start.yaw_rad)
    return (
        start.x_m + cosine * forward_m - sine * left_m,
        start.y_m + sine * forward_m + cosine * left_m,
    )


def relative_goal_from_world_waypoint(
    robot: RobotPose2D,
    *,
    target_x_m: float,
    target_y_m: float,
) -> RelativeGoalFields:
    if not all(math.isfinite(value) for value in (target_x_m, target_y_m)):
        raise ValueError("world waypoint must be finite")
    delta_x = target_x_m - robot.x_m
    delta_y = target_y_m - robot.y_m
    cosine = math.cos(robot.yaw_rad)
    sine = math.sin(robot.yaw_rad)
    forward = cosine * delta_x + sine * delta_y
    left = -sine * delta_x + cosine * delta_y
    rho_m = math.hypot(forward, left)
    if rho_m <= 1e-9:
        return RelativeGoalFields(0.0, 0.0, 1.0)
    return RelativeGoalFields(rho_m, left / rho_m, forward / rho_m)


class LockedRelativeWaypoint:
    """Resolve one body-frame offset once, then update its relative bearing."""

    def __init__(self, *, forward_m: float, left_m: float) -> None:
        if not all(math.isfinite(value) for value in (forward_m, left_m)):
            raise ValueError("waypoint offsets must be finite")
        self.forward_m = float(forward_m)
        self.left_m = float(left_m)
        self.target_xy_m: tuple[float, float] | None = None

    def update(self, robot: RobotPose2D) -> RelativeGoalFields:
        if self.target_xy_m is None:
            self.target_xy_m = resolve_body_offset_waypoint(
                robot,
                forward_m=self.forward_m,
                left_m=self.left_m,
            )
        return relative_goal_from_world_waypoint(
            robot,
            target_x_m=self.target_xy_m[0],
            target_y_m=self.target_xy_m[1],
        )
