"""Pure deterministic 2D geometry for the renderer-independent sim LiDAR.

This module intentionally contains no sensor noise.  The versioned measured
LDS-03 observation process lives in :mod:`livifuser_sim.observation_model`, one
layer downstream from ray casting, so nominal C0 sensing and later corruption
conditions cannot alter collision or expert geometry.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class LaserSpecification:
    beam_count: int
    angle_min_rad: float
    angle_max_rad: float
    scan_time_sec: float
    range_min_m: float
    range_max_m: float
    frame_id: str

    @property
    def angle_increment_rad(self) -> float:
        # Match the observed LDS-03 driver convention, including its +1 divisor.
        return (self.angle_max_rad - self.angle_min_rad) / (self.beam_count + 1)


@dataclass(frozen=True)
class BoxObstacle:
    name: str
    center_x_m: float
    center_y_m: float
    size_x_m: float
    size_y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class CircleObstacle:
    name: str
    center_x_m: float
    center_y_m: float
    radius_m: float


Obstacle: TypeAlias = BoxObstacle | CircleObstacle


@dataclass(frozen=True)
class AnalyticLidarGeometry:
    schema_version: int
    source: str
    laser: LaserSpecification
    obstacles: tuple[Obstacle, ...]


def _finite(value: object, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _xy(value: object, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must contain exactly two numbers")
    return _finite(value[0], field), _finite(value[1], field)


def load_geometry(path: Path) -> AnalyticLidarGeometry:
    """Load and validate the versioned simulation geometry contract."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported analytic LiDAR geometry schema")
    laser_data = payload["laser"]
    beam_count = int(laser_data["beam_count"])
    laser = LaserSpecification(
        beam_count=beam_count,
        angle_min_rad=_finite(laser_data["angle_min_rad"], "angle_min_rad"),
        angle_max_rad=_finite(laser_data["angle_max_rad"], "angle_max_rad"),
        scan_time_sec=_finite(laser_data["scan_time_sec"], "scan_time_sec"),
        range_min_m=_finite(laser_data["range_min_m"], "range_min_m"),
        range_max_m=_finite(laser_data["range_max_m"], "range_max_m"),
        frame_id=str(laser_data["frame_id"]),
    )
    if beam_count <= 0 or laser.scan_time_sec <= 0.0:
        raise ValueError("beam_count and scan_time_sec must be positive")
    if not 0.0 < laser.range_min_m < laser.range_max_m:
        raise ValueError("range limits must be positive and ordered")
    if laser.angle_max_rad <= laser.angle_min_rad:
        raise ValueError("angle limits must be ordered")

    obstacles: list[Obstacle] = []
    names: set[str] = set()
    for item in payload["obstacles"]:
        name = str(item["name"])
        if name in names:
            raise ValueError(f"duplicate obstacle name: {name}")
        names.add(name)
        center_x, center_y = _xy(item["center_xy_m"], f"{name}.center_xy_m")
        if item["type"] == "box":
            size_x, size_y = _xy(item["size_xy_m"], f"{name}.size_xy_m")
            if size_x <= 0.0 or size_y <= 0.0:
                raise ValueError(f"{name} box dimensions must be positive")
            obstacles.append(
                BoxObstacle(
                    name,
                    center_x,
                    center_y,
                    size_x,
                    size_y,
                    _finite(item["yaw_rad"], f"{name}.yaw_rad"),
                )
            )
        elif item["type"] == "circle":
            radius = _finite(item["radius_m"], f"{name}.radius_m")
            if radius <= 0.0:
                raise ValueError(f"{name} radius must be positive")
            obstacles.append(CircleObstacle(name, center_x, center_y, radius))
        else:
            raise ValueError(f"unsupported obstacle type for {name}")
    if not obstacles:
        raise ValueError("at least one obstacle is required")
    return AnalyticLidarGeometry(
        schema_version=1,
        source=str(payload["source"]),
        laser=laser,
        obstacles=tuple(obstacles),
    )


def _circle_intersection(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    obstacle: CircleObstacle,
) -> float | None:
    offset_x = obstacle.center_x_m - origin_x
    offset_y = obstacle.center_y_m - origin_y
    projected = offset_x * direction_x + offset_y * direction_y
    discriminant = projected * projected - (
        offset_x * offset_x + offset_y * offset_y - obstacle.radius_m**2
    )
    if discriminant < 0.0:
        return None
    root = math.sqrt(discriminant)
    near = projected - root
    far = projected + root
    if near >= 0.0:
        return near
    return far if far >= 0.0 else None


def _box_intersection(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    obstacle: BoxObstacle,
) -> float | None:
    cosine = math.cos(obstacle.yaw_rad)
    sine = math.sin(obstacle.yaw_rad)
    delta_x = origin_x - obstacle.center_x_m
    delta_y = origin_y - obstacle.center_y_m
    local_origin = (
        cosine * delta_x + sine * delta_y,
        -sine * delta_x + cosine * delta_y,
    )
    local_direction = (
        cosine * direction_x + sine * direction_y,
        -sine * direction_x + cosine * direction_y,
    )
    lower = (-obstacle.size_x_m / 2.0, -obstacle.size_y_m / 2.0)
    upper = (obstacle.size_x_m / 2.0, obstacle.size_y_m / 2.0)
    near = -math.inf
    far = math.inf
    for origin, direction, minimum, maximum in zip(
        local_origin, local_direction, lower, upper, strict=True
    ):
        if abs(direction) < 1e-12:
            if origin < minimum or origin > maximum:
                return None
            continue
        first = (minimum - origin) / direction
        second = (maximum - origin) / direction
        near = max(near, min(first, second))
        far = min(far, max(first, second))
        if near > far:
            return None
    if near >= 0.0:
        return near
    return far if far >= 0.0 else None


def raycast_distance(
    geometry: AnalyticLidarGeometry,
    pose: Pose2D,
    beam_angle_rad: float,
) -> float:
    """Return the nearest hit or positive infinity when no obstacle is in range."""

    world_angle = pose.yaw_rad + beam_angle_rad
    direction_x = math.cos(world_angle)
    direction_y = math.sin(world_angle)
    intersections = []
    for obstacle in geometry.obstacles:
        if isinstance(obstacle, CircleObstacle):
            distance = _circle_intersection(
                pose.x_m, pose.y_m, direction_x, direction_y, obstacle
            )
        else:
            distance = _box_intersection(
                pose.x_m, pose.y_m, direction_x, direction_y, obstacle
            )
        if distance is not None and geometry.laser.range_min_m <= distance:
            intersections.append(distance)
    if not intersections:
        return math.inf
    nearest = min(intersections)
    return nearest if nearest <= geometry.laser.range_max_m else math.inf


def simulate_ranges(
    geometry: AnalyticLidarGeometry,
    pose: Pose2D,
    *,
    specification: LaserSpecification | None = None,
) -> tuple[float, ...]:
    """Return ideal ranges for either the legacy or a sampled scan geometry."""

    laser = geometry.laser if specification is None else specification
    return tuple(
        raycast_distance(
            geometry,
            pose,
            laser.angle_min_rad + index * laser.angle_increment_rad,
        )
        for index in range(laser.beam_count)
    )
