"""Deterministic static-map rasterization for the bounded Nav2 probe.

The map contains collision-layer geometry except profile-switchable obstacles.
Those obstacles are the paired C0/C4 intervention and must be sensed online;
putting them in the static map would leak C4 ground truth into Nav2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .analytic_lidar import BoxObstacle, CircleObstacle, Obstacle
from .world_layers import LAYER_COLLISION, LAYER_LIDAR, LayeredWorld

FREE = 254
OCCUPIED = 0


@dataclass(frozen=True)
class Nav2Map:
    width: int
    height: int
    resolution_m: float
    origin_xy_m: tuple[float, float]
    pixels_top_down: bytes
    included_obstacles: tuple[str, ...]
    excluded_switchable_obstacles: tuple[str, ...]

    def pgm_bytes(self) -> bytes:
        header = f"P5\n{self.width} {self.height}\n255\n".encode("ascii")
        return header + self.pixels_top_down

    def yaml_text(self, image_name: str) -> str:
        origin_x, origin_y = self.origin_xy_m
        return (
            f"image: {image_name}\n"
            "mode: trinary\n"
            f"resolution: {self.resolution_m:.9f}\n"
            f"origin: [{origin_x:.9f}, {origin_y:.9f}, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.25\n"
        )

    def pixel_at_world(self, x_m: float, y_m: float) -> int:
        column = int(math.floor((x_m - self.origin_xy_m[0]) / self.resolution_m))
        row_from_bottom = int(
            math.floor((y_m - self.origin_xy_m[1]) / self.resolution_m)
        )
        if not 0 <= column < self.width or not 0 <= row_from_bottom < self.height:
            raise ValueError("world coordinate is outside the map")
        row_from_top = self.height - 1 - row_from_bottom
        return self.pixels_top_down[row_from_top * self.width + column]


def _contains(obstacle: Obstacle, x_m: float, y_m: float) -> bool:
    if isinstance(obstacle, CircleObstacle):
        return math.hypot(x_m - obstacle.center_x_m, y_m - obstacle.center_y_m) <= (
            obstacle.radius_m
        )
    if isinstance(obstacle, BoxObstacle):
        dx = x_m - obstacle.center_x_m
        dy = y_m - obstacle.center_y_m
        cosine = math.cos(obstacle.yaw_rad)
        sine = math.sin(obstacle.yaw_rad)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        return (
            abs(local_x) <= obstacle.size_x_m / 2.0
            and abs(local_y) <= obstacle.size_y_m / 2.0
        )
    raise TypeError(f"unsupported obstacle type: {type(obstacle).__name__}")


def rasterize_nav2_map(world: LayeredWorld, resolution_m: float = 0.025) -> Nav2Map:
    """Rasterize the permanent collision geometry without the C0/C4 probe set."""

    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("resolution_m must be positive and finite")
    origin_x, origin_y = world.bounds_min_xy_m
    maximum_x, maximum_y = world.bounds_max_xy_m
    width = int(math.ceil((maximum_x - origin_x) / resolution_m))
    height = int(math.ceil((maximum_y - origin_y) / resolution_m))
    if width <= 0 or height <= 0:
        raise ValueError("world bounds produce an empty map")

    included_entries = tuple(
        entry
        for entry in world.obstacles
        if entry.in_layer(LAYER_COLLISION)
        and not entry.profile_switchable
        and entry.in_layer(LAYER_LIDAR)
    )
    excluded = tuple(
        entry.name
        for entry in world.obstacles
        if entry.in_layer(LAYER_COLLISION)
        and (entry.profile_switchable or not entry.in_layer(LAYER_LIDAR))
    )
    if not included_entries:
        raise ValueError("Nav2 map requires permanent collision geometry")

    pixels = bytearray([FREE]) * (width * height)
    for row_from_bottom in range(height):
        y_m = origin_y + (row_from_bottom + 0.5) * resolution_m
        row_from_top = height - 1 - row_from_bottom
        row_offset = row_from_top * width
        for column in range(width):
            x_m = origin_x + (column + 0.5) * resolution_m
            if any(_contains(entry.obstacle, x_m, y_m) for entry in included_entries):
                pixels[row_offset + column] = OCCUPIED

    return Nav2Map(
        width=width,
        height=height,
        resolution_m=resolution_m,
        origin_xy_m=(origin_x, origin_y),
        pixels_top_down=bytes(pixels),
        included_obstacles=tuple(entry.name for entry in included_entries),
        excluded_switchable_obstacles=excluded,
    )
