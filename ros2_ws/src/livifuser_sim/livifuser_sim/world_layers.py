"""Five separately addressable geometry layers for the sensor-failure study.

Preregistration §7 requires that collision geometry, expert geometry, the
camera-visible world, the LiDAR-visible world, and the corrupted policy
observation stay distinct. Conflating any two invalidates the study, so the
separation is expressed in the world description itself rather than in
convention: every obstacle declares which layers it belongs to, and the loader
refuses worlds that violate the invariants below.

The condition this exists to make constructible is C4 — an obstacle that is
physically real, visible to the camera, known to the privileged expert, and
absent from the planar LiDAR. On the real platform that is not a contrivance:
the measured ``base_scan -> camera`` translation puts the camera 8.4 cm below
the scan plane, so a low obstacle genuinely is camera-visible and
LiDAR-invisible.

Stdlib only, Python 3.10 compatible: this runs on the ROS host as well as the
Windows analysis host.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .analytic_lidar import (
    AnalyticLidarGeometry,
    BoxObstacle,
    CircleObstacle,
    LaserSpecification,
    Obstacle,
    _finite,
    _xy,
)

LAYER_COLLISION = "collision"
LAYER_EXPERT = "expert"
LAYER_CAMERA = "camera"
LAYER_LIDAR = "lidar"

#: Declarable layers. The fifth preregistered layer, the corrupted policy
#: observation, is deliberately absent: it is produced downstream by the
#: corruption model (§13.2) and is never a property of the world.
DECLARABLE_LAYERS = (LAYER_COLLISION, LAYER_EXPERT, LAYER_CAMERA, LAYER_LIDAR)

#: Layers that are ground truth and may never be corrupted (§7).
GROUND_TRUTH_LAYERS = (LAYER_COLLISION, LAYER_EXPERT)


@dataclass(frozen=True)
class LayeredObstacle:
    """An obstacle together with the layers it participates in."""

    obstacle: Obstacle
    layers: frozenset[str]
    profile_switchable: bool = False

    @property
    def name(self) -> str:
        return self.obstacle.name

    def in_layer(self, layer: str) -> bool:
        return layer in self.layers


@dataclass(frozen=True)
class LayeredWorld:
    """A versioned world description with explicit per-layer membership."""

    schema_version: int
    name: str
    source: str
    laser: LaserSpecification
    obstacles: tuple[LayeredObstacle, ...]
    bounds_min_xy_m: tuple[float, float]
    bounds_max_xy_m: tuple[float, float]
    start_pose_xy_yaw: tuple[float, float, float] | None = None
    goal_xy_m: tuple[float, float] | None = None
    group: str | None = None
    seed: int | None = None
    archetype: str | None = None
    condition: str = "C0"

    def layer(self, layer: str) -> tuple[Obstacle, ...]:
        """Return the obstacles visible to one layer, in declaration order."""

        if layer not in DECLARABLE_LAYERS:
            raise ValueError(f"unknown layer: {layer}")
        return tuple(
            entry.obstacle for entry in self.obstacles if entry.in_layer(layer)
        )

    @property
    def lidar_invisible(self) -> tuple[Obstacle, ...]:
        """Obstacles that collide but do not appear in the scan — the C4 set."""

        return tuple(
            entry.obstacle
            for entry in self.obstacles
            if entry.in_layer(LAYER_COLLISION) and not entry.in_layer(LAYER_LIDAR)
        )

    @property
    def profile_switchable(self) -> tuple[Obstacle, ...]:
        """Obstacles C4 hides from the scan, in declaration order.

        These are present in every layer in the matched ID condition. C4 removes
        them from the LiDAR layer alone, leaving collision geometry and expert
        labels untouched (§8).
        """

        return tuple(
            entry.obstacle for entry in self.obstacles if entry.profile_switchable
        )


def _parse_layers(item: dict, name: str) -> frozenset[str]:
    raw = item.get("layers")
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must declare a layers object")
    unknown = set(raw) - set(DECLARABLE_LAYERS)
    if unknown:
        raise ValueError(f"{name} declares unknown layers: {sorted(unknown)}")
    missing = set(DECLARABLE_LAYERS) - set(raw)
    if missing:
        raise ValueError(
            f"{name} must declare every layer explicitly; missing {sorted(missing)}"
        )
    layers = set()
    for layer in DECLARABLE_LAYERS:
        value = raw[layer]
        if not isinstance(value, bool):
            raise ValueError(f"{name}.layers.{layer} must be a boolean")
        if value:
            layers.add(layer)
    return frozenset(layers)


def _build_obstacle(item: dict, name: str) -> Obstacle:
    center_x, center_y = _xy(item["center_xy_m"], f"{name}.center_xy_m")
    if item["type"] == "box":
        size_x, size_y = _xy(item["size_xy_m"], f"{name}.size_xy_m")
        if size_x <= 0.0 or size_y <= 0.0:
            raise ValueError(f"{name} box dimensions must be positive")
        return BoxObstacle(
            name,
            center_x,
            center_y,
            size_x,
            size_y,
            _finite(item["yaw_rad"], f"{name}.yaw_rad"),
        )
    if item["type"] == "circle":
        radius = _finite(item["radius_m"], f"{name}.radius_m")
        if radius <= 0.0:
            raise ValueError(f"{name} radius must be positive")
        return CircleObstacle(name, center_x, center_y, radius)
    raise ValueError(f"unsupported obstacle type for {name}")


def load_world(path: Path) -> LayeredWorld:
    """Load and validate a schema-2 layered world description."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_world(payload)


def parse_world(payload: dict) -> LayeredWorld:
    """Validate a layered world payload and enforce the §7 invariants."""

    if payload.get("schema_version") != 2:
        raise ValueError("unsupported layered world schema; expected version 2")

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

    minimum = _xy(payload["bounds_min_xy_m"], "bounds_min_xy_m")
    maximum = _xy(payload["bounds_max_xy_m"], "bounds_max_xy_m")
    if maximum[0] <= minimum[0] or maximum[1] <= minimum[1]:
        raise ValueError("world bounds must be ordered and non-degenerate")

    obstacles: list[LayeredObstacle] = []
    names: set[str] = set()
    for item in payload["obstacles"]:
        name = str(item["name"])
        if name in names:
            raise ValueError(f"duplicate obstacle name: {name}")
        names.add(name)
        layers = _parse_layers(item, name)
        if not layers:
            raise ValueError(f"{name} belongs to no layer")
        # Collision and expert are both ground truth (§7). Allowing them to
        # differ would let the expert either drive into something real or avoid
        # something that is not there, and either writes a wrong label.
        if (LAYER_COLLISION in layers) != (LAYER_EXPERT in layers):
            raise ValueError(
                f"{name} must share collision and expert membership; "
                "the privileged expert may not disagree with true geometry"
            )
        switchable = bool(item.get("profile_switchable", False))
        if switchable and not layers.issuperset(DECLARABLE_LAYERS):
            raise ValueError(
                f"{name} is profile_switchable, so it must start in every layer; "
                "C4 is produced by removing it from the LiDAR layer, not by "
                "authoring it as already hidden"
            )
        obstacles.append(
            LayeredObstacle(_build_obstacle(item, name), layers, switchable)
        )

    if not obstacles:
        raise ValueError("at least one obstacle is required")
    if not any(entry.in_layer(LAYER_COLLISION) for entry in obstacles):
        raise ValueError("a world must contain at least one colliding obstacle")

    start_pose = payload.get("start_pose_xy_yaw")
    if start_pose is not None:
        if not isinstance(start_pose, list) or len(start_pose) != 3:
            raise ValueError("start_pose_xy_yaw must contain x, y, and yaw")
        start_pose = tuple(
            _finite(value, "start_pose_xy_yaw") for value in start_pose
        )
    goal = payload.get("goal_xy_m")
    if goal is not None:
        goal = _xy(goal, "goal_xy_m")

    seed = payload.get("seed")
    condition = str(payload.get("condition", "C0"))
    if condition not in {"C0", "C1", "C4"}:
        raise ValueError(f"unsupported world condition: {condition}")
    return LayeredWorld(
        schema_version=2,
        name=str(payload["name"]),
        source=str(payload["source"]),
        laser=laser,
        obstacles=tuple(obstacles),
        bounds_min_xy_m=minimum,
        bounds_max_xy_m=maximum,
        start_pose_xy_yaw=start_pose,
        goal_xy_m=goal,
        group=None if payload.get("group") is None else str(payload["group"]),
        seed=None if seed is None else int(seed),
        archetype=(
            None if payload.get("archetype") is None else str(payload["archetype"])
        ),
        condition=condition,
    )


def geometry_for_layer(world: LayeredWorld, layer: str) -> AnalyticLidarGeometry:
    """Adapt one layer of a layered world to the analytic ray caster.

    This is the **only** supported way to ray-cast against a layered world.
    Casting against the full obstacle set would put LiDAR-invisible obstacles
    back into the scan and silently destroy condition C4 — the failure would
    surface as "fusion confers no benefit at C4", a plausible-looking null
    result on the primary thesis produced by plumbing rather than physics.

    ``tests/test_sim_privileged_expert.py`` pins the guarantee: casting a world
    that has LiDAR-invisible obstacles must give ranges identical to casting the
    same world with those obstacles deleted outright.
    """

    return AnalyticLidarGeometry(
        schema_version=1,
        source=f"{world.source} [layer={layer}]",
        laser=world.laser,
        obstacles=world.layer(layer),
    )


def point_clearance(obstacles: tuple[Obstacle, ...], x_m: float, y_m: float) -> float:
    """Return the distance from a point to the nearest obstacle surface.

    Zero inside an obstacle. Positive infinity when there are no obstacles,
    which keeps a caller's ``min`` well behaved on an empty layer.
    """

    nearest = math.inf
    for obstacle in obstacles:
        if isinstance(obstacle, CircleObstacle):
            offset_x = x_m - obstacle.center_x_m
            offset_y = y_m - obstacle.center_y_m
            distance = math.hypot(offset_x, offset_y) - obstacle.radius_m
        else:
            cosine = math.cos(obstacle.yaw_rad)
            sine = math.sin(obstacle.yaw_rad)
            delta_x = x_m - obstacle.center_x_m
            delta_y = y_m - obstacle.center_y_m
            local_x = cosine * delta_x + sine * delta_y
            local_y = -sine * delta_x + cosine * delta_y
            outside_x = abs(local_x) - obstacle.size_x_m / 2.0
            outside_y = abs(local_y) - obstacle.size_y_m / 2.0
            if outside_x > 0.0 or outside_y > 0.0:
                distance = math.hypot(max(outside_x, 0.0), max(outside_y, 0.0))
            else:
                distance = max(outside_x, outside_y)
        nearest = min(nearest, distance)
    return max(nearest, 0.0) if nearest != math.inf else math.inf
