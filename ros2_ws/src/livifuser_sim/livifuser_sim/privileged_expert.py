"""Privileged simulation expert: labels derived from geometry, never from sensors.

Preregistration §6 disqualifies the previous expert. ``reactive_expert_node``
subscribes to ``/scan`` and derives its command from that same scan, so under any
LiDAR corruption it would go blind to the obstacle, drive into it, and write the
resulting unsafe action out as the training label. Every sensor-failure label
would be corrupt, and the study with it.

The defence here is structural rather than procedural. **This module's public
functions cannot accept a sensor reading — there is no parameter to pass one
through.** Labels are a pure function of

    (world geometry, robot pose, goal)

so no corruption model, however configured, can reach them. The label-invariance
regression test in ``tests/test_sim_privileged_expert.py`` then confirms the
property end to end rather than establishing it.

Determinism is likewise structural: no RNG is used anywhere, the A* frontier
breaks ties on a monotonic insertion counter rather than on heap-internal
ordering, and neighbour expansion order is fixed. Identical inputs give bitwise
identical commands on repeated runs within a build.

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .world_layers import LAYER_EXPERT, LayeredWorld, point_clearance

#: TurtleBot3 Burger. Matches the bounds enforced by the command watchdog and
#: used by every recorded physical episode.
MAX_LINEAR_MPS = 0.08
MAX_ANGULAR_RADPS = 0.40


@dataclass(frozen=True)
class PrivilegedCommand:
    """A bounded expert command plus the privileged quantities that produced it."""

    linear_mps: float
    angular_radps: float
    reason: str
    clearance_m: float


@dataclass(frozen=True)
class PlannerSpecification:
    """Grid and footprint parameters for the privileged planner."""

    cell_size_m: float = 0.05
    robot_radius_m: float = 0.105
    safety_margin_m: float = 0.04
    clearance_preference_m: float = 0.40

    @property
    def inflation_m(self) -> float:
        return self.robot_radius_m + self.safety_margin_m


@dataclass(frozen=True)
class ExpertLimits:
    """Bounded control limits and the thresholds that shape expert behaviour."""

    max_linear_mps: float = MAX_LINEAR_MPS
    max_angular_radps: float = MAX_ANGULAR_RADPS
    goal_tolerance_m: float = 0.25
    lookahead_m: float = 0.35
    angular_gain: float = 1.5
    caution_clearance_m: float = 0.35
    stop_clearance_m: float = 0.16
    minimum_speed_scale: float = 0.25


@dataclass(frozen=True)
class ClearanceField:
    """Distance from each cell centre to the nearest expert-layer obstacle."""

    cell_size_m: float
    origin_x_m: float
    origin_y_m: float
    width: int
    height: int
    values: tuple[float, ...]
    inflation_m: float

    def index_of(self, column: int, row: int) -> int:
        return row * self.width + column

    def cell_centre(self, column: int, row: int) -> tuple[float, float]:
        return (
            self.origin_x_m + (column + 0.5) * self.cell_size_m,
            self.origin_y_m + (row + 0.5) * self.cell_size_m,
        )

    def cell_of(self, x_m: float, y_m: float) -> tuple[int, int]:
        column = int(math.floor((x_m - self.origin_x_m) / self.cell_size_m))
        row = int(math.floor((y_m - self.origin_y_m) / self.cell_size_m))
        return (
            min(max(column, 0), self.width - 1),
            min(max(row, 0), self.height - 1),
        )

    def clearance_at(self, x_m: float, y_m: float) -> float:
        column, row = self.cell_of(x_m, y_m)
        return self.values[self.index_of(column, row)]

    def traversable(self, column: int, row: int) -> bool:
        return self.values[self.index_of(column, row)] >= self.inflation_m


def build_clearance_field(
    world: LayeredWorld, specification: PlannerSpecification | None = None
) -> ClearanceField:
    """Compute the clearance field from the **expert** layer only.

    The expert layer is ground truth and is never corrupted (§7), so this field
    is identical across every sensor condition generated from a given world.
    """

    spec = specification or PlannerSpecification()
    obstacles = world.layer(LAYER_EXPERT)
    origin_x, origin_y = world.bounds_min_xy_m
    maximum_x, maximum_y = world.bounds_max_xy_m
    width = max(1, int(math.ceil((maximum_x - origin_x) / spec.cell_size_m)))
    height = max(1, int(math.ceil((maximum_y - origin_y) / spec.cell_size_m)))

    values: list[float] = []
    for row in range(height):
        centre_y = origin_y + (row + 0.5) * spec.cell_size_m
        for column in range(width):
            centre_x = origin_x + (column + 0.5) * spec.cell_size_m
            values.append(point_clearance(obstacles, centre_x, centre_y))
    return ClearanceField(
        cell_size_m=spec.cell_size_m,
        origin_x_m=origin_x,
        origin_y_m=origin_y,
        width=width,
        height=height,
        values=tuple(values),
        inflation_m=spec.inflation_m,
    )


#: Fixed expansion order. Deterministic traversal depends on it.
_NEIGHBOURS = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def plan_path(
    field: ClearanceField,
    start_xy_m: tuple[float, float],
    goal_xy_m: tuple[float, float],
    specification: PlannerSpecification | None = None,
) -> tuple[tuple[float, float], ...] | None:
    """Plan a ground-truth path with A*, or return ``None`` when none exists.

    Costs penalise proximity to obstacles so the path prefers the middle of free
    space, which keeps the followed trajectory away from the inflation boundary
    where small pose errors would otherwise register as collisions.
    """

    spec = specification or PlannerSpecification()
    start_cell = field.cell_of(*start_xy_m)
    goal_cell = field.cell_of(*goal_xy_m)
    if not field.traversable(*goal_cell) or not field.traversable(*start_cell):
        return None
    if start_cell == goal_cell:
        return (tuple(start_xy_m), tuple(goal_xy_m))

    start_index = field.index_of(*start_cell)
    goal_index = field.index_of(*goal_cell)
    goal_centre = field.cell_centre(*goal_cell)

    def heuristic(column: int, row: int) -> float:
        centre_x, centre_y = field.cell_centre(column, row)
        return math.hypot(goal_centre[0] - centre_x, goal_centre[1] - centre_y)

    best_cost = {start_index: 0.0}
    came_from: dict[int, int] = {}
    counter = 0
    frontier: list[tuple[float, int, int, int, int]] = [
        (heuristic(*start_cell), counter, start_index, start_cell[0], start_cell[1])
    ]
    visited: set[int] = set()

    while frontier:
        _, _, index, column, row = heapq.heappop(frontier)
        if index in visited:
            continue
        visited.add(index)
        if index == goal_index:
            break
        current_cost = best_cost[index]
        for delta_column, delta_row in _NEIGHBOURS:
            next_column = column + delta_column
            next_row = row + delta_row
            if not (0 <= next_column < field.width and 0 <= next_row < field.height):
                continue
            if not field.traversable(next_column, next_row):
                continue
            next_index = field.index_of(next_column, next_row)
            if next_index in visited:
                continue
            step = field.cell_size_m * math.hypot(delta_column, delta_row)
            clearance = field.values[next_index]
            shortfall = max(0.0, spec.clearance_preference_m - clearance)
            penalty = 1.0 + 2.0 * shortfall / spec.clearance_preference_m
            candidate = current_cost + step * penalty
            if candidate < best_cost.get(next_index, math.inf):
                best_cost[next_index] = candidate
                came_from[next_index] = index
                counter += 1
                heapq.heappush(
                    frontier,
                    (
                        candidate + heuristic(next_column, next_row),
                        counter,
                        next_index,
                        next_column,
                        next_row,
                    ),
                )

    if goal_index not in best_cost:
        return None

    reversed_cells = [goal_index]
    while reversed_cells[-1] != start_index:
        reversed_cells.append(came_from[reversed_cells[-1]])
    reversed_cells.reverse()

    path = [tuple(start_xy_m)]
    for index in reversed_cells[1:-1]:
        path.append(field.cell_centre(index % field.width, index // field.width))
    path.append(tuple(goal_xy_m))
    return tuple(path)


def _carrot(
    path: tuple[tuple[float, float], ...],
    x_m: float,
    y_m: float,
    lookahead_m: float,
) -> tuple[float, float]:
    """Return the pure-pursuit target: the first path point beyond lookahead."""

    nearest_index = 0
    nearest_distance = math.inf
    for index, (point_x, point_y) in enumerate(path):
        distance = math.hypot(point_x - x_m, point_y - y_m)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index
    for point_x, point_y in path[nearest_index:]:
        if math.hypot(point_x - x_m, point_y - y_m) >= lookahead_m:
            return point_x, point_y
    return path[-1]


def follow_path(
    path: tuple[tuple[float, float], ...] | None,
    pose_x_m: float,
    pose_y_m: float,
    pose_yaw_rad: float,
    goal_xy_m: tuple[float, float],
    clearance_m: float,
    limits: ExpertLimits | None = None,
) -> PrivilegedCommand:
    """Convert a privileged path into a bounded command.

    ``clearance_m`` is privileged ground truth from the expert layer, not a
    sensor reading.
    """

    bounds = limits or ExpertLimits()
    goal_distance = math.hypot(goal_xy_m[0] - pose_x_m, goal_xy_m[1] - pose_y_m)
    if goal_distance <= bounds.goal_tolerance_m:
        return PrivilegedCommand(0.0, 0.0, "goal_reached", clearance_m)
    if path is None or len(path) < 2:
        return PrivilegedCommand(0.0, 0.0, "no_path", clearance_m)
    if clearance_m <= bounds.stop_clearance_m:
        return PrivilegedCommand(0.0, 0.0, "clearance_stop", clearance_m)

    target_x, target_y = _carrot(path, pose_x_m, pose_y_m, bounds.lookahead_m)
    heading = math.atan2(target_y - pose_y_m, target_x - pose_x_m)
    alpha = math.atan2(
        math.sin(heading - pose_yaw_rad), math.cos(heading - pose_yaw_rad)
    )

    angular = max(
        -bounds.max_angular_radps,
        min(bounds.max_angular_radps, bounds.angular_gain * alpha),
    )
    turn_scale = max(
        bounds.minimum_speed_scale,
        1.0 - abs(angular) / bounds.max_angular_radps,
    )
    span = bounds.caution_clearance_m - bounds.stop_clearance_m
    if span <= 0.0:
        clearance_scale = 1.0
    else:
        clearance_scale = min(
            1.0, max(0.0, (clearance_m - bounds.stop_clearance_m) / span)
        )
    linear = bounds.max_linear_mps * turn_scale * clearance_scale

    reason = "track_goal" if clearance_scale >= 1.0 else "cautious_advance"
    return PrivilegedCommand(linear, angular, reason, clearance_m)


@dataclass(frozen=True)
class EpisodeRollout:
    """The result of driving one world under privileged labels."""

    labels: tuple[tuple[float, float, str], ...]
    poses: tuple[tuple[float, float, float], ...]
    reached: bool
    duration_s: float
    minimum_clearance_m: float


def simulate_expert_episode(
    world: LayeredWorld,
    start_pose_xy_yaw: tuple[float, float, float],
    goal_xy_m: tuple[float, float],
    *,
    control_period_s: float = 0.1,
    max_ticks: int = 1200,
    field: ClearanceField | None = None,
    specification: PlannerSpecification | None = None,
    limits: ExpertLimits | None = None,
) -> EpisodeRollout:
    """Roll a full episode under privileged labels, integrating differential drive.

    Plans once from ground-truth geometry and then follows, which is the usage
    episode generation should prefer over replanning every tick.

    Like every other entry point here this takes no sensor input, so the rollout
    is identical across sensor conditions derived from the same world.
    """

    spec = specification or PlannerSpecification()
    clearance_field = field or build_clearance_field(world, spec)
    x_m, y_m, yaw_rad = start_pose_xy_yaw
    path = plan_path(clearance_field, (x_m, y_m), goal_xy_m, spec)

    labels: list[tuple[float, float, str]] = []
    poses: list[tuple[float, float, float]] = []
    minimum_clearance = math.inf
    reached = False
    for _ in range(max_ticks):
        clearance = clearance_field.clearance_at(x_m, y_m)
        minimum_clearance = min(minimum_clearance, clearance)
        command = follow_path(
            path, x_m, y_m, yaw_rad, goal_xy_m, clearance, limits
        )
        labels.append((command.linear_mps, command.angular_radps, command.reason))
        poses.append((x_m, y_m, yaw_rad))
        if command.reason == "goal_reached":
            reached = True
            break
        if command.reason == "no_path":
            break
        x_m += command.linear_mps * math.cos(yaw_rad) * control_period_s
        y_m += command.linear_mps * math.sin(yaw_rad) * control_period_s
        yaw_rad += command.angular_radps * control_period_s

    return EpisodeRollout(
        labels=tuple(labels),
        poses=tuple(poses),
        reached=reached,
        duration_s=len(labels) * control_period_s,
        minimum_clearance_m=(
            0.0 if minimum_clearance == math.inf else minimum_clearance
        ),
    )


def privileged_command(
    world: LayeredWorld,
    pose_x_m: float,
    pose_y_m: float,
    pose_yaw_rad: float,
    goal_xy_m: tuple[float, float],
    *,
    field: ClearanceField | None = None,
    specification: PlannerSpecification | None = None,
    limits: ExpertLimits | None = None,
) -> PrivilegedCommand:
    """Label one tick from geometry alone.

    Note the signature: world, pose, goal. There is no sensor parameter, so no
    corruption condition can influence the result (§6).

    Callers generating episodes should build ``field`` once per world and pass
    it in; rebuilding it every tick is correct but wasteful.
    """

    spec = specification or PlannerSpecification()
    clearance_field = field or build_clearance_field(world, spec)
    clearance = clearance_field.clearance_at(pose_x_m, pose_y_m)
    path = plan_path(clearance_field, (pose_x_m, pose_y_m), goal_xy_m, spec)
    return follow_path(
        path,
        pose_x_m,
        pose_y_m,
        pose_yaw_rad,
        goal_xy_m,
        clearance,
        limits,
    )
