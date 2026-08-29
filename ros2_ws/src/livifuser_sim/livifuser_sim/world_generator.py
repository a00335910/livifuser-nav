"""Deterministic procedural generator for schema-2 layered worlds.

Produces the disjoint world groups required by preregistration §5, sized so that
an episode fits the §12.2 budget: at the locked 0.08 m/s a 45 s episode covers
about 3.2 m, so goals are drawn near 3 m rather than the 4.5 m the development
corridor used.

Three properties this module exists to guarantee:

**Determinism.** A world is a pure function of ``(group, index)``. The seed is
derived from those alone, drawn in fixed order, and recorded in the emitted
payload. Regenerating a group reproduces it byte for byte.

**Seed disjointness.** Each group occupies its own numeric block, so a training
seed can never collide with a test seed no matter how the counts change. §5
requires disjoint geometry, layout, and generation seed; the block allocation
makes the last of those true by construction rather than by inspection.

**Validated feasibility.** A generated world is rejected unless the privileged
planner can actually reach the goal and unless the task requires steering. An
infeasible world wastes a rollout; a trivial one silently weakens every
comparison by letting a straight-line policy score. Rejection redraws
deterministically, so validation does not break reproducibility.

Every world carries a profile-switchable obstacle set: present in all four
layers here, removed from the LiDAR layer alone by :func:`derive_condition` to
build C4 (§8). C4 therefore changes exactly one factor and leaves expert labels
untouched.

C1 is also derived from the matched world. It attaches only the checksum-pinned
camera-scene contract consumed by the SDF materializer; geometry and labels are
unchanged.

Stdlib only, Python 3.10 compatible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from .privileged_expert import (
    ExpertLimits,
    PlannerSpecification,
    build_clearance_field,
    plan_path,
    simulate_expert_episode,
)
from .visual_conditions import c1_condition_descriptor
from .visual_skin import visual_skin_descriptor
from .world_layers import (
    DECLARABLE_LAYERS,
    LAYER_LIDAR,
    LayeredWorld,
    parse_world,
    point_clearance,
)

#: Disjoint numeric blocks, one per split group (§5). Chosen so that no
#: plausible growth in episode counts can make two groups overlap.
GROUP_SEED_BLOCKS = {
    "dev": 1_000_000,
    "train": 2_000_000,
    "val_id": 3_000_000,
    "test_id": 4_000_000,
}

#: `test_ood` deliberately has no block: §5 requires it to reuse `test_id`
#: geometry exactly so a corruption differs from its matched ID condition in one
#: factor. Conditions are derived, never generated.
DERIVED_GROUPS = ("test_ood",)

ARCHETYPES = (
    "straight_corridor",
    "dogleg_corridor",
    "doorway",
    "t_junction",
    "island_room",
    "cluttered_room",
)

#: Goal distance window. Centred near 3 m so a typical episode completes near
#: the §12.2 budget at the locked 0.08 m/s with margin for the detour. Straight-
#: line distance is only a coarse proxy; the binding gate is measured duration.
MIN_GOAL_DISTANCE_M = 2.5
MAX_GOAL_DISTANCE_M = 3.4

#: A world must require at least this much detour over the straight line, or it
#: does not exercise obstacle avoidance at all.
MIN_PATH_EXCESS_RATIO = 1.04

#: Pathology guard, deliberately **not** a budget. Capping near the §12.2 45 s
#: figure would reject turn-heavy topologies — a 90-degree turn at 0.40 rad/s
#: with speed scaling is expensive — and would quietly bias the world set toward
#: easy geometry, weakening exactly the diversity §5 asks for. Duration is
#: measured and reported instead, so §12.2 can be re-derived from the observed
#: mean rather than the world set being bent to fit an estimate.
MAX_EPISODE_SECONDS = 120.0

# The real Burger scan plane is 0.172 m above the ground.  Every switchable
# obstacle is rendered below it in both members of the paired C0/C4 probe; C0
# supplies the analytic scan with an oracle return and C4 removes that return.
# Keeping the rendered object byte-identical isolates LiDAR observability while
# making the C4 member physically realizable and camera-visible.
LIDAR_SCAN_HEIGHT_M = 0.172
LOW_OBSTACLE_HEIGHT_M = 0.120
WALL_HEIGHT_M = 1.0
OBSTACLE_HEIGHT_M = 0.60

# A switchable object may not merely exist in a corner.  The privileged rollout
# must pass close enough that a scan-driven controller can plausibly interact
# with it.  This is a generator-level anti-vacuity gate; the learned-policy C4
# and C3 gates remain separate.
MAX_SWITCHABLE_PATH_CLEARANCE_M = 0.45

# Counterfactual route-intersection gate. Remove the switchable obstacles from
# the expert world, roll out the same deterministic controller, then require its
# robot footprint to intersect at least one removed obstacle. This is stronger
# than checking the avoiding expert's proximity: a blind route that misses the
# obstacle cannot demonstrate the C4 sensing limitation. The threshold is the
# measured Burger collision radius, so it has a physical rather than tuned unit.
MAX_BLIND_ROUTE_SWITCHABLE_CLEARANCE_M = PlannerSpecification().robot_radius_m

WALL_THICKNESS_M = 0.1
MAX_DRAWS = 200


class WorldGenerationError(RuntimeError):
    """Raised when no valid world could be drawn within the retry budget."""


@dataclass(frozen=True)
class ValidationReport:
    feasible: bool
    goal_distance_m: float
    path_length_m: float
    path_excess_ratio: float
    reason: str
    episode_duration_s: float = 0.0
    minimum_clearance_m: float = 0.0
    switchable_path_clearance_m: float = math.inf
    blind_route_switchable_clearance_m: float = math.inf

    @property
    def accepted(self) -> bool:
        return self.reason == "accepted"


def seed_for(group: str, index: int) -> int:
    """Return the disjoint seed for one world."""

    if group in DERIVED_GROUPS:
        raise ValueError(f"{group} worlds are derived from test_id, never generated")
    if group not in GROUP_SEED_BLOCKS:
        raise ValueError(f"unknown world group: {group}")
    if index < 0:
        raise ValueError("world index must be non-negative")
    if index >= GROUP_SEED_BLOCKS[group] // 2:
        raise ValueError("world index would overflow its seed block")
    return GROUP_SEED_BLOCKS[group] + index


def _all_layers() -> dict:
    return {layer: True for layer in DECLARABLE_LAYERS}


def _render_profile(*, switchable: bool, height_m: float) -> dict:
    if switchable:
        height_m = LOW_OBSTACLE_HEIGHT_M
        color = [0.92, 0.24, 0.08, 1.0]
    else:
        color = [0.36, 0.55, 0.78, 1.0]
    return {"height_m": height_m, "color_rgba": color}


def _box(
    name,
    center,
    size,
    yaw=0.0,
    *,
    switchable=False,
    height_m=WALL_HEIGHT_M,
) -> dict:
    item = {
        "name": name,
        "type": "box",
        "center_xy_m": [round(center[0], 4), round(center[1], 4)],
        "size_xy_m": [round(size[0], 4), round(size[1], 4)],
        "yaw_rad": round(yaw, 6),
        "layers": _all_layers(),
        "render": _render_profile(switchable=switchable, height_m=height_m),
    }
    if switchable:
        item["profile_switchable"] = True
    return item


def _circle(
    name,
    center,
    radius,
    *,
    switchable=False,
    height_m=OBSTACLE_HEIGHT_M,
) -> dict:
    item = {
        "name": name,
        "type": "circle",
        "center_xy_m": [round(center[0], 4), round(center[1], 4)],
        "radius_m": round(radius, 4),
        "layers": _all_layers(),
        "render": _render_profile(switchable=switchable, height_m=height_m),
    }
    if switchable:
        item["profile_switchable"] = True
    return item


def _corridor_walls(length_m: float, half_width_m: float, prefix: str = "") -> list:
    centre = length_m / 2.0
    return [
        _box(
            f"{prefix}left_wall",
            (centre, half_width_m + WALL_THICKNESS_M / 2.0),
            (length_m, WALL_THICKNESS_M),
        ),
        _box(
            f"{prefix}right_wall",
            (centre, -half_width_m - WALL_THICKNESS_M / 2.0),
            (length_m, WALL_THICKNESS_M),
        ),
        _box(
            f"{prefix}end_wall",
            (length_m + WALL_THICKNESS_M / 2.0, 0.0),
            (WALL_THICKNESS_M, 2.0 * half_width_m + 2.0 * WALL_THICKNESS_M),
        ),
        _box(
            f"{prefix}start_wall",
            (-WALL_THICKNESS_M / 2.0, 0.0),
            (WALL_THICKNESS_M, 2.0 * half_width_m + 2.0 * WALL_THICKNESS_M),
        ),
    ]


def _room_walls(width_m: float, height_m: float) -> list:
    return [
        _box("north_wall", (width_m / 2.0, height_m / 2.0 + WALL_THICKNESS_M / 2.0),
             (width_m + 2.0 * WALL_THICKNESS_M, WALL_THICKNESS_M)),
        _box("south_wall", (width_m / 2.0, -height_m / 2.0 - WALL_THICKNESS_M / 2.0),
             (width_m + 2.0 * WALL_THICKNESS_M, WALL_THICKNESS_M)),
        _box("east_wall", (width_m + WALL_THICKNESS_M / 2.0, 0.0),
             (WALL_THICKNESS_M, height_m + 2.0 * WALL_THICKNESS_M)),
        _box("west_wall", (-WALL_THICKNESS_M / 2.0, 0.0),
             (WALL_THICKNESS_M, height_m + 2.0 * WALL_THICKNESS_M)),
    ]


def _bounds(width_m: float, y_min: float, y_max: float) -> tuple[list, list]:
    pad = 0.5
    return (
        [round(-pad, 4), round(y_min - pad, 4)],
        [round(width_m + pad, 4), round(y_max + pad, 4)],
    )


def _straight_corridor(rng: random.Random) -> dict:
    length = rng.uniform(3.2, 3.8)
    half_width = rng.uniform(0.75, 1.05)
    obstacles = _corridor_walls(length, half_width)
    count = rng.randint(2, 3)
    for index in range(count):
        span = length / (count + 1)
        x_m = span * (index + 1) + rng.uniform(-0.15, 0.15)
        y_m = rng.uniform(-half_width + 0.42, half_width - 0.42)
        switchable = index == rng.randrange(count)
        if rng.random() < 0.5:
            obstacles.append(
                _circle(f"pillar_{index}", (x_m, y_m), rng.uniform(0.16, 0.26),
                        switchable=switchable)
            )
        else:
            side = rng.uniform(0.34, 0.5)
            obstacles.append(
                _box(f"crate_{index}", (x_m, y_m), (side, side),
                     rng.uniform(-0.4, 0.4), switchable=switchable)
            )
    minimum, maximum = _bounds(length, -half_width - WALL_THICKNESS_M,
                               half_width + WALL_THICKNESS_M)
    return {
        "obstacles": obstacles,
        "bounds_min_xy_m": minimum,
        "bounds_max_xy_m": maximum,
        "start_pose_xy_yaw": [0.4, round(rng.uniform(-0.2, 0.2), 4), 0.0],
        "goal_xy_m": [round(length - 0.4, 4), round(rng.uniform(-0.3, 0.3), 4)],
    }


def _dogleg_corridor(rng: random.Random) -> dict:
    length = rng.uniform(3.3, 3.9)
    half_width = rng.uniform(0.8, 1.05)
    offset = rng.choice((-1.0, 1.0)) * rng.uniform(0.55, 0.8)
    obstacles = _corridor_walls(length, half_width + abs(offset))
    bend_x = length * rng.uniform(0.42, 0.58)
    obstacles.append(
        _box("bend_block",
             (bend_x, offset + math.copysign(half_width, offset)),
             (rng.uniform(0.7, 1.0), 2.0 * half_width * 0.9),
             0.0)
    )
    obstacles.append(
        _circle("bend_pillar",
                (bend_x + rng.uniform(0.5, 0.85), -offset * rng.uniform(0.3, 0.6)),
                rng.uniform(0.17, 0.25),
                switchable=True)
    )
    limit = half_width + abs(offset) + WALL_THICKNESS_M
    minimum, maximum = _bounds(length, -limit, limit)
    return {
        "obstacles": obstacles,
        "bounds_min_xy_m": minimum,
        "bounds_max_xy_m": maximum,
        "start_pose_xy_yaw": [0.4, round(-offset * 0.3, 4), 0.0],
        "goal_xy_m": [round(length - 0.4, 4), round(offset * 0.35, 4)],
    }


def _doorway(rng: random.Random) -> dict:
    length = rng.uniform(3.2, 3.7)
    half_width = rng.uniform(0.95, 1.25)
    obstacles = _corridor_walls(length, half_width)
    wall_x = length * rng.uniform(0.45, 0.58)
    gap_centre = rng.uniform(-0.35, 0.35)
    gap_half = rng.uniform(0.32, 0.44)
    upper = half_width - (gap_centre + gap_half)
    lower = (gap_centre - gap_half) + half_width
    obstacles.append(
        _box("divider_upper",
             (wall_x, gap_centre + gap_half + upper / 2.0),
             (WALL_THICKNESS_M * 2.0, max(upper, 0.05)))
    )
    obstacles.append(
        _box("divider_lower",
             (wall_x, gap_centre - gap_half - lower / 2.0),
             (WALL_THICKNESS_M * 2.0, max(lower, 0.05)))
    )
    obstacles.append(
        _circle("approach_obstacle",
                (wall_x - rng.uniform(0.7, 1.0), gap_centre + rng.uniform(-0.4, 0.4)),
                rng.uniform(0.15, 0.22),
                switchable=True)
    )
    minimum, maximum = _bounds(length, -half_width - WALL_THICKNESS_M,
                               half_width + WALL_THICKNESS_M)
    return {
        "obstacles": obstacles,
        "bounds_min_xy_m": minimum,
        "bounds_max_xy_m": maximum,
        "start_pose_xy_yaw": [0.4, round(gap_centre * 0.4, 4), 0.0],
        "goal_xy_m": [round(length - 0.4, 4), round(gap_centre * 0.6, 4)],
    }


def _t_junction(rng: random.Random) -> dict:
    """Stem along +x opening into a perpendicular passage spanning +/- arm.

    Free space is [0, stem] x [-hw, hw] joined to
    [stem, stem + depth] x [-arm, arm]. Dimensions are drawn so the
    start-to-goal straight line always lands inside the §12 goal window; a
    geometry that cannot reach 2.5 m is unbuildable rather than merely unlucky.
    """

    stem = rng.uniform(2.4, 2.9)
    depth = rng.uniform(0.8, 1.0)
    half_width = rng.uniform(0.55, 0.72)
    arm = rng.uniform(1.4, 1.7)
    branch = rng.choice((-1.0, 1.0))
    half = WALL_THICKNESS_M / 2.0
    obstacles = [
        _box("stem_left", (stem / 2.0, half_width + half), (stem, WALL_THICKNESS_M)),
        _box("stem_right", (stem / 2.0, -half_width - half), (stem, WALL_THICKNESS_M)),
        _box("start_wall", (-half, 0.0),
             (WALL_THICKNESS_M, 2.0 * half_width + 2.0 * WALL_THICKNESS_M)),
        _box("cross_far", (stem + depth + half, 0.0),
             (WALL_THICKNESS_M, 2.0 * arm + 2.0 * WALL_THICKNESS_M)),
        _box("cross_top", (stem + depth / 2.0, arm + half),
             (depth, WALL_THICKNESS_M)),
        _box("cross_bottom", (stem + depth / 2.0, -arm - half),
             (depth, WALL_THICKNESS_M)),
        _box("near_upper", (stem - half, (half_width + arm) / 2.0),
             (WALL_THICKNESS_M, arm - half_width)),
        _box("near_lower", (stem - half, -(half_width + arm) / 2.0),
             (WALL_THICKNESS_M, arm - half_width)),
        _circle("junction_obstacle",
                (stem * rng.uniform(0.5, 0.7), rng.uniform(-0.2, 0.2)),
                rng.uniform(0.14, 0.20),
                switchable=True),
    ]
    minimum, maximum = _bounds(stem + depth, -arm - WALL_THICKNESS_M,
                               arm + WALL_THICKNESS_M)
    return {
        "obstacles": obstacles,
        "bounds_min_xy_m": minimum,
        "bounds_max_xy_m": maximum,
        "start_pose_xy_yaw": [0.4, 0.0, 0.0],
        "goal_xy_m": [
            round(stem + depth / 2.0, 4),
            round(branch * (arm - 0.4), 4),
        ],
    }


def _island_room(rng: random.Random) -> dict:
    width = rng.uniform(3.4, 4.0)
    height = rng.uniform(2.6, 3.2)
    obstacles = _room_walls(width, height)
    count = rng.randint(2, 3)
    switch_index = rng.randrange(count)
    for index in range(count):
        x_m = width * (index + 1) / (count + 1) + rng.uniform(-0.2, 0.2)
        y_m = rng.uniform(-height / 2.0 + 0.5, height / 2.0 - 0.5)
        obstacles.append(
            _box(f"island_{index}", (x_m, y_m),
                 (rng.uniform(0.4, 0.7), rng.uniform(0.4, 0.7)),
                 rng.uniform(-0.5, 0.5),
                 switchable=index == switch_index)
        )
    minimum, maximum = _bounds(width, -height / 2.0 - WALL_THICKNESS_M,
                               height / 2.0 + WALL_THICKNESS_M)
    start_y = rng.uniform(-height / 4.0, height / 4.0)
    return {
        "obstacles": obstacles,
        "bounds_min_xy_m": minimum,
        "bounds_max_xy_m": maximum,
        "start_pose_xy_yaw": [0.45, round(start_y, 4), 0.0],
        "goal_xy_m": [round(width - 0.45, 4), round(-start_y * 0.8, 4)],
    }


def _cluttered_room(rng: random.Random) -> dict:
    width = rng.uniform(3.3, 3.8)
    height = rng.uniform(2.4, 3.0)
    obstacles = _room_walls(width, height)
    count = rng.randint(4, 6)
    switch_index = rng.randrange(count)
    for index in range(count):
        x_m = rng.uniform(0.9, width - 0.7)
        y_m = rng.uniform(-height / 2.0 + 0.45, height / 2.0 - 0.45)
        obstacles.append(
            _circle(f"clutter_{index}", (x_m, y_m), rng.uniform(0.13, 0.22),
                    switchable=index == switch_index)
        )
    minimum, maximum = _bounds(width, -height / 2.0 - WALL_THICKNESS_M,
                               height / 2.0 + WALL_THICKNESS_M)
    return {
        "obstacles": obstacles,
        "bounds_min_xy_m": minimum,
        "bounds_max_xy_m": maximum,
        "start_pose_xy_yaw": [0.45, round(rng.uniform(-0.4, 0.4), 4), 0.0],
        "goal_xy_m": [round(width - 0.45, 4), round(rng.uniform(-0.5, 0.5), 4)],
    }


_BUILDERS = {
    "straight_corridor": _straight_corridor,
    "dogleg_corridor": _dogleg_corridor,
    "doorway": _doorway,
    "t_junction": _t_junction,
    "island_room": _island_room,
    "cluttered_room": _cluttered_room,
}


def validate_world(
    world: LayeredWorld,
    specification: PlannerSpecification | None = None,
    limits: ExpertLimits | None = None,
) -> ValidationReport:
    """Reject worlds that cannot be driven or that need no steering."""

    spec = specification or PlannerSpecification()
    bounds = limits or ExpertLimits()
    if world.start_pose_xy_yaw is None or world.goal_xy_m is None:
        return ValidationReport(False, 0.0, 0.0, 0.0, "missing_start_or_goal")

    start_x, start_y, _ = world.start_pose_xy_yaw
    goal = world.goal_xy_m
    straight = math.hypot(goal[0] - start_x, goal[1] - start_y)
    if not MIN_GOAL_DISTANCE_M <= straight <= MAX_GOAL_DISTANCE_M:
        return ValidationReport(False, straight, 0.0, 0.0, "goal_distance_out_of_range")
    if straight <= bounds.goal_tolerance_m:
        return ValidationReport(False, straight, 0.0, 0.0, "goal_inside_tolerance")

    field = build_clearance_field(world, spec)
    if not field.traversable(*field.cell_of(start_x, start_y)):
        return ValidationReport(False, straight, 0.0, 0.0, "start_blocked")
    if not field.traversable(*field.cell_of(*goal)):
        return ValidationReport(False, straight, 0.0, 0.0, "goal_blocked")

    path = plan_path(field, (start_x, start_y), goal, spec)
    if path is None:
        return ValidationReport(False, straight, 0.0, 0.0, "no_path")

    length = sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(path, path[1:], strict=False)
    )
    ratio = length / straight if straight > 0.0 else 0.0
    if ratio < MIN_PATH_EXCESS_RATIO:
        return ValidationReport(False, straight, length, ratio, "trivially_straight")
    if not world.profile_switchable:
        return ValidationReport(False, straight, length, ratio, "no_switchable_obstacle")

    # The binding gate: drive the world under privileged labels and require the
    # expert itself to finish. A path existing is not the same as the bounded
    # controller following it to the goal.
    rollout = simulate_expert_episode(
        world,
        world.start_pose_xy_yaw,
        goal,
        field=field,
        specification=spec,
        limits=bounds,
    )
    if not rollout.reached:
        return ValidationReport(
            False, straight, length, ratio, "expert_did_not_reach_goal",
            rollout.duration_s, rollout.minimum_clearance_m,
        )
    if rollout.duration_s > MAX_EPISODE_SECONDS:
        return ValidationReport(
            False, straight, length, ratio, "episode_too_long",
            rollout.duration_s, rollout.minimum_clearance_m,
        )

    switchable_obstacles = tuple(
        entry.obstacle for entry in world.obstacles if entry.profile_switchable
    )
    switchable_clearance = min(
        point_clearance(switchable_obstacles, x_m, y_m)
        for x_m, y_m, _ in rollout.poses
    )
    if switchable_clearance > MAX_SWITCHABLE_PATH_CLEARANCE_M:
        return ValidationReport(
            False,
            straight,
            length,
            ratio,
            "switchable_obstacle_not_route_relevant",
            rollout.duration_s,
            rollout.minimum_clearance_m,
            switchable_clearance,
        )

    # Development Nav2 C4 runs showed why the proximity check above is not
    # sufficient: a planner may pass near an obstacle while its blind route
    # still misses the footprint. The counterfactual removes only switchable
    # obstacles, then asks whether a robot following that blind route would
    # physically intersect one of them.
    blind_world = replace(
        world,
        obstacles=tuple(
            entry for entry in world.obstacles if not entry.profile_switchable
        ),
    )
    blind_field = build_clearance_field(blind_world, spec)
    blind_rollout = simulate_expert_episode(
        blind_world,
        blind_world.start_pose_xy_yaw,
        goal,
        field=blind_field,
        specification=spec,
        limits=bounds,
    )
    if not blind_rollout.reached:
        return ValidationReport(
            False,
            straight,
            length,
            ratio,
            "blind_route_did_not_reach_goal",
            rollout.duration_s,
            rollout.minimum_clearance_m,
            switchable_clearance,
        )
    blind_route_clearance = min(
        point_clearance(switchable_obstacles, x_m, y_m)
        for x_m, y_m, _ in blind_rollout.poses
    )
    if blind_route_clearance > MAX_BLIND_ROUTE_SWITCHABLE_CLEARANCE_M:
        return ValidationReport(
            False,
            straight,
            length,
            ratio,
            "switchable_obstacle_not_on_blind_route",
            rollout.duration_s,
            rollout.minimum_clearance_m,
            switchable_clearance,
            blind_route_clearance,
        )
    return ValidationReport(
        True,
        straight,
        length,
        ratio,
        "accepted",
        rollout.duration_s,
        rollout.minimum_clearance_m,
        switchable_clearance,
        blind_route_clearance,
    )


def generate_world(group: str, index: int, *, archetype: str | None = None) -> dict:
    """Return a validated schema-2 world payload for ``(group, index)``.

    Deterministic: the same arguments always produce the same payload, including
    the redraws that validation forces.
    """

    seed = seed_for(group, index)
    rng = random.Random(seed)
    chosen = archetype or ARCHETYPES[index % len(ARCHETYPES)]
    if chosen not in _BUILDERS:
        raise ValueError(f"unknown archetype: {chosen}")

    last = "not_attempted"
    for draw in range(MAX_DRAWS):
        body = _BUILDERS[chosen](rng)
        payload = {
            "schema_version": 2,
            "name": f"{group}_{chosen}_{index:03d}",
            "source": (
                "Procedurally generated by livifuser_sim.world_generator; "
                "analytic simulation LiDAR, not a Gazebo GPU sensor. "
                "Profile-switchable obstacles are present in every layer here; "
                "C4 removes them from the LiDAR layer alone."
            ),
            "group": group,
            "seed": seed,
            "archetype": chosen,
            "draw": draw,
            "visual_skin": visual_skin_descriptor(group),
            "laser": {
                "beam_count": 400,
                "angle_min_rad": 0.0,
                "angle_max_rad": 2.0 * math.pi,
                "scan_time_sec": 0.1,
                "range_min_m": 0.12,
                "range_max_m": 8.0,
                "frame_id": "base_scan",
            },
            **body,
        }
        try:
            world = parse_world(payload)
        except ValueError as error:
            last = f"invalid_payload: {error}"
            continue
        report = validate_world(world)
        if report.accepted:
            payload["validation"] = {
                "goal_distance_m": round(report.goal_distance_m, 4),
                "path_length_m": round(report.path_length_m, 4),
                "path_excess_ratio": round(report.path_excess_ratio, 4),
                "expert_episode_duration_s": round(report.episode_duration_s, 4),
                "expert_minimum_clearance_m": round(report.minimum_clearance_m, 4),
                "switchable_path_clearance_m": round(
                    report.switchable_path_clearance_m, 4
                ),
                "blind_route_switchable_clearance_m": round(
                    report.blind_route_switchable_clearance_m, 4
                ),
            }
            return payload
        last = report.reason
    raise WorldGenerationError(
        f"no valid {chosen} world for {group}[{index}] in {MAX_DRAWS} draws; "
        f"last rejection: {last}"
    )


def derive_condition(payload: dict, condition: str) -> dict:
    """Derive a single-factor condition variant from a matched ID world payload.

    ``C0`` returns the world unchanged. ``C1`` attaches the checksum-pinned
    camera-visible scene contract without changing geometry. ``C4`` removes the
    profile-switchable obstacles from the LiDAR layer and nothing else, so
    collision geometry, expert labels, start pose, goal, and seed are all
    preserved (§8).

    C3 is applied by the analytic observation model rather than the world.
    """

    if condition == "C0":
        return {key: value for key, value in payload.items()}
    if payload.get("condition") is not None:
        raise ValueError(
            f"cannot derive {condition} from existing condition "
            f"{payload['condition']}; conditions are matched independently to C0"
        )
    if condition == "C1":
        variant = {key: value for key, value in payload.items()}
        variant["condition"] = "C1"
        variant["camera_condition"] = c1_condition_descriptor()
        return variant
    if condition != "C4":
        raise ValueError(
            f"{condition} is not a world-level condition; C3 acts on the "
            "analytic observation model"
        )

    variant = {key: value for key, value in payload.items()}
    variant["condition"] = "C4"
    obstacles = []
    hidden = []
    for item in payload["obstacles"]:
        entry = {key: value for key, value in item.items()}
        entry["layers"] = dict(item["layers"])
        if item.get("profile_switchable", False):
            entry["layers"][LAYER_LIDAR] = False
            # The obstacle is no longer switchable in the derived world: it is
            # already hidden, and re-deriving C4 from a C4 world must not be
            # mistaken for a second single-factor change.
            entry.pop("profile_switchable", None)
            hidden.append(item["name"])
        obstacles.append(entry)
    if not hidden:
        raise ValueError("world has no profile-switchable obstacle to hide for C4")
    variant["obstacles"] = obstacles
    variant["c4_hidden_from_lidar"] = hidden
    return variant
