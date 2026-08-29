"""Scan-driven reactive expert. DISQUALIFIED for sensor-failure labelling.

**Do not use this module to generate episodes for the sensor-failure study.**
Preregistration §6 disqualifies it: it derives its command from the same
``/scan`` the policy sees, so under any LiDAR corruption it goes blind to the
obstacle, drives into it, and writes the resulting unsafe action out as the
training label.

Use ``privileged_expert.privileged_command`` instead, which reads ground-truth
geometry and cannot accept a sensor reading.

Retained only because the pre-corruption simulation gates
(`docs/PROJECT_STATE.md`, 2026-08-08) were recorded against it, and deleting it
would strand that evidence. It labels nothing new.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertCommand:
    linear_mps: float
    angular_radps: float
    reason: str


def _finite_clearance(
    ranges: Iterable[float],
    angles: Iterable[float],
    minimum_angle: float,
    maximum_angle: float,
    range_cap_m: float,
) -> float:
    selected = [
        min(float(distance), range_cap_m)
        for distance, angle in zip(ranges, angles, strict=True)
        if minimum_angle <= angle <= maximum_angle
        and math.isfinite(distance)
        and distance > 0.0
    ]
    return min(selected) if selected else 0.0


def reactive_command(
    ranges: Iterable[float],
    angles: Iterable[float],
    *,
    goal_rho_m: float,
    goal_alpha_rad: float,
    obstacle_threshold_m: float = 0.45,
    max_linear_mps: float = 0.08,
    max_angular_radps: float = 0.40,
) -> ExpertCommand:
    """Return a bounded goal-seeking command with deterministic obstacle avoidance."""

    ranges_tuple = tuple(float(value) for value in ranges)
    angles_tuple = tuple(math.atan2(math.sin(value), math.cos(value)) for value in angles)
    if len(ranges_tuple) != len(angles_tuple) or not ranges_tuple:
        return ExpertCommand(0.0, 0.0, "scan_invalid")
    if not math.isfinite(goal_rho_m) or not math.isfinite(goal_alpha_rad):
        return ExpertCommand(0.0, 0.0, "goal_invalid")
    if goal_rho_m <= 0.15:
        return ExpertCommand(0.0, 0.0, "goal_reached")

    front = _finite_clearance(ranges_tuple, angles_tuple, -0.30, 0.30, 8.0)
    if front < obstacle_threshold_m:
        left = _finite_clearance(ranges_tuple, angles_tuple, 0.25, 1.20, 8.0)
        right = _finite_clearance(ranges_tuple, angles_tuple, -1.20, -0.25, 8.0)
        turn = max_angular_radps if left >= right else -max_angular_radps
        return ExpertCommand(0.02, turn, "avoid_obstacle")

    turn = max(-max_angular_radps, min(max_angular_radps, 1.5 * goal_alpha_rad))
    speed_scale = max(0.25, 1.0 - abs(turn) / max_angular_radps)
    return ExpertCommand(max_linear_mps * speed_scale, turn, "track_goal")
