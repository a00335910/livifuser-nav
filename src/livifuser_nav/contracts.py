"""Canonical Stage 1 data contract from architecture specification v1.1."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StampedValue(Generic[T]):
    """A sensor value timestamped in nanoseconds on one shared clock."""

    timestamp_ns: int
    value: T

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")


@dataclass(frozen=True, slots=True)
class RelativeGoal:
    """Mandatory relative waypoint in the robot frame."""

    rho_m: float
    sin_alpha: float
    cos_alpha: float

    def __post_init__(self) -> None:
        values = (self.rho_m, self.sin_alpha, self.cos_alpha)
        if not all(isfinite(value) for value in values):
            raise ValueError("goal values must be finite")
        if self.rho_m < 0:
            raise ValueError("rho_m must be non-negative")
        norm = self.sin_alpha**2 + self.cos_alpha**2
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("sin_alpha and cos_alpha must describe a unit direction")


@dataclass(frozen=True, slots=True)
class RobotState:
    linear_velocity_mps: float
    angular_velocity_radps: float

    def __post_init__(self) -> None:
        values = (self.linear_velocity_mps, self.angular_velocity_radps)
        if not all(isfinite(value) for value in values):
            raise ValueError("robot state values must be finite")


@dataclass(frozen=True, slots=True)
class PilotSample:
    """One synchronized 10 Hz training view; payloads retain source message types."""

    timestamp_ns: int
    rgb: object
    lidar: object
    action: object
    robot_state: RobotState
    goal: RelativeGoal
    lidar_delta_ns: int

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.lidar_delta_ns < 0:
            raise ValueError("lidar_delta_ns must be an absolute duration")
