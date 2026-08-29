"""Platform-neutral fail-safe policy for the Stage 1 command watchdog."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class DecisionReason(str, Enum):
    FRESH = "fresh"
    MISSING = "intent_missing"
    STALE = "intent_stale"
    INVALID = "intent_invalid"
    CLOCK_REGRESSION = "monotonic_clock_regression"
    PUBLISHER_CONFLICT = "cmd_vel_publisher_conflict"
    GRAPH_UNKNOWN = "cmd_vel_graph_unknown"
    GRAPH_STALE = "cmd_vel_graph_stale"


@dataclass(frozen=True)
class VelocityIntent:
    linear_mps: float
    angular_radps: float
    structurally_valid: bool = True

    @property
    def is_valid(self) -> bool:
        return (
            self.structurally_valid
            and math.isfinite(self.linear_mps)
            and math.isfinite(self.angular_radps)
        )


ZERO_VELOCITY = VelocityIntent(0.0, 0.0)


def stale_detection_bound_ms(
    intent_timeout_ms: float,
    max_command_gap_ms: float,
) -> float:
    """Return the latest acceptable stale detection at the admitted tick gap."""

    values = (intent_timeout_ms, max_command_gap_ms)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("stale-detection inputs must be finite and positive")
    return intent_timeout_ms + max_command_gap_ms


def intent_from_message_fields(
    *,
    linear_x: float,
    linear_y: float,
    linear_z: float,
    angular_x: float,
    angular_y: float,
    angular_z: float,
    frame_id: str,
    stamp_is_set: bool,
    expected_frame_id: str,
) -> VelocityIntent:
    """Validate the planar stamped-intent boundary without importing ROS."""

    unsupported = (linear_y, linear_z, angular_x, angular_y)
    structurally_valid = (
        frame_id == expected_frame_id
        and stamp_is_set
        and all(math.isfinite(value) and abs(value) <= 1e-9 for value in unsupported)
    )
    return VelocityIntent(float(linear_x), float(angular_z), structurally_valid)


def count_external_publishers(
    publisher_nodes: list[tuple[str, str]],
    own_node: tuple[str, str],
) -> int:
    """Count command publishers other than exactly one endpoint owned by this node."""

    own_count = sum(identity == own_node for identity in publisher_nodes)
    return len(publisher_nodes) - min(1, own_count)


@dataclass(frozen=True)
class WatchdogLimits:
    timeout_s: float = 0.25
    max_abs_linear_mps: float = 0.10
    max_abs_angular_radps: float = 0.50

    def __post_init__(self) -> None:
        values = (self.timeout_s, self.max_abs_linear_mps, self.max_abs_angular_radps)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("watchdog timeout and velocity limits must be finite and positive")


@dataclass(frozen=True)
class CommandDecision:
    output: VelocityIntent
    requested: VelocityIntent
    reason: DecisionReason
    intent_present: bool
    clamped: bool
    intent_age_s: float


def decide_command(
    intent: VelocityIntent | None,
    received_monotonic_s: float | None,
    now_monotonic_s: float,
    limits: WatchdogLimits,
) -> CommandDecision:
    """Select a bounded output, failing to zero for every uncertain state."""

    if intent is None or received_monotonic_s is None:
        return CommandDecision(
            ZERO_VELOCITY,
            ZERO_VELOCITY,
            DecisionReason.MISSING,
            False,
            False,
            -1.0,
        )

    if not intent.is_valid:
        return CommandDecision(
            ZERO_VELOCITY,
            intent,
            DecisionReason.INVALID,
            True,
            False,
            max(0.0, now_monotonic_s - received_monotonic_s),
        )

    age_s = now_monotonic_s - received_monotonic_s
    if not math.isfinite(age_s) or age_s < 0.0:
        return CommandDecision(
            ZERO_VELOCITY,
            intent,
            DecisionReason.CLOCK_REGRESSION,
            True,
            False,
            age_s,
        )
    if age_s >= limits.timeout_s:
        return CommandDecision(
            ZERO_VELOCITY,
            intent,
            DecisionReason.STALE,
            True,
            False,
            age_s,
        )

    linear = max(-limits.max_abs_linear_mps, min(limits.max_abs_linear_mps, intent.linear_mps))
    angular = max(
        -limits.max_abs_angular_radps,
        min(limits.max_abs_angular_radps, intent.angular_radps),
    )
    output = VelocityIntent(linear, angular)
    return CommandDecision(
        output,
        intent,
        DecisionReason.FRESH,
        True,
        output != intent,
        age_s,
    )


def publisher_conflict_decision(
    previous: CommandDecision,
) -> CommandDecision:
    """Force zero when another node can bypass the watchdog on `/cmd_vel`."""

    return _forced_zero(previous, DecisionReason.PUBLISHER_CONFLICT)


def _forced_zero(previous: CommandDecision, reason: DecisionReason) -> CommandDecision:
    return CommandDecision(
        ZERO_VELOCITY,
        previous.requested,
        reason,
        previous.intent_present,
        previous.clamped,
        previous.intent_age_s,
    )


@dataclass(frozen=True)
class GraphCache:
    """Last known `/cmd_vel` publisher picture, sampled off the command path.

    Discovery queries can block for seconds under DDS load, so the command timer
    never performs one. It consumes this snapshot instead, and treats an absent
    or aged snapshot as unsafe rather than as "no conflict".
    """

    external_publishers: int | None = None
    probed_monotonic_s: float | None = None


def apply_graph_cache(
    previous: CommandDecision,
    cache: GraphCache,
    now_monotonic_s: float,
    max_age_s: float,
) -> CommandDecision:
    """Fold the cached publisher picture into a decision, failing to zero.

    Unknown and stale both force zero: never having probed is not evidence that
    no other publisher exists, and neither is a probe from a minute ago.
    """

    if cache.external_publishers is None or cache.probed_monotonic_s is None:
        return _forced_zero(previous, DecisionReason.GRAPH_UNKNOWN)

    age_s = now_monotonic_s - cache.probed_monotonic_s
    if not math.isfinite(age_s) or age_s < 0.0 or age_s > max_age_s:
        return _forced_zero(previous, DecisionReason.GRAPH_STALE)

    if cache.external_publishers > 0:
        return publisher_conflict_decision(previous)

    return previous
