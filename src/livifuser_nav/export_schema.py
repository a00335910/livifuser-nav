"""Versioned schema for the Stage 1 goal-conditioned training export.

The exporter writes this schema version into every manifest. Any change to
sample assembly, association policy, or rejection semantics requires a version
bump so that a dataset can always be traced to the rules that produced it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

#: Bumped to 1.3.0: every newly generated export declares whether its source is
#: physical hardware or simulation. Sample assembly semantics are unchanged, so
#: historical 1.1.0 and 1.2.0 manifests remain valid.
EXPORT_SCHEMA_VERSION = "1.3.0"

#: Locked policy loop period from architecture v1.1 section 7.1.
GRID_PERIOD_NS = 100_000_000


class AssociationPolicy(str, Enum):
    """How a stream is sampled relative to the reference observation time."""

    #: Closest source message in either direction. May select a future message.
    NEAREST = "nearest"
    #: Most recent source message at or before the reference time. Causal.
    LATEST_AT_OR_BEFORE = "latest_at_or_before"
    #: Causal, and the held value remains in effect until superseded.
    ZERO_ORDER_HOLD = "zero_order_hold"
    #: Constant for the whole run; validated once rather than per sample.
    RUN_LEVEL = "run_level"


class TimestampSource(str, Enum):
    """Which clock a stream's timestamps actually came from."""

    #: `header.stamp` set by the publisher at capture time.
    HEADER_STAMP = "header_stamp"
    #: rosbag2 receive time. Only a proxy for publication time.
    BAG_RECEIVE = "bag_receive_timestamp"


class RejectionCode(str, Enum):
    """Stable reason codes for excluding a grid tick from the training view.

    A sample records every code that applied plus one primary code, chosen by
    :data:`REJECTION_PRIORITY`, so that summary counts stay stable even when a
    sample fails several checks at once.
    """

    CAMERA_MISSING = "camera_missing"
    CAMERA_STALE = "camera_stale"
    DUPLICATE_CAMERA_FRAME = "duplicate_camera_frame"
    CAMERA_PAYLOAD_INVALID = "camera_payload_invalid"

    LIDAR_MISSING = "lidar_missing"
    LIDAR_STALE = "lidar_stale"
    LIDAR_PAYLOAD_INVALID = "lidar_payload_invalid"

    ODOM_MISSING = "odom_missing"
    ODOM_STALE = "odom_stale"
    ODOM_INVALID = "odom_invalid"

    GOAL_MISSING = "goal_missing"
    GOAL_STALE = "goal_stale"
    GOAL_INVALID = "goal_invalid"

    ACTION_MISSING = "action_missing"
    ACTION_STALE = "action_stale"
    ACTION_INVALID = "action_invalid"

    TF_UNAVAILABLE = "tf_unavailable"
    CALIBRATION_MISMATCH = "calibration_mismatch"
    TIMESTAMP_REGRESSION = "timestamp_regression"

    # Enforced by the training-time windower using exporter contiguity data,
    # not by per-tick export. Defined here so both stages share one vocabulary.
    CONTEXT_INCOMPLETE = "context_incomplete"
    ACTION_HORIZON_INCOMPLETE = "action_horizon_incomplete"
    SEQUENCE_CROSSES_GAP = "sequence_crosses_gap"


#: Most diagnostic reason first. A missing stream explains more than a stale one,
#: and a run-level calibration fault explains more than any per-sample timing.
REJECTION_PRIORITY: tuple[RejectionCode, ...] = (
    RejectionCode.CALIBRATION_MISMATCH,
    RejectionCode.TIMESTAMP_REGRESSION,
    RejectionCode.CAMERA_MISSING,
    RejectionCode.CAMERA_PAYLOAD_INVALID,
    RejectionCode.DUPLICATE_CAMERA_FRAME,
    RejectionCode.CAMERA_STALE,
    RejectionCode.LIDAR_MISSING,
    RejectionCode.LIDAR_PAYLOAD_INVALID,
    RejectionCode.LIDAR_STALE,
    RejectionCode.ACTION_MISSING,
    RejectionCode.ACTION_STALE,
    RejectionCode.ACTION_INVALID,
    RejectionCode.GOAL_MISSING,
    RejectionCode.GOAL_INVALID,
    RejectionCode.GOAL_STALE,
    RejectionCode.ODOM_MISSING,
    RejectionCode.ODOM_INVALID,
    RejectionCode.ODOM_STALE,
    RejectionCode.TF_UNAVAILABLE,
)


#: The only run-level code an operator override may downgrade to a warning.
#:
#: Deliberately a one-element set rather than a broad "clear everything" switch.
#: A timestamp regression, a missing transform, or a mid-run geometry change are
#: not calibration disagreements and must survive any calibration override.
OVERRIDABLE_RUN_LEVEL_CODES: frozenset[RejectionCode] = frozenset(
    {RejectionCode.CALIBRATION_MISMATCH}
)


def apply_run_level_override(
    codes: Sequence[RejectionCode], *, allow_calibration_mismatch: bool
) -> tuple[tuple[RejectionCode, ...], tuple[RejectionCode, ...]]:
    """Split run-level codes into those retained and those downgraded.

    Returns ``(retained, downgraded)``. Only codes in
    :data:`OVERRIDABLE_RUN_LEVEL_CODES` can ever appear in ``downgraded``, so an
    override cannot silently excuse an unrelated fault.
    """

    unique = tuple(dict.fromkeys(codes))
    if not allow_calibration_mismatch:
        return unique, ()
    retained = tuple(
        code for code in unique if code not in OVERRIDABLE_RUN_LEVEL_CODES
    )
    downgraded = tuple(code for code in unique if code in OVERRIDABLE_RUN_LEVEL_CODES)
    return retained, downgraded


def primary_rejection(codes: list[RejectionCode]) -> RejectionCode | None:
    """Return the single stable reason code for a set of applied codes."""

    if not codes:
        return None
    applied = set(codes)
    for candidate in REJECTION_PRIORITY:
        if candidate in applied:
            return candidate
    # Any code outside the priority table still yields a deterministic answer.
    return sorted(applied, key=lambda code: code.value)[0]


@dataclass(frozen=True, slots=True)
class StreamRule:
    """Association policy and eligibility bound for one recorded stream."""

    topic: str
    policy: AssociationPolicy
    timestamp_source: TimestampSource
    max_delta_ns: int | None = None
    message_type: str = ""

    def as_manifest(self) -> dict[str, object]:
        return {
            "topic": self.topic,
            "message_type": self.message_type,
            "policy": self.policy.value,
            "timestamp_source": self.timestamp_source.value,
            "max_delta_ms": (
                None if self.max_delta_ns is None else self.max_delta_ns / 1_000_000
            ),
        }


@dataclass(frozen=True, slots=True)
class ExportPolicy:
    """The complete, manifest-serializable rule set for one export run."""

    grid_period_ns: int = GRID_PERIOD_NS
    camera: StreamRule = field(
        default_factory=lambda: StreamRule(
            topic="/camera/image_raw",
            policy=AssociationPolicy.NEAREST,
            timestamp_source=TimestampSource.HEADER_STAMP,
            max_delta_ns=50_000_000,
            message_type="sensor_msgs/msg/Image",
        )
    )
    lidar: StreamRule = field(
        default_factory=lambda: StreamRule(
            topic="/scan",
            policy=AssociationPolicy.NEAREST,
            timestamp_source=TimestampSource.HEADER_STAMP,
            max_delta_ns=75_000_000,
            message_type="sensor_msgs/msg/LaserScan",
        )
    )
    odometry: StreamRule = field(
        default_factory=lambda: StreamRule(
            topic="/odom",
            policy=AssociationPolicy.LATEST_AT_OR_BEFORE,
            timestamp_source=TimestampSource.HEADER_STAMP,
            max_delta_ns=100_000_000,
            message_type="nav_msgs/msg/Odometry",
        )
    )
    goal: StreamRule = field(
        default_factory=lambda: StreamRule(
            topic="/livifuser/goal_relative",
            policy=AssociationPolicy.LATEST_AT_OR_BEFORE,
            timestamp_source=TimestampSource.HEADER_STAMP,
            max_delta_ns=150_000_000,
            message_type="livifuser_interfaces/msg/RelativeGoal",
        )
    )
    action: StreamRule = field(
        default_factory=lambda: StreamRule(
            topic="/cmd_vel",
            policy=AssociationPolicy.ZERO_ORDER_HOLD,
            timestamp_source=TimestampSource.BAG_RECEIVE,
            max_delta_ns=150_000_000,
            message_type="geometry_msgs/msg/Twist",
        )
    )

    def streams(self) -> dict[str, StreamRule]:
        return {
            "camera": self.camera,
            "lidar": self.lidar,
            "odometry": self.odometry,
            "goal": self.goal,
            "action": self.action,
        }

    def as_manifest(self) -> dict[str, object]:
        return {
            "grid_period_ms": self.grid_period_ns / 1_000_000,
            "grid_rate_hz": 1_000_000_000 / self.grid_period_ns,
            "streams": {name: rule.as_manifest() for name, rule in self.streams().items()},
            "camera_info": {
                "policy": AssociationPolicy.RUN_LEVEL.value,
                "note": "Calibration identity validated once per run, not per sample.",
            },
            "static_tf": {
                "policy": AssociationPolicy.RUN_LEVEL.value,
                "note": (
                    "base_scan -> camera -> camera_optical_frame is a static chain; "
                    "it is checked against the accepted extrinsics once per run."
                ),
            },
        }
