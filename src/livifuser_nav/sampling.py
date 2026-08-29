"""Assemble the locked 10 Hz training view from recorded sensor streams.

This module is deliberately free of ROS and numpy imports so it can be unit
tested on Windows. The rosbag2 reader converts each message into a
:class:`Payload` of plain Python data plus a validity verdict, and everything
here operates on those.

Association semantics, per architecture v1.1 section 7:

* The 10 Hz grid is an artifact of our own construction, so the camera frame
  nearest each tick becomes the sample's *observation timestamp*.
* Every other stream is then associated against that observation timestamp, not
  against the grid tick, because the camera is the observation reference.
* Odometry, goal, and action use causal lookups so the training view can never
  contain a value the online policy could not have seen.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .association import Selection, TimeSeries
from .export_schema import (
    GRID_PERIOD_NS,
    AssociationPolicy,
    ExportPolicy,
    RejectionCode,
    StreamRule,
    primary_rejection,
)

#: Streams whose failure never rejects a sample in the sensor-only view.
ACTION_CODES = frozenset(
    {
        RejectionCode.ACTION_MISSING,
        RejectionCode.ACTION_STALE,
        RejectionCode.ACTION_INVALID,
    }
)


@dataclass(frozen=True, slots=True)
class Payload:
    """One decoded source message, already checked for structural validity."""

    data: Mapping[str, object]
    valid: bool = True
    invalid_code: RejectionCode | None = None


@dataclass(frozen=True, slots=True)
class StreamSelection:
    """The association outcome for one stream at one grid tick."""

    name: str
    policy: AssociationPolicy
    source_timestamp_ns: int | None = None
    signed_delta_ns: int | None = None
    source_index: int | None = None
    eligible: bool = False
    payload: Payload | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "policy": self.policy.value,
            "source_timestamp_ns": self.source_timestamp_ns,
            "signed_delta_ms": (
                None
                if self.signed_delta_ns is None
                else self.signed_delta_ns / 1_000_000
            ),
            "abs_delta_ms": (
                None
                if self.signed_delta_ns is None
                else abs(self.signed_delta_ns) / 1_000_000
            ),
            "from_future": (
                None if self.signed_delta_ns is None else self.signed_delta_ns > 0
            ),
            "source_index": self.source_index,
            "eligible": self.eligible,
        }


@dataclass(frozen=True, slots=True)
class GridSample:
    """One candidate training sample, accepted or rejected with reasons."""

    grid_index: int
    grid_timestamp_ns: int
    observation_timestamp_ns: int | None
    selections: Mapping[str, StreamSelection]
    rejection_codes: tuple[RejectionCode, ...] = ()
    advisory_codes: tuple[RejectionCode, ...] = ()
    segment_id: int | None = None

    @property
    def accepted(self) -> bool:
        return not self.rejection_codes

    @property
    def primary_rejection_code(self) -> RejectionCode | None:
        return primary_rejection(list(self.rejection_codes))

    def as_record(self) -> dict[str, object]:
        return {
            "grid_index": self.grid_index,
            "grid_timestamp_ns": self.grid_timestamp_ns,
            "observation_timestamp_ns": self.observation_timestamp_ns,
            "grid_offset_ms": (
                None
                if self.observation_timestamp_ns is None
                else (self.observation_timestamp_ns - self.grid_timestamp_ns) / 1_000_000
            ),
            "accepted": self.accepted,
            "segment_id": self.segment_id,
            "primary_rejection_code": (
                None
                if self.primary_rejection_code is None
                else self.primary_rejection_code.value
            ),
            "rejection_codes": [code.value for code in self.rejection_codes],
            "advisory_codes": [code.value for code in self.advisory_codes],
            "streams": {
                name: selection.as_record() for name, selection in self.selections.items()
            },
        }


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Every candidate tick, plus the contiguous run structure of the accepted ones."""

    samples: tuple[GridSample, ...]
    segment_lengths: tuple[int, ...] = ()
    grid_start_ns: int = 0
    grid_period_ns: int = GRID_PERIOD_NS
    streams_present: Mapping[str, int] = field(default_factory=dict)

    @property
    def accepted(self) -> tuple[GridSample, ...]:
        return tuple(sample for sample in self.samples if sample.accepted)

    def rejection_counts(self) -> dict[str, int]:
        """Count of samples per primary rejection code."""

        counts: dict[str, int] = {}
        for sample in self.samples:
            code = sample.primary_rejection_code
            if code is not None:
                counts[code.value] = counts.get(code.value, 0) + 1
        return dict(sorted(counts.items()))

    def all_rejection_counts(self) -> dict[str, int]:
        """Count of every applied code, so overlapping failures stay visible."""

        counts: dict[str, int] = {}
        for sample in self.samples:
            for code in sample.rejection_codes:
                counts[code.value] = counts.get(code.value, 0) + 1
        return dict(sorted(counts.items()))

    def advisory_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in self.samples:
            for code in sample.advisory_codes:
                counts[code.value] = counts.get(code.value, 0) + 1
        return dict(sorted(counts.items()))

    def windowable_count(self, context_k: int, horizon_h: int) -> int:
        """Samples usable as a window origin for GRU context K and ACT horizon H.

        A window needs ``context_k`` samples up to and including the origin and
        ``horizon_h`` actions from the origin forward, all inside one contiguous
        segment. This is what makes ``context_incomplete``,
        ``action_horizon_incomplete``, and ``sequence_crosses_gap`` enforceable
        at training time without re-reading the bag.
        """

        needed = context_k + horizon_h - 1
        return sum(max(0, length - needed + 1) for length in self.segment_lengths)


def build_grid(start_ns: int, end_ns: int, period_ns: int = GRID_PERIOD_NS) -> list[int]:
    """Inclusive 10 Hz tick sequence covering ``[start_ns, end_ns]``."""

    if period_ns <= 0:
        raise ValueError("period_ns must be positive")
    if start_ns < 0:
        raise ValueError("start_ns must be non-negative")
    if end_ns < start_ns:
        return []
    count = (end_ns - start_ns) // period_ns + 1
    return [start_ns + index * period_ns for index in range(count)]


def _select(
    series: TimeSeries[Payload] | None,
    rule: StreamRule,
    reference_ns: int,
    *,
    override_policy: AssociationPolicy | None = None,
) -> Selection[Payload] | None:
    if series is None or not series:
        return None
    policy = override_policy or rule.policy
    if policy is AssociationPolicy.NEAREST:
        return series.nearest(reference_ns)
    return series.latest_at_or_before(reference_ns)


def _resolve(
    name: str,
    series: TimeSeries[Payload] | None,
    rule: StreamRule,
    reference_ns: int,
    codes: tuple[RejectionCode, RejectionCode],
    *,
    override_policy: AssociationPolicy | None = None,
) -> tuple[StreamSelection, list[RejectionCode]]:
    """Associate one stream and return its selection plus any failure codes."""

    missing_code, stale_code = codes
    policy = override_policy or rule.policy
    selection = _select(series, rule, reference_ns, override_policy=override_policy)
    if selection is None:
        return StreamSelection(name=name, policy=policy), [missing_code]

    applied: list[RejectionCode] = []
    eligible = selection.within(rule.max_delta_ns)
    if not eligible:
        applied.append(stale_code)
    payload = selection.sample.value
    if not payload.valid and payload.invalid_code is not None:
        applied.append(payload.invalid_code)

    return (
        StreamSelection(
            name=name,
            policy=policy,
            source_timestamp_ns=selection.sample.timestamp_ns,
            signed_delta_ns=selection.signed_delta_ns,
            source_index=selection.index,
            eligible=eligible,
            payload=payload,
        ),
        applied,
    )


def assemble_samples(
    *,
    camera: TimeSeries[Payload],
    lidar: TimeSeries[Payload] | None = None,
    odometry: TimeSeries[Payload] | None = None,
    goal: TimeSeries[Payload] | None = None,
    action: TimeSeries[Payload] | None = None,
    policy: ExportPolicy | None = None,
    require_action: bool = True,
    lidar_causal: bool = False,
    run_level_codes: tuple[RejectionCode, ...] = (),
) -> ExportResult:
    """Build the 10 Hz training view from pre-decoded streams.

    ``require_action`` distinguishes the two legitimate views of a bag. With it
    enabled (the policy view) a sample without a usable command is rejected. With
    it disabled (the sensor view) action failures are recorded as advisory only,
    which is how a stationary bag remains useful for replay, preprocessing,
    timing, and OOD work while producing no action-valid training samples.

    ``lidar_causal`` swaps the spec-locked nearest LiDAR association for a causal
    one. It exists to measure the train/deploy asymmetry of nearest association,
    not as the default.
    """

    rules = (policy or ExportPolicy())
    period_ns = rules.grid_period_ns
    if not camera:
        return ExportResult(samples=(), grid_period_ns=period_ns)

    grid = build_grid(camera.timestamps[0], camera.timestamps[-1], period_ns)
    lidar_override = (
        AssociationPolicy.LATEST_AT_OR_BEFORE if lidar_causal else None
    )

    # First pass: associate every stream, deferring duplicate-frame resolution.
    drafts: list[tuple[int, int, dict[str, StreamSelection], list[RejectionCode]]] = []
    for grid_index, tick_ns in enumerate(grid):
        codes: list[RejectionCode] = list(run_level_codes)
        selections: dict[str, StreamSelection] = {}

        camera_selection, camera_codes = _resolve(
            "camera",
            camera,
            rules.camera,
            tick_ns,
            (RejectionCode.CAMERA_MISSING, RejectionCode.CAMERA_STALE),
        )
        selections["camera"] = camera_selection
        codes.extend(camera_codes)

        # Fall back to the grid tick so other streams still report honest deltas
        # even when the camera itself failed.
        observation_ns = camera_selection.source_timestamp_ns
        reference_ns = observation_ns if observation_ns is not None else tick_ns

        for name, rule, pair, override in (
            (
                "lidar",
                rules.lidar,
                (RejectionCode.LIDAR_MISSING, RejectionCode.LIDAR_STALE),
                lidar_override,
            ),
            (
                "odometry",
                rules.odometry,
                (RejectionCode.ODOM_MISSING, RejectionCode.ODOM_STALE),
                None,
            ),
            (
                "goal",
                rules.goal,
                (RejectionCode.GOAL_MISSING, RejectionCode.GOAL_STALE),
                None,
            ),
            (
                "action",
                rules.action,
                (RejectionCode.ACTION_MISSING, RejectionCode.ACTION_STALE),
                None,
            ),
        ):
            series = {"lidar": lidar, "odometry": odometry, "goal": goal, "action": action}[
                name
            ]
            selection, applied = _resolve(
                name, series, rule, reference_ns, pair, override_policy=override
            )
            selections[name] = selection
            codes.extend(applied)

        drafts.append((grid_index, tick_ns, selections, codes))

    # Second pass: a camera frame may only anchor one sample. During a dropout
    # two ticks can select the same frame; the closer tick keeps it.
    best_tick_for_frame: dict[int, int] = {}
    for grid_index, _tick_ns, selections, _codes in drafts:
        camera_selection = selections["camera"]
        frame_index = camera_selection.source_index
        if frame_index is None or camera_selection.signed_delta_ns is None:
            continue
        incumbent = best_tick_for_frame.get(frame_index)
        if incumbent is None:
            best_tick_for_frame[frame_index] = grid_index
            continue
        incumbent_delta = abs(drafts[incumbent][2]["camera"].signed_delta_ns or 0)
        if abs(camera_selection.signed_delta_ns) < incumbent_delta:
            best_tick_for_frame[frame_index] = grid_index

    samples: list[GridSample] = []
    for grid_index, tick_ns, selections, codes in drafts:
        frame_index = selections["camera"].source_index
        if frame_index is not None and best_tick_for_frame.get(frame_index) != grid_index:
            codes.append(RejectionCode.DUPLICATE_CAMERA_FRAME)

        if require_action:
            rejections = codes
            advisories: list[RejectionCode] = []
        else:
            rejections = [code for code in codes if code not in ACTION_CODES]
            advisories = [code for code in codes if code in ACTION_CODES]

        samples.append(
            GridSample(
                grid_index=grid_index,
                grid_timestamp_ns=tick_ns,
                observation_timestamp_ns=selections["camera"].source_timestamp_ns,
                selections=selections,
                rejection_codes=tuple(dict.fromkeys(rejections)),
                advisory_codes=tuple(dict.fromkeys(advisories)),
            )
        )

    samples_with_segments, segment_lengths = _assign_segments(samples)
    return ExportResult(
        samples=tuple(samples_with_segments),
        segment_lengths=tuple(segment_lengths),
        grid_start_ns=grid[0] if grid else 0,
        grid_period_ns=period_ns,
        streams_present={
            "camera": len(camera),
            "lidar": len(lidar) if lidar else 0,
            "odometry": len(odometry) if odometry else 0,
            "goal": len(goal) if goal else 0,
            "action": len(action) if action else 0,
        },
    )


def _assign_segments(
    samples: list[GridSample],
) -> tuple[list[GridSample], list[int]]:
    """Group accepted samples into runs of consecutive grid ticks.

    Any rejected tick breaks the run, so a downstream K=8 / H=8 window can never
    silently span a sensor dropout.
    """

    resolved: list[GridSample] = []
    lengths: list[int] = []
    segment_id = -1
    previous_index: int | None = None

    for sample in samples:
        if not sample.accepted:
            resolved.append(sample)
            previous_index = None
            continue
        if previous_index is None or sample.grid_index != previous_index + 1:
            segment_id += 1
            lengths.append(0)
        lengths[segment_id] += 1
        previous_index = sample.grid_index
        resolved.append(
            GridSample(
                grid_index=sample.grid_index,
                grid_timestamp_ns=sample.grid_timestamp_ns,
                observation_timestamp_ns=sample.observation_timestamp_ns,
                selections=sample.selections,
                rejection_codes=sample.rejection_codes,
                advisory_codes=sample.advisory_codes,
                segment_id=segment_id,
            )
        )
    return resolved, lengths
