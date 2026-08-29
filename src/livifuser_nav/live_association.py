"""Deterministic causal association for the frozen live-policy contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STREAMS = ("rgb", "scan", "odometry", "goal")


@dataclass(frozen=True, slots=True)
class ArrivedMessage:
    stamp_ns: int
    arrival_sequence: int
    arrival_monotonic_ns: int
    payload: Any


@dataclass(frozen=True, slots=True)
class AssociatedContext:
    rgb: ArrivedMessage
    scan: ArrivedMessage
    odometry: ArrivedMessage
    goal: ArrivedMessage


@dataclass(frozen=True, slots=True)
class AssociationResult:
    accepted: bool
    reason: str
    context: AssociatedContext | None = None


class LiveAssociator:
    """Buffer only already-arrived messages and consume each RGB stamp once.

    The caller owns payload validation. A duplicate or per-stream header
    regression is reported as an integrity failure and clears every buffer.
    """

    def __init__(
        self,
        *,
        scan_max_delta_ns: int = 75_000_000,
        odometry_max_age_ns: int = 100_000_000,
        goal_max_age_ns: int = 150_000_000,
        maximum_messages_per_stream: int = 256,
    ) -> None:
        if min(scan_max_delta_ns, odometry_max_age_ns, goal_max_age_ns) < 0:
            raise ValueError("association tolerances must be non-negative")
        if maximum_messages_per_stream < 2:
            raise ValueError("maximum_messages_per_stream must be at least two")
        self.scan_max_delta_ns = int(scan_max_delta_ns)
        self.odometry_max_age_ns = int(odometry_max_age_ns)
        self.goal_max_age_ns = int(goal_max_age_ns)
        self.maximum_messages_per_stream = int(maximum_messages_per_stream)
        self._arrival_sequence = 0
        self._buffers: dict[str, list[ArrivedMessage]] = {name: [] for name in STREAMS}
        self._last_stamp: dict[str, int | None] = {name: None for name in STREAMS}
        self._last_tick_ns: int | None = None
        self._last_consumed_rgb_ns: int | None = None
        self.reset_count = 0

    def reset(self) -> None:
        for values in self._buffers.values():
            values.clear()
        for name in STREAMS:
            self._last_stamp[name] = None
        self._last_tick_ns = None
        self._last_consumed_rgb_ns = None
        self.reset_count += 1

    def push(
        self,
        stream: str,
        *,
        stamp_ns: int,
        arrival_monotonic_ns: int,
        payload: Any,
    ) -> AssociationResult:
        if stream not in self._buffers:
            raise ValueError(f"unknown stream: {stream}")
        if stamp_ns < 0 or arrival_monotonic_ns < 0:
            self.reset()
            return AssociationResult(False, f"{stream}_negative_time")
        previous = self._last_stamp[stream]
        if previous is not None and stamp_ns <= previous:
            reason = f"{stream}_{'duplicate' if stamp_ns == previous else 'regression'}"
            self.reset()
            return AssociationResult(False, reason)
        self._arrival_sequence += 1
        message = ArrivedMessage(
            stamp_ns=int(stamp_ns),
            arrival_sequence=self._arrival_sequence,
            arrival_monotonic_ns=int(arrival_monotonic_ns),
            payload=payload,
        )
        self._buffers[stream].append(message)
        self._buffers[stream] = self._buffers[stream][-self.maximum_messages_per_stream :]
        self._last_stamp[stream] = int(stamp_ns)
        return AssociationResult(True, "buffered")

    def select(self, tick_ns: int) -> AssociationResult:
        tick_ns = int(tick_ns)
        if tick_ns < 0:
            self.reset()
            return AssociationResult(False, "clock_regression_or_duplicate_tick")
        if self._last_tick_ns is not None and tick_ns <= self._last_tick_ns:
            # Duplicate sim-time timer catch-up. Reject the tick but keep
            # buffers: resetting here wipes the K=8 history on every overfire.
            return AssociationResult(False, "clock_regression_or_duplicate_tick")
        self._last_tick_ns = tick_ns
        eligible_rgb = [
            message
            for message in self._buffers["rgb"]
            if message.stamp_ns <= tick_ns
            and (
                self._last_consumed_rgb_ns is None
                or message.stamp_ns > self._last_consumed_rgb_ns
            )
        ]
        if not eligible_rgb:
            self.reset()
            return AssociationResult(False, "rgb_missing")
        rgb = max(eligible_rgb, key=lambda item: (item.stamp_ns, item.arrival_sequence))
        self._last_consumed_rgb_ns = rgb.stamp_ns

        scans = self._buffers["scan"]
        if not scans:
            self.reset()
            return AssociationResult(False, "scan_missing")
        scan = min(
            scans,
            key=lambda item: (
                abs(item.stamp_ns - rgb.stamp_ns),
                item.stamp_ns,
                item.arrival_sequence,
            ),
        )
        if abs(scan.stamp_ns - rgb.stamp_ns) > self.scan_max_delta_ns:
            self.reset()
            return AssociationResult(False, "scan_stale")

        odometry = self._latest_at_or_before("odometry", rgb.stamp_ns)
        if odometry is None:
            self.reset()
            return AssociationResult(False, "odometry_missing")
        if rgb.stamp_ns - odometry.stamp_ns > self.odometry_max_age_ns:
            self.reset()
            return AssociationResult(False, "odometry_stale")

        goal = self._latest_at_or_before("goal", rgb.stamp_ns)
        if goal is None:
            self.reset()
            return AssociationResult(False, "goal_missing")
        if rgb.stamp_ns - goal.stamp_ns > self.goal_max_age_ns:
            self.reset()
            return AssociationResult(False, "goal_stale")

        return AssociationResult(
            True,
            "accepted",
            AssociatedContext(rgb=rgb, scan=scan, odometry=odometry, goal=goal),
        )

    def _latest_at_or_before(self, stream: str, stamp_ns: int) -> ArrivedMessage | None:
        eligible = [item for item in self._buffers[stream] if item.stamp_ns <= stamp_ns]
        if not eligible:
            return None
        return max(eligible, key=lambda item: (item.stamp_ns, item.arrival_sequence))
