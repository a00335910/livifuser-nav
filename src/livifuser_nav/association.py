"""Multi-rate timestamp association used by the rosbag2 export path.

Two selection policies are provided because the streams differ in kind:

* :meth:`TimeSeries.nearest` may return a message recorded *after* the reference
  time. It is correct for the LiDAR scan, which architecture v1.1 section 7.2
  locks to nearest-timestamp association.
* :meth:`TimeSeries.latest_at_or_before` never looks forward. It is required for
  odometry, goal, and action, because a future value was not available to the
  online policy and using one would leak information into the training view.

Signed deltas follow one convention throughout: ``source - reference``, so a
positive delta means the selected message came from the future.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from .contracts import StampedValue

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Selection(Generic[T]):
    """One selected source message, with its offset from the reference time."""

    sample: StampedValue[T]
    signed_delta_ns: int
    index: int

    @property
    def delta_ns(self) -> int:
        """Absolute offset from the reference time."""

        return abs(self.signed_delta_ns)

    @property
    def is_from_future(self) -> bool:
        """True when the selected message post-dates the reference time."""

        return self.signed_delta_ns > 0

    def within(self, max_delta_ns: int | None) -> bool:
        """Whether this selection satisfies an eligibility bound."""

        return max_delta_ns is None or self.delta_ns <= max_delta_ns


def _validate_sorted(timestamps: Sequence[int]) -> None:
    """Reject unsorted or negative timestamps in a single pass."""

    previous: int | None = None
    for timestamp in timestamps:
        if timestamp < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if previous is not None and timestamp < previous:
            raise ValueError("samples must be sorted by timestamp_ns")
        previous = timestamp


class TimeSeries(Generic[T]):
    """An immutable, pre-validated stream supporting O(log n) lookups.

    Sortedness is checked once here rather than on every lookup, so associating
    a full bag stays O(m log n) instead of degrading to O(m * n log n).
    """

    __slots__ = ("_samples", "_timestamps")

    def __init__(self, samples: Iterable[StampedValue[T]]) -> None:
        self._samples: tuple[StampedValue[T], ...] = tuple(samples)
        self._timestamps: tuple[int, ...] = tuple(
            sample.timestamp_ns for sample in self._samples
        )
        _validate_sorted(self._timestamps)

    def __len__(self) -> int:
        return len(self._samples)

    def __bool__(self) -> bool:
        return bool(self._samples)

    @property
    def samples(self) -> tuple[StampedValue[T], ...]:
        return self._samples

    @property
    def timestamps(self) -> tuple[int, ...]:
        return self._timestamps

    def nearest(self, reference_ns: int) -> Selection[T] | None:
        """Closest message in either direction, preferring the earlier on a tie.

        The whole stream is searched. Candidates are never pre-trimmed to an
        overlap window, because discarding a bracketing message just outside the
        window silently biases the reported offset.
        """

        if not self._samples:
            return None
        if reference_ns < 0:
            raise ValueError("reference_ns must be non-negative")

        index = bisect_left(self._timestamps, reference_ns)
        if index == 0:
            chosen = 0
        elif index == len(self._samples):
            chosen = len(self._samples) - 1
        else:
            before_delta = reference_ns - self._timestamps[index - 1]
            after_delta = self._timestamps[index] - reference_ns
            chosen = index - 1 if before_delta <= after_delta else index
        return self._selection(chosen, reference_ns)

    def latest_at_or_before(self, reference_ns: int) -> Selection[T] | None:
        """Most recent message at or before the reference time, else ``None``.

        Returning ``None`` distinguishes "no value had ever been published" from
        "a value exists but is too old", which the exporter reports as separate
        rejection codes.
        """

        if not self._samples:
            return None
        if reference_ns < 0:
            raise ValueError("reference_ns must be non-negative")

        index = bisect_right(self._timestamps, reference_ns)
        if index == 0:
            return None
        return self._selection(index - 1, reference_ns)

    def _selection(self, index: int, reference_ns: int) -> Selection[T]:
        sample = self._samples[index]
        return Selection(
            sample=sample,
            signed_delta_ns=sample.timestamp_ns - reference_ns,
            index=index,
        )


def nearest_sample(
    timestamp_ns: int,
    samples: Sequence[StampedValue[T]],
    *,
    max_delta_ns: int | None = None,
) -> StampedValue[T]:
    """Return the nearest sample, preferring the earlier one on an exact tie.

    Raises :class:`ValueError` when the nearest sample violates ``max_delta_ns``.
    Prefer :class:`TimeSeries` when associating many reference times against one
    stream; this helper revalidates its input on every call.
    """

    if not samples:
        raise ValueError("samples must not be empty")
    if timestamp_ns < 0:
        raise ValueError("timestamp_ns must be non-negative")
    if max_delta_ns is not None and max_delta_ns < 0:
        raise ValueError("max_delta_ns must be non-negative")

    selection = TimeSeries(samples).nearest(timestamp_ns)
    if selection is None:  # unreachable: samples is non-empty
        raise ValueError("samples must not be empty")
    if not selection.within(max_delta_ns):
        raise ValueError(
            f"nearest sample is {selection.delta_ns} ns away (limit: {max_delta_ns} ns)"
        )
    return selection.sample
