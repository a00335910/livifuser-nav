"""Offline characterization of the physical LDS-03 observation process.

This module turns recorded ``sensor_msgs/msg/LaserScan`` content into measured
statistics that a simulated scan source can later be asked to reproduce. It is
deliberately platform-neutral and ROS-free: everything that decides whether an
observation is usable lives here, where it is unit tested, rather than in the
decoding script.

Three distinctions carry most of the weight and are easy to get wrong:

*Bearing is not beam index.* The driver spreads however many returns it obtained
evenly over the full circle, emitting ``angle_increment = 2*pi/(beam_count+1)``.
Beam ``i`` therefore has no fixed bearing across scans, and beam counts have been
observed from 396 to 404 within single runs. Every cross-scan comparison in this
module aligns by physical bearing computed from each scan's own increment.

*No-return is not dropout.* A beam reporting no return may be aimed at open space
or at a surface beyond the sensor's usable range, which is correct behaviour and
not a sensor fault. Only a bearing where a stable in-range surface is normally
present can support a stochastic missing-return estimate, so the two quantities
are computed separately and named differently throughout.

*Repeatability is not accuracy.* Comparing a stationary sensor's repeated
observations of the same surface measures its spread, not its distance from
truth. No ground-truth distance exists in this data, so no accuracy figure is
produced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

FULL_TURN_RAD = 2.0 * math.pi

#: Observed no-return encodings. The driver does not use ``inf``; it emits either
#: ``NaN`` or exactly ``0.0``. These are counted separately because they are
#: distinct driver codes and may not share a physical cause.
NO_RETURN_NAN = "nan"
NO_RETURN_ZERO = "zero"


class CharacterizationError(ValueError):
    """Raised when scan geometry is structurally unusable."""


@dataclass(frozen=True, slots=True)
class ScanRecord:
    """One decoded scan, reduced to what characterization needs.

    ``stamp_ns`` is the header stamp, not the bag receive time: header order is
    what any online consumer would have seen, and the two differ by about 1.4 ms
    on this sensor.
    """

    stamp_ns: int
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray

    @property
    def beam_count(self) -> int:
        return int(self.ranges.size)


@dataclass(frozen=True, slots=True)
class MotionFreeInterval:
    """A half-open ``[start_ns, end_ns)`` window verified to contain no motion."""

    start_ns: int
    end_ns: int
    sample_count: int
    max_abs_linear_mps: float
    max_abs_angular_radps: float

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9


def expected_angle_increment(beam_count: int) -> float:
    """The increment this driver is expected to emit for ``beam_count`` returns.

    The ``+1`` is the whole point: it is what makes the emitted arc fall one step
    short of a closed circle, and assuming ``2*pi/beam_count`` instead misplaces
    the far beam by a full step.
    """

    if beam_count < 1:
        raise CharacterizationError(f"beam_count must be positive, got {beam_count}")
    return FULL_TURN_RAD / (beam_count + 1)


def beam_bearings(angle_min: float, angle_increment: float, beam_count: int) -> np.ndarray:
    """Bearings of every beam in one scan, from that scan's own increment."""

    if beam_count < 1:
        raise CharacterizationError(f"beam_count must be positive, got {beam_count}")
    if not math.isfinite(angle_min) or not math.isfinite(angle_increment):
        raise CharacterizationError("angle_min and angle_increment must be finite")
    if angle_increment <= 0.0:
        raise CharacterizationError(f"angle_increment must be positive, got {angle_increment}")
    return angle_min + np.arange(beam_count, dtype=np.float64) * angle_increment


def wrap_to_turn(bearings: np.ndarray) -> np.ndarray:
    """Normalize bearings into ``[0, 2*pi)`` so scans can be compared."""

    return np.mod(np.asarray(bearings, dtype=np.float64), FULL_TURN_RAD)


def increment_convention_residual(beam_count: int, angle_increment: float) -> dict[str, float]:
    """Compare an observed increment against the ``2*pi/(beams+1)`` relation."""

    expected = expected_angle_increment(beam_count)
    absolute = abs(angle_increment - expected)
    return {
        "beam_count": float(beam_count),
        "observed_increment_rad": float(angle_increment),
        "expected_increment_rad": float(expected),
        "absolute_error_rad": float(absolute),
        "relative_error": float(absolute / expected),
    }


def sector_indices(bearings: np.ndarray, sector_count: int) -> np.ndarray:
    """Assign each bearing to one of ``sector_count`` equal physical sectors.

    Binning by physical angle is what makes scans with different beam counts
    comparable at all. Sector 0 is centred on the scan frame's zero bearing.
    """

    if sector_count < 1:
        raise CharacterizationError(f"sector_count must be positive, got {sector_count}")
    wrapped = wrap_to_turn(bearings)
    width = FULL_TURN_RAD / sector_count
    # Offset by half a sector so sector 0 brackets zero rather than starting at it.
    shifted = np.mod(wrapped + 0.5 * width, FULL_TURN_RAD)
    return np.floor(shifted / width).astype(np.int64) % sector_count


def classify_returns(record: ScanRecord) -> dict[str, np.ndarray]:
    """Split one scan's returns into valid and each distinct no-return code.

    ``range_max`` is deliberately not used as an upper validity bound. This
    driver declares ``range_max = 100.0 m``, far beyond anything the LDS-03 can
    actually measure, so testing against it would classify nothing and imply a
    reach the sensor does not have.
    """

    ranges = np.asarray(record.ranges, dtype=np.float64)
    is_nan = np.isnan(ranges)
    is_zero = ranges == 0.0
    is_inf = np.isinf(ranges)
    below_declared_min = np.zeros_like(is_nan)
    finite_positive = np.isfinite(ranges) & (ranges > 0.0)
    if math.isfinite(record.range_min):
        below_declared_min = finite_positive & (ranges < record.range_min)
    valid = finite_positive & ~below_declared_min
    return {
        "valid": valid,
        NO_RETURN_NAN: is_nan,
        NO_RETURN_ZERO: is_zero,
        "inf": is_inf,
        "below_declared_range_min": below_declared_min,
    }


def detect_range_quantization(
    records: Sequence[ScanRecord],
    candidate_steps_m: Sequence[float] = (0.001, 0.002, 0.005, 0.01),
    *,
    tolerance_steps: float = 0.01,
) -> dict[str, object]:
    """Find the lattice, if any, that reported ranges fall on.

    This matters more than it looks. If ranges are reported on a lattice, every
    robust spread statistic derived from them is *also* lattice-valued — a median
    absolute deviation can only land on a multiple of the step. An estimate that
    looks impressively repeatable across independent recordings may simply be
    reporting the same integer number of steps each time, so the resolution
    limit has to be stated alongside any noise figure.

    Values are compared in float64 but were transported as float32, so the
    tolerance absorbs representation error rather than real off-lattice spread.
    """

    pooled = np.concatenate(
        [
            np.asarray(record.ranges, dtype=np.float64)[
                np.isfinite(record.ranges) & (np.asarray(record.ranges) > 0.0)
            ]
            for record in records
        ]
    ) if records else np.empty(0, dtype=np.float64)

    if pooled.size == 0:
        return {"sample_count": 0, "detected_step_m": None}

    results = []
    for step in candidate_steps_m:
        offsets = np.abs(pooled / step - np.round(pooled / step))
        results.append(
            {
                "step_m": float(step),
                "max_offset_steps": float(np.max(offsets)),
                "consistent": bool(np.max(offsets) <= tolerance_steps),
            }
        )
    consistent = [entry for entry in results if entry["consistent"]]
    detected = max((entry["step_m"] for entry in consistent), default=None)

    distinct = np.unique(pooled)
    gaps = np.diff(distinct)
    gaps = gaps[gaps > 1e-9]
    return {
        "sample_count": int(pooled.size),
        "distinct_value_count": int(distinct.size),
        "min_gap_between_distinct_values_m": float(np.min(gaps)) if gaps.size else None,
        "median_gap_between_distinct_values_m": float(np.median(gaps)) if gaps.size else None,
        "candidates": results,
        "detected_step_m": detected,
        "implication": (
            None
            if detected is None
            else (
                "reported ranges lie on a lattice; MAD and any robust sigma derived "
                "from it are quantized to multiples of this step, so the repeatability "
                "estimate cannot resolve differences finer than roughly one step"
            )
        ),
    }


def interval_statistics(
    stamps_ns: Sequence[int],
    gap_thresholds_s: Sequence[float],
) -> dict[str, object]:
    """Header-interval distribution and counts above fixed documented thresholds.

    Reports the tail explicitly. A mean interval alone hides exactly the stalls
    that matter for an association contract.
    """

    ordered = np.asarray(sorted(int(value) for value in stamps_ns), dtype=np.int64)
    if ordered.size < 2:
        return {
            "sample_count": int(ordered.size),
            "interval_count": 0,
            "intervals_s": None,
            "gaps_above_threshold": {f"{t:g}": 0 for t in gap_thresholds_s},
        }
    intervals = np.diff(ordered) / 1e9
    return {
        "sample_count": int(ordered.size),
        "interval_count": int(intervals.size),
        "intervals_s": {
            "min": float(np.min(intervals)),
            "median": float(np.median(intervals)),
            "mean": float(np.mean(intervals)),
            "p95": float(np.quantile(intervals, 0.95)),
            "p99": float(np.quantile(intervals, 0.99)),
            "max": float(np.max(intervals)),
        },
        "implied_rate_hz": float(1.0 / np.median(intervals)),
        "gaps_above_threshold": {
            f"{threshold:g}": int(np.sum(intervals > threshold))
            for threshold in gap_thresholds_s
        },
    }


def pooled_interval_statistics(
    groups: Sequence[Sequence[int]],
    gap_thresholds_s: Sequence[float],
) -> dict[str, object]:
    """Pool header intervals across recordings without spanning their boundaries.

    Concatenating stamps from separate bags and differencing the result would
    invent one enormous interval per boundary — a gap that no sensor ever
    experienced. Differences are therefore taken strictly within a recording and
    only the resulting intervals are pooled.
    """

    collected: list[np.ndarray] = []
    contributing = 0
    for group in groups:
        ordered = np.asarray(sorted(int(value) for value in group), dtype=np.int64)
        if ordered.size < 2:
            continue
        collected.append(np.diff(ordered) / 1e9)
        contributing += 1
    if not collected:
        return {
            "group_count": len(groups),
            "contributing_group_count": 0,
            "interval_count": 0,
            "intervals_s": None,
            "gaps_above_threshold": {f"{t:g}": 0 for t in gap_thresholds_s},
        }

    intervals = np.concatenate(collected)
    return {
        "group_count": len(groups),
        "contributing_group_count": contributing,
        "interval_count": int(intervals.size),
        "intervals_s": {
            "min": float(np.min(intervals)),
            "median": float(np.median(intervals)),
            "mean": float(np.mean(intervals)),
            "p95": float(np.quantile(intervals, 0.95)),
            "p99": float(np.quantile(intervals, 0.99)),
            "max": float(np.max(intervals)),
        },
        "implied_rate_hz": float(1.0 / np.median(intervals)),
        "gaps_above_threshold": {
            f"{threshold:g}": int(np.sum(intervals > threshold))
            for threshold in gap_thresholds_s
        },
    }


def find_motion_free_intervals(
    stamps_ns: Sequence[int],
    linear_mps: Sequence[float],
    angular_radps: Sequence[float],
    *,
    linear_tolerance_mps: float,
    angular_tolerance_radps: float,
    min_duration_s: float,
) -> list[MotionFreeInterval]:
    """Maximal windows in which every odometry sample reports no motion.

    Stationarity is established from recorded evidence rather than from a bag's
    name. A run called "stationary" may still contain a startup transient, and a
    moving run generally contains genuinely motion-free stretches that are
    perfectly usable.
    """

    stamps = np.asarray([int(value) for value in stamps_ns], dtype=np.int64)
    linear = np.asarray(linear_mps, dtype=np.float64)
    angular = np.asarray(angular_radps, dtype=np.float64)
    if not (stamps.size == linear.size == angular.size):
        raise CharacterizationError("odometry stamp/linear/angular lengths must match")
    if stamps.size == 0:
        return []

    order = np.argsort(stamps, kind="stable")
    stamps, linear, angular = stamps[order], linear[order], angular[order]
    still = (
        np.isfinite(linear)
        & np.isfinite(angular)
        & (np.abs(linear) <= linear_tolerance_mps)
        & (np.abs(angular) <= angular_tolerance_radps)
    )

    intervals: list[MotionFreeInterval] = []
    start = None
    for index in range(still.size + 1):
        inside = bool(still[index]) if index < still.size else False
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            stop = index - 1
            duration = (stamps[stop] - stamps[start]) / 1e9
            if stop > start and duration >= min_duration_s:
                intervals.append(
                    MotionFreeInterval(
                        start_ns=int(stamps[start]),
                        end_ns=int(stamps[stop]),
                        sample_count=int(stop - start + 1),
                        max_abs_linear_mps=float(np.max(np.abs(linear[start : stop + 1]))),
                        max_abs_angular_radps=float(np.max(np.abs(angular[start : stop + 1]))),
                    )
                )
            start = None
    return intervals


#: Evidence basis for an accepted motion-free interval, strongest first.
EVIDENCE_ODOMETRY_AND_COMMAND = "odometry_and_zero_command"
EVIDENCE_ODOMETRY_ONLY = "odometry_only_no_command_topic_recorded"


def confirm_intervals_by_zero_command(
    intervals: Sequence[MotionFreeInterval],
    stamps_ns: Sequence[int],
    linear_mps: Sequence[float],
    angular_radps: Sequence[float],
    *,
    command_topic_recorded: bool = True,
) -> tuple[list[MotionFreeInterval], list[dict[str, object]], str]:
    """Keep only intervals in which every recorded command was exactly zero.

    Odometry alone can under-report motion, so an interval is accepted as
    motion-free only when the independent command record agrees. Commands are
    compared against exact zero rather than a tolerance: this platform's command
    path emits exact zeros when stopped, so any nonzero value is a real command
    rather than noise.

    ``command_topic_recorded=False`` covers a genuinely different case, not a
    missing measurement. A recording made with no velocity publisher present at
    all cannot contain a command, and its absence is stronger evidence of
    stillness than a stream of zeros would be. Those intervals are accepted, but
    the returned evidence label says which basis was used so the distinction
    survives into the artifact instead of being flattened.

    Returns the confirmed intervals, a rejection record for the rest, and the
    evidence label that applies to the confirmed set.
    """

    if not command_topic_recorded:
        return list(intervals), [], EVIDENCE_ODOMETRY_ONLY

    stamps = np.asarray([int(value) for value in stamps_ns], dtype=np.int64)
    linear = np.asarray(linear_mps, dtype=np.float64)
    angular = np.asarray(angular_radps, dtype=np.float64)
    if not (stamps.size == linear.size == angular.size):
        raise CharacterizationError("command stamp/linear/angular lengths must match")

    confirmed: list[MotionFreeInterval] = []
    rejected: list[dict[str, object]] = []
    for interval in intervals:
        inside = (stamps >= interval.start_ns) & (stamps < interval.end_ns)
        if not np.any(inside):
            rejected.append(
                {
                    "start_ns": interval.start_ns,
                    "duration_s": interval.duration_s,
                    "reason": "no command sample inside interval",
                }
            )
            continue
        moving = (linear[inside] != 0.0) | (angular[inside] != 0.0)
        if np.any(moving):
            rejected.append(
                {
                    "start_ns": interval.start_ns,
                    "duration_s": interval.duration_s,
                    "reason": "nonzero command inside interval",
                    "nonzero_command_count": int(np.sum(moving)),
                    "command_sample_count": int(np.sum(inside)),
                }
            )
            continue
        confirmed.append(interval)
    return confirmed, rejected, EVIDENCE_ODOMETRY_AND_COMMAND


def no_return_occupancy_by_sector(
    records: Sequence[ScanRecord],
    sector_count: int,
) -> dict[str, object]:
    """Occupancy of each no-return code per physical angular sector.

    Named occupancy rather than dropout on purpose. A high figure here is
    consistent with open space or an out-of-reach surface, and on its own is not
    evidence of a sensor fault.
    """

    totals = np.zeros(sector_count, dtype=np.int64)
    valid = np.zeros(sector_count, dtype=np.int64)
    nan_counts = np.zeros(sector_count, dtype=np.int64)
    zero_counts = np.zeros(sector_count, dtype=np.int64)

    for record in records:
        bearings = beam_bearings(record.angle_min, record.angle_increment, record.beam_count)
        sectors = sector_indices(bearings, sector_count)
        classes = classify_returns(record)
        np.add.at(totals, sectors, 1)
        np.add.at(valid, sectors, classes["valid"].astype(np.int64))
        np.add.at(nan_counts, sectors, classes[NO_RETURN_NAN].astype(np.int64))
        np.add.at(zero_counts, sectors, classes[NO_RETURN_ZERO].astype(np.int64))

    observed = totals > 0
    occupancy = np.zeros(sector_count, dtype=np.float64)
    np.divide(totals - valid, totals, out=occupancy, where=observed)
    return {
        "sector_count": int(sector_count),
        "sector_width_rad": float(FULL_TURN_RAD / sector_count),
        "observations": totals.tolist(),
        "valid": valid.tolist(),
        "no_return_nan": nan_counts.tolist(),
        "no_return_zero": zero_counts.tolist(),
        "no_return_occupancy": [
            float(value) if flag else None
            for value, flag in zip(occupancy, observed, strict=True)
        ],
        "aggregate": {
            "observations": int(totals.sum()),
            "valid": int(valid.sum()),
            "no_return_nan": int(nan_counts.sum()),
            "no_return_zero": int(zero_counts.sum()),
            "no_return_occupancy": (
                float((totals.sum() - valid.sum()) / totals.sum()) if totals.sum() else None
            ),
        },
    }


def sector_range_series(
    records: Sequence[ScanRecord],
    sector_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-scan, per-sector median valid range, validity, and beam coverage.

    Returns ``(values, valid, covered)`` each shaped ``(len(records),
    sector_count)``. Taking the median within a sector collapses the one-or-two
    beams that land there without letting beam-count drift change the comparison.

    ``covered`` is reported separately and is not a formality. If a sector
    received no beam at all in some scan, treating that as a missing return
    would manufacture dropout out of angular binning. Every rate downstream
    therefore divides by coverage, not by scan count.
    """

    values = np.full((len(records), sector_count), np.nan, dtype=np.float64)
    valid = np.zeros((len(records), sector_count), dtype=bool)
    covered = np.zeros((len(records), sector_count), dtype=bool)
    for row, record in enumerate(records):
        bearings = beam_bearings(record.angle_min, record.angle_increment, record.beam_count)
        sectors = sector_indices(bearings, sector_count)
        good = classify_returns(record)["valid"]
        ranges = np.asarray(record.ranges, dtype=np.float64)
        covered[row, np.unique(sectors)] = True
        for sector in np.unique(sectors[good]):
            selected = ranges[good & (sectors == sector)]
            if selected.size:
                values[row, sector] = float(np.median(selected))
                valid[row, sector] = True
    return values, valid, covered


def stable_bearing_eligibility(
    values: np.ndarray,
    valid: np.ndarray,
    covered: np.ndarray,
    *,
    min_valid_fraction: float,
    min_observations: int,
    min_range_m: float,
    max_range_m: float,
    max_neighbor_step_m: float,
    max_mad_m: float,
    max_half_split_drift_m: float,
) -> dict[str, object]:
    """Select sectors whose surface is stable enough to measure repeatability.

    Five exclusions, each removing a different way the estimate would be wrong:
    a sector must be *usually* returning (or its spread is really a mixture of
    hit and miss), within a plausible range band, not adjacent to a depth
    discontinuity (an object edge makes the sensor straddle two surfaces and
    reads as enormous noise), not drifting, and not wildly dispersed.

    Drift is tested separately from dispersion on purpose. Thresholding spread
    alone would be circular — it discards sectors for having exactly the
    property being measured — so a first-half versus second-half median shift
    carries the "something in the scene moved" test, while the dispersion cap is
    left loose and only catches bimodal straddling.

    The valid fraction is taken over scans that actually placed a beam in the
    sector, so angular binning cannot masquerade as sensor behaviour.
    """

    scan_count, sector_count = values.shape
    counts = valid.sum(axis=0)
    coverage = covered.sum(axis=0)
    fractions = np.zeros(sector_count, dtype=np.float64)
    np.divide(counts, coverage, out=fractions, where=coverage > 0)

    medians = np.full(sector_count, np.nan, dtype=np.float64)
    mads = np.full(sector_count, np.nan, dtype=np.float64)
    drift = np.full(sector_count, np.nan, dtype=np.float64)
    midpoint = scan_count // 2
    for sector in range(sector_count):
        column = values[valid[:, sector], sector]
        if column.size:
            median = float(np.median(column))
            medians[sector] = median
            mads[sector] = float(np.median(np.abs(column - median)))
        early = values[: midpoint, sector][valid[: midpoint, sector]]
        late = values[midpoint :, sector][valid[midpoint :, sector]]
        if early.size and late.size:
            drift[sector] = abs(float(np.median(late)) - float(np.median(early)))

    with np.errstate(invalid="ignore"):
        enough = (fractions >= min_valid_fraction) & (counts >= min_observations)
        in_band = (medians >= min_range_m) & (medians <= max_range_m)
        settled = np.isfinite(drift) & (drift <= max_half_split_drift_m)
        steady = (mads <= max_mad_m) & settled

    previous = np.roll(medians, 1)
    following = np.roll(medians, -1)
    with np.errstate(invalid="ignore"):
        step = np.maximum(np.abs(medians - previous), np.abs(medians - following))
        interior = step <= max_neighbor_step_m
    interior &= np.isfinite(previous) & np.isfinite(following)

    eligible = enough & in_band & steady & interior & np.isfinite(medians) & (coverage > 0)
    return {
        "eligible": eligible,
        "valid_fraction": fractions,
        "coverage_count": coverage,
        "median_range_m": medians,
        "mad_m": mads,
        "half_split_drift_m": drift,
        "excluded_counts": {
            "insufficient_returns": int(np.sum(~enough)),
            "outside_range_band": int(np.sum(enough & ~in_band)),
            "unstable_surface": int(np.sum(enough & in_band & ~steady)),
            "angular_discontinuity": int(np.sum(enough & in_band & steady & ~interior)),
        },
        "eligible_sector_count": int(np.sum(eligible)),
    }


def robust_repeatability(
    values: np.ndarray,
    valid: np.ndarray,
    eligible: np.ndarray,
    *,
    range_bin_edges_m: Sequence[float] | None = None,
    min_bin_samples: int = 200,
) -> dict[str, object]:
    """Spread of repeated stationary observations about each sector's median.

    This is repeatability, not accuracy: it says how tightly the sensor repeats
    itself, and nothing about whether the reported distance is correct. Robust
    statistics are used because a single straddled edge produces outliers that
    would dominate a standard deviation.
    """

    residuals: list[np.ndarray] = []
    medians_for_residual: list[np.ndarray] = []
    for sector in np.flatnonzero(eligible):
        column = values[valid[:, sector], sector]
        if column.size < 2:
            continue
        median = float(np.median(column))
        residuals.append(column - median)
        medians_for_residual.append(np.full(column.size, median, dtype=np.float64))

    if not residuals:
        return {
            "sample_count": 0,
            "overall": None,
            "range_binned": None,
            "note": "no eligible sector supplied at least two valid observations",
        }

    flat = np.concatenate(residuals)
    reference = np.concatenate(medians_for_residual)
    summary = _residual_summary(flat)

    binned: list[dict[str, object]] | None = None
    if range_bin_edges_m is not None:
        binned = []
        edges = [float(edge) for edge in range_bin_edges_m]
        for low, high in zip(edges, edges[1:], strict=False):
            selected = flat[(reference >= low) & (reference < high)]
            if selected.size < min_bin_samples:
                binned.append(
                    {
                        "range_low_m": low,
                        "range_high_m": high,
                        "sample_count": int(selected.size),
                        "summary": None,
                        "note": "below min_bin_samples; not reported",
                    }
                )
                continue
            binned.append(
                {
                    "range_low_m": low,
                    "range_high_m": high,
                    "sample_count": int(selected.size),
                    "summary": _residual_summary(selected),
                }
            )

    return {
        "sample_count": int(flat.size),
        "contributing_sector_count": int(len(residuals)),
        "overall": summary,
        "range_binned": binned,
        "interpretation": "repeatability about each sector's own median; not absolute accuracy",
    }


def _residual_summary(residuals: np.ndarray) -> dict[str, float]:
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    return {
        "median_residual_m": median,
        "mad_m": mad,
        "robust_sigma_m": float(1.4826 * mad),
        "p95_abs_residual_m": float(np.quantile(np.abs(residuals), 0.95)),
        "max_abs_residual_m": float(np.max(np.abs(residuals))),
        "sample_count": int(residuals.size),
    }


def stochastic_missing_return(
    valid: np.ndarray,
    covered: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, object]:
    """Missing-return probability restricted to bearings with a stable surface.

    This is the only place a dropout-like number is legitimate. Restricting to
    eligible sectors is what separates "the sensor intermittently failed to
    return from a surface that is there" from "there was nothing to hit".
    """

    chosen = np.flatnonzero(eligible)
    if chosen.size == 0:
        return {"eligible_sector_count": 0, "estimate": None, "note": "no eligible sector"}

    hits = valid[:, chosen].sum(axis=0).astype(np.float64)
    looks = covered[:, chosen].sum(axis=0).astype(np.float64)
    if not np.all(looks > 0):
        raise CharacterizationError("eligible sector reported zero beam coverage")
    per_sector = 1.0 - hits / looks
    return {
        "eligible_sector_count": int(chosen.size),
        "observation_count": int(looks.sum()),
        "estimate": {
            "pooled_probability": float(1.0 - hits.sum() / looks.sum()),
            "per_sector_median": float(np.median(per_sector)),
            "per_sector_p95": float(np.quantile(per_sector, 0.95)),
            "per_sector_max": float(np.max(per_sector)),
        },
        "scope": "stable in-range surfaces only; not comparable to no-return occupancy",
    }
