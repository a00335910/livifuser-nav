"""Run-level invariants that must hold across a whole recording.

These are distinct from per-sample validation in :mod:`livifuser_nav.decode`:
they check properties of the *run*, so a fault here invalidates every sample
rather than one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def count_timestamp_regressions(stamps: Sequence[int]) -> int:
    """Count adjacent backward steps in arrival order.

    This counts regression *events*, not displaced positions. Sorting-based
    counts overstate badly: a single early element out of place displaces every
    later position, so one fault can read as hundreds.
    """

    return sum(
        1
        for earlier, later in zip(stamps, stamps[1:], strict=False)
        if later < earlier
    )


@dataclass(frozen=True, slots=True)
class AngularFrame:
    """The parameters that fix the bearing of each beam index.

    Without these the stored range array cannot be turned into the locked
    ``[r, sin(theta), cos(theta), validity]`` token form at all, so they are part
    of the dataset contract rather than optional metadata.

    Beam count and angle increment are deliberately excluded, because on this
    scanner they covary: the driver spreads however many returns it obtained
    evenly over the full circle, emitting ``angle_increment = 2*pi/(beams+1)``.
    So beam ``i`` does **not** have a fixed bearing across scans, and a single
    global bearing table would be wrong by up to ~0.9 degrees on the far side.
    Bearings must be computed per scan from that scan's own increment.
    """

    angle_min: float
    angle_max: float
    range_min: float
    range_max: float

    def bearing_rad(self, beam_index: int, angle_increment: float) -> float:
        """Bearing of one beam, using that scan's own increment."""

        return self.angle_min + beam_index * angle_increment

    def as_manifest(self) -> dict[str, object]:
        return {
            "angle_min_rad": self.angle_min,
            "angle_max_rad": self.angle_max,
            "range_min_m": self.range_min,
            "range_max_m": self.range_max,
        }


@dataclass(frozen=True, slots=True)
class GeometryReport:
    """Whether the angular frame held constant, plus observed beam-count jitter."""

    frame: AngularFrame | None
    frame_variants: tuple[AngularFrame, ...]
    beam_counts: tuple[int, ...] = ()
    angle_increments: tuple[float, ...] = ()

    @property
    def is_constant(self) -> bool:
        """True when one angular frame explains the whole run."""

        return len(self.frame_variants) == 1

    @property
    def max_beam_count(self) -> int:
        return max(self.beam_counts, default=0)

    def as_manifest(self) -> dict[str, object]:
        return {
            "angular_frame_constant": self.is_constant,
            "angular_frame_variant_count": len(self.frame_variants),
            "angular_frame": None if self.frame is None else self.frame.as_manifest(),
            "angular_frame_variants": (
                [variant.as_manifest() for variant in self.frame_variants]
                if not self.is_constant
                else []
            ),
            "beam_counts_observed": list(self.beam_counts),
            "max_beam_count": self.max_beam_count,
            "beam_count_varies": len(self.beam_counts) > 1,
            "angle_increments_observed_rad": list(self.angle_increments),
            "angle_increment_varies": len(self.angle_increments) > 1,
            "max_bearing_spread_deg": self.max_bearing_spread_deg,
            "note": (
                "Bearing of beam i is angle_min + i * angle_increment, using THAT "
                "scan's increment. Increment covaries with beam count on this "
                "scanner, so a single global bearing table is invalid; per-scan "
                "increment and beam count are stored alongside the ranges."
            ),
        }

    @property
    def max_bearing_spread_deg(self) -> float:
        """Worst-case bearing disagreement at the far beam across increments."""

        if len(self.angle_increments) < 2 or not self.beam_counts:
            return 0.0
        far_index = self.max_beam_count - 1
        spread = (max(self.angle_increments) - min(self.angle_increments)) * far_index
        return math.degrees(spread)


def scan_geometry_report(payload_data: Sequence[dict[str, object]]) -> GeometryReport:
    """Collect the distinct angular frames and beam counts present in a run.

    A change in the angular frame (origin, span, or range limits) means the run
    cannot be tokenized coherently. Variation in beam count and its covarying
    increment is normal on this scanner and is reported rather than rejected.
    """

    variants: list[AngularFrame] = []
    seen: set[tuple[float, ...]] = set()
    beam_counts: set[int] = set()
    increments: set[float] = set()

    for data in payload_data:
        beam_counts.add(int(data["beam_count"]))  # type: ignore[arg-type]
        increments.add(float(data["angle_increment"]))  # type: ignore[arg-type]
        candidate = AngularFrame(
            angle_min=float(data["angle_min"]),  # type: ignore[arg-type]
            angle_max=float(data["angle_max"]),  # type: ignore[arg-type]
            range_min=float(data["range_min"]),  # type: ignore[arg-type]
            range_max=float(data["range_max"]),  # type: ignore[arg-type]
        )
        key = (
            candidate.angle_min,
            candidate.angle_max,
            candidate.range_min,
            candidate.range_max,
        )
        if key not in seen:
            seen.add(key)
            variants.append(candidate)

    return GeometryReport(
        frame=variants[0] if len(variants) == 1 else None,
        frame_variants=tuple(variants),
        beam_counts=tuple(sorted(beam_counts)),
        angle_increments=tuple(sorted(increments)),
    )


def _normalized(quaternion: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in quaternion)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("quaternion must have non-zero norm")
    return tuple(value / norm for value in values)


@dataclass(frozen=True, slots=True)
class TransformComparison:
    """Numerical agreement between a recorded transform and the accepted one."""

    translation_error_m: float
    rotation_error_rad: float
    translation_tolerance_m: float
    rotation_tolerance_rad: float

    @property
    def matches(self) -> bool:
        return (
            self.translation_error_m <= self.translation_tolerance_m
            and self.rotation_error_rad <= self.rotation_tolerance_rad
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "translation_error_mm": self.translation_error_m * 1000.0,
            "rotation_error_deg": math.degrees(self.rotation_error_rad),
            "translation_tolerance_mm": self.translation_tolerance_m * 1000.0,
            "rotation_tolerance_deg": math.degrees(self.rotation_tolerance_rad),
        }


def compare_transform(
    recorded_translation: Sequence[float],
    recorded_quaternion_xyzw: Sequence[float],
    accepted_translation: Sequence[float],
    accepted_quaternion_xyzw: Sequence[float],
    *,
    translation_tolerance_m: float = 0.002,
    rotation_tolerance_rad: float = math.radians(0.5),
) -> TransformComparison:
    """Compare a live transform against the accepted calibration numerically.

    Checking that the TF frame *names* exist proves only that something is
    publishing; it does not prove the published numbers are the accepted ones.

    Quaternion sign is ignored because ``q`` and ``-q`` are the same rotation, so
    the comparison uses ``|dot|``.
    """

    translation_error = math.sqrt(
        sum(
            (float(recorded) - float(accepted)) ** 2
            for recorded, accepted in zip(
                recorded_translation, accepted_translation, strict=True
            )
        )
    )
    # Normalize first: quaternions stored in YAML are rounded and so are not
    # exactly unit norm, and acos amplifies that error near dot = 1.
    recorded_unit = _normalized(recorded_quaternion_xyzw)
    accepted_unit = _normalized(accepted_quaternion_xyzw)
    dot = abs(
        sum(
            recorded * accepted
            for recorded, accepted in zip(recorded_unit, accepted_unit, strict=True)
        )
    )
    rotation_error = 2.0 * math.acos(min(1.0, max(0.0, dot)))
    return TransformComparison(
        translation_error_m=translation_error,
        rotation_error_rad=rotation_error,
        translation_tolerance_m=translation_tolerance_m,
        rotation_tolerance_rad=rotation_tolerance_rad,
    )
